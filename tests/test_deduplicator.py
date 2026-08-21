"""
Phase 9 tests — app.services.deduplicator.

Most tests build FindingRecord instances directly from controlled
field values (via the _make_record helper below) for precise control
over evidence combinations. A few integration tests run the real
Phase 5 -> 7 -> 8 -> 9 pipeline end-to-end using existing fixtures.
Nothing here is presented as a real crash finding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.asan_parser import parse_asan_report                       # noqa: E402
from app.services.feature_extractor import (                                  # noqa: E402
    extract_features, CrashFeatures, FindingState, ReproductionStatus,
)
from app.services.stack_normalizer import normalize_crash_features_stack, NormalizedStack  # noqa: E402
from app.services.deduplicator import (                                       # noqa: E402
    deduplicate, compare_evidence, choose_representative,
    FindingRecord, DedupDecision, EvidenceComparison,
)

ASAN_FIXTURES = Path(__file__).parent / "fixtures" / "asan"


def _make_record(
    identifier,
    state=FindingState.CRASH,
    signature="SIG_A",
    version="1.0",
    error_type="heap-buffer-overflow",
    access_type="READ",
    function="foo",
    source_file="a.c",
    source_line=100,
    access_size=None,
    memory_region=None,
    reproduction_status=ReproductionStatus.NOT_ATTEMPTED,
    stack_depth=None,
    artifact_path=None,
):
    features = CrashFeatures(
        finding_id=identifier,
        finding_state=state,
        reproduction_status=reproduction_status,
        error_type=error_type,
        crash_type=error_type,
        access_type=access_type,
        access_size=access_size,
        faulting_function=function,
        source_file=source_file,
        source_line=source_line,
        memory_region=memory_region,
        stack_depth=stack_depth,
        artifact_path=artifact_path or identifier,
    )
    stack = NormalizedStack(stack_signature=signature, stack_signature_version=version) if signature else None
    return FindingRecord(features=features, stack=stack, identifier=identifier)


# ---------------------------------------------------------------------------
# TEST 1 — identical findings -> one group
# ---------------------------------------------------------------------------

def test_identical_findings_one_group():
    a = _make_record("A")
    b = _make_record("B")
    result = deduplicate([a, b])
    assert len(result.groups) == 1
    assert result.groups[0].count == 2
    assert set(result.groups[0].finding_ids) == {"A", "B"}


# ---------------------------------------------------------------------------
# TEST 2 — same stack signature + same ASan evidence -> one group
# ---------------------------------------------------------------------------

def test_same_signature_same_asan_evidence_one_group():
    a = _make_record("A", access_size=4, memory_region="HEAP")
    b = _make_record("B", access_size=4, memory_region="HEAP")
    result = deduplicate([a, b])
    assert len(result.groups) == 1
    assert "access_size=4" in result.groups[0].reason
    assert "memory_region=HEAP" in result.groups[0].reason


# ---------------------------------------------------------------------------
# TEST 3 — different stack signatures -> separate
# ---------------------------------------------------------------------------

def test_different_stack_signatures_separate():
    a = _make_record("A", signature="SIG_A")
    b = _make_record("B", signature="SIG_B")
    result = deduplicate([a, b])
    assert len(result.groups) == 2
    assert all(g.count == 1 for g in result.groups)


# ---------------------------------------------------------------------------
# TEST 4 — different ASan error types -> separate
# ---------------------------------------------------------------------------

def test_different_error_types_separate():
    a = _make_record("A", signature="SIG_ABC", error_type="heap-buffer-overflow")
    b = _make_record("B", signature="SIG_ABC", error_type="use-after-free")
    decision = compare_evidence(a, b)
    assert decision.decision == DedupDecision.SEPARATE
    assert "error_type" in decision.reason
    result = deduplicate([a, b])
    assert len(result.groups) == 2


# ---------------------------------------------------------------------------
# TEST 5 — READ vs WRITE -> separate
# ---------------------------------------------------------------------------

def test_read_vs_write_separate():
    a = _make_record("A", signature="SIG_ABC", access_type="READ")
    b = _make_record("B", signature="SIG_ABC", access_type="WRITE")
    decision = compare_evidence(a, b)
    assert decision.decision == DedupDecision.SEPARATE
    assert "access_type" in decision.reason


# ---------------------------------------------------------------------------
# TEST 6 — different faulting functions -> separate
# ---------------------------------------------------------------------------

def test_different_faulting_functions_separate():
    a = _make_record("A", signature="SIG_XYZ", function="foo")
    b = _make_record("B", signature="SIG_XYZ", function="bar")
    decision = compare_evidence(a, b)
    assert decision.decision == DedupDecision.SEPARATE
    assert "faulting_function" in decision.reason


# ---------------------------------------------------------------------------
# TEST 7 — CRASH vs HANG -> separate
# ---------------------------------------------------------------------------

def test_crash_vs_hang_separate_even_with_matching_signature():
    a = _make_record("A", state=FindingState.CRASH, signature="SIG_SAME")
    b = _make_record("B", state=FindingState.HANG, signature="SIG_SAME")
    decision = compare_evidence(a, b)
    assert decision.decision == DedupDecision.SEPARATE
    assert "finding_state" in decision.reason
    result = deduplicate([a, b])
    assert len(result.groups) == 2


# ---------------------------------------------------------------------------
# TEST 8 — CRASH vs NORMAL -> separate
# ---------------------------------------------------------------------------

def test_crash_vs_normal_separate():
    a = _make_record("A", state=FindingState.CRASH, signature="SIG_SAME")
    b = _make_record("B", state=FindingState.NORMAL, signature="SIG_SAME")
    decision = compare_evidence(a, b)
    assert decision.decision == DedupDecision.SEPARATE


# ---------------------------------------------------------------------------
# TEST 9 — missing access type -> deterministic behavior
# ---------------------------------------------------------------------------

def test_missing_access_type_is_insufficient_not_duplicate():
    a = _make_record("A", signature="SIG_ABC", access_type="READ")
    b = _make_record("B", signature="SIG_ABC", access_type=None)
    decision = compare_evidence(a, b)
    assert decision.evidence["access_type"] == EvidenceComparison.UNKNOWN
    assert decision.decision == DedupDecision.INSUFFICIENT_EVIDENCE
    result = deduplicate([a, b])
    assert len(result.groups) == 2  # not merged
    assert all(g.count == 1 for g in result.groups)


# ---------------------------------------------------------------------------
# TEST 10 — missing source location -> deterministic behavior
# ---------------------------------------------------------------------------

def test_missing_source_location_near_duplicate_example():
    """
    The exact "near-duplicate example" from the spec: same signature,
    same error type, same access type, but B is missing source_file
    and source_line. Must NOT be silently treated as a match.
    """
    a = _make_record("A", signature="SIG_ABC", source_file="a.c", source_line=100)
    b = _make_record("B", signature="SIG_ABC", source_file=None, source_line=None)
    decision = compare_evidence(a, b)
    assert decision.evidence["source_file"] == EvidenceComparison.UNKNOWN
    assert decision.evidence["source_line"] == EvidenceComparison.UNKNOWN
    assert decision.decision == DedupDecision.INSUFFICIENT_EVIDENCE
    assert "source_file" in decision.reason or "source_line" in decision.reason


# ---------------------------------------------------------------------------
# TEST 11 — missing stack signature -> safe behavior
# ---------------------------------------------------------------------------

def test_missing_stack_signature_never_merges():
    a = _make_record("A", signature=None)
    b = _make_record("B", signature=None)
    result = deduplicate([a, b])
    assert len(result.groups) == 2
    assert result.groups[0].reason.startswith("Independent finding — no stack signature")


# ---------------------------------------------------------------------------
# TEST 12 — different signature versions -> safe behavior
# ---------------------------------------------------------------------------

def test_different_signature_versions_not_merged():
    a = _make_record("A", signature="SIG_ABC", version="1.0")
    b = _make_record("B", signature="SIG_ABC", version="2.0")
    decision = compare_evidence(a, b)
    # Same raw signature string, but different version -> the pair is
    # not directly comparable -> CONFLICT (documented: never silently
    # assume comparability across signature algorithm versions).
    assert decision.evidence["stack_signature"] == EvidenceComparison.CONFLICT
    assert decision.decision == DedupDecision.SEPARATE
    result = deduplicate([a, b])
    assert len(result.groups) == 2


# ---------------------------------------------------------------------------
# TEST 13 — multiple identical findings -> one group with correct count
# ---------------------------------------------------------------------------

def test_multiple_identical_findings_correct_count():
    records = [_make_record(f"crash_{i:03d}") for i in range(17)]
    result = deduplicate(records)
    assert len(result.groups) == 1
    assert result.groups[0].count == 17
    assert len(result.groups[0].finding_ids) == 17


# ---------------------------------------------------------------------------
# TEST 14 — different artifact IDs but identical evidence -> same group
# ---------------------------------------------------------------------------

def test_different_artifact_ids_identical_evidence_same_group():
    a = _make_record("A", artifact_path="/afl/crashes/id:000001,sig:06")
    b = _make_record("B", artifact_path="/afl/crashes/id:000002,sig:06")
    result = deduplicate([a, b])
    assert len(result.groups) == 1
    assert set(result.groups[0].artifact_ids) == {
        "/afl/crashes/id:000001,sig:06", "/afl/crashes/id:000002,sig:06",
    }


# ---------------------------------------------------------------------------
# TEST 15 — reproduction failure preserved
# ---------------------------------------------------------------------------

def test_reproduction_failure_preserved_as_valid_crash():
    a = _make_record("A", state=FindingState.CRASH, reproduction_status=ReproductionStatus.NOT_REPRODUCED)
    result = deduplicate([a])
    assert len(result.groups) == 1
    assert result.groups[0].count == 1
    assert "A" in result.groups[0].finding_ids


# ---------------------------------------------------------------------------
# TEST 16 — reproduced finding selected deterministically as representative
# ---------------------------------------------------------------------------

def test_reproduced_finding_preferred_as_representative():
    a = _make_record("A", reproduction_status=ReproductionStatus.NOT_REPRODUCED)
    b = _make_record("B", reproduction_status=ReproductionStatus.REPRODUCED)
    rep = choose_representative([a, b])
    assert rep.identifier == "B"


def test_more_complete_evidence_preferred_when_reproduction_tied():
    a = _make_record("A", reproduction_status=ReproductionStatus.NOT_ATTEMPTED,
                      access_size=None, memory_region=None)
    b = _make_record("B", reproduction_status=ReproductionStatus.NOT_ATTEMPTED,
                      access_size=4, memory_region="HEAP")
    rep = choose_representative([a, b])
    assert rep.identifier == "B"


# ---------------------------------------------------------------------------
# TEST 17 — representative selection is stable
# ---------------------------------------------------------------------------

def test_representative_selection_is_stable_across_orderings():
    a = _make_record("A")
    b = _make_record("B")
    c = _make_record("C")
    rep_1 = choose_representative([a, b, c]).identifier
    rep_2 = choose_representative([c, a, b]).identifier
    rep_3 = choose_representative([b, c, a]).identifier
    assert rep_1 == rep_2 == rep_3


# ---------------------------------------------------------------------------
# TEST 18 — input order independence
# ---------------------------------------------------------------------------

def test_input_order_independence():
    a = _make_record("A")
    b = _make_record("B")
    c = _make_record("C", signature="SIG_DIFFERENT")

    result_1 = deduplicate([a, b, c])
    result_2 = deduplicate([c, a, b])
    result_3 = deduplicate([b, c, a])

    def summary(result):
        return sorted((g.group_id, tuple(sorted(g.finding_ids))) for g in result.groups)

    assert summary(result_1) == summary(result_2) == summary(result_3)


# ---------------------------------------------------------------------------
# TEST 19 — idempotence
# ---------------------------------------------------------------------------

def test_idempotence_same_input_twice():
    records = [_make_record("A"), _make_record("B"), _make_record("C", signature="SIG_X")]
    result_1 = deduplicate(records)
    result_2 = deduplicate(records)
    ids_1 = sorted((g.group_id, tuple(sorted(g.finding_ids))) for g in result_1.groups)
    ids_2 = sorted((g.group_id, tuple(sorted(g.finding_ids))) for g in result_2.groups)
    assert ids_1 == ids_2


def test_idempotence_rerunning_on_representatives_does_not_further_merge():
    records = [_make_record(f"crash_{i}") for i in range(5)] + \
              [_make_record(f"other_{i}", signature="SIG_OTHER") for i in range(3)]
    first_pass = deduplicate(records)
    representatives = [
        FindingRecord(
            features=next(r.features for r in records if r.identifier == g.representative_identifier),
            stack=next(r.stack for r in records if r.identifier == g.representative_identifier),
            identifier=g.representative_identifier,
        )
        for g in first_pass.groups
    ]
    second_pass = deduplicate(representatives)
    assert len(second_pass.groups) == len(first_pass.groups)
    assert all(g.count == 1 for g in second_pass.groups)


# ---------------------------------------------------------------------------
# TEST 20 — raw artifact references preserved
# ---------------------------------------------------------------------------

def test_all_artifacts_preserved_across_groups():
    records = [_make_record(f"id_{i}", artifact_path=f"/afl/crashes/{i}") for i in range(10)]
    result = deduplicate(records)
    assert result.total_input_count == 10
    assert result.total_grouped_count() == 10
    all_referenced = {fid for g in result.groups for fid in g.finding_ids}
    assert all_referenced == {f"id_{i}" for i in range(10)}


# ---------------------------------------------------------------------------
# TEST 21 — raw ASan evidence preserved (no mutation)
# ---------------------------------------------------------------------------

def test_raw_asan_report_not_mutated_by_deduplication():
    asan = parse_asan_report((ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text())
    features = extract_features(asan=asan)
    stack = normalize_crash_features_stack(features)
    record = FindingRecord(features=features, stack=stack, identifier="real1")

    original_raw = features.raw_asan_report
    deduplicate([record])
    assert features.raw_asan_report == original_raw


# ---------------------------------------------------------------------------
# TEST 22 — no source evidence is mutated
# ---------------------------------------------------------------------------

def test_crash_features_object_not_mutated():
    a = _make_record("A")
    b = _make_record("B")
    before_a = (a.features.finding_state, a.features.error_type, a.features.access_type)
    before_b = (b.features.finding_state, b.features.error_type, b.features.access_type)

    deduplicate([a, b])

    assert (a.features.finding_state, a.features.error_type, a.features.access_type) == before_a
    assert (b.features.finding_state, b.features.error_type, b.features.access_type) == before_b


# ---------------------------------------------------------------------------
# TEST 23 — explainability
# ---------------------------------------------------------------------------

def test_group_reason_names_actual_evidence_not_vague():
    a = _make_record("A", error_type="heap-buffer-overflow", access_type="WRITE", function="decode_mcu_block")
    b = _make_record("B", error_type="heap-buffer-overflow", access_type="WRITE", function="decode_mcu_block")
    result = deduplicate([a, b])
    reason = result.groups[0].reason
    assert "heap-buffer-overflow" in reason
    assert "WRITE" in reason
    assert "decode_mcu_block" in reason
    assert reason != "Similar crash."


def test_separate_reason_names_which_dimension_conflicted():
    a = _make_record("A", signature="SIG_S", error_type="heap-buffer-overflow")
    b = _make_record("B", signature="SIG_S", error_type="use-after-free")
    decision = compare_evidence(a, b)
    assert "error_type" in decision.reason
    assert "conflict" in decision.reason.lower()


# ---------------------------------------------------------------------------
# TEST 24 — conflicting sanitizer evidence prevents merging
# ---------------------------------------------------------------------------

def test_adversarial_same_signature_conflicting_error_type():
    a = _make_record("A", signature="ABC", error_type="heap-buffer-overflow", access_type="READ", function="foo")
    b = _make_record("B", signature="ABC", error_type="use-after-free", access_type="READ", function="foo")
    result = deduplicate([a, b])
    assert len(result.groups) == 2


# ---------------------------------------------------------------------------
# TEST 25 — conflicting access evidence prevents merging
# ---------------------------------------------------------------------------

def test_adversarial_same_signature_conflicting_access_type():
    a = _make_record("A", signature="ABC", error_type="heap-buffer-overflow", access_type="READ")
    b = _make_record("B", signature="ABC", error_type="heap-buffer-overflow", access_type="WRITE")
    result = deduplicate([a, b])
    assert len(result.groups) == 2


def test_adversarial_different_signature_same_everything_else():
    a = _make_record("A", signature="ABC", error_type="heap-buffer-overflow", function="foo")
    b = _make_record("B", signature="XYZ", error_type="heap-buffer-overflow", function="foo")
    result = deduplicate([a, b])
    assert len(result.groups) == 2


# ---------------------------------------------------------------------------
# TEST 26 — incomplete/malformed finding does not crash the deduplicator
# ---------------------------------------------------------------------------

def test_fully_empty_crash_features_does_not_crash():
    empty_features = CrashFeatures()  # every field at its default
    record = FindingRecord(features=empty_features, stack=None, identifier="empty1")
    result = deduplicate([record, _make_record("normal1")])
    assert result.total_input_count == 2
    assert result.total_grouped_count() == 2


def test_record_with_no_identifier_and_no_finding_id_still_works():
    features = CrashFeatures(finding_state=FindingState.CRASH, error_type="double-free")
    record = FindingRecord(features=features, stack=None, identifier=None)
    result = deduplicate([record])
    assert len(result.groups) == 1
    assert result.groups[0].finding_ids[0].startswith("content:")


# ---------------------------------------------------------------------------
# TEST 27 — deterministic group IDs
# ---------------------------------------------------------------------------

def test_group_id_deterministic_and_content_based():
    a = _make_record("A")
    b = _make_record("B")
    result_1 = deduplicate([a, b])
    result_2 = deduplicate([b, a])
    assert result_1.groups[0].group_id == result_2.groups[0].group_id

    # A group with different members must have a different id.
    c = _make_record("C", signature="SIG_DIFF")
    result_3 = deduplicate([a, b, c])
    ids_with_c = {g.group_id for g in result_3.groups}
    assert result_1.groups[0].group_id in ids_with_c  # {A,B} group id unchanged by C's presence


# ---------------------------------------------------------------------------
# TEST 28 — repeated execution gives identical output (determinism, broader)
# ---------------------------------------------------------------------------

def test_full_repeated_execution_identical_output():
    records = [_make_record(f"n{i}") for i in range(6)]
    results = [deduplicate(records) for _ in range(5)]
    signatures = [
        sorted((g.group_id, g.representative_identifier, tuple(sorted(g.finding_ids))) for g in r.groups)
        for r in results
    ]
    assert all(s == signatures[0] for s in signatures)


# ---------------------------------------------------------------------------
# TEST 29 — empty input
# ---------------------------------------------------------------------------

def test_empty_input():
    result = deduplicate([])
    assert result.groups == []
    assert result.total_input_count == 0
    assert result.total_grouped_count() == 0


# ---------------------------------------------------------------------------
# TEST 30 — single finding
# ---------------------------------------------------------------------------

def test_single_finding():
    result = deduplicate([_make_record("solo")])
    assert len(result.groups) == 1
    assert result.groups[0].count == 1
    assert result.groups[0].representative_identifier == "solo"


# ---------------------------------------------------------------------------
# Additional: transitivity safety proof (the key correctness property
# this whole design depends on) and real end-to-end integration
# ---------------------------------------------------------------------------

def test_transitivity_is_never_violated_even_with_partial_unknowns():
    """
    Worked counter-example proving why blocking dimensions require
    actual MATCH (not merely non-CONFLICT): if UNKNOWN were allowed to
    pass a blocking check, A~B and B~C could both look like duplicates
    while A and C directly conflict. With the actual implementation,
    B's missing `function` blocks BOTH A-B and B-C from ever reaching
    DUPLICATE, so no group can form that would need to be split.
    """
    a = _make_record("A", signature="S", function="foo")
    b = _make_record("B", signature="S", function=None)
    c = _make_record("C", signature="S", function="bar")

    assert compare_evidence(a, b).decision != DedupDecision.DUPLICATE
    assert compare_evidence(b, c).decision != DedupDecision.DUPLICATE
    assert compare_evidence(a, c).decision == DedupDecision.SEPARATE

    result = deduplicate([a, b, c])
    assert len(result.groups) == 3  # nothing incorrectly merged


def test_end_to_end_real_pipeline_integration():
    """Phase 5 -> 7 -> 8 -> 9, using real fixture ASan reports."""
    text = (ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text()

    asan_1 = parse_asan_report(text)
    features_1 = extract_features(asan=asan_1)
    stack_1 = normalize_crash_features_stack(features_1)

    asan_2 = parse_asan_report(text)  # same raw report, independently parsed
    features_2 = extract_features(asan=asan_2)
    stack_2 = normalize_crash_features_stack(features_2)

    record_1 = FindingRecord(features=features_1, stack=stack_1, identifier="pipeline_1")
    record_2 = FindingRecord(features=features_2, stack=stack_2, identifier="pipeline_2")

    result = deduplicate([record_1, record_2])
    assert len(result.groups) == 1
    assert result.groups[0].count == 2
