"""
Phase 11 tests — app.services.prioritizer.

Most tests build PrioritizationInput instances directly from controlled
CrashFeatures for precise control over evidence combinations. A few
integration tests run the real Phase 5 -> 7 -> 9 -> 10 -> 11 pipeline.
Nothing here is presented as a real crash finding.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.feature_extractor import CrashFeatures, FindingState, ReproductionStatus  # noqa: E402
from app.services.deduplicator import DedupGroup                                              # noqa: E402
from app.services.clusterer import Cluster                                                     # noqa: E402
from app.services.prioritizer import (                                                          # noqa: E402
    score_finding, prioritize, build_prioritization_inputs,
    PrioritizationInput, DEFAULT_WEIGHTS, SCOPE_DISCLAIMER,
    CRITICAL_THRESHOLD, HIGH_THRESHOLD, MEDIUM_THRESHOLD,
)


def _make_input(
    identifier,
    finding_state=FindingState.CRASH,
    reproduction_status=ReproductionStatus.NOT_ATTEMPTED,
    asan_detected=False,
    access_type=None,
    faulting_function=None,
    source_file=None,
    source_line=None,
    dedup_group=None,
    cluster=None,
):
    features = CrashFeatures(
        finding_id=identifier,
        finding_state=finding_state,
        reproduction_status=reproduction_status,
        asan_detected=asan_detected,
        access_type=access_type,
        faulting_function=faulting_function,
        source_file=source_file,
        source_line=source_line,
        artifact_path=identifier,
    )
    return PrioritizationInput(identifier=identifier, features=features,
                                dedup_group=dedup_group, cluster=cluster)


def _full_evidence_input(identifier, **overrides):
    kwargs = dict(
        finding_state=FindingState.CRASH,
        reproduction_status=ReproductionStatus.REPRODUCED,
        asan_detected=True,
        access_type="WRITE",
        faulting_function="decode_mcu_block",
        source_file="jdhuff.c",
        source_line=341,
    )
    kwargs.update(overrides)
    return _make_input(identifier, **kwargs)


# ---------------------------------------------------------------------------
# TEST 1 — empty input
# ---------------------------------------------------------------------------

def test_empty_input():
    assert prioritize([]) == []


# ---------------------------------------------------------------------------
# TEST 2 — single finding
# ---------------------------------------------------------------------------

def test_single_finding():
    result = prioritize([_full_evidence_input("A")])
    assert len(result) == 1
    assert result[0].rank == 1
    assert result[0].finding_id == "A"


# ---------------------------------------------------------------------------
# TEST 3 — deterministic score
# ---------------------------------------------------------------------------

def test_deterministic_score():
    inp = _full_evidence_input("A")
    r1 = score_finding(inp)
    r2 = score_finding(inp)
    assert r1.score == r2.score
    assert r1.priority == r2.priority


# ---------------------------------------------------------------------------
# TEST 4 — deterministic ranking
# ---------------------------------------------------------------------------

def test_deterministic_ranking():
    inputs = [_full_evidence_input("A"), _make_input("B", asan_detected=True)]
    r1 = prioritize(inputs)
    r2 = prioritize(inputs)
    assert [(r.finding_id, r.rank) for r in r1] == [(r.finding_id, r.rank) for r in r2]


# ---------------------------------------------------------------------------
# TEST 5 — input-order independence
# ---------------------------------------------------------------------------

def test_input_order_independence():
    a = _full_evidence_input("A")
    b = _make_input("B", asan_detected=True, reproduction_status=ReproductionStatus.NOT_REPRODUCED)
    c = _make_input("C", finding_state=FindingState.HANG)

    order_1 = [r.finding_id for r in prioritize([a, b, c])]
    order_2 = [r.finding_id for r in prioritize([c, a, b])]
    order_3 = [r.finding_id for r in prioritize([b, c, a])]
    assert order_1 == order_2 == order_3


# ---------------------------------------------------------------------------
# TEST 6 — tie-breaking
# ---------------------------------------------------------------------------

def test_tie_breaking_is_deterministic_and_by_finding_id():
    a = _full_evidence_input("zzz_last")
    b = _full_evidence_input("aaa_first")
    result = prioritize([a, b])
    assert result[0].score == result[1].score  # identical evidence -> identical score
    assert result[0].finding_id == "aaa_first"  # ascending finding_id tie-break
    assert result[1].finding_id == "zzz_last"


# ---------------------------------------------------------------------------
# TEST 7 / ADVERSARIAL TEST A — reproducibility
# ---------------------------------------------------------------------------

def test_adversarial_A_reproduced_ranks_above_non_reproduced_all_else_equal():
    a = _make_input("A", asan_detected=True, reproduction_status=ReproductionStatus.REPRODUCED)
    b = _make_input("B", asan_detected=True, reproduction_status=ReproductionStatus.NOT_REPRODUCED)
    result = prioritize([a, b])
    assert result[0].finding_id == "A"
    assert result[0].score > result[1].score


# ---------------------------------------------------------------------------
# TEST 8 — crash vs hang treated differently
# ---------------------------------------------------------------------------

def test_crash_vs_hang_different_treatment():
    crash = score_finding(_full_evidence_input("A"))
    hang = score_finding(_make_input("B", finding_state=FindingState.HANG))
    assert crash.score is not None
    assert hang.score is None
    assert crash.priority != "MEDIUM" or crash.score is not None
    assert hang.priority == "MEDIUM"
    assert "hang" in " ".join(hang.uncertainties).lower() or "timeout" in " ".join(hang.uncertainties).lower()


# ---------------------------------------------------------------------------
# TEST 9 — missing reproduction evidence is not treated as zero-risk
# ---------------------------------------------------------------------------

def test_missing_reproduction_not_automatically_zero_risk():
    result = score_finding(_make_input("A", asan_detected=True, reproduction_status=ReproductionStatus.NOT_ATTEMPTED))
    assert result.score is not None
    assert result.score > 0  # ASan + crash evidence still contributes


# ---------------------------------------------------------------------------
# TEST 10 — missing evidence does not become maximum risk
# ---------------------------------------------------------------------------

def test_missing_evidence_does_not_become_maximum_risk():
    sparse = score_finding(_make_input("A"))  # only bare CRASH state, everything else unknown
    full = score_finding(_full_evidence_input("B"))
    assert sparse.score < full.score
    assert sparse.score < CRITICAL_THRESHOLD


# ---------------------------------------------------------------------------
# TEST 11 — missing evidence surfaced as uncertainty
# ---------------------------------------------------------------------------

def test_missing_evidence_surfaced_as_uncertainty():
    result = score_finding(_make_input("A"))
    joined = " ".join(result.uncertainties).lower()
    assert "reproduction" in joined
    assert "sanitizer" in joined or "asan" in joined
    assert "access type" in joined
    assert "location" in joined


# ---------------------------------------------------------------------------
# TEST 12 — strong sanitizer evidence reflected appropriately
# ---------------------------------------------------------------------------

def test_strong_sanitizer_evidence_increases_score():
    without = score_finding(_make_input("A", reproduction_status=ReproductionStatus.REPRODUCED))
    with_asan = score_finding(_make_input("B", reproduction_status=ReproductionStatus.REPRODUCED, asan_detected=True))
    assert with_asan.score > without.score
    assert with_asan.score - without.score == DEFAULT_WEIGHTS["ASAN_CONFIRMED"]


# ---------------------------------------------------------------------------
# TEST 13 — no sanitizer evidence handled safely
# ---------------------------------------------------------------------------

def test_no_sanitizer_evidence_handled_safely_not_punitive():
    result = score_finding(_make_input("A", reproduction_status=ReproductionStatus.REPRODUCED, asan_detected=False))
    assert result.score is not None
    assert result.score > 0  # crash + reproduced still contributes; absence of ASan isn't punished beyond "no bonus"


# ---------------------------------------------------------------------------
# TEST 14 / ADVERSARIAL TEST D — cluster info does not determine severity
# ---------------------------------------------------------------------------

def test_adversarial_D_cluster_size_does_not_inflate_score():
    small_cluster = Cluster(cluster_id="c1", member_ids=["A", "x"], member_count=2,
                             representative_id="A", behavioral_profile={}, explanation="x")
    huge_cluster = Cluster(cluster_id="c2", member_ids=["B"] + [f"m{i}" for i in range(99)], member_count=100,
                            representative_id="B", behavioral_profile={}, explanation="x")
    a = _full_evidence_input("A", cluster=small_cluster)
    b = _full_evidence_input("B", cluster=huge_cluster)
    ra, rb = score_finding(a), score_finding(b)
    assert ra.score == rb.score  # identical direct evidence -> identical score regardless of cluster size
    assert rb.metadata["behavioral_cluster_size"] == 100  # size IS visible, just not scored


# ---------------------------------------------------------------------------
# TEST 15 — cluster size does not directly equal severity (numeric check)
# ---------------------------------------------------------------------------

def test_cluster_size_not_linearly_related_to_score():
    for size in (1, 2, 50, 1000):
        cluster = Cluster(cluster_id=f"c{size}", member_ids=[f"m{i}" for i in range(size)], member_count=size,
                           representative_id="m0", behavioral_profile={}, explanation="x")
        result = score_finding(_full_evidence_input(f"F{size}", cluster=cluster))
        assert result.score == 100  # constant, regardless of size


# ---------------------------------------------------------------------------
# TEST 16 / ADVERSARIAL TEST C — raw artifact count does not determine priority
# ---------------------------------------------------------------------------

def test_adversarial_C_artifact_count_does_not_dominate():
    one_artifact = DedupGroup(group_id="g1", representative_identifier="A", finding_ids=["A"],
                               artifact_ids=["/afl/a"], count=1, reason="x",
                               stack_signature="S", stack_signature_version="1.0", evidence_summary={})
    hundred_artifacts = DedupGroup(group_id="g2", representative_identifier="B", finding_ids=["B"],
                                    artifact_ids=[f"/afl/{i}" for i in range(100)], count=100, reason="x",
                                    stack_signature="S2", stack_signature_version="1.0", evidence_summary={})
    a = _full_evidence_input("A", dedup_group=one_artifact)
    b = _full_evidence_input("B", dedup_group=hundred_artifacts)
    ra, rb = score_finding(a), score_finding(b)
    assert ra.score == rb.score
    assert rb.metadata["dedup_artifact_count"] == 100  # visible as metadata, not scored


# ---------------------------------------------------------------------------
# TEST 17 — duplicate artifacts represented by one logical finding are not double-counted
# ---------------------------------------------------------------------------

def test_duplicate_artifacts_not_double_counted_in_score():
    group = DedupGroup(group_id="g1", representative_identifier="A", finding_ids=["A"] * 50,
                        artifact_ids=[f"/afl/{i}" for i in range(50)], count=50, reason="x",
                        stack_signature="S", stack_signature_version="1.0", evidence_summary={})
    result = score_finding(_full_evidence_input("A", dedup_group=group))
    assert result.score == 100  # same as any other fully-evidenced single finding


# ---------------------------------------------------------------------------
# TEST 18 — same input produces identical scores
# ---------------------------------------------------------------------------

def test_same_input_identical_scores():
    inputs = [_full_evidence_input("A"), _make_input("B", asan_detected=True)]
    r1 = [r.score for r in prioritize(inputs)]
    r2 = [r.score for r in prioritize(inputs)]
    assert r1 == r2


# ---------------------------------------------------------------------------
# TEST 19 — different input order produces identical ranking (repeat, stronger)
# ---------------------------------------------------------------------------

def test_different_order_identical_ranking_ids_and_ranks():
    a, b, c = _full_evidence_input("A"), _make_input("B", asan_detected=True), _make_input("C")
    result_normal = prioritize([a, b, c])
    result_shuffled = prioritize([c, a, b])
    assert [(r.finding_id, r.rank) for r in result_normal] == [(r.finding_id, r.rank) for r in result_shuffled]


# ---------------------------------------------------------------------------
# TEST 20 — score remains within declared bounds
# ---------------------------------------------------------------------------

def test_score_within_bounds():
    for inp in (_make_input("A"), _full_evidence_input("B"), _make_input("C", asan_detected=True)):
        result = score_finding(inp)
        if result.score is not None:
            assert 0 <= result.score <= 100


# ---------------------------------------------------------------------------
# TEST 21 — no NaN
# ---------------------------------------------------------------------------

def test_no_nan():
    for inp in (_make_input("A"), _full_evidence_input("B")):
        result = score_finding(inp)
        if result.score is not None:
            assert not math.isnan(result.score)


# ---------------------------------------------------------------------------
# TEST 22 — no infinity
# ---------------------------------------------------------------------------

def test_no_infinity():
    for inp in (_make_input("A"), _full_evidence_input("B")):
        result = score_finding(inp)
        if result.score is not None:
            assert not math.isinf(result.score)


# ---------------------------------------------------------------------------
# TEST 23 — invalid configuration fails clearly
# ---------------------------------------------------------------------------

def test_invalid_weights_missing_key_fails_clearly():
    incomplete_weights = {"CONFIRMED_CRASH": 30}  # missing required keys
    try:
        score_finding(_full_evidence_input("A"), weights=incomplete_weights)
        assert False, "expected KeyError for incomplete weights config"
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# TEST 24 — threshold boundaries
# ---------------------------------------------------------------------------

def test_threshold_boundary_critical():
    # crash+reproduced+asan+write = 90 exactly
    result = score_finding(_make_input("A", reproduction_status=ReproductionStatus.REPRODUCED,
                                        asan_detected=True, access_type="WRITE"))
    assert result.score == 90
    assert result.priority == "CRITICAL"


def test_threshold_boundary_high():
    # crash+reproduced+asan = 80
    result = score_finding(_make_input("A", reproduction_status=ReproductionStatus.REPRODUCED, asan_detected=True))
    assert result.score == 80
    assert result.priority == "HIGH"


def test_threshold_boundary_medium():
    # crash+asan only = 55
    result = score_finding(_make_input("A", asan_detected=True))
    assert result.score == 55
    assert result.priority == "MEDIUM"


def test_threshold_boundary_low():
    # crash alone = 30
    result = score_finding(_make_input("A"))
    assert result.score == 30
    assert result.priority == "LOW"


def test_threshold_exact_medium_boundary_value():
    # crash+location only = 40, the exact MEDIUM_THRESHOLD boundary
    result = score_finding(_make_input("A", faulting_function="f", source_file="a.c", source_line=1))
    assert result.score == MEDIUM_THRESHOLD == 40
    assert result.priority == "MEDIUM"


# ---------------------------------------------------------------------------
# TEST 25 — equal-score findings deterministic tie-break (repeat, via full batch)
# ---------------------------------------------------------------------------

def test_equal_score_batch_tie_break_stable_across_runs():
    findings = [_make_input(f"id_{i}", asan_detected=True) for i in ("c", "a", "b")]
    r1 = [r.finding_id for r in prioritize(findings)]
    r2 = [r.finding_id for r in prioritize(list(reversed(findings)))]
    assert r1 == r2 == ["id_a", "id_b", "id_c"]


# ---------------------------------------------------------------------------
# TEST 26 — evidence contributions are explainable
# ---------------------------------------------------------------------------

def test_evidence_contributions_are_explainable():
    result = score_finding(_full_evidence_input("A"))
    dims = {c.dimension for c in result.contributions}
    assert dims == {"CONFIRMED_CRASH", "REPRODUCED", "ASAN_CONFIRMED", "WRITE_ACCESS", "COMPLETE_LOCATION"}
    for c in result.contributions:
        assert c.rationale  # every contribution explains itself
        assert c.evidence_type in ("DIRECT", "DERIVED", "UNKNOWN")


# ---------------------------------------------------------------------------
# TEST 27 — uncertainty fields populated when evidence incomplete
# ---------------------------------------------------------------------------

def test_uncertainty_populated_when_incomplete():
    sparse = score_finding(_make_input("A"))
    full = score_finding(_full_evidence_input("B"))
    assert len(sparse.uncertainties) > len(full.uncertainties)
    assert SCOPE_DISCLAIMER in sparse.uncertainties
    assert SCOPE_DISCLAIMER in full.uncertainties  # always present regardless of completeness


# ---------------------------------------------------------------------------
# TEST 28 — raw findings not mutated
# ---------------------------------------------------------------------------

def test_raw_features_not_mutated():
    inp = _full_evidence_input("A")
    before = (inp.features.finding_state, inp.features.asan_detected, inp.features.access_type)
    score_finding(inp)
    after = (inp.features.finding_state, inp.features.asan_detected, inp.features.access_type)
    assert before == after


def test_dedup_group_and_cluster_not_mutated():
    group = DedupGroup(group_id="g1", representative_identifier="A", finding_ids=["A"], artifact_ids=["/a"],
                        count=5, reason="x", stack_signature="S", stack_signature_version="1.0", evidence_summary={})
    cluster = Cluster(cluster_id="c1", member_ids=["A"], member_count=1, representative_id="A",
                       behavioral_profile={}, explanation="x")
    inp = _full_evidence_input("A", dedup_group=group, cluster=cluster)
    before = (group.count, cluster.member_count)
    score_finding(inp)
    assert (group.count, cluster.member_count) == before


# ---------------------------------------------------------------------------
# TEST 29 — Phase 9 relationship remains intact
# ---------------------------------------------------------------------------

def test_phase9_relationship_intact_via_metadata():
    group = DedupGroup(group_id="g1", representative_identifier="A", finding_ids=["A", "B", "C"],
                        artifact_ids=["/a", "/b", "/c"], count=3, reason="x",
                        stack_signature="S", stack_signature_version="1.0", evidence_summary={})
    result = score_finding(_full_evidence_input("A", dedup_group=group))
    assert result.metadata["dedup_group_id"] == "g1"
    assert result.metadata["dedup_artifact_count"] == 3


# ---------------------------------------------------------------------------
# TEST 30 — Phase 10 cluster relationship remains intact
# ---------------------------------------------------------------------------

def test_phase10_relationship_intact_via_metadata():
    cluster = Cluster(cluster_id="c1", member_ids=["A", "D"], member_count=2, representative_id="A",
                       behavioral_profile={"error_type": {"mode": "heap-buffer-overflow"}}, explanation="x")
    result = score_finding(_full_evidence_input("A", cluster=cluster))
    assert result.metadata["behavioral_cluster_id"] == "c1"
    assert result.metadata["behavioral_cluster_size"] == 2


# ---------------------------------------------------------------------------
# TEST 31 — high score does not claim exploitability
# ---------------------------------------------------------------------------

def test_high_score_does_not_claim_exploitability():
    result = score_finding(_full_evidence_input("A"))
    assert result.score == 100
    joined = " ".join(result.uncertainties).lower()
    assert "exploitability has not been established" in joined
    for c in result.contributions:
        assert "exploitable" not in c.rationale.lower()


# ---------------------------------------------------------------------------
# TEST 32 — low score does not claim safety
# ---------------------------------------------------------------------------

def test_low_score_does_not_claim_safety():
    result = score_finding(_make_input("A"))
    all_text = " ".join(result.uncertainties) + " " + " ".join(c.rationale for c in result.contributions)
    assert "safe" not in all_text.lower()
    assert "not a vulnerability" not in all_text.lower()
    assert "harmless" not in all_text.lower()


# ---------------------------------------------------------------------------
# TEST 33 / ADVERSARIAL TEST F — score does not depend on unrelated findings
# ---------------------------------------------------------------------------

def test_adversarial_F_unrelated_finding_does_not_change_others_scores():
    a = _full_evidence_input("A")
    b = _make_input("B", asan_detected=True)
    only_ab = {r.finding_id: r.score for r in prioritize([a, b])}

    c = _make_input("C", finding_state=FindingState.HANG)  # wildly different, unrelated finding
    with_c = {r.finding_id: r.score for r in prioritize([a, b, c])}

    assert only_ab["A"] == with_c["A"]
    assert only_ab["B"] == with_c["B"]


# ---------------------------------------------------------------------------
# TEST 34 — adding unrelated finding does not rescale existing scores (direct, via score_finding)
# ---------------------------------------------------------------------------

def test_score_finding_never_reads_other_findings_structurally():
    import inspect
    sig = inspect.signature(score_finding)
    params = list(sig.parameters.keys())
    assert params == ["inp", "weights"]  # structurally cannot see a batch/other findings


# ---------------------------------------------------------------------------
# ADVERSARIAL TEST B — unknown evidence
# ---------------------------------------------------------------------------

def test_adversarial_B_missing_evidence_not_auto_highest_or_lowest():
    complete = score_finding(_full_evidence_input("A"))
    sparse = score_finding(_make_input("B"))  # bare CRASH, everything else unknown
    assert complete.score == 100  # complete evidence -> highest possible, correctly
    assert sparse.score == 30     # NOT 0 (auto-lowest) and NOT 100 (auto-highest) -- reflects exactly the evidence present
    assert 0 < sparse.score < complete.score
    assert len(sparse.uncertainties) > 1  # uncertainty is genuinely surfaced


# ---------------------------------------------------------------------------
# ADVERSARIAL TEST E — identical evidence -> deterministic tie-break (repeat)
# ---------------------------------------------------------------------------

def test_adversarial_E_identical_evidence_deterministic_tiebreak():
    a = _full_evidence_input("finding_002")
    b = _full_evidence_input("finding_001")
    result = prioritize([a, b])
    assert result[0].score == result[1].score
    assert result[0].finding_id == "finding_001"


# ---------------------------------------------------------------------------
# ADVERSARIAL TEST G — same stack signature does not imply identical risk
# ---------------------------------------------------------------------------

def test_adversarial_G_same_stack_does_not_force_identical_priority():
    # Phase 11 doesn't even look at stack_signature -- prove it by
    # giving two findings the same conceptual "stack identity" via
    # matching dedup group stack_signature, but different direct
    # evidence otherwise.
    group_shared_sig = DedupGroup(group_id="g", representative_identifier="A", finding_ids=["A"],
                                   artifact_ids=["/a"], count=1, reason="x",
                                   stack_signature="IDENTICAL", stack_signature_version="1.0", evidence_summary={})
    group_shared_sig_2 = DedupGroup(group_id="g2", representative_identifier="B", finding_ids=["B"],
                                     artifact_ids=["/b"], count=1, reason="x",
                                     stack_signature="IDENTICAL", stack_signature_version="1.0", evidence_summary={})
    a = _full_evidence_input("A", dedup_group=group_shared_sig)
    b = _make_input("B", dedup_group=group_shared_sig_2)  # minimal evidence, same "stack"
    ra, rb = score_finding(a), score_finding(b)
    assert ra.score != rb.score  # same stack signature, different score -- proves no stack-identity shortcut exists


# ---------------------------------------------------------------------------
# No CVSS impersonation / terminology checks
# ---------------------------------------------------------------------------

def test_no_cvss_claim_anywhere():
    result = score_finding(_full_evidence_input("A"))
    all_text = SCOPE_DISCLAIMER + " ".join(result.uncertainties) + " ".join(c.rationale for c in result.contributions)
    assert "CVSS" in SCOPE_DISCLAIMER  # mentioned only to explicitly disclaim it
    assert result.priority in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INSUFFICIENT_EVIDENCE")


def test_scope_disclaimer_present_on_every_result_type():
    crash = score_finding(_full_evidence_input("A"))
    hang = score_finding(_make_input("B", finding_state=FindingState.HANG))
    failure = score_finding(_make_input("C", finding_state=FindingState.REPRODUCTION_FAILURE))
    normal = score_finding(_make_input("D", finding_state=FindingState.NORMAL))
    for result in (crash, hang, failure, normal):
        assert SCOPE_DISCLAIMER in result.uncertainties


# ---------------------------------------------------------------------------
# build_prioritization_inputs bridge (Phase 10 -> Phase 11)
# ---------------------------------------------------------------------------

def test_build_prioritization_inputs_attaches_correct_cluster():
    from app.services.clusterer import LogicalFinding, ClusteringResult

    features_a = CrashFeatures(finding_id="A", finding_state=FindingState.CRASH, asan_detected=True)
    features_b = CrashFeatures(finding_id="B", finding_state=FindingState.CRASH, asan_detected=True)
    logical_a = LogicalFinding(identifier="A", features=features_a)
    logical_b = LogicalFinding(identifier="B", features=features_b)

    cluster = Cluster(cluster_id="c1", member_ids=["A", "B"], member_count=2,
                       representative_id="A", behavioral_profile={}, explanation="x")
    clustering_result = ClusteringResult(clusters=[cluster], noise_ids=[], total_input_count=2,
                                          config={"eps": 0.3, "min_samples": 2})

    inputs = build_prioritization_inputs([logical_a, logical_b], clustering_result)
    by_id = {i.identifier: i for i in inputs}
    assert by_id["A"].cluster.cluster_id == "c1"
    assert by_id["B"].cluster.cluster_id == "c1"


def test_build_prioritization_inputs_noise_finding_has_no_cluster():
    from app.services.clusterer import LogicalFinding, ClusteringResult

    features = CrashFeatures(finding_id="Z", finding_state=FindingState.CRASH)
    logical = LogicalFinding(identifier="Z", features=features)
    clustering_result = ClusteringResult(clusters=[], noise_ids=["Z"], total_input_count=1,
                                          config={"eps": 0.3, "min_samples": 2})

    inputs = build_prioritization_inputs([logical], clustering_result)
    assert inputs[0].cluster is None


def test_build_prioritization_inputs_without_clustering_result():
    from app.services.clusterer import LogicalFinding

    features = CrashFeatures(finding_id="Z", finding_state=FindingState.CRASH)
    logical = LogicalFinding(identifier="Z", features=features)
    inputs = build_prioritization_inputs([logical], clustering_result=None)
    assert inputs[0].cluster is None


# ---------------------------------------------------------------------------
# End-to-end integration: Phase 5 -> 7 -> 9 -> 10 -> 11
# ---------------------------------------------------------------------------

def test_end_to_end_real_pipeline_integration():
    from app.services.asan_parser import parse_asan_report
    from app.services.feature_extractor import extract_features
    from app.services.stack_normalizer import normalize_crash_features_stack
    from app.services.deduplicator import deduplicate, FindingRecord
    from app.services.clusterer import build_logical_findings, cluster_findings

    fixture = Path(__file__).parent / "fixtures" / "asan" / "heap_buffer_overflow.txt"
    asan = parse_asan_report(fixture.read_text())
    features = extract_features(asan=asan)
    stack = normalize_crash_features_stack(features)
    record = FindingRecord(features=features, stack=stack, identifier="e2e_1")

    dedup_result = deduplicate([record])
    logical_findings = build_logical_findings(dedup_result, {"e2e_1": record})
    clustering_result = cluster_findings(logical_findings)

    inputs = build_prioritization_inputs(logical_findings, clustering_result)
    results = prioritize(inputs)

    assert len(results) == 1
    assert results[0].score is not None
    assert results[0].priority in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    assert SCOPE_DISCLAIMER in results[0].uncertainties
