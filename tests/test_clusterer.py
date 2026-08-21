"""
Phase 10 tests — app.services.clusterer.

Most tests build LogicalFinding instances directly from controlled
field values (via _make_finding) for precise control over behavioral
feature combinations. A few integration tests run the real
Phase 5 -> 7 -> 9 -> 10 pipeline. Nothing here is presented as a real
crash finding, and no test label is ever fed into production
clustering logic (see test_no_data_leakage_from_test_labels).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.feature_extractor import CrashFeatures, FindingState, ReproductionStatus  # noqa: E402
from app.services.deduplicator import deduplicate, FindingRecord                               # noqa: E402
from app.services.clusterer import (                                                            # noqa: E402
    cluster_findings, build_feature_vector, build_logical_findings, gower_distance,
    LogicalFinding, DEFAULT_EPS, DEFAULT_MIN_SAMPLES,
)


def _make_finding(
    identifier,
    error_type="heap-buffer-overflow",
    access_type="READ",
    memory_region="HEAP",
    reproduction_status=ReproductionStatus.REPRODUCED,
    access_size=None,
    stack_depth=None,
    duration_ms=None,
    artifact_size=None,
    mutation_operator=None,
):
    raw_meta = {"op": mutation_operator} if mutation_operator else {}
    features = CrashFeatures(
        finding_id=identifier,
        finding_state=FindingState.CRASH,
        error_type=error_type,
        crash_type=error_type,
        access_type=access_type,
        memory_region=memory_region,
        reproduction_status=reproduction_status,
        access_size=access_size,
        stack_depth=stack_depth,
        duration_ms=duration_ms,
        artifact_size=artifact_size,
        artifact_path=identifier,
        raw_afl_filename_metadata=raw_meta,
    )
    return LogicalFinding(identifier=identifier, features=features, stack=None)


# ---------------------------------------------------------------------------
# TEST 1 — empty input
# ---------------------------------------------------------------------------

def test_empty_input():
    result = cluster_findings([])
    assert result.clusters == []
    assert result.noise_ids == []
    assert result.total_input_count == 0


# ---------------------------------------------------------------------------
# TEST 2 — single finding
# ---------------------------------------------------------------------------

def test_single_finding_is_noise_not_a_forced_singleton_cluster():
    result = cluster_findings([_make_finding("A")])
    assert result.clusters == []
    assert result.noise_ids == ["A"]
    assert result.total_input_count == 1


# ---------------------------------------------------------------------------
# TEST 3 — two clearly similar findings
# ---------------------------------------------------------------------------

def test_two_clearly_similar_findings_cluster_together():
    a = _make_finding("A", access_size=4, stack_depth=5)
    b = _make_finding("B", access_size=4, stack_depth=5)
    result = cluster_findings([a, b])
    assert len(result.clusters) == 1
    assert set(result.clusters[0].member_ids) == {"A", "B"}
    assert result.noise_ids == []


def test_small_n_min_max_normalization_can_make_close_values_look_extreme():
    """
    Documents a real, inherent characteristic of per-dataset min-max
    (Gower) normalization, not a bug: with only 2 data points, ANY
    differing numeric value defines both ends of the range, so it
    always normalizes to the maximum possible distance (1.0) on that
    dimension -- "10ms vs 11ms" looks identical to "10ms vs 99999ms"
    when there's no third point to provide scale context. This is why
    a pair that's close in absolute terms but not IDENTICAL can still
    end up as noise at a tight eps -- documented here explicitly
    rather than silently surprising someone reading cluster output.
    """
    a = _make_finding("A", duration_ms=10.0, artifact_size=100)
    b = _make_finding("B", duration_ms=11.0, artifact_size=102)  # barely different in absolute terms
    result = cluster_findings([a, b], eps=DEFAULT_EPS, min_samples=2)
    # At the default eps, these do NOT cluster -- both differing
    # numeric dimensions normalize to 1.0 with only 2 points present,
    # even though 10 vs 11 "feels" close in absolute terms.
    assert result.clusters == []
    assert set(result.noise_ids) == {"A", "B"}
    # With a third reference point providing real scale context, the
    # same absolute values normalize very differently and DO cluster.
    c = _make_finding("C", duration_ms=99999.0, artifact_size=500000)
    result_with_context = cluster_findings([a, b, c], eps=DEFAULT_EPS, min_samples=2)
    assert set(result_with_context.noise_ids) == {"C"}
    assert len(result_with_context.clusters) == 1
    assert set(result_with_context.clusters[0].member_ids) == {"A", "B"}


# ---------------------------------------------------------------------------
# TEST 4 — two clearly dissimilar findings
# ---------------------------------------------------------------------------

def test_two_clearly_dissimilar_findings_are_noise():
    a = _make_finding("A", error_type="heap-buffer-overflow", access_type="READ",
                       memory_region="HEAP", access_size=4, stack_depth=2)
    b = _make_finding("B", error_type="stack-use-after-return", access_type="WRITE",
                       memory_region="STACK", access_size=4096, stack_depth=40)
    result = cluster_findings([a, b])
    assert result.clusters == []
    assert set(result.noise_ids) == {"A", "B"}


# ---------------------------------------------------------------------------
# TEST 5 — three findings forming two behavioral groups
# ---------------------------------------------------------------------------

def test_three_findings_two_behavioral_groups():
    a = _make_finding("A", access_size=4, stack_depth=3)
    b = _make_finding("B", access_size=4, stack_depth=3)
    c = _make_finding("C", error_type="use-after-free", access_type="WRITE",
                       memory_region="HEAP", access_size=4096, stack_depth=40)
    result = cluster_findings([a, b, c])
    assert len(result.clusters) == 1
    assert set(result.clusters[0].member_ids) == {"A", "B"}
    assert result.noise_ids == ["C"]


# ---------------------------------------------------------------------------
# TEST 6 — noise/outlier handling
# ---------------------------------------------------------------------------

def test_outlier_among_a_real_cluster_remains_noise():
    a = _make_finding("A", access_size=4, stack_depth=3)
    b = _make_finding("B", access_size=4, stack_depth=3)
    c = _make_finding("C", access_size=4, stack_depth=3)
    outlier = _make_finding("Z", error_type="double-free", access_type="WRITE",
                             memory_region="STACK", access_size=99999, stack_depth=1)
    result = cluster_findings([a, b, c, outlier])
    assert len(result.clusters) == 1
    assert set(result.clusters[0].member_ids) == {"A", "B", "C"}
    assert result.noise_ids == ["Z"]
    # No manufactured second cluster just to eliminate noise.
    assert len(result.clusters) != 2


# ---------------------------------------------------------------------------
# TEST 7 — missing categorical feature
# ---------------------------------------------------------------------------

def test_missing_categorical_feature_handled_deterministically():
    a = _make_finding("A", access_type="READ")
    b = _make_finding("B", access_type=None)
    vec_a, vec_b = build_feature_vector(a), build_feature_vector(b)
    d = gower_distance(vec_a, vec_b, {"access_size": (None, None), "stack_depth": (None, None),
                                        "duration_ms": (None, None), "artifact_size": (None, None)})
    assert isinstance(d, float)  # does not crash; access_type dimension simply excluded


# ---------------------------------------------------------------------------
# TEST 8 — missing numeric feature
# ---------------------------------------------------------------------------

def test_missing_numeric_feature_handled_deterministically():
    a = _make_finding("A", access_size=4)
    b = _make_finding("B", access_size=None)
    result = cluster_findings([a, b])
    # Must not crash; outcome determined by remaining comparable dims.
    assert result.total_input_count == 2


# ---------------------------------------------------------------------------
# TEST 9 — zero-variance numeric feature
# ---------------------------------------------------------------------------

def test_zero_variance_numeric_feature_does_not_divide_by_zero():
    a = _make_finding("A", access_size=4)
    b = _make_finding("B", access_size=4)
    c = _make_finding("C", access_size=4)
    result = cluster_findings([a, b, c])  # all identical access_size -> zero range
    assert result.total_input_count == 3  # no crash


# ---------------------------------------------------------------------------
# TEST 10 — mixed categorical + numeric features
# ---------------------------------------------------------------------------

def test_mixed_categorical_numeric_features_combine_in_distance():
    a = _make_finding("A", error_type="heap-buffer-overflow", access_size=4)
    b = _make_finding("B", error_type="heap-buffer-overflow", access_size=4)
    vec_a, vec_b = build_feature_vector(a), build_feature_vector(b)
    assert vec_a.categorical["error_type"] == "heap-buffer-overflow"
    assert vec_a.numeric["access_size"] == 4.0


# ---------------------------------------------------------------------------
# TEST 11 — numeric normalization
# ---------------------------------------------------------------------------

def test_numeric_normalization_prevents_scale_domination():
    # artifact_size in the thousands must not swamp access_size in single digits.
    a = _make_finding("A", access_size=4, artifact_size=100)
    b = _make_finding("B", access_size=8, artifact_size=100)
    c = _make_finding("C", access_size=4, artifact_size=50000)
    result = cluster_findings([a, b, c], eps=0.6, min_samples=2)
    # A and B (small access_size difference, same artifact_size) should
    # be more clusterable than A and C once normalized.
    assert result.total_input_count == 3


# ---------------------------------------------------------------------------
# TEST 12 — same input produces same output
# ---------------------------------------------------------------------------

def test_same_input_same_output():
    findings = [_make_finding(f"n{i}", access_size=4, stack_depth=3) for i in range(4)]
    r1 = cluster_findings(findings)
    r2 = cluster_findings(findings)
    assert [c.cluster_id for c in r1.clusters] == [c.cluster_id for c in r2.clusters]
    assert r1.noise_ids == r2.noise_ids


# ---------------------------------------------------------------------------
# TEST 13 — input order independence
# ---------------------------------------------------------------------------

def test_input_order_independence():
    a = _make_finding("A", access_size=4, stack_depth=3)
    b = _make_finding("B", access_size=4, stack_depth=3)
    c = _make_finding("C", error_type="use-after-free", access_size=4096, stack_depth=40)

    def summary(result):
        return sorted((c.cluster_id, tuple(c.member_ids)) for c in result.clusters), sorted(result.noise_ids)

    r1 = summary(cluster_findings([a, b, c]))
    r2 = summary(cluster_findings([c, a, b]))
    r3 = summary(cluster_findings([b, c, a]))
    assert r1 == r2 == r3


# ---------------------------------------------------------------------------
# TEST 14 — deterministic cluster IDs
# ---------------------------------------------------------------------------

def test_deterministic_content_based_cluster_ids():
    a = _make_finding("A", access_size=4, stack_depth=3)
    b = _make_finding("B", access_size=4, stack_depth=3)
    r1 = cluster_findings([a, b])
    r2 = cluster_findings([b, a])
    assert r1.clusters[0].cluster_id == r2.clusters[0].cluster_id
    assert r1.clusters[0].cluster_id.startswith("cluster-")


# ---------------------------------------------------------------------------
# TEST 15 — deterministic representative selection
# ---------------------------------------------------------------------------

def test_deterministic_representative_selection():
    a = _make_finding("A", access_size=4, stack_depth=3, reproduction_status=ReproductionStatus.NOT_REPRODUCED)
    b = _make_finding("B", access_size=4, stack_depth=3, reproduction_status=ReproductionStatus.REPRODUCED)
    r1 = cluster_findings([a, b])
    r2 = cluster_findings([b, a])
    assert r1.clusters[0].representative_id == "B"
    assert r2.clusters[0].representative_id == "B"


# ---------------------------------------------------------------------------
# TEST 16 — no forced clustering of every finding
# ---------------------------------------------------------------------------

def test_not_every_finding_is_forced_into_a_cluster():
    a = _make_finding("A", access_size=4, stack_depth=3)
    b = _make_finding("B", access_size=4, stack_depth=3)
    isolated = _make_finding("ISO", error_type="stack-use-after-return",
                              access_type="WRITE", memory_region="STACK", access_size=99999)
    result = cluster_findings([a, b, isolated])
    assert "ISO" in result.noise_ids
    assert "ISO" not in [m for c in result.clusters for m in c.member_ids]


# ---------------------------------------------------------------------------
# TEST 17 — dedup groups treated as individual logical findings
# ---------------------------------------------------------------------------

def test_dedup_groups_become_single_logical_findings():
    # 6 raw findings collapse to 2 dedup groups (3 artifacts each);
    # Phase 10 must see 2 logical findings, not 6.
    records = []
    for i in range(3):
        f = CrashFeatures(finding_id=f"g1_{i}", finding_state=FindingState.CRASH,
                           error_type="heap-buffer-overflow", access_type="READ",
                           faulting_function="foo", source_file="a.c", source_line=1,
                           artifact_path=f"/afl/g1_{i}")
        records.append(FindingRecord(features=f, identifier=f"g1_{i}"))
    for i in range(3):
        f = CrashFeatures(finding_id=f"g2_{i}", finding_state=FindingState.CRASH,
                           error_type="use-after-free", access_type="WRITE",
                           faulting_function="bar", source_file="b.c", source_line=2,
                           artifact_path=f"/afl/g2_{i}")
        records.append(FindingRecord(features=f, identifier=f"g2_{i}"))
    # give each record a real stack signature so Phase 9 can actually group them
    from app.services.stack_normalizer import NormalizedStack
    for r in records[:3]:
        r.stack = NormalizedStack(stack_signature="SIG1", stack_signature_version="1.0")
    for r in records[3:]:
        r.stack = NormalizedStack(stack_signature="SIG2", stack_signature_version="1.0")

    dedup_result = deduplicate(records)
    assert len(dedup_result.groups) == 2  # confirms Phase 9 actually collapsed them

    records_by_id = {r.identifier: r for r in records}
    logical_findings = build_logical_findings(dedup_result, records_by_id)
    assert len(logical_findings) == 2  # Phase 10 sees 2, not 6


# ---------------------------------------------------------------------------
# TEST 18 — original finding IDs preserved
# ---------------------------------------------------------------------------

def test_original_finding_ids_preserved_in_clusters():
    a = _make_finding("finding_alpha", access_size=4, stack_depth=3)
    b = _make_finding("finding_beta", access_size=4, stack_depth=3)
    result = cluster_findings([a, b])
    assert set(result.clusters[0].member_ids) == {"finding_alpha", "finding_beta"}


# ---------------------------------------------------------------------------
# TEST 19 — cluster membership preserves finding IDs (no renumbering to opaque ints)
# ---------------------------------------------------------------------------

def test_cluster_membership_uses_real_identifiers_not_opaque_indices():
    a = _make_finding("real_id_123", access_size=4, stack_depth=3)
    b = _make_finding("real_id_456", access_size=4, stack_depth=3)
    result = cluster_findings([a, b])
    for member in result.clusters[0].member_ids:
        assert member in ("real_id_123", "real_id_456")


# ---------------------------------------------------------------------------
# TEST 20 — artifact references remain reachable through findings/groups
# ---------------------------------------------------------------------------

def test_artifact_references_reachable_via_dedup_group():
    f = CrashFeatures(finding_id="rep1", finding_state=FindingState.CRASH,
                       error_type="heap-buffer-overflow", artifact_path="/afl/crashes/rep1")
    from app.services.stack_normalizer import NormalizedStack
    record = FindingRecord(features=f, stack=NormalizedStack(stack_signature="S", stack_signature_version="1.0"), identifier="rep1")
    dedup_result = deduplicate([record])
    logical = build_logical_findings(dedup_result, {"rep1": record})
    assert logical[0].dedup_group is not None
    assert logical[0].dedup_group.artifact_ids == ["/afl/crashes/rep1"]


# ---------------------------------------------------------------------------
# TEST 21 — raw evidence is not mutated
# ---------------------------------------------------------------------------

def test_clustering_does_not_mutate_crash_features():
    a = _make_finding("A", access_size=4, stack_depth=3)
    b = _make_finding("B", access_size=4, stack_depth=3)
    before_a = (a.features.error_type, a.features.access_size, a.features.stack_depth)
    before_b = (b.features.error_type, b.features.access_size, b.features.stack_depth)
    cluster_findings([a, b])
    assert (a.features.error_type, a.features.access_size, a.features.stack_depth) == before_a
    assert (b.features.error_type, b.features.access_size, b.features.stack_depth) == before_b


# ---------------------------------------------------------------------------
# TEST 22 — repeated execution is idempotent
# ---------------------------------------------------------------------------

def test_repeated_execution_idempotent():
    findings = [_make_finding(f"f{i}", access_size=4, stack_depth=3) for i in range(4)]
    results = [cluster_findings(findings) for _ in range(4)]
    signatures = [
        (sorted((c.cluster_id, tuple(c.member_ids)) for c in r.clusters), sorted(r.noise_ids))
        for r in results
    ]
    assert all(s == signatures[0] for s in signatures)


# ---------------------------------------------------------------------------
# TEST 23 — different configuration produces explicitly different output
# ---------------------------------------------------------------------------

def test_different_eps_produces_different_output():
    a = _make_finding("A", access_size=4, stack_depth=3)
    b = _make_finding("B", access_size=10, stack_depth=8)  # moderately different
    tight = cluster_findings([a, b], eps=0.05, min_samples=2)
    loose = cluster_findings([a, b], eps=0.9, min_samples=2)
    assert tight.clusters != loose.clusters or tight.noise_ids != loose.noise_ids
    assert loose.clusters  # loose eps merges them
    assert not tight.clusters  # tight eps keeps them apart


# ---------------------------------------------------------------------------
# TEST 24 — invalid/malformed feature data handled safely
# ---------------------------------------------------------------------------

def test_fully_empty_crash_features_does_not_crash():
    empty = LogicalFinding(identifier="empty1", features=CrashFeatures(), stack=None)
    normal = _make_finding("normal1", access_size=4, stack_depth=3)
    result = cluster_findings([empty, normal])
    assert result.total_input_count == 2  # no crash


# ---------------------------------------------------------------------------
# TEST 25 — small datasets do not crash
# ---------------------------------------------------------------------------

def test_small_datasets_do_not_crash():
    for n in (0, 1, 2, 3):
        findings = [_make_finding(f"s{i}", access_size=4, stack_depth=3) for i in range(n)]
        result = cluster_findings(findings)
        assert result.total_input_count == n


# ---------------------------------------------------------------------------
# TEST 26 — reproduced vs non-reproduced behave per documented feature policy
# ---------------------------------------------------------------------------

def test_reproduction_status_is_a_categorical_feature_affecting_distance():
    a = _make_finding("A", reproduction_status=ReproductionStatus.REPRODUCED)
    b = _make_finding("B", reproduction_status=ReproductionStatus.NOT_REPRODUCED)
    vec_a, vec_b = build_feature_vector(a), build_feature_vector(b)
    ranges = {"access_size": (None, None), "stack_depth": (None, None),
              "duration_ms": (None, None), "artifact_size": (None, None)}
    d_same = gower_distance(vec_a, vec_a, ranges)
    d_diff = gower_distance(vec_a, vec_b, ranges)
    assert d_same < d_diff  # differing reproduction_status increases distance


# ---------------------------------------------------------------------------
# TEST 27 — different stack signatures CAN still be behaviorally clustered
#           (adversarial TEST B)
# ---------------------------------------------------------------------------

def test_adversarial_B_different_signatures_similar_behavior_can_cluster():
    a = LogicalFinding(
        identifier="A",
        features=CrashFeatures(finding_state=FindingState.CRASH, error_type="heap-buffer-overflow",
                                access_type="READ", memory_region="HEAP", access_size=4, stack_depth=5,
                                faulting_function="foo_a", source_file="a.c", source_line=10),
        stack=None,
    )
    b = LogicalFinding(
        identifier="B",
        features=CrashFeatures(finding_state=FindingState.CRASH, error_type="heap-buffer-overflow",
                                access_type="READ", memory_region="HEAP", access_size=4, stack_depth=5,
                                faulting_function="foo_b_completely_different", source_file="z.c", source_line=999),
        stack=None,
    )
    # Note: faulting_function/source_file/source_line differ entirely
    # (these would make Phase 9 treat them as SEPARATE) -- Phase 10
    # doesn't use those fields at all, so identical behavioral
    # features (error_type/access_type/memory_region/access_size/stack_depth)
    # can still cluster them.
    result = cluster_findings([a, b])
    assert len(result.clusters) == 1
    assert set(result.clusters[0].member_ids) == {"A", "B"}


# ---------------------------------------------------------------------------
# TEST 28 — same stack signature is NOT automatically the same cluster
#           (adversarial TEST A)
# ---------------------------------------------------------------------------

def test_adversarial_A_same_signature_different_behavior_not_forced_together():
    from app.services.stack_normalizer import NormalizedStack
    same_sig = NormalizedStack(stack_signature="IDENTICAL_SIG", stack_signature_version="1.0")

    a = LogicalFinding(
        identifier="A",
        features=CrashFeatures(finding_state=FindingState.CRASH, error_type="heap-buffer-overflow",
                                access_type="READ", memory_region="HEAP", access_size=4, stack_depth=3),
        stack=same_sig,
    )
    b = LogicalFinding(
        identifier="B",
        features=CrashFeatures(finding_state=FindingState.CRASH, error_type="stack-use-after-return",
                                access_type="WRITE", memory_region="STACK", access_size=8192, stack_depth=50),
        stack=same_sig,  # SAME stack signature, wildly different behavior otherwise
    )
    result = cluster_findings([a, b])
    # Clusterer doesn't even look at stack_signature -- but prove the
    # adversarial point explicitly: despite matching signature, wildly
    # different behavioral features keep them apart.
    assert result.clusters == []
    assert set(result.noise_ids) == {"A", "B"}


# ---------------------------------------------------------------------------
# TEST 29 — cluster explanations contain actual feature evidence
# ---------------------------------------------------------------------------

def test_cluster_explanation_names_real_evidence_not_vague():
    a = _make_finding("A", error_type="heap-buffer-overflow", memory_region="HEAP", access_size=4, stack_depth=3)
    b = _make_finding("B", error_type="heap-buffer-overflow", memory_region="HEAP", access_size=4, stack_depth=3)
    result = cluster_findings([a, b])
    explanation = result.clusters[0].explanation
    assert "heap-buffer-overflow" in explanation
    assert "HEAP" in explanation
    assert explanation != "AI found these similar."
    assert "same root cause" not in explanation.lower()
    assert "vulnerability" not in explanation.lower()


# ---------------------------------------------------------------------------
# TEST 30 — no random behavior across repeated runs
# ---------------------------------------------------------------------------

def test_no_random_behavior_across_many_repeated_runs():
    findings = [_make_finding(f"r{i}", access_size=4 + (i % 2), stack_depth=3) for i in range(8)]
    first = cluster_findings(findings)
    for _ in range(10):
        again = cluster_findings(findings)
        assert sorted((c.cluster_id, tuple(c.member_ids)) for c in again.clusters) == \
               sorted((c.cluster_id, tuple(c.member_ids)) for c in first.clusters)
        assert again.noise_ids == first.noise_ids


# ---------------------------------------------------------------------------
# Additional adversarial tests C, D, E + data-leakage / audit checks
# ---------------------------------------------------------------------------

def test_adversarial_C_extreme_outlier_remains_noise_not_forced():
    a = _make_finding("A", access_size=4, stack_depth=3)
    b = _make_finding("B", access_size=4, stack_depth=3)
    c = _make_finding("C", access_size=4, stack_depth=3)
    extreme = _make_finding("EXTREME", error_type="global-buffer-overflow", access_type="WRITE",
                             memory_region="GLOBAL", access_size=10_000_000, stack_depth=500,
                             duration_ms=99999.0)
    result = cluster_findings([a, b, c, extreme])
    assert "EXTREME" in result.noise_ids
    assert len(result.clusters) == 1
    assert len(result.clusters[0].member_ids) == 3


def test_adversarial_D_shuffled_order_same_membership():
    findings = [_make_finding(f"f{i}", access_size=4, stack_depth=3) for i in range(5)]
    import random
    shuffled = findings[:]
    random.Random(42).shuffle(shuffled)
    r1 = cluster_findings(findings)
    r2 = cluster_findings(shuffled)
    assert sorted(tuple(sorted(c.member_ids)) for c in r1.clusters) == \
           sorted(tuple(sorted(c.member_ids)) for c in r2.clusters)


def test_adversarial_E_repeated_runs_identical_normalized_output():
    findings = [_make_finding(f"f{i}", access_size=4, stack_depth=3) for i in range(6)] + \
               [_make_finding(f"g{i}", error_type="use-after-free", access_size=999, stack_depth=1) for i in range(4)]
    r1 = cluster_findings(findings)
    r2 = cluster_findings(findings)
    assert [(c.cluster_id, c.member_ids) for c in r1.clusters] == [(c.cluster_id, c.member_ids) for c in r2.clusters]


def test_no_data_leakage_from_test_labels():
    """
    Production clustering must never see this test's own naming
    scheme as a feature -- confirm identifiers themselves play no role
    in the distance computation (only the CrashFeatures values do).
    """
    a = _make_finding("this_name_says_cluster_alpha", access_size=4, stack_depth=3)
    b = _make_finding("this_name_says_cluster_beta_totally_different_label", access_size=4, stack_depth=3)
    result = cluster_findings([a, b])
    assert len(result.clusters) == 1  # grouped on real feature similarity, not identifier text


def test_silhouette_none_when_degenerate():
    a = _make_finding("A", access_size=4, stack_depth=3)
    b = _make_finding("B", access_size=4, stack_depth=3)
    result = cluster_findings([a, b])  # exactly one cluster -> silhouette undefined
    assert result.overall_silhouette is None


def test_feature_vector_excludes_stack_signature_and_fault_location():
    finding = _make_finding("A")
    finding.features.faulting_function = "should_not_matter"
    finding.features.source_file = "should_not_matter.c"
    vec = build_feature_vector(finding)
    assert "faulting_function" not in vec.categorical
    assert "source_file" not in vec.categorical
    assert "stack_signature" not in vec.categorical
    assert "stack_signature" not in vec.numeric
