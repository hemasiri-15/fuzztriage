"""
Phase 10 — behavioral crash clustering.

Answers a DIFFERENT question than Phase 9:

    Phase 9  : "Are these findings sufficiently evidenced to represent
                the same logical finding?"          (exact/strong evidence)
    Phase 10 : "Among DISTINCT logical findings, which exhibit similar
                BEHAVIORAL characteristics?"          (similarity, not identity)

Phase 10 consumes Phase 9's *logical findings* (one representative per
DedupGroup), never raw AFL++ artifacts directly — otherwise mutation
volume (a Phase 9 group with 17 artifacts vs. one with 1) would
artificially dominate the clustering.

----------------------------------------------------------------------
Inspection note (repository reality)
----------------------------------------------------------------------
DedupGroup (Phase 9) carries only `representative_identifier: str` and
a small `evidence_summary` dict — NOT the representative's full
CrashFeatures/NormalizedStack (confirmed by inspecting deduplicator.py
before writing this file). Behavioral features like access_size,
stack_depth, duration_ms, and artifact_size are NOT in
evidence_summary. `build_logical_findings()` below is the explicit
bridge: given a DeduplicationResult and the original FindingRecords
(keyed by identifier), it looks up each representative's full evidence
so Phase 10 has real behavioral data to work with.

----------------------------------------------------------------------
Feature vector (see FEATURE_AUDIT at the bottom of this file for the
full documented table)
----------------------------------------------------------------------
Deliberately EXCLUDED: stack_signature, error_type-as-primary-key,
faulting_function, source_file, source_line as *grouping keys* the way
Phase 9 uses them. error_type/access_type/memory_region ARE used, but
only as similarity-contributing features in a distance computation —
never as an exact-match grouping key — which is what keeps Phase 10
genuinely different from Phase 9 (see adversarial tests A/B).

CATEGORICAL : error_type, access_type, memory_region,
              reproduction_status, mutation_operator (sparse/optional)
NUMERIC     : access_size, stack_depth, duration_ms, artifact_size

----------------------------------------------------------------------
Distance metric: Gower distance
----------------------------------------------------------------------
A standard, citable (Gower, 1971) method for mixed categorical/numeric
data that handles missing values principled-ly: for each feature
dimension, if either side is missing that value, the dimension is
EXCLUDED from that pair's comparison (contributes neither similarity
nor dissimilarity) rather than defaulting to a guessed value. Every
comparable dimension gets equal weight (1 / number of comparable
dimensions) — no arbitrary hand-tuned weights. Numeric features are
range-normalized (min-max over the actual dataset) before comparison
so no feature dominates purely due to units. A pair with ZERO
comparable dimensions gets maximal distance (1.0) — "nothing to
compare" must never be treated as "similar."

----------------------------------------------------------------------
Algorithm: DBSCAN, hand-rolled (no new dependency)
----------------------------------------------------------------------
Why DBSCAN: does not require choosing a cluster count in advance
(unlike K-Means), naturally produces a NOISE/unclustered outcome for
points with no sufficiently similar neighbors (required — findings
must not be force-assigned to a cluster), and has no random
initialization step (unlike K-Means), so it's deterministic by
construction given a fixed eps/min_samples.

Why hand-rolled instead of scikit-learn: the project's own stated
scale ("hundreds of logical findings") makes an O(n^2) precomputed
distance matrix plus a ~40-line DBSCAN loop entirely sufficient, and
implementing it directly avoids adding a large new dependency
(scikit-learn + numpy) for a lightweight MVP project, per this
project's explicit "avoid over-engineering / no large dependency just
because it's fashionable" principle. Every step is order-independent
by construction: points are always processed and expanded in sorted-
identifier order (never input-list order), and final cluster IDs are
a content hash of the member set (never the algorithm's raw integer
labels, which scikit-learn's own docs note are not stable across
runs/implementations either).

----------------------------------------------------------------------
Security / purity
----------------------------------------------------------------------
No execution, no subprocess, no shell, no network, no filesystem
access beyond what's already in memory. Read-only with respect to all
CrashFeatures/NormalizedStack/DedupGroup evidence passed in.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from app.services.feature_extractor import CrashFeatures
from app.services.stack_normalizer import NormalizedStack
from app.services.deduplicator import (
    DedupGroup, DeduplicationResult, FindingRecord, choose_representative,
)

DEFAULT_EPS = 0.3
DEFAULT_MIN_SAMPLES = 2

_CATEGORICAL_FEATURES = (
    "error_type", "access_type", "memory_region",
    "reproduction_status", "mutation_operator",
)
_NUMERIC_FEATURES = (
    "access_size", "stack_depth", "duration_ms", "artifact_size",
)


# ---------------------------------------------------------------------------
# Input bundle — the Phase 9 -> Phase 10 boundary
# ---------------------------------------------------------------------------

@dataclass
class LogicalFinding:
    """One deduplicated logical finding, ready for behavioral clustering."""
    identifier: str
    features: CrashFeatures
    stack: Optional[NormalizedStack] = None
    dedup_group: Optional[DedupGroup] = None   # None if constructed directly (e.g. in tests)


def build_logical_findings(
    dedup_result: DeduplicationResult,
    records_by_id: dict,
) -> list:
    """
    Bridge Phase 9's output into Phase 10's input: one LogicalFinding
    per DedupGroup, carrying that group's REPRESENTATIVE's full
    evidence (looked up from the original FindingRecords, since
    DedupGroup itself doesn't carry it — see module docstring).

    A group whose representative_identifier isn't found in
    `records_by_id` is skipped rather than raising — a caller wiring
    mismatch should not crash clustering for every other finding.
    """
    findings = []
    for group in dedup_result.groups:
        record = records_by_id.get(group.representative_identifier)
        if record is None:
            continue
        findings.append(LogicalFinding(
            identifier=group.representative_identifier,
            features=record.features,
            stack=record.stack,
            dedup_group=group,
        ))
    return findings


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

@dataclass
class BehavioralFeatureVector:
    categorical: dict
    numeric: dict


def build_feature_vector(finding: LogicalFinding) -> BehavioralFeatureVector:
    f = finding.features
    mutation_operator = None
    if f.raw_afl_filename_metadata:
        mutation_operator = f.raw_afl_filename_metadata.get("op")

    return BehavioralFeatureVector(
        categorical={
            "error_type": f.error_type,
            "access_type": f.access_type,
            "memory_region": f.memory_region,
            "reproduction_status": f.reproduction_status.value if f.reproduction_status else None,
            "mutation_operator": mutation_operator,
        },
        numeric={
            "access_size": float(f.access_size) if f.access_size is not None else None,
            "stack_depth": float(f.stack_depth) if f.stack_depth is not None else None,
            "duration_ms": float(f.duration_ms) if f.duration_ms is not None else None,
            "artifact_size": float(f.artifact_size) if f.artifact_size is not None else None,
        },
    )


def _numeric_ranges(vectors: list) -> dict:
    """min-max range per numeric feature, over whatever values are
    actually present. A feature present in fewer than 2 distinct
    values (including entirely absent) has no discriminative range —
    handled explicitly at comparison time, not by dividing by zero."""
    ranges = {}
    for name in _NUMERIC_FEATURES:
        values = [v.numeric[name] for v in vectors if v.numeric.get(name) is not None]
        if values:
            ranges[name] = (min(values), max(values))
        else:
            ranges[name] = (None, None)
    return ranges


def gower_distance(a: BehavioralFeatureVector, b: BehavioralFeatureVector, numeric_ranges: dict) -> float:
    """
    Gower-style mixed-type distance in [0, 1]. See module docstring
    for the full missing-data / normalization policy. Returns 1.0
    (maximal distance) if no dimension was comparable at all.
    """
    contributions = []

    for name in _CATEGORICAL_FEATURES:
        va, vb = a.categorical.get(name), b.categorical.get(name)
        if va is None or vb is None:
            continue
        contributions.append(0.0 if va == vb else 1.0)

    for name in _NUMERIC_FEATURES:
        va, vb = a.numeric.get(name), b.numeric.get(name)
        if va is None or vb is None:
            continue
        lo, hi = numeric_ranges.get(name, (None, None))
        if lo is None or hi is None or hi == lo:
            continue  # zero-variance or no reference range -> no discriminative signal
        contributions.append(abs(va - vb) / (hi - lo))

    if not contributions:
        return 1.0
    return sum(contributions) / len(contributions)


# ---------------------------------------------------------------------------
# Hand-rolled, deterministic DBSCAN over a precomputed distance matrix
# ---------------------------------------------------------------------------

def _dbscan(ids: list, distance: dict, eps: float, min_samples: int) -> dict:
    """
    Returns {id: label}, label is a positive int for a real cluster or
    -1 for noise. Every iteration order (outer loop, neighbor
    expansion) uses `sorted(...)`, never input list order, so the
    resulting raw labeling is deterministic and independent of the
    order `ids` was given in. (Final cluster identity for callers is
    still a content hash, computed after this — see cluster_findings —
    which is an additional, independent guarantee.)
    """
    sorted_ids = sorted(ids)

    def neighbors(p):
        return [q for q in sorted_ids if q != p and distance[(p, q)] <= eps]

    labels = {p: None for p in sorted_ids}
    visited = set()
    next_cluster_id = 0

    for p in sorted_ids:
        if p in visited:
            continue
        visited.add(p)
        p_neighbors = neighbors(p)
        if len(p_neighbors) + 1 < min_samples:
            labels[p] = -1
            continue

        next_cluster_id += 1
        labels[p] = next_cluster_id
        seed_set = sorted(p_neighbors)
        i = 0
        while i < len(seed_set):
            q = seed_set[i]
            i += 1
            if q not in visited:
                visited.add(q)
                q_neighbors = neighbors(q)
                if len(q_neighbors) + 1 >= min_samples:
                    for nn in sorted(q_neighbors):
                        if nn not in seed_set:
                            seed_set.append(nn)
            if labels[q] is None or labels[q] == -1:
                labels[q] = next_cluster_id

    return labels


# ---------------------------------------------------------------------------
# Cluster summary / explanation
# ---------------------------------------------------------------------------

def _mode(values: list) -> Optional[tuple]:
    """(mode_value, count, total_present), deterministic tie-break by
    string value. None if nothing present."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    counts: dict = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    top_count = max(counts.values())
    candidates = sorted((v for v, c in counts.items() if c == top_count), key=str)
    return (candidates[0], top_count, len(present))


def _summarize(members: list) -> dict:
    profile: dict = {}
    for name in _CATEGORICAL_FEATURES:
        result = _mode([m.categorical.get(name) for m in members])
        if result:
            value, count, total = result
            profile[name] = {"mode": value, "unanimous": count == len(members), "present_in": total}
    for name in _NUMERIC_FEATURES:
        values = [m.numeric.get(name) for m in members if m.numeric.get(name) is not None]
        if values:
            profile[name] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values), "present_in": len(values)}
    return profile


def _build_explanation(profile: dict, member_count: int) -> str:
    parts = []
    for name in _CATEGORICAL_FEATURES:
        info = profile.get(name)
        if info and info["present_in"] == member_count and info["unanimous"]:
            parts.append(f"same {name} ({info['mode']})")
    for name in _NUMERIC_FEATURES:
        info = profile.get(name)
        if info and info["present_in"] == member_count:
            parts.append(f"similar {name} (mean {info['mean']:.1f}, range {info['min']:.1f}-{info['max']:.1f})")
    if not parts:
        return "Behaviorally clustered findings; individual feature values were too sparse to summarize confidently."
    return "Behavioral cluster: findings share " + ", ".join(parts) + "."


def _content_id(member_ids: list) -> str:
    canonical = "|".join(sorted(member_ids))
    return "cluster-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    cluster_id: str
    member_ids: list
    member_count: int
    representative_id: Optional[str]
    behavioral_profile: dict
    explanation: str
    mean_intra_cluster_distance: Optional[float] = None


@dataclass
class ClusteringResult:
    clusters: list             # list[Cluster] — noise is NOT included here
    noise_ids: list
    total_input_count: int
    config: dict                # {"eps": ..., "min_samples": ...} used for this run
    overall_silhouette: Optional[float] = None


def _silhouette(ids: list, labels: dict, distance: dict) -> Optional[float]:
    """
    Standard silhouette coefficient, computed only over clustered
    (non-noise) points, only when there are at least 2 real clusters
    and every real cluster has at least 2 members (silhouette is
    undefined for singleton clusters). Returns None rather than a
    fabricated number when the dataset is too small/degenerate for a
    meaningful score.
    """
    clustered = [i for i in ids if labels[i] != -1]
    cluster_ids = set(labels[i] for i in clustered)
    if len(cluster_ids) < 2:
        return None
    by_cluster = {c: [i for i in clustered if labels[i] == c] for c in cluster_ids}
    if any(len(members) < 2 for members in by_cluster.values()):
        return None

    scores = []
    for i in clustered:
        own_cluster = labels[i]
        own_members = [m for m in by_cluster[own_cluster] if m != i]
        a_i = sum(distance[(i, m)] for m in own_members) / len(own_members)
        b_i = min(
            sum(distance[(i, m)] for m in members) / len(members)
            for c, members in by_cluster.items() if c != own_cluster
        )
        s_i = 0.0 if max(a_i, b_i) == 0 else (b_i - a_i) / max(a_i, b_i)
        scores.append(s_i)
    return sum(scores) / len(scores)


def cluster_findings(
    findings: list,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> ClusteringResult:
    """
    Behaviorally cluster `findings` (list[LogicalFinding]).

    Deterministic and order-independent: cluster_findings(findings)
    and cluster_findings(list(reversed(findings))) always produce the
    same set of clusters (same membership, same cluster_ids, same
    noise set) for the same eps/min_samples.

    A finding with no sufficiently similar neighbor (per eps/
    min_samples) is reported in `noise_ids`, never forced into the
    nearest cluster.
    """
    if not findings:
        return ClusteringResult(clusters=[], noise_ids=[], total_input_count=0,
                                 config={"eps": eps, "min_samples": min_samples})

    by_id = {f.identifier: f for f in findings}
    vectors = {f.identifier: build_feature_vector(f) for f in findings}
    ranges = _numeric_ranges(list(vectors.values()))

    ids = sorted(by_id.keys())
    distance = {}
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                distance[(ids[i], ids[j])] = 0.0
            else:
                distance[(ids[i], ids[j])] = gower_distance(vectors[ids[i]], vectors[ids[j]], ranges)

    labels = _dbscan(ids, distance, eps=eps, min_samples=min_samples)

    grouped: dict = {}
    noise_ids = []
    for identifier in ids:
        label = labels[identifier]
        if label == -1:
            noise_ids.append(identifier)
        else:
            grouped.setdefault(label, []).append(identifier)

    clusters = []
    for member_ids in grouped.values():
        members_vectors = [vectors[m] for m in member_ids]
        records = [FindingRecord(features=by_id[m].features, stack=by_id[m].stack, identifier=m) for m in member_ids]
        representative = choose_representative(records)

        pair_distances = [
            distance[(a, b)] for idx, a in enumerate(member_ids) for b in member_ids[idx + 1:]
        ]
        mean_intra = sum(pair_distances) / len(pair_distances) if pair_distances else 0.0

        profile = _summarize(members_vectors)
        clusters.append(Cluster(
            cluster_id=_content_id(member_ids),
            member_ids=sorted(member_ids),
            member_count=len(member_ids),
            representative_id=representative.identifier,
            behavioral_profile=profile,
            explanation=_build_explanation(profile, len(member_ids)),
            mean_intra_cluster_distance=mean_intra,
        ))

    clusters.sort(key=lambda c: c.cluster_id)
    overall_silhouette = _silhouette(ids, labels, distance)

    return ClusteringResult(
        clusters=clusters,
        noise_ids=sorted(noise_ids),
        total_input_count=len(findings),
        config={"eps": eps, "min_samples": min_samples},
        overall_silhouette=overall_silhouette,
    )


# ---------------------------------------------------------------------------
# FEATURE AUDIT (documented, not just in prose — kept in sync with the
# actual _CATEGORICAL_FEATURES / _NUMERIC_FEATURES tuples above)
# ---------------------------------------------------------------------------
#
# Feature              | Type        | Source                          | Missing-data behavior          | Normalization        | Used? | Reason
# ---------------------|-------------|----------------------------------|---------------------------------|-----------------------|-------|----------------------------------------------------------
# error_type           | categorical | CrashFeatures.error_type         | excluded from pair if either None | none (nominal match)  | yes   | broad behavioral category of the memory-safety failure
# access_type          | categorical | CrashFeatures.access_type        | excluded from pair if either None | none                   | yes   | read vs. write access pattern
# memory_region        | categorical | CrashFeatures.memory_region      | excluded from pair if either None | none                   | yes   | which memory area was involved (supporting evidence in Ph9, genuine behavioral signal here)
# reproduction_status  | categorical | CrashFeatures.reproduction_status| excluded from pair if either None | none                   | yes   | reliability/nature of triggering the crash
# mutation_operator    | categorical | CrashFeatures.raw_afl_filename_metadata["op"] | excluded (very commonly absent) | none    | yes   | AFL++ mutation metadata; sparse, documented as optional
# access_size          | numeric     | CrashFeatures.access_size        | excluded from pair if either None | min-max range (Gower)  | yes   | magnitude of the out-of-bounds/misused access
# stack_depth          | numeric     | CrashFeatures.stack_depth        | excluded from pair if either None | min-max range (Gower)  | yes   | structural depth of the fault, independent of exact function identity
# duration_ms          | numeric     | ReproductionResult.duration_ms (via CrashFeatures) | excluded if either None | min-max range (Gower) | yes | runtime/timing behavior of triggering the crash
# artifact_size         | numeric     | ArtifactRecord.size_bytes (via CrashFeatures) | excluded if either None | min-max range (Gower) | yes  | size of the triggering input; correlates with malformation complexity
# stack_signature        | (excluded)  | NormalizedStack                  | n/a                              | n/a                    | NO    | this is Phase 9's exact-identity dimension; using it here would make clustering redundant with deduplication
# faulting_function       | (excluded)  | CrashFeatures.faulting_function   | n/a                              | n/a                    | NO    | exact fault-location identity belongs to Phase 9, not behavioral similarity
# source_file / source_line | (excluded) | CrashFeatures                  | n/a                              | n/a                    | NO    | same reasoning as faulting_function
# reproducible (bool)     | (excluded)  | CrashFeatures.reproducible        | n/a                              | n/a                    | NO    | redundant with reproduction_status (same underlying signal, avoids double-counting)
# asan_detected (bool)    | (excluded)  | CrashFeatures.asan_detected       | n/a                              | n/a                    | NO    | constant (True) across any realistic CRASH-only input set — zero discriminative power
# timed_out (bool)         | (excluded)  | CrashFeatures.timed_out           | n/a                              | n/a                    | NO    | constant (False) for CRASH-state findings by construction (Phase 7 routes timeouts to HANG)
