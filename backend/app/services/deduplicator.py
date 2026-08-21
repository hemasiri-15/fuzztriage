"""
Phase 9 — evidence-based crash deduplication.

Groups findings that have strong, compatible evidence of representing
the same underlying failure. This is NOT clustering (Phase 10), NOT
priority scoring (Phase 11), and NOT proof that grouped findings are
"the same vulnerability" — it is a conservative, explainable grouping
decision based on concrete, already-extracted evidence.

Core principle (stated explicitly because it drives every design
choice below): FALSE MERGING IS MORE DANGEROUS THAN UNDER-MERGING.
When evidence is ambiguous, findings stay separate.

----------------------------------------------------------------------
Inspection note (repository reality vs. assumption)
----------------------------------------------------------------------
CrashFeatures (Phase 7) does NOT carry stack_signature / normalized_stack
/ stack_signature_version fields — Phase 8's normalize_stack() returns
those in a separate NormalizedStack object, and Phase 7 was never
modified to embed them. Confirmed by inspecting both modules before
writing this file. Phase 9 therefore takes CrashFeatures paired with
its (optional) NormalizedStack via the small FindingRecord bundle
below, rather than assuming they were already merged onto one object.

----------------------------------------------------------------------
Evidence model
----------------------------------------------------------------------
Every comparison between two findings, for each evidence dimension, is
one of three states:

    MATCH     both sides have a value, and the values are equal
    CONFLICT  both sides have a value, and the values differ
    UNKNOWN   at least one side is missing that value

UNKNOWN is never treated as positive evidence. Missing data is not the
same as matching data.

"Blocking" dimensions are REQUIRED to be an actual MATCH (not merely
"not CONFLICT") for two findings to be declared DUPLICATE:

    finding_state
    stack_signature (+ stack_signature_version, as a pair)
    error_type            (ASan/sanitizer error class)
    access_type            (READ/WRITE)
    faulting_function
    source_file
    source_line

Requiring actual equality (not just "non-conflicting") on every
blocking dimension is a deliberate design choice: equality is
transitive, so building final groups via Union-Find over pairwise
DUPLICATE decisions is provably safe from the classic pairwise-dedup
transitivity trap (A~B, B~C, but A!~C) with no extra verification
pass needed. If UNKNOWN were allowed to "pass" a blocking dimension,
that safety property would break (see module tests for a worked
counter-example proving this).

"Supporting" evidence is included in the explanation but never blocks
or requires a merge:

    access_size     — legitimately varies (partial reads/writes,
                       alignment) even for the same underlying bug
    memory_region    — spec explicitly calls this supporting evidence

Any CONFLICT on a blocking dimension -> SEPARATE, regardless of how
many other dimensions match (a matching stack signature never
overrides a real conflict elsewhere).

If nothing conflicts but at least one blocking dimension is UNKNOWN on
either side -> INSUFFICIENT_EVIDENCE (not proven same, not proven
different).

Only when every blocking dimension is an actual MATCH -> DUPLICATE.

----------------------------------------------------------------------
Security / purity
----------------------------------------------------------------------
This module performs no execution, no subprocess, no shell commands,
no network access, and does not modify its inputs. It reads
already-extracted CrashFeatures/NormalizedStack objects and returns a
new, independent grouping structure.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.services.feature_extractor import CrashFeatures, ReproductionStatus
from app.services.stack_normalizer import NormalizedStack


# ---------------------------------------------------------------------------
# Evidence comparison primitives
# ---------------------------------------------------------------------------

class EvidenceComparison(str, Enum):
    MATCH = "MATCH"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"


class DedupDecision(str, Enum):
    DUPLICATE = "DUPLICATE"
    SEPARATE = "SEPARATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def _compare(a: Optional[object], b: Optional[object]) -> EvidenceComparison:
    if a is None or b is None:
        return EvidenceComparison.UNKNOWN
    return EvidenceComparison.MATCH if a == b else EvidenceComparison.CONFLICT


# Dimensions required to be an actual MATCH for DUPLICATE. Order here
# also defines the pipeline stages named in the design: state ->
# signature -> sanitizer -> access -> fault-location.
_BLOCKING_DIMENSIONS = (
    "finding_state",
    "stack_signature",          # combined signature+version, see _signature_pair
    "error_type",
    "access_type",
    "faulting_function",
    "source_file",
    "source_line",
)


# ---------------------------------------------------------------------------
# Input bundle
# ---------------------------------------------------------------------------

@dataclass
class FindingRecord:
    """
    Pairs one CrashFeatures with its (optional) Phase 8 NormalizedStack.
    `identifier` is what this record is referenced by in group output;
    if omitted, a deterministic fallback is computed from `features`
    (see _stable_identifier) so grouping never depends on omitted
    identifiers to remain deterministic.
    """
    features: CrashFeatures
    stack: Optional[NormalizedStack] = None
    identifier: Optional[str] = None


def _stable_identifier(record: FindingRecord) -> str:
    """
    Deterministic, order-independent identity for one finding record.

    Preference order: explicit identifier -> CrashFeatures.finding_id
    -> CrashFeatures.artifact_path -> a content hash of key evidence
    fields (guarantees every record has a usable identifier, even a
    fully synthetic one with no artifact behind it, without ever
    depending on input list position).
    """
    if record.identifier:
        return record.identifier
    if record.features.finding_id:
        return record.features.finding_id
    if record.features.artifact_path:
        return record.features.artifact_path

    f = record.features
    key = "|".join(str(x) for x in (
        f.finding_state.value, f.error_type, f.access_type,
        f.faulting_function, f.source_file, f.source_line,
        record.stack.stack_signature if record.stack else None,
    ))
    return "content:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _signature_pair(record: FindingRecord) -> Optional[tuple]:
    """(stack_signature, stack_signature_version) as one comparable unit,
    or None if no signature is available (Phase 8 wasn't run, or the
    stack was empty)."""
    if record.stack is None or record.stack.stack_signature is None:
        return None
    return (record.stack.stack_signature, record.stack.stack_signature_version)


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

@dataclass
class PairDecision:
    decision: DedupDecision
    reason: str
    evidence: dict  # dimension name -> EvidenceComparison


def compare_evidence(a: FindingRecord, b: FindingRecord) -> PairDecision:
    """
    Compare two findings across every blocking dimension and return a
    single deterministic decision. Never mutates `a` or `b`.
    """
    evidence: dict = {}

    evidence["finding_state"] = _compare(a.features.finding_state, b.features.finding_state)
    evidence["stack_signature"] = _compare(_signature_pair(a), _signature_pair(b))
    evidence["error_type"] = _compare(a.features.error_type, b.features.error_type)
    evidence["access_type"] = _compare(a.features.access_type, b.features.access_type)
    evidence["faulting_function"] = _compare(a.features.faulting_function, b.features.faulting_function)
    evidence["source_file"] = _compare(a.features.source_file, b.features.source_file)
    evidence["source_line"] = _compare(a.features.source_line, b.features.source_line)

    # Supporting evidence — recorded for explainability, never gates the decision.
    evidence["access_size"] = _compare(a.features.access_size, b.features.access_size)
    evidence["memory_region"] = _compare(a.features.memory_region, b.features.memory_region)

    conflicts = [dim for dim in _BLOCKING_DIMENSIONS if evidence[dim] == EvidenceComparison.CONFLICT]
    if conflicts:
        return PairDecision(
            decision=DedupDecision.SEPARATE,
            reason="Not merged: " + ", ".join(f"{dim} conflicts" for dim in conflicts) + ".",
            evidence=evidence,
        )

    unknowns = [dim for dim in _BLOCKING_DIMENSIONS if evidence[dim] == EvidenceComparison.UNKNOWN]
    if unknowns:
        return PairDecision(
            decision=DedupDecision.INSUFFICIENT_EVIDENCE,
            reason=(
                "Insufficient evidence to merge: no conflicts found, but "
                + ", ".join(unknowns) + " unavailable on at least one finding."
            ),
            evidence=evidence,
        )

    return PairDecision(
        decision=DedupDecision.DUPLICATE,
        reason=_build_match_reason(a, b, evidence),
        evidence=evidence,
    )


def _build_match_reason(a: FindingRecord, b: FindingRecord, evidence: dict) -> str:
    parts = [
        f"same stack signature",
        f"same ASan error type ({a.features.error_type})",
        f"same access type ({a.features.access_type})",
        f"same faulting function ({a.features.faulting_function})",
        f"same source location ({a.features.source_file}:{a.features.source_line})",
    ]
    supporting = []
    if evidence["access_size"] == EvidenceComparison.MATCH:
        supporting.append(f"access_size={a.features.access_size}")
    if evidence["memory_region"] == EvidenceComparison.MATCH:
        supporting.append(f"memory_region={a.features.memory_region}")
    text = "Merged: " + ", ".join(parts) + "."
    if supporting:
        text += " Supporting: " + ", ".join(supporting) + "."
    return text


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

@dataclass
class DedupGroup:
    group_id: str
    representative_identifier: str
    finding_ids: list          # list[str] — see _stable_identifier
    artifact_ids: list         # list[Optional[str]], parallel meaning to finding_ids' source
    count: int
    reason: str
    stack_signature: Optional[str]
    stack_signature_version: Optional[str]
    evidence_summary: dict


@dataclass
class DeduplicationResult:
    groups: list                # list[DedupGroup]
    total_input_count: int

    def total_grouped_count(self) -> int:
        return sum(g.count for g in self.groups)


def _blocking_key(record: FindingRecord):
    """
    Deterministic bucket key. Records sharing this key are the only
    ones ever pairwise-compared for DUPLICATE — safe because
    stack_signature (+version) and finding_state are themselves
    required-MATCH blocking dimensions, so two records that don't
    share this key can never be DUPLICATE anyway. This turns full
    O(n^2) pairwise comparison into O(n) bucketing plus O(k^2) within
    each bucket of size k, which is far smaller in practice since
    genuinely-identical stack signatures are the exception, not the
    rule, across a whole campaign.

    Records with no stack signature available (Phase 8 not run, or an
    empty stack) get a bucket key of None and are never merged with
    anything — they are always singleton groups. This is correct, not
    just an optimization: DUPLICATE requires an actual signature
    MATCH, which is impossible when either side has no signature.
    """
    sig = _signature_pair(record)
    if sig is None:
        return None
    return (record.features.finding_state.value, sig[0], sig[1])


class _UnionFind:
    def __init__(self, items: list):
        self._parent = {item: item for item in items}

    def find(self, item):
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic merge direction so the result never
            # depends on call order.
            if ra < rb:
                self._parent[rb] = ra
            else:
                self._parent[ra] = rb


def _completeness_score(features: CrashFeatures) -> int:
    """Higher = more complete ASan evidence. Used only for representative selection."""
    fields = (
        features.error_type, features.access_type, features.access_size,
        features.fault_address, features.memory_region, features.faulting_function,
        features.source_file, features.source_line,
    )
    return sum(1 for f in fields if f is not None)


def choose_representative(records: list) -> FindingRecord:
    """
    Deterministic representative selection, in this exact priority
    order (documented, not arbitrary):

      1. reproduction_status == REPRODUCED preferred over anything else
      2. more complete ASan evidence preferred (see _completeness_score)
      3. deeper/more complete stack preferred (stack_depth; None treated as 0)
      4. stable identifier, ascending, as the final deterministic tie-breaker

    Choosing a representative never modifies any of the other records
    in the group — it only selects which one to surface as primary.
    """
    def sort_key(record: FindingRecord):
        reproduced_first = 0 if record.features.reproduction_status == ReproductionStatus.REPRODUCED else 1
        completeness = -_completeness_score(record.features)  # more complete first -> negate for ascending sort
        depth = -(record.features.stack_depth or 0)
        return (reproduced_first, completeness, depth, _stable_identifier(record))

    return sorted(records, key=sort_key)[0]


def _group_id(identifiers: list) -> str:
    """
    Deterministic, order-independent group id: a hash of the SET of
    member identifiers (sorted before hashing), never of insertion
    order, timestamps, or randomness.
    """
    canonical = "|".join(sorted(identifiers))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _build_group(records: list) -> DedupGroup:
    identifiers = [_stable_identifier(r) for r in records]
    representative = choose_representative(records)
    rep_id = _stable_identifier(representative)

    sig_pair = _signature_pair(representative)

    if len(records) == 1:
        sig = _signature_pair(records[0])
        if sig is None:
            reason = "Independent finding — no stack signature available for comparison."
        else:
            reason = "Independent finding — no other finding shared compatible evidence."
    else:
        # All pairs in a merged group are DUPLICATE by construction
        # (Union-Find only unions on DUPLICATE edges); reuse one
        # representative pairwise comparison for the group-level reason
        # since every blocking dimension is, by definition, identical
        # across the whole group.
        reason = _build_match_reason(records[0], records[1], compare_evidence(records[0], records[1]).evidence)

    return DedupGroup(
        group_id=_group_id(identifiers),
        representative_identifier=rep_id,
        finding_ids=sorted(identifiers),
        artifact_ids=sorted({r.features.artifact_path for r in records if r.features.artifact_path is not None}),
        count=len(records),
        reason=reason,
        stack_signature=sig_pair[0] if sig_pair else None,
        stack_signature_version=sig_pair[1] if sig_pair else None,
        evidence_summary={
            "finding_state": records[0].features.finding_state.value,
            "error_type": representative.features.error_type,
            "access_type": representative.features.access_type,
            "faulting_function": representative.features.faulting_function,
            "source_file": representative.features.source_file,
            "source_line": representative.features.source_line,
        },
    )


def deduplicate(records: list) -> DeduplicationResult:
    """
    Group `records` (list[FindingRecord]) into DedupGroups using
    conservative, evidence-based comparison.

    Deterministic and order-independent: deduplicate(records) and
    deduplicate(list(reversed(records))) always produce the same set
    of groups (same membership, same group_ids) — group_id and
    representative selection are both computed from record content,
    never from list position.

    Every input record is preserved and referenced by exactly one
    output group (verified by DeduplicationResult.total_grouped_count()
    always equaling total_input_count) — deduplication only creates
    relationships, it never discards, merges, or mutates the
    underlying CrashFeatures/NormalizedStack evidence.
    """
    if not records:
        return DeduplicationResult(groups=[], total_input_count=0)

    identifiers = [_stable_identifier(r) for r in records]
    by_id = dict(zip(identifiers, records))
    uf = _UnionFind(identifiers)

    buckets: dict = {}
    for record, identifier in zip(records, identifiers):
        key = _blocking_key(record)
        if key is None:
            continue  # no signature -> can never be DUPLICATE with anything, stays singleton
        buckets.setdefault(key, []).append((identifier, record))

    for bucket_records in buckets.values():
        for i in range(len(bucket_records)):
            id_a, record_a = bucket_records[i]
            for j in range(i + 1, len(bucket_records)):
                id_b, record_b = bucket_records[j]
                decision = compare_evidence(record_a, record_b)
                if decision.decision == DedupDecision.DUPLICATE:
                    uf.union(id_a, id_b)

    components: dict = {}
    for identifier in identifiers:
        root = uf.find(identifier)
        components.setdefault(root, []).append(by_id[identifier])

    groups = [_build_group(members) for members in components.values()]
    groups.sort(key=lambda g: g.group_id)  # deterministic output order

    return DeduplicationResult(groups=groups, total_input_count=len(records))
