"""
Phase 11 — explainable security risk / triage prioritization.

Answers a DIFFERENT question than Phases 9 and 10:

    Phase 9  : "Which artifacts represent the same logical finding?"
    Phase 10 : "Which DISTINCT findings exhibit similar behavior?"
    Phase 11 : "Which findings should a security engineer investigate
                FIRST, based on the evidence currently available?"

This is EXPLAINABLE TRIAGE PRIORITIZATION, not vulnerability proof,
not exploitability proof, not a CVSS replacement, not root-cause
analysis. Every result carries the phrase "investigation priority",
never "confirmed severity" — see PriorityResult.SCOPE_DISCLAIMER,
which is attached to every single result regardless of score.

----------------------------------------------------------------------
Inspection note (repository reality)
----------------------------------------------------------------------
`Crash.priority_score` / `Crash.severity` and `Cluster.priority_score`
/ `Cluster.severity` already exist as nullable columns in models.py,
explicitly commented "# Phase 11 — not populated yet" (confirmed by
inspection before writing this file). Phase 11 does NOT write to them
here — it returns structured PriorityResult objects, exactly like
Phase 9/10 return DeduplicationResult/ClusteringResult without
touching the database themselves. Wiring persistence is Phase 12's
job (pipeline orchestration + API), keeping this module a pure,
side-effect-free, independently testable service layer.

----------------------------------------------------------------------
Scoring model — every point is documented, nothing is arbitrary
----------------------------------------------------------------------
Total scale: 0-100 integer points (no fake precision), built from five
independent, non-double-counted evidence dimensions. "Independent"
means each measures a genuinely distinct fact:

  CONFIRMED_CRASH   (+30, DIRECT)   — did the target actually terminate
                       abnormally at all? The foundational fact that
                       makes a finding worth investigating.
  REPRODUCED        (+25, DIRECT)   — was the failure independently
                       re-triggered under controlled conditions? Higher
                       confidence this is real, not a fluke/environment
                       artifact. NEVER implies "therefore exploitable".
  ASAN_CONFIRMED    (+25, DIRECT)   — did a sanitizer independently
                       classify a specific memory-safety violation,
                       vs. a bare, unclassified signal? Measures
                       diagnostic DEPTH, not "how scary the ASan
                       category sounds" — no per-error-type weighting
                       exists anywhere in this module.
  WRITE_ACCESS      (+10, DERIVED)  — a write-primitive memory-safety
                       violation is conventionally treated as carrying
                       higher exploitation-relevant risk than a
                       read-primitive one in standard memory-safety
                       triage practice (a wild write can corrupt
                       control-flow-relevant data; a wild read
                       typically cannot). Only awarded when access_type
                       is actually known — UNKNOWN access_type gets
                       zero contribution, never guessed.
  COMPLETE_LOCATION (+10, DIRECT)   — is the finding fully localized
                       (function + file + line all present)? Measures
                       how actionable/well-understood the finding is
                       for an investigating engineer.

  30 + 25 + 25 + 10 + 10 = 100 — the scale's own maximum IS the sum of
  every defensible contribution; nothing was scaled/curve-fit to hit
  100, it simply falls out of the five components.

Explicitly EXCLUDED from scoring (see FEATURE_AUDIT below for the full
table with reasons): memory_region (no defensible universal ordering),
stack_depth (a Phase 10 behavioral feature, not risk evidence),
Phase 9 dedup_group.count / raw artifact volume (explicitly forbidden
by the project's adversarial Test C — exposed as metadata only),
Phase 10 cluster_size (explicitly forbidden by adversarial Test D —
exposed as metadata only), AFL++ execution/coverage counts (explicitly
forbidden assumption per the spec — "more executions != more
dangerous").

----------------------------------------------------------------------
HANG / REPRODUCTION_FAILURE / NORMAL handling
----------------------------------------------------------------------
Only CRASH-state findings go through the 5-dimension score above.
HANG is a structurally different evidence class (could be a legitimate
slow path OR a real DoS-relevant issue) that this module does not yet
have a defensible way to further rank against crashes or against other
hangs on the same numeric scale — HANG findings get priority=MEDIUM
by a fixed, documented, non-computed rule, with score=None (not 0,
not guessed). REPRODUCTION_FAILURE and NORMAL findings get
priority=INSUFFICIENT_EVIDENCE with score=None — we either don't know
if it's even a crash (REPRODUCTION_FAILURE), or it isn't one (NORMAL),
so a crash-investigation-priority score does not apply.

----------------------------------------------------------------------
Determinism / batch-independence
----------------------------------------------------------------------
score_finding() takes exactly ONE PrioritizationInput and returns
exactly one PriorityResult — it has no visibility into any other
finding, so it is structurally impossible for one finding's score to
be affected by another finding being present, absent, or having any
particular value. This is the direct, provable fix for the same class
of batch-dependence problem identified and corrected in Phase 10.

----------------------------------------------------------------------
Security / purity
----------------------------------------------------------------------
No execution, no subprocess, no network, no LLM, no ML classifier, no
external vulnerability database lookups. Read-only with respect to all
CrashFeatures/DedupGroup/Cluster evidence passed in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.services.feature_extractor import CrashFeatures, FindingState, ReproductionStatus
from app.services.deduplicator import DedupGroup
from app.services.clusterer import Cluster, ClusteringResult, LogicalFinding

# ---------------------------------------------------------------------------
# Documented, configurable weights — see module docstring for the full
# justification of each. Exposed as a module-level dict (not buried in
# function bodies) so a caller can inspect or override them explicitly;
# overriding requires passing a full replacement dict to score_finding's
# `weights` parameter, not silent monkeypatching.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "CONFIRMED_CRASH": 30,
    "REPRODUCED": 25,
    "ASAN_CONFIRMED": 25,
    "WRITE_ACCESS": 10,
    "COMPLETE_LOCATION": 10,
}
assert sum(DEFAULT_WEIGHTS.values()) == 100  # the scale's max IS the sum — enforced, not asserted-and-hoped

# Threshold boundaries, documented by which evidence combination lands
# in each band (see module docstring's worked combinations):
#   CRITICAL: >=90  — essentially every dimension positive at once
#             (e.g. crash+reproduced+asan+one of {write,location} = 90)
#   HIGH:     65-89 — strong evidence, one meaningful dimension short
#             (e.g. crash+reproduced+asan = 80)
#   MEDIUM:   40-64 — real but partial direct evidence
#             (e.g. crash+asan only = 55, or crash+reproduced only = 55)
#   LOW:      1-39  — minimal confirmed evidence
#             (e.g. crash alone = 30)
CRITICAL_THRESHOLD = 90
HIGH_THRESHOLD = 65
MEDIUM_THRESHOLD = 40

SCOPE_DISCLAIMER = (
    "This is an internal evidence-based triage priority, not a formal "
    "CVSS assessment, confirmed vulnerability severity, or exploitability "
    "determination."
)


# ---------------------------------------------------------------------------
# Input bundle — the Phase 9/10 -> Phase 11 boundary
# ---------------------------------------------------------------------------

@dataclass
class PrioritizationInput:
    """One finding, ready for prioritization, with its Phase 9/10 context attached."""
    identifier: str
    features: CrashFeatures
    dedup_group: Optional[DedupGroup] = None
    cluster: Optional[Cluster] = None   # None if noise/unclustered, or Phase 10 wasn't run


def build_prioritization_inputs(
    logical_findings: list,
    clustering_result: Optional[ClusteringResult] = None,
) -> list:
    """
    Bridge Phase 10's output into Phase 11's input: attach each
    LogicalFinding's Cluster (if it ended up in one) by looking it up
    from ClusteringResult -- LogicalFinding itself doesn't carry a
    back-reference to its cluster (ClusteringResult only maps
    cluster -> member_ids, not member -> cluster), so this function is
    the explicit bridge, mirroring how Phase 10 bridged Phase 9's
    output via build_logical_findings().
    """
    member_to_cluster = {}
    if clustering_result is not None:
        for cluster in clustering_result.clusters:
            for member_id in cluster.member_ids:
                member_to_cluster[member_id] = cluster

    return [
        PrioritizationInput(
            identifier=finding.identifier,
            features=finding.features,
            dedup_group=finding.dedup_group,
            cluster=member_to_cluster.get(finding.identifier),
        )
        for finding in logical_findings
    ]


# ---------------------------------------------------------------------------
# Evidence / result types
# ---------------------------------------------------------------------------

@dataclass
class EvidenceContribution:
    dimension: str
    value: Optional[object]
    contribution: int
    evidence_type: str   # "DIRECT" | "DERIVED" | "UNKNOWN"
    rationale: str


@dataclass
class PriorityResult:
    finding_id: str
    score: Optional[int]        # 0-100, or None for HANG / INSUFFICIENT_EVIDENCE paths
    priority: str                # CRITICAL | HIGH | MEDIUM | LOW | INSUFFICIENT_EVIDENCE
    contributions: list           # list[EvidenceContribution]
    uncertainties: list           # list[str] — always includes SCOPE_DISCLAIMER
    metadata: dict                 # contextual, non-scoring info (dedup count, cluster size, etc.)
    rank: Optional[int] = None    # filled in by prioritize(), None if scored alone via score_finding()


def _priority_band(score: int) -> str:
    if score >= CRITICAL_THRESHOLD:
        return "CRITICAL"
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _evidence_completeness_count(contributions: list) -> int:
    """Number of DIRECT/DERIVED contributions that actually fired (contribution > 0).
    Used only for deterministic tie-breaking, never for the score itself."""
    return sum(1 for c in contributions if c.evidence_type != "UNKNOWN" and c.contribution > 0)


def score_finding(inp: PrioritizationInput, weights: dict = None) -> PriorityResult:
    """
    Score exactly ONE finding. Takes no batch/context beyond `inp`
    itself -- see module docstring's "Determinism / batch-independence"
    section for why this structurally guarantees one finding's score
    can never be affected by any other finding.
    """
    weights = weights or DEFAULT_WEIGHTS
    f = inp.features
    metadata = _build_metadata(inp)

    if f.finding_state == FindingState.HANG:
        return PriorityResult(
            finding_id=inp.identifier, score=None, priority="MEDIUM",
            contributions=[],
            uncertainties=[
                "Timeout/hang behavior, not a confirmed crash — may indicate a "
                "denial-of-service-relevant issue or a legitimate slow path; this "
                "module does not yet have a defensible basis to rank hangs "
                "numerically against crashes or against each other.",
                SCOPE_DISCLAIMER,
            ],
            metadata=metadata,
        )

    if f.finding_state in (FindingState.REPRODUCTION_FAILURE, FindingState.NORMAL):
        reason = (
            "Reproduction could not be attempted — unknown whether this "
            "artifact represents a real crash at all."
            if f.finding_state == FindingState.REPRODUCTION_FAILURE else
            "Not a confirmed crash or hang — no investigation priority applies."
        )
        return PriorityResult(
            finding_id=inp.identifier, score=None, priority="INSUFFICIENT_EVIDENCE",
            contributions=[], uncertainties=[reason, SCOPE_DISCLAIMER], metadata=metadata,
        )

    # --- CRASH-state findings: the five-dimension scored path ---
    contributions = []
    uncertainties = [SCOPE_DISCLAIMER]

    contributions.append(EvidenceContribution(
        dimension="CONFIRMED_CRASH", value=True, contribution=weights["CONFIRMED_CRASH"],
        evidence_type="DIRECT",
        rationale="Target terminated abnormally under a confirmed CRASH finding state.",
    ))

    if f.reproduction_status == ReproductionStatus.REPRODUCED:
        contributions.append(EvidenceContribution(
            dimension="REPRODUCED", value=True, contribution=weights["REPRODUCED"],
            evidence_type="DIRECT",
            rationale="Failure was independently re-triggered under controlled reproduction.",
        ))
    else:
        contributions.append(EvidenceContribution(
            dimension="REPRODUCED", value=f.reproduction_status.value if f.reproduction_status else None,
            contribution=0, evidence_type="UNKNOWN",
            rationale="Reproduction not confirmed — treated as unknown, not penalized.",
        ))
        uncertainties.append(
            f"Reproduction status is {f.reproduction_status.value if f.reproduction_status else 'unknown'} "
            "— not independently confirmed to be consistently reproducible."
        )

    if f.asan_detected:
        contributions.append(EvidenceContribution(
            dimension="ASAN_CONFIRMED", value=True, contribution=weights["ASAN_CONFIRMED"],
            evidence_type="DIRECT",
            rationale="Sanitizer independently classified a specific memory-safety violation.",
        ))
    else:
        contributions.append(EvidenceContribution(
            dimension="ASAN_CONFIRMED", value=False, contribution=0, evidence_type="UNKNOWN",
            rationale="No sanitizer diagnostics available (signal-only crash, or ASan not attached).",
        ))
        uncertainties.append("No sanitizer (ASan) diagnostics available for this finding.")

    if f.access_type == "WRITE":
        contributions.append(EvidenceContribution(
            dimension="WRITE_ACCESS", value="WRITE", contribution=weights["WRITE_ACCESS"],
            evidence_type="DERIVED",
            rationale="Write-primitive memory-safety violations are conventionally treated as "
                      "carrying higher exploitation-relevant risk than read-primitive violations.",
        ))
    elif f.access_type == "READ":
        contributions.append(EvidenceContribution(
            dimension="WRITE_ACCESS", value="READ", contribution=0, evidence_type="DERIVED",
            rationale="Read-primitive access — no write-access bonus applies.",
        ))
    else:
        contributions.append(EvidenceContribution(
            dimension="WRITE_ACCESS", value=None, contribution=0, evidence_type="UNKNOWN",
            rationale="Access type not known — not guessed, no contribution.",
        ))
        uncertainties.append("Memory access type (read/write) is not known for this finding.")

    location_complete = bool(f.faulting_function and f.source_file and f.source_line is not None)
    if location_complete:
        contributions.append(EvidenceContribution(
            dimension="COMPLETE_LOCATION", value=True, contribution=weights["COMPLETE_LOCATION"],
            evidence_type="DIRECT",
            rationale="Finding is fully localized to a specific function, file, and line.",
        ))
    else:
        contributions.append(EvidenceContribution(
            dimension="COMPLETE_LOCATION", value=False, contribution=0, evidence_type="UNKNOWN",
            rationale="Fault location is incomplete (function/file/line not all present).",
        ))
        uncertainties.append("Fault location is incomplete — investigation may require extra effort to localize.")

    uncertainties.append("Exploitability has not been established.")
    uncertainties.append("No external vulnerability (e.g. CVE) correlation has been performed.")

    score = sum(c.contribution for c in contributions)
    score = max(0, min(100, score))  # defensive clamp; unreachable given the weight table, but bounds are a hard guarantee

    return PriorityResult(
        finding_id=inp.identifier, score=score, priority=_priority_band(score),
        contributions=contributions, uncertainties=uncertainties, metadata=metadata,
    )


def _build_metadata(inp: PrioritizationInput) -> dict:
    """
    Contextual information that is deliberately NEVER used in scoring:
    raw artifact/dedup count and cluster size are explicitly excluded
    from the score (see module docstring) but are still useful context
    for a human reviewing the result -- exposed here, separately, so
    there is no ambiguity about whether they influenced the number.
    """
    metadata = {}
    if inp.dedup_group is not None:
        metadata["dedup_artifact_count"] = inp.dedup_group.count
        metadata["dedup_group_id"] = inp.dedup_group.group_id
    if inp.cluster is not None:
        metadata["behavioral_cluster_id"] = inp.cluster.cluster_id
        metadata["behavioral_cluster_size"] = inp.cluster.member_count
    return metadata


# ---------------------------------------------------------------------------
# Batch ranking
# ---------------------------------------------------------------------------

def _rank_sort_key(result: PriorityResult):
    """
    Deterministic ranking key:
      1. score descending (None treated as lower than any real score)
      2. evidence completeness descending (more fired DIRECT/DERIVED
         contributions ranks first among equal scores)
      3. finding_id ascending (final deterministic tie-breaker; never
         timestamp, random UUID, or insertion order)
    """
    score_for_sort = result.score if result.score is not None else -1
    completeness = _evidence_completeness_count(result.contributions)
    return (-score_for_sort, -completeness, result.finding_id)


def prioritize(inputs: list, weights: dict = None) -> list:
    """
    Score every input independently (see score_finding's determinism
    guarantee), then produce a single deterministic ranking across the
    whole batch. Input order has no effect on the output ranking.
    """
    results = [score_finding(inp, weights=weights) for inp in inputs]
    results.sort(key=_rank_sort_key)
    for i, result in enumerate(results, start=1):
        result.rank = i
    return results


# ---------------------------------------------------------------------------
# FEATURE_AUDIT — documented, kept in sync with the code above
# ---------------------------------------------------------------------------
#
# Dimension           | Source                              | Type      | Direct/Derived/Unknown | Used in score? | Reason                                                              | Double-counting risk
# --------------------|--------------------------------------|-----------|------------------------|-----------------|----------------------------------------------------------------------|---------------------------------------------
# finding_state        | CrashFeatures.finding_state          | categorical | DIRECT                | yes (gate + base) | determines which scoring path applies at all; CRASH is the foundational fact | none — used once, as the gate
# reproduction_status  | CrashFeatures.reproduction_status     | categorical | DIRECT/UNKNOWN         | yes             | independent re-confirmation of the failure                          | none — distinct fact from asan_detected
# asan_detected        | CrashFeatures.asan_detected           | boolean     | DIRECT/UNKNOWN         | yes             | diagnostic depth/specificity, not "how scary the category sounds"    | none — never combined with error_type severity
# access_type          | CrashFeatures.access_type              | categorical | DERIVED/UNKNOWN        | yes (small)     | documented security-triage convention (write > read)                 | none — orthogonal to asan_detected (whether vs. what kind)
# fault location complete | CrashFeatures.faulting_function/source_file/source_line | boolean (derived) | DIRECT/UNKNOWN | yes | actionability/investigability of the finding | checked: correlated with asan_detected in practice (same ASan report) but measures a DIFFERENT fact (localization vs. classification) — kept as a separate, smaller-weighted dimension deliberately
# error_type (specific)| CrashFeatures.error_type              | categorical | n/a                    | NO (metadata only) | no per-category severity table exists or is defensible without inventing one | would double-count with asan_detected if scored
# memory_region        | CrashFeatures.memory_region            | categorical | n/a                    | NO (metadata only) | no defensible universal severity ordering (HEAP vs STACK vs GLOBAL) | n/a
# stack_depth          | CrashFeatures.stack_depth               | numeric     | n/a                    | NO                | Phase 10 behavioral feature, not risk evidence                       | n/a
# dedup_group.count    | DedupGroup (Phase 9)                    | numeric     | n/a                    | NO (metadata only) | explicitly forbidden — raw artifact volume must not inflate priority (adversarial Test C) | would double-count fuzzer mutation volume as "danger"
# cluster.member_count | Cluster (Phase 10)                       | numeric     | n/a                    | NO (metadata only) | explicitly forbidden — cluster size is relationship context, not severity (adversarial Test D) | would amplify the same underlying artifact volume via a second path
# AFL++ execs/time      | CrashFeatures.raw_afl_filename_metadata | numeric     | n/a                    | NO                | "more executions/coverage = more dangerous" is an explicitly forbidden assumption | n/a
