"""
Phase 12 — real end-to-end FuzzTriage pipeline orchestration.

This module is an ORCHESTRATOR ONLY. It contains no parsing logic, no
feature-derivation logic, no deduplication logic, no clustering
algorithm, and no scoring formula — every one of those already has
exactly one authoritative implementation in Phases 3-11, and this
module calls them, never reimplements them. See the module-level
integration map in the accompanying commit/report for the full
service-by-service input/output/failure-mode audit performed before
writing any code here.

Pipeline:

    AFL++ campaign directory
            |
            v
    artifact_collector.collect_artifacts()      (Phase 4)
            |
            v
    afl_parser.parse_fuzzer_stats()             (Phase 3, campaign metadata only)
    per-crash: reproducer.reproduce_crash()      (Phase 6, OPTIONAL)
    per-crash: asan_parser.parse_asan_report()   (Phase 5, only if reproduction produced stderr)
    per-artifact: feature_extractor.extract_features()  (Phase 7)
    per-finding: stack_normalizer.normalize_crash_features_stack()  (Phase 8)
            |
            v
    deduplicator.deduplicate()                  (Phase 9)
            |
            v
    clusterer.build_logical_findings() + cluster_findings()  (Phase 10)
            |
            v
    prioritizer.build_prioritization_inputs() + prioritize()  (Phase 11)
            |
            v
    PipelineResult

----------------------------------------------------------------------
The hang-classification adapter (documented up front, not buried)
----------------------------------------------------------------------
Phase 7's `_derive_finding_state()` can only produce FindingState.HANG
from a REAL ReproductionResult with timed_out=True. Reproducing every
hang artifact by default would mean deliberately re-running
potentially long/infinite-looping inputs for every hang in a campaign
-- this pipeline does not do that unless the caller explicitly opts in
via `reproduce_hangs=True`. By default, hang artifacts get
FindingState.HANG assigned directly from Phase 4's own real
classification (their presence in the campaign's hangs/ directory IS
direct, non-fabricated evidence -- AFL++ determined this during actual
fuzzing execution, not something this module invented). This never
constructs a fake ReproductionResult; when reproduce_hangs is False,
no reproduction object is created for hangs at all.

----------------------------------------------------------------------
Fatal vs. artifact-level errors
----------------------------------------------------------------------
FATAL (raises immediately, no PipelineResult produced):
    - fuzz_output_dir is not a valid Path/str
    - eps/min_samples/weights are structurally invalid configuration
      (these are caller configuration mistakes, not data-quality
      problems -- Phase 10/11 already validate and raise for these,
      this module does not swallow that)

ARTIFACT-LEVEL (caught, recorded as an ArtifactError, that one
artifact excluded from findings, campaign processing continues):
    - any exception raised while extracting features, normalizing the
      stack, or (if enabled) reproducing ONE specific artifact

----------------------------------------------------------------------
Security
----------------------------------------------------------------------
All artifact paths come only from artifact_collector's own discovery
(Path.iterdir() under the configured fuzz_output_dir) -- this module
never constructs a path from untrusted string concatenation, never
invokes a shell, and passes every path to reproduce_crash() as a
Path object exactly as artifact_collector produced it, never modified.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from app.services.afl_parser import parse_fuzzer_stats
from app.services.artifact_collector import ArtifactRecord, collect_artifacts
from app.services.asan_parser import parse_asan_report
from app.services.reproducer import TargetCommand, reproduce_crash
from app.services.feature_extractor import CrashFeatures, FindingState, extract_features
from app.services.stack_normalizer import NormalizedStack, normalize_crash_features_stack
from app.services.deduplicator import FindingRecord, deduplicate
from app.services.clusterer import (
    DEFAULT_EPS, DEFAULT_MIN_SAMPLES, build_logical_findings, cluster_findings,
)
from app.services.prioritizer import DEFAULT_WEIGHTS, build_prioritization_inputs, prioritize

logger = logging.getLogger("fuzztriage.pipeline")


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class ArtifactError:
    artifact_id: str
    stage: str
    error_type: str
    message: str


@dataclass
class StageStatus:
    stage: str
    status: str          # "COMPLETED" | "SKIPPED"
    processed_count: int
    error_count: int


@dataclass
class CampaignMetadata:
    fuzz_output_dir: str
    campaign_id: Optional[str]
    afl_stats: dict            # AflStats.raw -- context only, NEVER used in scoring (see prioritizer's FEATURE_AUDIT)
    queue_count: int
    crash_artifact_count: int
    hang_artifact_count: int


@dataclass
class PipelineResult:
    campaign: CampaignMetadata
    findings: list              # list[CrashFeatures] that survived extraction (pre-dedup, for transparency)
    deduplication: dict          # {"groups": [...], "total_input_count": int} -- see _dedup_summary
    clustering: dict             # {"clusters": [...], "noise_ids": [...], ...} -- see _clustering_summary
    priorities: list             # list[PriorityResult]
    artifact_errors: list        # list[ArtifactError]
    stages: list                 # list[StageStatus]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    fuzz_output_dir,
    campaign_id: Optional[str] = None,
    target_command: Optional[TargetCommand] = None,
    timeout_seconds: float = 5.0,
    reproduce_hangs: bool = False,
    cluster_eps: float = DEFAULT_EPS,
    cluster_min_samples: int = DEFAULT_MIN_SAMPLES,
    priority_weights: dict = None,
) -> PipelineResult:
    """
    Run the complete FuzzTriage pipeline over one AFL++ campaign
    output directory. See module docstring for the fatal-vs-artifact-
    level error model and the hang-classification adapter.

    `target_command` is optional and, by design, the ONLY way this
    pipeline ever executes anything: if it is None, no binary is ever
    launched, and crash findings are still produced (with
    reproduction_status=NOT_ATTEMPTED, asan_detected=False, honestly
    reflecting the absence of that evidence source) -- consistent with
    "do not launch binaries without an explicit existing contract
    requiring it."
    """
    if not isinstance(fuzz_output_dir, (str, Path)):
        raise TypeError(f"fuzz_output_dir must be a str or Path, got {type(fuzz_output_dir)!r}")

    fuzz_output_dir = Path(fuzz_output_dir)
    stages: list = []
    artifact_errors: list = []

    # --- Stage: artifact discovery (Phase 4) ---
    collection = collect_artifacts(fuzz_output_dir)
    stages.append(StageStatus(
        stage="artifact_discovery", status="COMPLETED",
        processed_count=collection.queue_count + collection.crash_count + collection.hang_count,
        error_count=0,
    ))

    # --- Stage: AFL stats parsing (Phase 3) — context only, never scored ---
    afl_stats = parse_fuzzer_stats(fuzz_output_dir / "fuzzer_stats")
    stages.append(StageStatus(stage="afl_stats_parsing", status="COMPLETED",
                               processed_count=0 if afl_stats.is_empty else 1, error_count=0))

    campaign_metadata = CampaignMetadata(
        fuzz_output_dir=str(fuzz_output_dir),
        campaign_id=campaign_id,
        afl_stats=dict(afl_stats.raw),
        queue_count=collection.queue_count,
        crash_artifact_count=collection.crash_count,
        hang_artifact_count=collection.hang_count,
    )

    # --- Stage: crash processing (Phases 5/6/7/8) ---
    records: list = []
    crash_processed, crash_errors = 0, 0
    for artifact in collection.crashes:
        crash_processed += 1
        try:
            record = _process_crash_artifact(artifact, target_command, timeout_seconds, campaign_id)
            records.append(record)
        except Exception as exc:  # noqa: BLE001 — artifact-level isolation is the whole point here
            crash_errors += 1
            artifact_errors.append(ArtifactError(
                artifact_id=artifact.path, stage="crash_processing",
                error_type=type(exc).__name__, message=str(exc),
            ))
            logger.warning("crash_processing failed campaign=%s artifact=%s error=%s",
                            campaign_id, artifact.filename, exc)
    stages.append(StageStatus(stage="crash_processing", status="COMPLETED",
                               processed_count=crash_processed, error_count=crash_errors))

    # --- Stage: hang processing (Phase 7, direct classification — see module docstring) ---
    hang_processed, hang_errors = 0, 0
    for artifact in collection.hangs:
        hang_processed += 1
        try:
            record = _process_hang_artifact(artifact, target_command, timeout_seconds,
                                             reproduce_hangs, campaign_id)
            records.append(record)
        except Exception as exc:  # noqa: BLE001
            hang_errors += 1
            artifact_errors.append(ArtifactError(
                artifact_id=artifact.path, stage="hang_processing",
                error_type=type(exc).__name__, message=str(exc),
            ))
            logger.warning("hang_processing failed campaign=%s artifact=%s error=%s",
                            campaign_id, artifact.filename, exc)
    stages.append(StageStatus(stage="hang_processing", status="COMPLETED",
                               processed_count=hang_processed, error_count=hang_errors))

    findings = [r.features for r in records]

    if not records:
        stages.append(StageStatus(stage="deduplication", status="SKIPPED", processed_count=0, error_count=0))
        stages.append(StageStatus(stage="clustering", status="SKIPPED", processed_count=0, error_count=0))
        stages.append(StageStatus(stage="prioritization", status="SKIPPED", processed_count=0, error_count=0))
        return PipelineResult(
            campaign=campaign_metadata, findings=findings,
            deduplication={"groups": [], "total_input_count": 0},
            clustering={"clusters": [], "noise_ids": [], "total_input_count": 0,
                        "config": {"eps": cluster_eps, "min_samples": cluster_min_samples}},
            priorities=[], artifact_errors=artifact_errors, stages=stages,
        )

    # --- Stage: deduplication (Phase 9) — fatal on real config errors, never on data ---
    dedup_result = deduplicate(records)
    stages.append(StageStatus(stage="deduplication", status="COMPLETED",
                               processed_count=len(records), error_count=0))

    # --- Stage: clustering (Phase 10) ---
    records_by_id = {r.identifier: r for r in records}
    logical_findings = build_logical_findings(dedup_result, records_by_id)
    clustering_result = cluster_findings(logical_findings, eps=cluster_eps, min_samples=cluster_min_samples)
    stages.append(StageStatus(stage="clustering", status="COMPLETED",
                               processed_count=len(logical_findings), error_count=0))

    # --- Stage: prioritization (Phase 11) ---
    prioritization_inputs = build_prioritization_inputs(logical_findings, clustering_result)
    priorities = prioritize(prioritization_inputs, weights=priority_weights or DEFAULT_WEIGHTS)
    stages.append(StageStatus(stage="prioritization", status="COMPLETED",
                               processed_count=len(priorities), error_count=0))

    return PipelineResult(
        campaign=campaign_metadata,
        findings=findings,
        deduplication=_dedup_summary(dedup_result),
        clustering=_clustering_summary(clustering_result),
        priorities=priorities,
        artifact_errors=artifact_errors,
        stages=stages,
    )


def _process_crash_artifact(
    artifact: ArtifactRecord,
    target_command: Optional[TargetCommand],
    timeout_seconds: float,
    campaign_id: Optional[str],
) -> FindingRecord:
    reproduction = None
    asan = None
    if target_command is not None:
        reproduction = reproduce_crash(target_command, Path(artifact.path), timeout_seconds=timeout_seconds)
        if reproduction.stderr:
            asan = parse_asan_report(reproduction.stderr)

    features = extract_features(artifact=artifact, reproduction=reproduction, asan=asan, campaign_id=campaign_id)
    stack = normalize_crash_features_stack(features)
    return FindingRecord(features=features, stack=stack, identifier=artifact.path)


def _process_hang_artifact(
    artifact: ArtifactRecord,
    target_command: Optional[TargetCommand],
    timeout_seconds: float,
    reproduce_hangs: bool,
    campaign_id: Optional[str],
) -> FindingRecord:
    if reproduce_hangs and target_command is not None:
        # Real reproduction attempted -- if it genuinely times out
        # again, Phase 7's own derivation correctly assigns HANG from
        # real evidence, no adapter needed on this path at all.
        reproduction = reproduce_crash(target_command, Path(artifact.path), timeout_seconds=timeout_seconds)
        asan = parse_asan_report(reproduction.stderr) if reproduction.stderr else None
        features = extract_features(artifact=artifact, reproduction=reproduction, asan=asan, campaign_id=campaign_id)
        if features.finding_state != FindingState.HANG:
            # It didn't reproduce as a hang this run (timing can be
            # flaky) -- Phase 4's own classification is still the more
            # reliable signal, so override the state explicitly rather
            # than silently reporting the wrong thing.
            features.finding_state = FindingState.HANG
    else:
        # No reproduction attempted -- construct CrashFeatures directly
        # from Phase 4's own real classification (see module docstring's
        # "hang-classification adapter"). Never fabricates a
        # ReproductionResult.
        features = CrashFeatures(
            finding_state=FindingState.HANG,
            artifact_path=artifact.path,
            artifact_filename=artifact.filename,
            artifact_size=artifact.size_bytes,
            artifact_type=artifact.artifact_type,
            campaign_id=campaign_id,
        )

    stack = normalize_crash_features_stack(features)
    return FindingRecord(features=features, stack=stack, identifier=artifact.path)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _dedup_summary(result) -> dict:
    return {
        "groups": [_to_jsonable(g) for g in result.groups],
        "total_input_count": result.total_input_count,
    }


def _clustering_summary(result) -> dict:
    return {
        "clusters": [_to_jsonable(c) for c in result.clusters],
        "noise_ids": list(result.noise_ids),
        "total_input_count": result.total_input_count,
        "config": dict(result.config),
        "overall_silhouette": result.overall_silhouette,
    }


def _to_jsonable(obj):
    """
    Recursively convert dataclasses / Enums / dicts / lists into
    JSON-compatible primitives. No raw Python objects, filesystem
    handles, or exception objects survive this conversion.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)  # last-resort fallback; tests assert this path is never hit for known types


def pipeline_result_to_dict(result: PipelineResult) -> dict:
    """Full PipelineResult -> JSON-safe dict. `json.dumps(pipeline_result_to_dict(result))`
    must always succeed -- this is verified directly in tests, not just assumed."""
    return _to_jsonable(result)
