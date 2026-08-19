"""
Phase 7 — crash feature extraction.

Combines Phase 4 artifact metadata + Phase 5 ASan parser output +
Phase 6 reproduction output into a single structured CrashFeatures
object. This module performs no execution, no shell commands, no
network access, and no randomness — it is a pure, deterministic
transformation:

    ArtifactRecord (Phase 4)
    ReproductionResult (Phase 6)
    AsanReport (Phase 5)
            |
            v
    extract_features()
            |
            v
    CrashFeatures

Data-integrity design
----------------------
Raw evidence, parsed evidence, and derived features are kept distinct:

  - Raw evidence  (never mutated, always copied, not re-derived):
        artifact_path, artifact_filename, raw_stderr, raw_asan_report,
        raw_stack_trace, raw_afl_filename_metadata
  - Parsed evidence (taken as-is from Phase 5/6, not re-interpreted):
        error_type, access_type, access_size, faulting_function,
        source_file, source_line, return_code, signal, timed_out,
        duration_ms
  - Derived features (computed here, deterministically, from the above):
        finding_id, finding_state, reproduction_status, crash_type,
        top_frame, stack_depth

Nothing here performs stack normalization, hashing, deduplication,
clustering, or priority/severity scoring — those are Phases 8-11.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

from app.services.artifact_collector import ArtifactRecord
from app.services.asan_parser import AsanReport
from app.services.reproducer import ReproductionResult

FEATURE_SCHEMA_VERSION = "1.0"


class FindingState(str, Enum):
    """
    What kind of finding this is. A closed set, deliberately small:

      CRASH                 — actual crash evidence exists (ASan
                               detected it, or the process died from a
                               signal)
      HANG                  — the reproduction attempt timed out.
                               NEVER CRASH, even if that's why AFL++
                               flagged the artifact — a hang is a
                               distinct finding class.
      NORMAL                — reproduction ran to completion with no
                               crash evidence. This includes a
                               non-zero exit code with no ASan report
                               and no signal (e.g. the target's own
                               "invalid input" error path) — a bare
                               non-zero return code is NOT, by itself,
                               evidence of a crash.
      REPRODUCTION_FAILURE  — reproduction could not even be attempted
                               (Phase 6 validation/launch error) or
                               otherwise did not execute. The original
                               artifact is not discarded in this case;
                               it is represented explicitly as this
                               state.
    """
    CRASH = "CRASH"
    HANG = "HANG"
    NORMAL = "NORMAL"
    REPRODUCTION_FAILURE = "REPRODUCTION_FAILURE"


class ReproductionStatus(str, Enum):
    """Richer than a boolean — mirrors exactly what ReproductionResult can tell us."""
    REPRODUCED = "REPRODUCED"
    NOT_REPRODUCED = "NOT_REPRODUCED"
    TIMED_OUT = "TIMED_OUT"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    ERROR = "ERROR"


@dataclass
class CrashFeatures:
    """
    Normalized, structured representation of one artifact's analysis.

    Every field defaults to None/empty/"unknown" — nothing is ever
    fabricated. A field is populated only when the corresponding
    source object (artifact / reproduction / asan) actually provided
    that information; see `provenance` for exactly which source each
    populated field came from.
    """
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    # --- finding identity (derived, but deterministic — NOT a stack
    #     hash; that's Phase 8's concept, this is an artifact-identity
    #     based id) ---
    finding_id: Optional[str] = None
    finding_state: FindingState = FindingState.NORMAL

    # --- identity: raw evidence from Phase 4 ---
    artifact_path: Optional[str] = None
    artifact_filename: Optional[str] = None
    artifact_size: Optional[int] = None
    artifact_type: Optional[str] = None

    # --- reproduction: parsed evidence from Phase 6, plus one derived status enum ---
    reproduction_status: ReproductionStatus = ReproductionStatus.NOT_ATTEMPTED
    reproducible: Optional[bool] = None
    return_code: Optional[int] = None
    signal: Optional[int] = None
    timed_out: Optional[bool] = None
    duration_ms: Optional[float] = None

    # --- ASan: parsed evidence from Phase 5 ---
    asan_detected: bool = False
    sanitizer: Optional[str] = None          # e.g. "AddressSanitizer"; None if no sanitizer fired
    error_type: Optional[str] = None         # == AsanReport.error_class, unrenamed
    crash_type: Optional[str] = None         # same value as error_type — see module note below
    access_type: Optional[str] = None
    access_size: Optional[int] = None
    fault_address: Optional[str] = None
    memory_region: Optional[str] = None

    # --- location: parsed evidence from Phase 5 ---
    faulting_function: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None

    # --- stack: derived summary + raw copy (NOT normalized/hashed — Phase 8) ---
    top_frame: Optional[str] = None
    stack_depth: Optional[int] = None
    raw_stack_trace: list = field(default_factory=list)   # list[dict], independent copy

    # --- metadata ---
    campaign_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    # --- raw evidence, preserved verbatim; never overwritten by
    #     normalized/derived values ---
    raw_stderr: Optional[str] = None
    raw_asan_report: Optional[str] = None
    raw_afl_filename_metadata: dict = field(default_factory=dict)

    # --- provenance: maps populated field name -> which source object
    #     it came from. Only fields actually populated appear here. ---
    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NOTE on crash_type vs error_type
# ---------------------------------------------------------------------------
# Two already-established names exist in this codebase for the same
# underlying concept: Phase 5's AsanReport calls it `error_class`, and
# Phase 2's Crash SQLAlchemy model (models.py) calls its column
# `crash_type`. Rather than silently picking one and renaming the
# other, CrashFeatures exposes both `error_type` (mirroring Phase 5's
# term) and `crash_type` (mirroring the DB column's term) — both set
# from the same value, never independently derived.


def _has_crash_evidence(reproduction: Optional[ReproductionResult], asan: Optional[AsanReport]) -> bool:
    """
    Crash evidence = a full ASan report (preferred, when available) or
    a signal-based termination, with Phase 6's cheap `asan_detected`
    flag as a fallback when no full AsanReport was supplied. A bare
    non-zero return code is deliberately NOT crash evidence on its own
    (a target's normal "invalid input" error path also exits non-zero
    without crashing).

    Used by both _derive_finding_state and _derive_reproduction_status
    so the two derivations can never disagree with each other.
    """
    if asan is not None and asan.is_asan:
        return True
    if asan is None and reproduction is not None and reproduction.asan_detected:
        return True
    if reproduction is not None and reproduction.signal is not None:
        return True
    return False


def _derive_reproduction_status(
    reproduction: Optional[ReproductionResult],
    asan: Optional[AsanReport],
) -> ReproductionStatus:
    if reproduction is None:
        return ReproductionStatus.NOT_ATTEMPTED
    if not reproduction.executed:
        return ReproductionStatus.ERROR
    if reproduction.timed_out:
        return ReproductionStatus.TIMED_OUT
    if _has_crash_evidence(reproduction, asan):
        return ReproductionStatus.REPRODUCED
    return ReproductionStatus.NOT_REPRODUCED


def _derive_finding_state(
    reproduction: Optional[ReproductionResult],
    asan: Optional[AsanReport],
) -> FindingState:
    if reproduction is not None and reproduction.timed_out:
        return FindingState.HANG
    if reproduction is not None and not reproduction.executed:
        return FindingState.REPRODUCTION_FAILURE
    if _has_crash_evidence(reproduction, asan):
        return FindingState.CRASH
    return FindingState.NORMAL


def _compute_finding_id(artifact: Optional[ArtifactRecord]) -> Optional[str]:
    """
    Deterministic identity for this finding, derived from the
    artifact's own path (already a stable, unique identifier per
    artifact). This is explicitly NOT a stack hash (Phase 8), NOT a
    database primary key (Phase 2, auto-generated on insert), and NOT
    a cluster id (Phase 10) — those are different concepts belonging
    to different phases.
    """
    if artifact is None or not artifact.path:
        return None
    return hashlib.sha256(artifact.path.encode("utf-8")).hexdigest()[:16]


def _format_top_frame(asan: Optional[AsanReport]) -> Optional[str]:
    if asan is None or not asan.stack_trace:
        return None
    frame = asan.stack_trace[0]
    if frame.function and frame.source_file and frame.source_line is not None:
        return f"{frame.function} ({frame.source_file}:{frame.source_line})"
    if frame.function:
        return frame.function
    return None


def _copy_stack_trace(asan: Optional[AsanReport]) -> list:
    """Independent list[dict] copy — never returns/aliases the original list."""
    if asan is None or not asan.stack_trace:
        return []
    return [
        {
            "index": f.index,
            "function": f.function,
            "source_file": f.source_file,
            "source_line": f.source_line,
        }
        for f in asan.stack_trace
    ]


def _build_provenance(
    artifact: Optional[ArtifactRecord],
    reproduction: Optional[ReproductionResult],
    asan: Optional[AsanReport],
) -> dict:
    provenance: dict = {}

    if artifact is not None:
        provenance["artifact_path"] = "artifact_metadata"
        provenance["artifact_filename"] = "artifact_metadata"
        provenance["artifact_size"] = "artifact_metadata"
        provenance["artifact_type"] = "artifact_metadata"

    if reproduction is not None:
        for name in ("return_code", "signal", "timed_out", "duration_ms"):
            if getattr(reproduction, name, None) is not None:
                provenance[name] = "reproduction_result"

    if asan is not None and asan.is_asan:
        for name in ("access_type", "access_size", "memory_region"):
            if getattr(asan, name, None) is not None:
                provenance[name] = "asan_report"
        if asan.faulting_function is not None:
            provenance["faulting_function"] = "asan_report"
        if asan.source_file is not None:
            provenance["source_file"] = "asan_report"
        if asan.source_line is not None:
            provenance["source_line"] = "asan_report"
        provenance["error_type"] = "asan_report"
        provenance["crash_type"] = "asan_report"

    return provenance


def extract_features(
    artifact: Optional[ArtifactRecord] = None,
    reproduction: Optional[ReproductionResult] = None,
    asan: Optional[AsanReport] = None,
    campaign_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> CrashFeatures:
    """
    Deterministically build a CrashFeatures object from whatever
    combination of Phase 4/5/6 evidence is available.

    All three evidence arguments are optional and independent — this
    supports, for example, extracting features for a hang (artifact +
    reproduction, no ASan report at all) or for an artifact that
    couldn't be reproduced (artifact + a failed ReproductionResult, no
    ASan report).

    Never executes anything, never touches the filesystem or network,
    never mutates artifact/reproduction/asan — read-only with respect
    to all source evidence.
    """
    features = CrashFeatures()

    features.finding_id = _compute_finding_id(artifact)
    features.finding_state = _derive_finding_state(reproduction, asan)
    features.campaign_id = campaign_id
    features.timestamp = timestamp

    # --- identity ---
    if artifact is not None:
        features.artifact_path = artifact.path
        features.artifact_filename = artifact.filename
        features.artifact_size = artifact.size_bytes
        features.artifact_type = artifact.artifact_type
        features.raw_afl_filename_metadata = dict(artifact.metadata)  # independent copy

    # --- reproduction ---
    features.reproduction_status = _derive_reproduction_status(reproduction, asan)
    features.reproducible = features.reproduction_status == ReproductionStatus.REPRODUCED
    if reproduction is not None:
        features.return_code = reproduction.return_code
        features.signal = reproduction.signal
        features.timed_out = reproduction.timed_out
        features.duration_ms = reproduction.duration_ms
        features.asan_detected = reproduction.asan_detected
        features.raw_stderr = reproduction.stderr

    # --- ASan / location / stack ---
    if asan is not None:
        features.asan_detected = features.asan_detected or asan.is_asan
        features.raw_asan_report = asan.raw_report
        if asan.is_asan:
            features.sanitizer = "AddressSanitizer"
            features.error_type = asan.error_class
            features.crash_type = asan.error_class  # same value; see module note above
            features.access_type = asan.access_type
            features.access_size = asan.access_size
            features.fault_address = asan.address
            features.memory_region = asan.memory_region
            features.faulting_function = asan.faulting_function
            features.source_file = asan.source_file
            features.source_line = asan.source_line
            features.top_frame = _format_top_frame(asan)
            features.stack_depth = len(asan.stack_trace)
            features.raw_stack_trace = _copy_stack_trace(asan)

    features.provenance = _build_provenance(artifact, reproduction, asan)

    return features


def features_to_dict(features: CrashFeatures) -> dict:
    """
    Convenience serialization helper (e.g. for logging or a future
    Phase 12 persistence step) — converts enums to their string value
    and datetimes to ISO format, everything else via dataclasses.asdict.
    Does not mutate `features`.
    """
    data = asdict(features)
    data["finding_state"] = features.finding_state.value
    data["reproduction_status"] = features.reproduction_status.value
    if features.timestamp is not None:
        data["timestamp"] = features.timestamp.isoformat()
    return data
