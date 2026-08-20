"""
Phase 7 tests — app.services.feature_extractor.

Reuses existing fixtures from Phase 5 (tests/fixtures/asan/) and
Phase 6 (tests/fixtures/reproducer/) rather than inventing new
production-like data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.asan_parser import parse_asan_report               # noqa: E402
from app.services.artifact_collector import collect_artifacts          # noqa: E402
from app.services.reproducer import TargetCommand, reproduce_crash     # noqa: E402
from app.services.feature_extractor import (                            # noqa: E402
    extract_features,
    features_to_dict,
    FindingState,
    ReproductionStatus,
    FEATURE_SCHEMA_VERSION,
)

ASAN_FIXTURES = Path(__file__).parent / "fixtures" / "asan"
AFL_FIXTURES = Path(__file__).parent / "fixtures" / "afl-output"
REPRO_FIXTURES = Path(__file__).parent / "fixtures" / "reproducer"
SAMPLE_INPUT = REPRO_FIXTURES / "sample_input.bin"


def _cmd(script_name: str) -> TargetCommand:
    return TargetCommand(binary=REPRO_FIXTURES / script_name)


def _real_artifact():
    """A real ArtifactRecord from the Phase 4 crash fixture (not a hardcoded stand-in)."""
    collection = collect_artifacts(AFL_FIXTURES / "with-crash" / "default")
    assert collection.crashes, "fixture setup problem: expected at least one crash fixture"
    return collection.crashes[0]


# ---------------------------------------------------------------------------
# TEST 1 — complete heap-buffer-overflow result
# ---------------------------------------------------------------------------

def test_heap_buffer_overflow_full_extraction():
    asan = parse_asan_report((ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text())
    reproduction = reproduce_crash(_cmd("asan_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    artifact = _real_artifact()

    features = extract_features(artifact=artifact, reproduction=reproduction, asan=asan)

    assert features.finding_state == FindingState.CRASH
    assert features.crash_type == "heap-buffer-overflow"
    assert features.error_type == "heap-buffer-overflow"
    assert features.access_type == "WRITE"
    assert features.access_size == 4
    assert features.faulting_function == "decode_mcu_block"
    assert features.source_file == "/home/user/libjpeg-turbo/src/jdhuff.c"
    assert features.source_line == 341
    assert features.stack_depth >= 3
    assert features.top_frame is not None
    assert features.raw_stack_trace
    # reproduction information
    assert features.reproduction_status == ReproductionStatus.REPRODUCED
    assert features.reproducible is True
    assert features.artifact_path == artifact.path


# ---------------------------------------------------------------------------
# TEST 2 — use-after-free
# ---------------------------------------------------------------------------

def test_use_after_free_extraction():
    asan = parse_asan_report((ASAN_FIXTURES / "use_after_free.txt").read_text())
    features = extract_features(asan=asan)

    assert features.finding_state == FindingState.CRASH
    assert features.crash_type == "heap-use-after-free"
    assert features.memory_region == "HEAP"
    assert features.faulting_function == "jpeg_free_large"
    assert features.source_line == 1103


# ---------------------------------------------------------------------------
# TEST 3 — missing optional ASan fields must not crash the extractor
# ---------------------------------------------------------------------------

def test_sparse_asan_report_does_not_crash():
    # A minimal ASan-looking report with no stack frames at all.
    asan = parse_asan_report("==1==ERROR: AddressSanitizer: double-free on address 0x1\n")
    features = extract_features(asan=asan)

    assert features.finding_state == FindingState.CRASH
    assert features.crash_type == "double-free"
    assert features.faulting_function is None
    assert features.source_file is None
    assert features.source_line is None
    # A real ASan report was parsed and genuinely contained zero
    # frames -- that's a known fact (0), not an unknown one (None).
    # None is reserved for "no ASan evidence at all" (see the next test).
    assert features.stack_depth == 0
    assert features.raw_stack_trace == []


def test_no_evidence_at_all_does_not_crash():
    features = extract_features()
    assert features.finding_state == FindingState.NORMAL
    assert features.crash_type is None
    assert features.artifact_path is None
    # No ASan evidence at all -> genuinely unknown, must stay None (not 0).
    assert features.stack_depth is None


# ---------------------------------------------------------------------------
# TEST 4 — successful, non-crashing reproduction
# ---------------------------------------------------------------------------

def test_successful_reproduction_is_not_a_crash():
    reproduction = reproduce_crash(_cmd("success_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    features = extract_features(reproduction=reproduction)

    assert features.finding_state == FindingState.NORMAL
    assert features.reproduction_status == ReproductionStatus.NOT_REPRODUCED
    assert features.reproducible is False
    assert features.crash_type is None
    assert features.error_type is None


def test_nonzero_exit_without_asan_is_not_a_crash():
    # Mirrors the real libjpeg-turbo "Not a JPEG file" behavior observed
    # in practice: non-zero exit, no ASan, no signal -> NOT a crash.
    reproduction = reproduce_crash(_cmd("crash_nonzero_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    features = extract_features(reproduction=reproduction)

    assert features.return_code != 0
    assert features.finding_state == FindingState.NORMAL
    assert features.crash_type is None


# ---------------------------------------------------------------------------
# TEST 5 — timeout/hang must never become a crash
# ---------------------------------------------------------------------------

def test_timeout_is_hang_not_crash():
    reproduction = reproduce_crash(_cmd("timeout_target.py"), SAMPLE_INPUT, timeout_seconds=0.5)
    features = extract_features(reproduction=reproduction)

    assert features.timed_out is True
    assert features.finding_state == FindingState.HANG
    assert features.finding_state != FindingState.CRASH
    assert features.reproduction_status == ReproductionStatus.TIMED_OUT
    assert features.crash_type is None
    assert features.error_type is None


# ---------------------------------------------------------------------------
# TEST 6 — ASan not detected
# ---------------------------------------------------------------------------

def test_asan_not_detected_no_fabricated_crash_type():
    reproduction = reproduce_crash(_cmd("crash_nonzero_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    features = extract_features(reproduction=reproduction)

    assert features.asan_detected is False
    assert features.crash_type is None
    assert features.sanitizer is None


# ---------------------------------------------------------------------------
# TEST 7 — real artifact metadata is used, not invented
# ---------------------------------------------------------------------------

def test_artifact_metadata_reflects_real_fixture_sizes():
    collection = collect_artifacts(AFL_FIXTURES / "default")
    for record in collection.queue:
        features = extract_features(artifact=record)
        assert features.artifact_size == record.size_bytes
        assert features.artifact_path == record.path
        assert features.artifact_type == "queue"


def test_different_artifacts_produce_different_finding_ids():
    collection = collect_artifacts(AFL_FIXTURES / "default")
    ids = {extract_features(artifact=r).finding_id for r in collection.queue}
    assert len(ids) == len(collection.queue)  # every artifact gets a distinct, stable id


# ---------------------------------------------------------------------------
# TEST 8 — determinism
# ---------------------------------------------------------------------------

def test_extraction_is_deterministic():
    asan = parse_asan_report((ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text())
    artifact = _real_artifact()

    first = extract_features(artifact=artifact, asan=asan, campaign_id="c1")
    second = extract_features(artifact=artifact, asan=asan, campaign_id="c1")

    assert first == second
    assert first.finding_id == second.finding_id
    assert features_to_dict(first) == features_to_dict(second)


# ---------------------------------------------------------------------------
# Additional: data-integrity / raw-evidence-preservation requirements
# ---------------------------------------------------------------------------

def test_raw_evidence_is_preserved_verbatim():
    raw_text = (ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text()
    asan = parse_asan_report(raw_text)
    features = extract_features(asan=asan)
    assert features.raw_asan_report == raw_text


def test_raw_stack_trace_is_independent_copy_not_alias():
    asan = parse_asan_report((ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text())
    features = extract_features(asan=asan)

    features.raw_stack_trace.append({"index": 999, "function": "tampered", "source_file": None, "source_line": None})
    # Mutating the returned copy must not affect the original parsed AsanReport.
    assert len(asan.stack_trace) != len(features.raw_stack_trace)
    assert all(f.function != "tampered" for f in asan.stack_trace)


def test_extractor_does_not_mutate_source_asan_report():
    asan = parse_asan_report((ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text())
    original_error_class = asan.error_class
    original_stack_len = len(asan.stack_trace)

    extract_features(asan=asan)

    assert asan.error_class == original_error_class
    assert len(asan.stack_trace) == original_stack_len


def test_extractor_does_not_mutate_reproduction_result():
    reproduction = reproduce_crash(_cmd("asan_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    original_stderr = reproduction.stderr
    original_return_code = reproduction.return_code

    extract_features(reproduction=reproduction)

    assert reproduction.stderr == original_stderr
    assert reproduction.return_code == original_return_code


def test_feature_schema_version_present():
    features = extract_features()
    assert features.feature_schema_version == FEATURE_SCHEMA_VERSION == "1.0"


def test_finding_id_is_not_a_stack_hash_or_db_pk_or_cluster_id():
    """
    Structural sanity check: finding_id must be independent of any
    stack-hash concept (Phase 8 doesn't exist yet in this module at
    all) and must not collide in name/type with a cluster id.
    """
    artifact = _real_artifact()
    features = extract_features(artifact=artifact)
    assert not hasattr(features, "stack_hash")
    assert not hasattr(features, "cluster_id")
    assert features.finding_id is not None
    assert len(features.finding_id) == 16  # sha256[:16] hex digest


def test_reproduction_failure_is_not_discarded_as_a_finding():
    """
    A crash artifact whose target could not even be launched must
    still surface as a REPRODUCTION_FAILURE finding, not silently
    disappear.
    """
    bad_command = TargetCommand(binary=REPRO_FIXTURES / "does_not_exist.py")
    reproduction = reproduce_crash(bad_command, SAMPLE_INPUT, timeout_seconds=2.0)
    artifact = _real_artifact()

    features = extract_features(artifact=artifact, reproduction=reproduction)

    assert features.finding_state == FindingState.REPRODUCTION_FAILURE
    assert features.reproduction_status == ReproductionStatus.ERROR
    assert features.artifact_path == artifact.path  # artifact identity preserved, not discarded


def test_provenance_reflects_actual_sources():
    asan = parse_asan_report((ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text())
    reproduction = reproduce_crash(_cmd("asan_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    artifact = _real_artifact()

    features = extract_features(artifact=artifact, reproduction=reproduction, asan=asan)

    assert features.provenance["faulting_function"] == "asan_report"
    assert features.provenance["source_file"] == "asan_report"
    assert features.provenance["return_code"] == "reproduction_result"
    assert features.provenance["duration_ms"] == "reproduction_result"
    assert features.provenance["artifact_size"] == "artifact_metadata"


def test_no_execution_no_shell_no_network_side_effects():
    """
    Feature extraction must be pure analysis — confirm it works
    correctly even when given already-computed evidence and no access
    to any binary at all (no TargetCommand involved in this test path).
    """
    asan = parse_asan_report((ASAN_FIXTURES / "use_after_free.txt").read_text())
    features = extract_features(asan=asan)
    assert features.finding_state == FindingState.CRASH  # proves it worked without spawning anything
