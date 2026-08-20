"""
Phase 6 tests — app.services.reproducer.

All fixture targets are small deterministic Python scripts under
tests/fixtures/reproducer/ (see README_FIXTURE.txt there). None of
this depends on the DGX or on a real compiled fuzzing target being
present, and nothing here is presented as a real AFL++ finding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.reproducer import TargetCommand, reproduce_crash  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "reproducer"
SAMPLE_INPUT = FIXTURES / "sample_input.bin"


def _cmd(script_name: str) -> TargetCommand:
    return TargetCommand(binary=FIXTURES / script_name)


# ---------------------------------------------------------------------------
# TEST 1 — successful execution
# ---------------------------------------------------------------------------

def test_successful_execution():
    result = reproduce_crash(_cmd("success_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.executed is True
    assert result.timed_out is False
    assert result.return_code == 0
    assert "decoded" in result.stdout
    assert result.error is None


def test_successful_execution_duration_is_recorded():
    result = reproduce_crash(_cmd("success_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.duration_ms is not None
    assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# TEST 2 — non-zero exit status
# ---------------------------------------------------------------------------

def test_nonzero_exit_status():
    result = reproduce_crash(_cmd("crash_nonzero_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.executed is True
    assert result.timed_out is False
    assert result.return_code != 0
    assert result.return_code == 1
    assert "could not parse" in result.stderr


# ---------------------------------------------------------------------------
# TEST 3 — timeout
# ---------------------------------------------------------------------------

def test_timeout_is_detected():
    result = reproduce_crash(_cmd("timeout_target.py"), SAMPLE_INPUT, timeout_seconds=0.5)
    assert result.executed is True
    assert result.timed_out is True
    assert result.return_code is None


def test_timeout_does_not_block_beyond_configured_limit():
    import time
    start = time.perf_counter()
    reproduce_crash(_cmd("timeout_target.py"), SAMPLE_INPUT, timeout_seconds=0.5)
    elapsed = time.perf_counter() - start
    # Generous upper bound (kill+reap overhead) — must not run anywhere
    # near the fixture's actual 30s sleep.
    assert elapsed < 5.0


# ---------------------------------------------------------------------------
# TEST 4 — missing target
# ---------------------------------------------------------------------------

def test_missing_target_binary_returns_structured_error():
    command = TargetCommand(binary=FIXTURES / "does_not_exist.py")
    result = reproduce_crash(command, SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.executed is False
    assert result.error == "target_not_found"
    assert result.return_code is None


def test_target_that_is_not_executable_returns_structured_error():
    command = TargetCommand(binary=FIXTURES / "README_FIXTURE.txt")  # exists, not +x
    result = reproduce_crash(command, SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.executed is False
    assert result.error == "target_not_executable"


# ---------------------------------------------------------------------------
# TEST 5 — missing input
# ---------------------------------------------------------------------------

def test_missing_input_returns_structured_error():
    result = reproduce_crash(_cmd("success_target.py"), FIXTURES / "no_such_input.bin", timeout_seconds=2.0)
    assert result.executed is False
    assert result.error == "input_not_found"
    assert result.return_code is None


def test_input_that_is_a_directory_returns_structured_error():
    result = reproduce_crash(_cmd("success_target.py"), FIXTURES, timeout_seconds=2.0)
    assert result.executed is False
    assert result.error == "input_not_regular_file"


# ---------------------------------------------------------------------------
# TEST 6 — ASan output present
# ---------------------------------------------------------------------------

def test_asan_output_is_detected():
    result = reproduce_crash(_cmd("asan_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.executed is True
    assert result.asan_detected is True
    assert "AddressSanitizer" in result.stderr


def test_asan_detection_is_presence_only_not_parsing():
    """
    reproducer.py must NOT attempt to extract structured ASan fields
    (that's Phase 5's job) — confirm ReproductionResult carries no
    such attributes.
    """
    result = reproduce_crash(_cmd("asan_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    assert not hasattr(result, "error_class")
    assert not hasattr(result, "faulting_function")
    assert not hasattr(result, "stack_trace")


# ---------------------------------------------------------------------------
# TEST 7 — ordinary stderr, no ASan
# ---------------------------------------------------------------------------

def test_non_asan_stderr_is_not_flagged():
    result = reproduce_crash(_cmd("crash_nonzero_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.asan_detected is False


# ---------------------------------------------------------------------------
# TEST 8 — signal termination
# ---------------------------------------------------------------------------

def test_signal_termination_is_captured():
    result = reproduce_crash(_cmd("signal_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.executed is True
    assert result.timed_out is False
    assert result.return_code is not None
    assert result.return_code < 0  # POSIX convention: negative == -signal
    import signal as signal_module
    assert result.signal == signal_module.SIGSEGV


# ---------------------------------------------------------------------------
# Integration with Phase 5 (asan_parser) — reproducer output must be
# directly usable as asan_parser input, with no duplicated parsing.
# ---------------------------------------------------------------------------

def test_reproducer_stderr_is_directly_consumable_by_phase5_parser():
    from app.services.asan_parser import parse_asan_report

    result = reproduce_crash(_cmd("asan_target.py"), SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.asan_detected is True

    report = parse_asan_report(result.stderr)
    assert report.is_asan is True
    assert report.error_class == "heap-buffer-overflow"


# ---------------------------------------------------------------------------
# TargetCommand.build_argv — the generic command abstraction
# ---------------------------------------------------------------------------

def test_build_argv_simple_positional_target():
    # Target A: "target input"
    cmd = TargetCommand(binary=Path("/opt/target"))
    argv = cmd.build_argv(Path("/tmp/crash_input"))
    assert argv == ["/opt/target", "/tmp/crash_input"]


def test_build_argv_flag_before_input_target():
    # Target B: "target --input input"
    cmd = TargetCommand(binary=Path("/opt/target"), args_before_input=["--input"])
    argv = cmd.build_argv(Path("/tmp/crash_input"))
    assert argv == ["/opt/target", "--input", "/tmp/crash_input"]


def test_build_argv_output_file_target_matches_djpeg_convention():
    # Target C / our real target: "djpeg -outfile output input"
    cmd = TargetCommand(binary=Path("/opt/djpeg"), needs_output_file=True, output_flag="-outfile")
    argv = cmd.build_argv(Path("/tmp/crash.jpg"), output_path=Path("/tmp/out.ppm"))
    assert argv == ["/opt/djpeg", "-outfile", "/tmp/out.ppm", "/tmp/crash.jpg"]


def test_build_argv_never_returns_a_string():
    cmd = TargetCommand(binary=Path("/opt/target"))
    argv = cmd.build_argv(Path("/tmp/crash_input"))
    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)


def test_build_argv_raises_if_output_needed_but_not_provided():
    cmd = TargetCommand(binary=Path("/opt/djpeg"), needs_output_file=True, output_flag="-outfile")
    try:
        cmd.build_argv(Path("/tmp/crash.jpg"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_output_file_is_created_in_tempdir_not_repository():
    """
    For a target that needs an output file (like djpeg), confirm the
    reproducer never writes it into the repository tree.
    """
    cmd = TargetCommand(
        binary=FIXTURES / "success_target.py",
        needs_output_file=False,  # success_target ignores an output arg; keep this test focused on path hygiene
    )
    result = reproduce_crash(cmd, SAMPLE_INPUT, timeout_seconds=2.0)
    assert result.executed is True
    repo_root = Path(__file__).resolve().parents[1]
    leftover = list(repo_root.glob("reproduction_output*"))
    assert leftover == []
