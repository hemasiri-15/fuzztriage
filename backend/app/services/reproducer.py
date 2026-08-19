"""
Phase 6 — safe crash reproduction.

Given a configured target (a TargetCommand — how to build argv for a
specific fuzzing target's CLI) and a saved AFL++ input file, safely
re-executes the target under subprocess and returns a structured
ReproductionResult (stdout, stderr, return code, timeout state, signal,
and a lightweight ASan-presence flag).

This module deliberately does NOT parse ASan output in any depth —
that is Phase 5's job (app.services.asan_parser). This module's only
sanitizer-related responsibility is a cheap substring check on stderr
so callers know whether it's worth handing stderr to the Phase 5
parser at all.

Security posture (see also the module-level SECURITY comment further
down):
    - subprocess is invoked with an explicit argv list, never a string
    - shell=True is never used
    - the target binary and input file are validated to exist and be
      the expected kind of filesystem object *before* anything is
      executed
    - only the explicitly configured target binary is ever executed —
      there is no code path that runs an arbitrary/uploaded executable
    - a hard timeout bounds every execution; a hung target cannot block
      the caller indefinitely
    - any temporary output file a target needs is created in an
      auto-cleaned TemporaryDirectory, never inside the repository
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class ReproductionResult:
    """
    Structured outcome of one reproduction attempt.

    `executed` is True as soon as the target process was actually
    launched — including cases where it then timed out or crashed.
    It is False only when reproduction never got as far as spawning a
    process at all (validation failure, or an OS-level launch error).

    Every field defaults to a value that means "unknown/not
    applicable" rather than a value that could be mistaken for a real
    observation (e.g. return_code stays None on validation failure —
    never 0, which would falsely imply a clean exit).
    """
    executed: bool
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: Optional[float] = None
    timed_out: bool = False
    signal: Optional[int] = None
    asan_detected: bool = False

    # Not in the spec's minimum field list, but necessary to make
    # validation failures (Phase 6 TEST 4 / TEST 5) inspectable without
    # requiring callers to catch an exception for an expected,
    # routine condition (a crash artifact whose target binary moved,
    # for example, is a normal operational event in this pipeline, not
    # an exceptional one).
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Target command abstraction
# ---------------------------------------------------------------------------

@dataclass
class TargetCommand:
    """
    Describes how to build a safe argv list for one fuzzing target.

    Different targets take their input differently on the command
    line. This is intentionally generic rather than hardcoding any
    single target's CLI (e.g. djpeg's `-outfile <path> <input>`)
    directly into the reproducer:

        Target A  "target input"
            TargetCommand(binary=...)

        Target B  "target --input input"
            TargetCommand(binary=..., args_before_input=["--input"])

        Target C  "target -outfile output input"  (our current djpeg target)
            TargetCommand(binary=..., needs_output_file=True, output_flag="-outfile")

    `binary` is resolved once at construction time; nothing about the
    real filesystem path (local or DGX) is hardcoded here — the caller
    supplies it, typically sourced from app.config.get_settings().target_binary.
    """
    binary: Path
    args_before_input: list[str] = field(default_factory=list)
    args_after_input: list[str] = field(default_factory=list)
    needs_output_file: bool = False
    output_flag: Optional[str] = None
    output_filename: str = "reproduction_output"

    def build_argv(self, input_path: Path, output_path: Optional[Path] = None) -> list[str]:
        """
        Build a safe argv list (never a shell string) for this target.

        Every element is a separate list entry — callers must pass
        this list straight to subprocess, never join it into a string.
        """
        if self.needs_output_file and output_path is None:
            raise ValueError(
                "TargetCommand.needs_output_file is True but no output_path was provided "
                "to build_argv(); the reproducer is responsible for allocating one."
            )

        argv: list[str] = [str(self.binary)]
        argv.extend(self.args_before_input)

        if self.needs_output_file:
            if self.output_flag:
                argv.append(self.output_flag)
            argv.append(str(output_path))

        argv.append(str(input_path))
        argv.extend(self.args_after_input)
        return argv


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_target(binary: Path) -> Optional[str]:
    """Returns an error code string, or None if the target is valid."""
    if not binary.exists():
        return "target_not_found"
    if not binary.is_file():
        return "target_not_regular_file"
    if not os.access(binary, os.X_OK):
        return "target_not_executable"
    return None


def _validate_input(input_path: Path) -> Optional[str]:
    """Returns an error code string, or None if the input is valid."""
    if not input_path.exists():
        return "input_not_found"
    if not input_path.is_file():
        return "input_not_regular_file"
    if not os.access(input_path, os.R_OK):
        return "input_not_readable"
    return None


_ASAN_MARKER = "AddressSanitizer"


def _detect_asan(stderr: str) -> bool:
    """
    Cheap presence check only — NOT parsing. Real ASan-report parsing
    (error class, stack trace, faulting function, ...) belongs to
    app.services.asan_parser (Phase 5); this reproducer only tells the
    caller whether it's worth handing stderr to that parser.
    """
    return _ASAN_MARKER in stderr


# ---------------------------------------------------------------------------
# Core reproduction
# ---------------------------------------------------------------------------

def reproduce_crash(
    command: TargetCommand,
    input_path: Path | str,
    timeout_seconds: float = 5.0,
) -> ReproductionResult:
    """
    Safely re-execute `command`'s target against a saved AFL++ input
    (a crash or hang artifact) and capture the outcome.

    Validation (target exists / is executable, input exists / is a
    regular readable file) happens before anything is spawned. On a
    validation failure, ReproductionResult.executed is False and
    .error names exactly which check failed — no subprocess is ever
    launched in that case.

    A timeout does not raise — it is a normal, expected outcome for a
    fuzzing artifact (that may be exactly why AFL++ flagged it as a
    hang) and is reported via timed_out=True, not an exception.
    """
    input_path = Path(input_path)

    target_error = _validate_target(command.binary)
    if target_error:
        return ReproductionResult(executed=False, error=target_error)

    input_error = _validate_input(input_path)
    if input_error:
        return ReproductionResult(executed=False, error=input_error)

    with tempfile.TemporaryDirectory(prefix="fuzztriage-repro-") as tmpdir:
        output_path = Path(tmpdir) / command.output_filename if command.needs_output_file else None
        argv = command.build_argv(input_path, output_path=output_path)

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                argv,                      # list, never a shell string
                shell=False,                # never shell=True
                capture_output=True,
                timeout=timeout_seconds,
                cwd=tmpdir,
            )
            duration_ms = (time.perf_counter() - start) * 1000

            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            return_code = proc.returncode
            signal_num = -return_code if return_code is not None and return_code < 0 else None

            return ReproductionResult(
                executed=True,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                timed_out=False,
                signal=signal_num,
                asan_detected=_detect_asan(stderr),
            )

        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            # subprocess.run() already killed and reaped the child
            # before re-raising; whatever it captured up to that point
            # is on the exception itself.
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            return ReproductionResult(
                executed=True,
                return_code=None,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                timed_out=True,
                signal=None,
                asan_detected=_detect_asan(stderr),
            )

        except (OSError, PermissionError) as exc:
            # Covers races (e.g. the binary was valid at validation
            # time but the launch itself failed) and any other
            # OS-level failure to actually start the process.
            return ReproductionResult(
                executed=False,
                error="execution_failed",
                stderr=str(exc),
            )
