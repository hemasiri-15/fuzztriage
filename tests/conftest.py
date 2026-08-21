"""
Test-suite-wide fixtures/setup.

This file exists specifically to defend against a real regression:
the Phase 6 reproducer fixture scripts under tests/fixtures/reproducer/
must be executable (chmod +x) for tests/test_reproducer.py to pass,
but that bit can be lost in transit depending on how the repository
was packaged/extracted/checked out (zip tools, some Windows unzip
utilities, and a `git add` performed on a checkout that already lost
the bit can all silently drop it — confirmed NOT to be a bug in
reproducer.py itself, which correctly reports target_not_executable
for exactly this reason when a fixture legitimately isn't executable).

This is a TEST-infrastructure fix, not a production code change:
app/services/reproducer.py's validation behavior is untouched and
must remain untouched — a target that is genuinely not executable
should keep reporting target_not_executable. This fixture only
repairs the known, fixed set of reproducer test fixtures themselves,
before the test session starts, so the test suite is robust to how it
was checked out.

If you hit `target_not_executable` failures in test_reproducer.py
outside of pytest (e.g. running the fixture scripts directly), fix it
permanently at the source with:

    chmod +x tests/fixtures/reproducer/*.py
    git update-index --chmod=+x tests/fixtures/reproducer/*.py
    git commit -m "Restore executable bit on reproducer fixtures"
"""
import os
import stat
from pathlib import Path

_REPRODUCER_FIXTURES = Path(__file__).parent / "fixtures" / "reproducer"
_EXECUTABLE_FIXTURES = (
    "success_target.py",
    "crash_nonzero_target.py",
    "timeout_target.py",
    "asan_target.py",
    "signal_target.py",
)


def pytest_configure(config):
    """Runs once, before test collection — repair known fixture
    permissions if they were lost, regardless of cause."""
    for name in _EXECUTABLE_FIXTURES:
        path = _REPRODUCER_FIXTURES / name
        if not path.is_file():
            continue  # missing entirely is a different problem; not this fixture's job to create files
        current_mode = path.stat().st_mode
        if not (current_mode & stat.S_IXUSR):
            path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
