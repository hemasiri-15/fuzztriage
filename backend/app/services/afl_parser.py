"""
Parser for AFL++'s `fuzzer_stats` file.

AFL++ writes this as simple `key      : value` lines (colon-separated,
padded with spaces for alignment). The exact field set has changed
across AFL++ versions and can vary by build/mode, so this parser:

  - extracts every key:value pair present, into `raw` (a dict[str, str])
  - additionally exposes a fixed set of well-known fields as typed,
    Optional attributes on AflStats, coercing to int/float where the
    field is numeric and leaving it None if absent or unparsable
  - never raises on a missing or unexpected field
  - never fabricates a value that wasn't actually in the file

This module has zero knowledge of crashes/hangs/artifacts — it only
understands the stats file's key:value text format. Artifact discovery
lives in artifact_collector.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Fields we specifically surface as typed attributes. Anything else
# found in the file still ends up in `raw`, just not promoted to a
# named attribute.
_INT_FIELDS = {
    "start_time", "last_update", "run_time", "cycles_done", "cycles_wo_finds",
    "execs_done", "corpus_count", "saved_crashes", "saved_hangs",
    "edges_found", "total_edges", "last_find", "last_crash", "last_hang",
    "total_crashes", "total_tmouts", "corpus_favored", "corpus_found",
    "corpus_imported", "pending_favs", "pending_total", "max_depth",
}
_FLOAT_FIELDS = {
    "execs_per_sec", "execs_ps_last_min",
}
# Percentage fields are stored as text like "100.00%" — strip the '%'
# and coerce to float.
_PERCENT_FIELDS = {
    "stability", "bitmap_cvg",
}


@dataclass
class AflStats:
    # Timing
    start_time: Optional[int] = None
    last_update: Optional[int] = None
    run_time: Optional[int] = None
    cycles_done: Optional[int] = None
    cycles_wo_finds: Optional[int] = None

    # Execution
    execs_done: Optional[int] = None
    execs_per_sec: Optional[float] = None

    # Corpus / coverage
    corpus_count: Optional[int] = None
    edges_found: Optional[int] = None
    total_edges: Optional[int] = None
    stability: Optional[float] = None      # percentage, e.g. 100.0
    bitmap_cvg: Optional[float] = None     # percentage, e.g. 4.62

    # Findings
    saved_crashes: Optional[int] = None
    saved_hangs: Optional[int] = None
    last_find: Optional[int] = None
    last_crash: Optional[int] = None
    last_hang: Optional[int] = None

    # Every key:value pair actually present in the file, unmodified
    # (values as raw strings). Useful for anything not promoted above,
    # and for version-difference debugging.
    raw: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True if nothing meaningful was parsed (e.g. file missing/blank)."""
        return not self.raw


def _coerce(key: str, value: str):
    value = value.strip()
    if key in _PERCENT_FIELDS:
        try:
            return float(value.rstrip("%").strip())
        except ValueError:
            return None
    if key in _INT_FIELDS:
        try:
            return int(float(value))  # tolerate "1234.0"-style values defensively
        except ValueError:
            return None
    if key in _FLOAT_FIELDS:
        try:
            return float(value)
        except ValueError:
            return None
    return value


def parse_fuzzer_stats_text(text: str) -> AflStats:
    """Parse fuzzer_stats file *contents* (already read into a string)."""
    stats = AflStats()
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        stats.raw[key] = value

        coerced = _coerce(key, value)
        if key in _PERCENT_FIELDS or key in _INT_FIELDS or key in _FLOAT_FIELDS:
            if hasattr(stats, key):
                setattr(stats, key, coerced)

    return stats


def parse_fuzzer_stats(path: Path | str) -> AflStats:
    """
    Parse a fuzzer_stats file from disk.

    Missing file -> returns an empty AflStats (is_empty=True), not an
    exception — a campaign that hasn't produced a stats file yet (or
    is pointed at a not-yet-existent FUZZ_OUTPUT_DIR) is a valid state,
    not an error condition, per the project's "no data source yet"
    handling elsewhere.
    """
    path = Path(path)
    if not path.is_file():
        return AflStats()
    text = path.read_text(errors="replace")
    return parse_fuzzer_stats_text(text)
