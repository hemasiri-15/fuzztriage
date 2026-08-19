"""
Parser for AddressSanitizer stderr/report text.

Not every crash artifact will have an ASan report (a plain SIGSEGV
with no sanitizer attached, or a target built without ASan, still has
to be representable). Every field on AsanReport is Optional and stays
None when the report text doesn't contain that information — nothing
here is inferred or guessed.

Supported error classes (per project spec):
    heap-buffer-overflow, stack-buffer-overflow, global-buffer-overflow,
    use-after-free, heap-use-after-free, double-free, invalid-free,
    stack-use-after-return

Unknown/unrecognized ASan error classes are still captured (the raw
matched string is kept), just not validated against the known set —
we don't want a future ASan version's new error class to make the
parser silently drop everything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_ERROR_LINE_RE = re.compile(
    r"ERROR:\s*AddressSanitizer:\s*(?P<error_class>[a-zA-Z0-9_-]+(?:-[a-zA-Z]+)*)"
    r"(?:.*?address\s+(?P<address>0x[0-9a-fA-F]+))?",
)

_ACCESS_RE = re.compile(
    r"(?P<access_type>READ|WRITE) of size (?P<access_size>\d+)",
)

# Primary crash-thread stack frames, e.g.:
#   #0 0x55a1b2c3d4e5 in decode_mcu_block jdhuff.c:341:12
#   #1 0x55a1b2c3d600 in decode_image jdmaster.c:78:5
#   #2 0x7f8a1b2c3d70 in __libc_start_main (libc.so.6+0x24d70)
_FRAME_RE = re.compile(
    r"^\s*#(?P<index>\d+)\s+0x[0-9a-fA-F]+\s+in\s+(?P<function>\S+)"
    r"(?:\s+(?P<location>[^\s(]+\.[a-zA-Z]+:\d+(?::\d+)?))?",
)

_SUMMARY_RE = re.compile(
    r"SUMMARY:\s*AddressSanitizer:\s*(?P<error_class>[a-zA-Z0-9_-]+)\s+"
    r"(?P<location>\S+\.[a-zA-Z]+:\d+(?::\d+)?)?\s*(?:in\s+(?P<function>\S+))?",
)

_ALLOCATED_BY_RE = re.compile(r"allocated by thread T\d+ here:")
_FREED_BY_RE = re.compile(r"freed by thread T\d+ here:")

_MEMORY_REGION_BY_ERROR_CLASS = {
    "heap-buffer-overflow": "HEAP",
    "heap-use-after-free": "HEAP",
    "use-after-free": "HEAP",
    "double-free": "HEAP",
    "invalid-free": "HEAP",
    "stack-buffer-overflow": "STACK",
    "stack-use-after-return": "STACK",
    "global-buffer-overflow": "GLOBAL",
}

KNOWN_ERROR_CLASSES = frozenset(_MEMORY_REGION_BY_ERROR_CLASS.keys())


@dataclass
class StackFrame:
    index: int
    function: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None


@dataclass
class AsanReport:
    is_asan: bool = False
    error_class: Optional[str] = None
    access_type: Optional[str] = None          # READ | WRITE
    access_size: Optional[int] = None
    address: Optional[str] = None
    memory_region: Optional[str] = None        # HEAP | STACK | GLOBAL

    faulting_function: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None

    stack_trace: list = field(default_factory=list)          # list[StackFrame], crash-thread
    allocation_stack: list = field(default_factory=list)     # list[StackFrame]
    deallocation_stack: list = field(default_factory=list)   # list[StackFrame]

    raw_report: str = ""


def _parse_frames(lines: list[str]) -> list[StackFrame]:
    frames = []
    for line in lines:
        m = _FRAME_RE.match(line)
        if not m:
            # A non-frame line (blank, or prose) ends this stack block.
            if frames:
                break
            continue
        location = m.group("location")
        source_file, source_line = None, None
        if location:
            parts = location.rsplit(":", 2) if location.count(":") >= 2 else location.split(":")
            if len(parts) >= 2:
                source_file = parts[0]
                try:
                    source_line = int(parts[1])
                except ValueError:
                    source_line = None
        frames.append(StackFrame(
            index=int(m.group("index")),
            function=m.group("function"),
            source_file=source_file,
            source_line=source_line,
        ))
    return frames


def parse_asan_report(text: str) -> AsanReport:
    """
    Parse ASan stderr/report text.

    A text blob with no recognizable ASan ERROR line returns
    AsanReport(is_asan=False) with every other field left at its
    default (None/empty) — this represents "not an ASan report" (e.g.
    a plain segfault, or a target not built with ASan), not a parse
    failure.
    """
    report = AsanReport(raw_report=text)

    error_match = _ERROR_LINE_RE.search(text)
    if not error_match:
        return report

    report.is_asan = True
    report.error_class = error_match.group("error_class")
    report.address = error_match.group("address")
    report.memory_region = _MEMORY_REGION_BY_ERROR_CLASS.get(report.error_class)

    access_match = _ACCESS_RE.search(text)
    if access_match:
        report.access_type = access_match.group("access_type")
        try:
            report.access_size = int(access_match.group("access_size"))
        except ValueError:
            report.access_size = None

    lines = text.splitlines()

    # Split the report into sections by "allocated by" / "freed by"
    # markers so the crash-thread stack doesn't get contaminated by
    # allocation/deallocation traces that follow it.
    alloc_idx = next((i for i, l in enumerate(lines) if _ALLOCATED_BY_RE.search(l)), None)
    freed_idx = next((i for i, l in enumerate(lines) if _FREED_BY_RE.search(l)), None)

    section_boundaries = [i for i in (alloc_idx, freed_idx) if i is not None]
    crash_section_end = min(section_boundaries) if section_boundaries else len(lines)

    report.stack_trace = _parse_frames(lines[:crash_section_end])
    if report.stack_trace:
        top = report.stack_trace[0]
        report.faulting_function = top.function
        report.source_file = top.source_file
        report.source_line = top.source_line

    if alloc_idx is not None:
        end = freed_idx if (freed_idx is not None and freed_idx > alloc_idx) else len(lines)
        report.allocation_stack = _parse_frames(lines[alloc_idx + 1:end])

    if freed_idx is not None:
        end = alloc_idx if (alloc_idx is not None and alloc_idx > freed_idx) else len(lines)
        report.deallocation_stack = _parse_frames(lines[freed_idx + 1:end])

    # Fall back to the SUMMARY line if no frame gave us a
    # faulting_function/source location (e.g. stripped stack frame text).
    if not report.faulting_function:
        summary_match = _SUMMARY_RE.search(text)
        if summary_match:
            report.faulting_function = summary_match.group("function")
            location = summary_match.group("location")
            if location:
                parts = location.split(":")
                if len(parts) >= 2:
                    report.source_file = parts[0]
                    try:
                        report.source_line = int(parts[1])
                    except ValueError:
                        pass

    return report
