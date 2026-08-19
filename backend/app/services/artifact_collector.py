"""
Discovers AFL++ artifacts (queue/, crashes/, hangs/) under a configured
FUZZ_OUTPUT_DIR and parses the metadata AFL++ encodes into each
artifact's filename.

Does NOT interpret whether something is a "vulnerability" — a hang is
a hang, a crash artifact is a crash artifact. Classification of
severity happens much later in the pipeline (Phase 11), and only for
actual crashes, never hangs.

An empty crashes/ directory, a missing hangs/ directory, or a
FUZZ_OUTPUT_DIR that doesn't exist yet are all valid states — this
module returns empty lists in those cases, never raises, and never
invents a placeholder artifact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# AFL++ names queue/crash/hang files with comma-separated key:value
# tokens, e.g.:
#   id:000001,src:000875,time:2153861,execs:332398,op:flip32,pos:183
#   id:000000,time:0,execs:0,orig:testorig.jpg
#   id:000042,sync:other-fuzzer,src:000012
# Not every token has a ':' (e.g. flags can appear bare in some AFL
# forks) — tolerate that by only splitting tokens that contain ':'.
_TOKEN_SPLIT_RE = re.compile(r",(?=[a-zA-Z+])")

# Files AFL++ writes into these directories that are NOT artifacts.
_IGNORED_FILENAMES = {"readme.txt", "readme", ".state"}


@dataclass
class ArtifactRecord:
    path: str
    filename: str
    artifact_type: str          # "queue" | "crash" | "hang"
    size_bytes: int
    metadata: dict = field(default_factory=dict)
    afl_id: Optional[str] = None
    src_id: Optional[str] = None
    op: Optional[str] = None
    orig: Optional[str] = None
    recognized: bool = True     # False if the filename didn't match AFL++'s convention


@dataclass
class ArtifactCollection:
    queue: list[ArtifactRecord] = field(default_factory=list)
    crashes: list[ArtifactRecord] = field(default_factory=list)
    hangs: list[ArtifactRecord] = field(default_factory=list)

    @property
    def queue_count(self) -> int:
        return len(self.queue)

    @property
    def crash_count(self) -> int:
        return len(self.crashes)

    @property
    def hang_count(self) -> int:
        return len(self.hangs)


def parse_afl_filename(filename: str) -> dict:
    """
    Parse AFL++'s comma-separated key:value filename metadata.

    Tolerant of:
      - unknown/extra keys (AFL forks and sync setups add their own)
      - missing keys
      - tokens without a ':' (kept under a "flags" list rather than dropped)
      - a leading "id:NNNNNN" that's actually part of a longer AFL-style name

    Never raises on malformed input — returns whatever it could parse,
    with unmatched fragments preserved under "flags" instead of being
    silently discarded.
    """
    result: dict = {"flags": []}
    if not filename:
        return result

    tokens = _TOKEN_SPLIT_RE.split(filename)
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            key, _, value = token.partition(":")
            key = key.strip()
            value = value.strip()
            if key:
                result[key] = value
        else:
            result["flags"].append(token)
    return result


def _classify_record(entry: Path, artifact_type: str) -> Optional[ArtifactRecord]:
    if entry.name.lower() in _IGNORED_FILENAMES:
        return None
    if entry.name.startswith("."):
        # AFL++ working files like .cur_input, .synced/ marker files, etc.
        return None
    if not entry.is_file():
        return None

    metadata = parse_afl_filename(entry.name)
    recognized = "id" in metadata

    return ArtifactRecord(
        path=str(entry),
        filename=entry.name,
        artifact_type=artifact_type,
        size_bytes=entry.stat().st_size,
        metadata=metadata,
        afl_id=metadata.get("id"),
        src_id=metadata.get("src"),
        op=metadata.get("op"),
        orig=metadata.get("orig"),
        recognized=recognized,
    )


def _collect_dir(directory: Path, artifact_type: str) -> list[ArtifactRecord]:
    if not directory.is_dir():
        return []
    records = []
    for entry in sorted(directory.iterdir()):
        record = _classify_record(entry, artifact_type)
        if record is not None:
            records.append(record)
    return records


def collect_artifacts(fuzz_output_dir: Path | str) -> ArtifactCollection:
    """
    Discover queue/, crashes/, hangs/ under fuzz_output_dir.

    Every subdirectory is independently optional — a campaign with an
    empty (but existing) crashes/ directory, or one that doesn't have
    a hangs/ directory at all yet, both return an empty list for that
    category rather than raising.
    """
    base = Path(fuzz_output_dir)
    return ArtifactCollection(
        queue=_collect_dir(base / "queue", "queue"),
        crashes=_collect_dir(base / "crashes", "crash"),
        hangs=_collect_dir(base / "hangs", "hang"),
    )
