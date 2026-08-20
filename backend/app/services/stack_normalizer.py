"""
Phase 8 — stack normalization and stable stack signature.

Answers exactly one question: "what is the canonical representation
of this stack?" It does NOT decide whether two findings are the same
vulnerability (Phase 9), does not cluster (Phase 10), and does not
score priority (Phase 11).

This module does not parse ASan text itself and does not duplicate
Phase 5's parsing logic. It consumes whatever frame-like objects
Phase 5 (AsanReport.stack_trace: list[StackFrame]) or Phase 7
(CrashFeatures.raw_stack_trace: list[dict]) already produced —
both use the identical field set (index, function, source_file,
source_line), confirmed by inspecting both modules before writing
this one.

Pipeline:
    raw frames (StackFrame objects OR dicts, either representation)
        -> canonicalize each frame (strip machine-specific path
           prefixes, never touch call order, never invent missing
           fields)
        -> serialize deterministically (one line per frame)
        -> SHA-256 of that serialized text -> stack_signature

Security: this module never executes anything, never opens a
subprocess, never touches the network, and never modifies its input —
it is pure, read-only text transformation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Union

STACK_SIGNATURE_VERSION = "1.0"

# Explicit sentinel for "this field was None on the source frame" in
# the human-readable serialization. Deliberately distinct from "??",
# which is what ASan itself sometimes emits for a genuinely-unknown
# frame (see FrameLike / _canonicalize_frame docstring) — the two must
# stay distinguishable, so we never fabricate one to look like the
# other.
_UNKNOWN = "<unknown>"

# Any object exposing these four attributes (StackFrame) or a dict
# with these four keys (CrashFeatures.raw_stack_trace) is accepted.
FrameLike = Union[object, dict]


@dataclass
class CanonicalFrame:
    """
    One canonicalized stack frame.

    `index` and `module` are carried through for inspection/future UI
    use but are deliberately EXCLUDED from the serialized text used to
    compute stack_signature (see _serialize_frame) — frame numbering
    must not affect the signature, and raw addresses were never part
    of the input data to begin with (Phase 5's StackFrame has no
    address field at all).

    `module` is forward-compatible: Phase 5's current StackFrame does
    not capture shared-library/module identity, so this is always None
    today. It is not fabricated — see the module docstring's
    "Known limitation" note.
    """
    index: Optional[int] = None
    module: Optional[str] = None
    function: Optional[str] = None
    source_file: Optional[str] = None   # normalized (path-shortened), not the raw absolute path
    source_line: Optional[int] = None


@dataclass
class NormalizedStack:
    """
    Result of normalizing one stack trace.

    stack_signature is None (not a hash of empty data) when there were
    no frames to normalize — an empty stack must never produce
    something that looks like a real fingerprint.
    """
    frames: list = field(default_factory=list)     # list[CanonicalFrame], call order preserved
    normalized_stack: str = ""                       # human-readable canonical text
    stack_signature: Optional[str] = None
    stack_signature_version: str = STACK_SIGNATURE_VERSION
    frame_count: int = 0


def _frame_field(raw_frame: FrameLike, key: str):
    """Read `key` from either a dict (CrashFeatures.raw_stack_trace shape) or an
    attribute-bearing object (Phase 5's StackFrame) — no parsing, just field access."""
    if isinstance(raw_frame, dict):
        return raw_frame.get(key)
    return getattr(raw_frame, key, None)


def _normalize_path(path: Optional[str]) -> Optional[str]:
    """
    Path normalization strategy (documented, not arbitrary):

    Keep the last two path segments (immediate parent directory +
    filename), discard everything before that.

        /home/user/libjpeg-turbo/src/jdhuff.c            -> src/jdhuff.c
        /dgxa_home/se24ucse043/project/src/jdhuff.c       -> src/jdhuff.c

    Both machine-specific absolute paths above normalize identically,
    which is the goal (Phase 8 spec's own example). A pure basename
    strategy (just "jdhuff.c") was deliberately rejected: two different
    files that happen to share a filename in different subdirectories
    of the real project (e.g. two different "utils.c") would
    incorrectly collide. Keeping one extra path segment is a
    documented trade-off toward precision over aggressive matching,
    consistent with this project's "high precision + explainability"
    principle for Phase 8.
    """
    if not path:
        return None
    posix_path = path.replace("\\", "/")
    parts = [p for p in posix_path.split("/") if p]
    if not parts:
        return None
    tail = parts[-2:] if len(parts) >= 2 else parts[-1:]
    return "/".join(tail)


def _canonicalize_frame(raw_frame: FrameLike) -> CanonicalFrame:
    """
    Build one CanonicalFrame from a raw frame-like object.

    Never fabricates a value: a missing/empty field stays None. A
    frame whose function is literally the string "??" (ASan's own
    "symbol unknown" marker, distinct from Phase 5 simply not having
    captured anything) is preserved verbatim as "??" — that is real
    information ASan itself reported, not something Phase 8 invented.
    """
    index = _frame_field(raw_frame, "index")
    module = _frame_field(raw_frame, "module") or None   # always None until Phase 5 captures it
    function = _frame_field(raw_frame, "function") or None
    source_file = _frame_field(raw_frame, "source_file")
    source_line = _frame_field(raw_frame, "source_line")

    return CanonicalFrame(
        index=index,
        module=module,
        function=function,
        source_file=_normalize_path(source_file),
        source_line=source_line,
    )


def _serialize_frame(frame: CanonicalFrame) -> str:
    """
    Deterministic, human-readable single-line representation:

        module!function|source_file:source_line

    Missing components render as the explicit "<unknown>" sentinel —
    never a fabricated function/file name, never silently omitted
    (omitting a component would make two different kinds of "missing"
    collide in the serialized text).
    """
    module = frame.module if frame.module else _UNKNOWN
    function = frame.function if frame.function else _UNKNOWN
    source_file = frame.source_file if frame.source_file else _UNKNOWN
    source_line = str(frame.source_line) if frame.source_line is not None else _UNKNOWN
    return f"{module}!{function}|{source_file}:{source_line}"


def normalize_stack(raw_frames: Optional[list]) -> NormalizedStack:
    """
    Normalize a list of frame-like objects (Phase 5 StackFrame
    instances, or Phase 7 CrashFeatures.raw_stack_trace dicts — either
    works, both expose the same four fields) into a NormalizedStack.

    Deterministic: the same input frames always produce the same
    normalized_stack text and the same stack_signature, in this
    process, in a different process, or on a different machine.

    An empty/missing frame list returns an explicit empty state
    (stack_signature=None) rather than hashing nothing and returning
    something that could be mistaken for a real fingerprint.

    Call order is preserved exactly as given — frames are never
    sorted, reordered, or deduplicated. Recursive/repeated frames
    remain repeated in both `frames` and `normalized_stack`.
    """
    if not raw_frames:
        return NormalizedStack(frames=[], normalized_stack="", stack_signature=None, frame_count=0)

    canonical_frames = [_canonicalize_frame(f) for f in raw_frames]
    normalized_text = "\n".join(_serialize_frame(f) for f in canonical_frames)
    signature = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    return NormalizedStack(
        frames=canonical_frames,
        normalized_stack=normalized_text,
        stack_signature=signature,
        frame_count=len(canonical_frames),
    )


def normalize_crash_features_stack(features) -> NormalizedStack:
    """
    Thin convenience wrapper for the natural Phase 7 -> Phase 8
    boundary: normalize a CrashFeatures object's raw_stack_trace
    directly, without the caller needing to know its internal shape.

        features: app.services.feature_extractor.CrashFeatures

    This is intentionally just `normalize_stack(features.raw_stack_trace)`
    — no additional logic, so there is exactly one normalization code
    path regardless of which entry point a caller uses.
    """
    return normalize_stack(getattr(features, "raw_stack_trace", None))
