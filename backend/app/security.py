"""
Phase 13 — API path security.

Kept as a small, independently testable module rather than inline in
main.py, per the "keep the API thin" principle -- this is genuinely
distinct, security-critical logic (path containment), not a route
handler concern.

The core guarantee: a client-supplied `campaign_path` string can NEVER
resolve to a filesystem location outside the configured data root,
regardless of '../' traversal, an absolute-path override, or a symlink
pointing outside the root. This never relies on string prefix checks
(e.g. `str(resolved).startswith(str(root))`), which are a well-known
insufficient check (e.g. root=/data, escape=/data-secret would pass a
naive prefix check) -- instead it uses Path.resolve() (which also
resolves symlinks on POSIX) plus Path.relative_to() for a structural
containment check.
"""
from __future__ import annotations

from pathlib import Path


class PathSecurityError(Exception):
    """Raised for any campaign_path validation failure. `code` is a
    stable machine-readable reason; the API layer maps it to an HTTP
    status code (see main.py's _HTTP_STATUS_BY_CODE)."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def resolve_campaign_path(data_root: Path, campaign_path: str) -> Path:
    """
    Safely resolve `campaign_path` (as supplied by an untrusted API
    client) against `data_root` (already-resolved, server-configured).

    Raises PathSecurityError with a specific `.code` on any violation;
    never returns a path outside data_root.
    """
    if not campaign_path or not isinstance(campaign_path, str):
        raise PathSecurityError("EMPTY_PATH", "campaign_path must be a non-empty string.")

    # Reject absolute paths BEFORE any join: pathlib's `/` operator
    # silently discards the left operand entirely when the right side
    # is absolute (Path("/data") / "/etc/passwd" == Path("/etc/passwd")) --
    # this check must happen first, not after constructing a candidate.
    if Path(campaign_path).is_absolute():
        raise PathSecurityError(
            "ABSOLUTE_PATH_REJECTED",
            "campaign_path must be relative to the configured data root, not absolute.",
        )

    candidate = (data_root / campaign_path).resolve()

    try:
        candidate.relative_to(data_root)
    except ValueError:
        # Path.resolve() already normalized any '..' segments and
        # resolved symlinks -- if the fully-resolved candidate still
        # isn't inside data_root, every escape class (traversal,
        # symlink escape) is caught here uniformly.
        raise PathSecurityError(
            "PATH_OUTSIDE_ROOT",
            "campaign_path resolves outside the permitted data root.",
        )

    if not candidate.exists():
        raise PathSecurityError("NOT_FOUND", "Campaign path does not exist.")

    if not candidate.is_dir():
        raise PathSecurityError("NOT_A_DIRECTORY", "campaign_path must be a directory, not a file.")

    return candidate
