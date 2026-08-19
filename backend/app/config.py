"""
Environment-based configuration.

This module is the ONLY place FUZZ_OUTPUT_DIR, TARGET_BINARY, SEED_CORPUS,
and DATABASE_URL are read from the environment. No other module should
call os.environ / os.getenv directly for these values, and no module
should ever hardcode a filesystem path (local or DGX) — that is the
entire point of this file existing.

Required variables (see .env.example / configs/example.env):
    FUZZ_OUTPUT_DIR
    DATABASE_URL
    TARGET_BINARY
    SEED_CORPUS

A .env file (if present, at the repository root) is loaded automatically
via python-dotenv. Real environment variables always take precedence
over .env file values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Repo root is two levels up from this file: backend/app/config.py -> repo/
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _DOTENV_PATH = _REPO_ROOT / ".env"
    if _DOTENV_PATH.exists():
        load_dotenv(_DOTENV_PATH)
except ImportError:
    # python-dotenv is a convenience, not a hard requirement — if it's
    # not installed, we simply rely on real environment variables.
    pass


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    fuzz_output_dir: Path
    database_url: str
    target_binary: Path
    seed_corpus: Path

    @property
    def afl_fuzzer_stats_path(self) -> Path:
        return self.fuzz_output_dir / "fuzzer_stats"

    @property
    def afl_queue_dir(self) -> Path:
        return self.fuzz_output_dir / "queue"

    @property
    def afl_crashes_dir(self) -> Path:
        return self.fuzz_output_dir / "crashes"

    @property
    def afl_hangs_dir(self) -> Path:
        return self.fuzz_output_dir / "hangs"


_REQUIRED_VARS = ("FUZZ_OUTPUT_DIR", "DATABASE_URL", "TARGET_BINARY", "SEED_CORPUS")


def load_settings(env: dict | None = None) -> Settings:
    """
    Build a Settings object from environment variables.

    Accepts an optional explicit `env` mapping (used by tests to avoid
    depending on process-wide environment state); defaults to
    os.environ.
    """
    source = env if env is not None else os.environ

    missing = [var for var in _REQUIRED_VARS if not source.get(var)]
    if missing:
        raise ConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill these in, "
              "or set them in your shell/DGX environment."
        )

    return Settings(
        fuzz_output_dir=Path(source["FUZZ_OUTPUT_DIR"]).expanduser(),
        database_url=source["DATABASE_URL"],
        target_binary=Path(source["TARGET_BINARY"]).expanduser(),
        seed_corpus=Path(source["SEED_CORPUS"]).expanduser(),
    )


# Module-level singleton, built lazily so importing this module never
# fails just because .env hasn't been created yet (e.g. during initial
# `pip install` / static analysis / doc generation).
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
