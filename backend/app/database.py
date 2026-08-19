"""
SQLAlchemy engine/session setup. DATABASE_URL always comes from
app.config — never hardcoded here.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import get_settings

Base = declarative_base()

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        _engine = create_engine(settings.database_url, connect_args=connect_args)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def init_db() -> None:
    """Create tables if they don't exist. Never inserts rows."""
    from app import models  # noqa: F401  (registers models on Base)
    Base.metadata.create_all(bind=get_engine())


def get_db():
    """FastAPI dependency — yields a session, closes it after the request."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()
