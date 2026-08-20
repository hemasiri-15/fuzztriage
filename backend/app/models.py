"""
SQLAlchemy ORM models for FuzzTriage.

Every table starts empty on a fresh database — nothing here seeds
rows. Field sets match the project spec exactly; fields that can only
be known after a real AFL++ campaign or the triage pipeline runs are
nullable, and stay NULL until that stage actually populates them.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, Boolean, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from app.database import Base


def gen_id() -> str:
    return uuid.uuid4().hex[:12]


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(String, primary_key=True, default=gen_id)
    name = Column(String, nullable=False)
    target = Column(String, nullable=False)
    binary = Column(String, nullable=False)
    corpus = Column(String, nullable=False)
    status = Column(String, nullable=False, default="CREATED")

    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)

    # Populated by afl_parser.py from a real fuzzer_stats file — NULL
    # until an actual campaign has been ingested.
    executions = Column(Integer, nullable=True)
    exec_per_sec = Column(Float, nullable=True)
    coverage = Column(Float, nullable=True)
    raw_crashes = Column(Integer, nullable=True)
    raw_hangs = Column(Integer, nullable=True)
    corpus_count = Column(Integer, nullable=True)
    new_edges = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    crashes = relationship("Crash", back_populates="campaign", cascade="all, delete-orphan")
    hangs = relationship("Hang", back_populates="campaign", cascade="all, delete-orphan")
    clusters = relationship("Cluster", back_populates="campaign", cascade="all, delete-orphan")
    pipeline_runs = relationship("PipelineRun", back_populates="campaign", cascade="all, delete-orphan")


class Crash(Base):
    __tablename__ = "crashes"

    id = Column(String, primary_key=True, default=gen_id)
    campaign_id = Column(String, ForeignKey("campaigns.id"), nullable=False)
    artifact_path = Column(String, nullable=False)

    crash_type = Column(String, nullable=True)
    access_type = Column(String, nullable=True)
    memory_region = Column(String, nullable=True)
    faulting_function = Column(String, nullable=True)
    source_file = Column(String, nullable=True)
    source_line = Column(Integer, nullable=True)

    stack_trace = Column(Text, nullable=True)        # JSON-encoded list of frames
    normalized_stack = Column(Text, nullable=True)    # Phase 8 — not populated yet
    stack_hash = Column(String, nullable=True)        # Phase 8 — not populated yet

    asan_report = Column(Text, nullable=True)
    reproducible = Column(Boolean, nullable=True)

    priority_score = Column(Float, nullable=True)     # Phase 11 — not populated yet
    severity = Column(String, nullable=True)          # Phase 11 — not populated yet

    cluster_id = Column(String, ForeignKey("clusters.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    campaign = relationship("Campaign", back_populates="crashes")
    cluster = relationship("Cluster", back_populates="crashes")


class Hang(Base):
    __tablename__ = "hangs"

    id = Column(String, primary_key=True, default=gen_id)
    campaign_id = Column(String, ForeignKey("campaigns.id"), nullable=False)
    artifact_path = Column(String, nullable=False)
    source_id = Column(String, nullable=True)   # e.g. AFL++ "src:NNNNNN" filename field
    timestamp = Column(DateTime, default=datetime.utcnow)
    timeout_ms = Column(Integer, nullable=True)
    size_bytes = Column(Integer, nullable=True)

    campaign = relationship("Campaign", back_populates="hangs")


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(String, primary_key=True, default=gen_id)
    campaign_id = Column(String, ForeignKey("campaigns.id"), nullable=False)
    signature = Column(String, nullable=True)
    crash_type = Column(String, nullable=True)
    dominant_function = Column(String, nullable=True)
    member_count = Column(Integer, nullable=True)
    priority_score = Column(Float, nullable=True)
    severity = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    campaign = relationship("Campaign", back_populates="clusters")
    crashes = relationship("Crash", back_populates="cluster")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(String, primary_key=True, default=gen_id)
    campaign_id = Column(String, ForeignKey("campaigns.id"), nullable=False)
    stage = Column(String, nullable=False)
    status = Column(String, nullable=False, default="PENDING")  # PENDING|RUNNING|SUCCESS|FAILED
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    message = Column(Text, nullable=True)

    campaign = relationship("Campaign", back_populates="pipeline_runs")
