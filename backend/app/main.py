"""
Phase 13 — FuzzTriage API.

Thin adapter around the existing, already-tested Phase 12 pipeline.
This file contains NO parsing logic, NO feature-derivation logic, NO
deduplication/clustering/scoring logic -- it validates a request,
calls app.services.pipeline.run_pipeline(), and serializes the result
via that same module's own pipeline_result_to_dict(). There is exactly
one authoritative implementation of the triage pipeline; this module
is not a second one.

Run locally with:
    uvicorn app.main:app --reload

Required environment variable (see app/config.py):
    FUZZTRIAGE_DATA_ROOT   -- the root directory that all campaign_path
                              request values are validated against.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import ConfigError, get_data_root, load_settings
from app.security import PathSecurityError, resolve_campaign_path
from app.services.clusterer import DEFAULT_EPS, DEFAULT_MIN_SAMPLES
from app.services.pipeline import pipeline_result_to_dict, run_pipeline
from app.services.reproducer import TargetCommand

logger = logging.getLogger("fuzztriage.api")

app = FastAPI(
    title="FuzzTriage API",
    version="1.0.0",
    description="Adapter API around the FuzzTriage Phase 1-12 triage pipeline. "
                 "Exposes campaign analysis; does not execute fuzzers or arbitrary binaries.",
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    campaign_path: str = Field(
        ..., description="Path to an AFL++ campaign output directory, relative to the "
                          "server-configured FUZZTRIAGE_DATA_ROOT. Never an absolute path.",
    )
    campaign_id: Optional[str] = Field(None, description="Optional caller-supplied campaign identifier.")
    reproduce_hangs: bool = Field(False, description="If true, attempt real reproduction of hang artifacts "
                                                        "(slower; off by default).")
    timeout_seconds: float = Field(
        5.0, gt=0, le=60.0,
        description="Per-artifact reproduction timeout in seconds. Bounded to prevent a client "
                    "from requesting a pathologically long per-artifact execution window.",
    )
    cluster_eps: float = Field(
        DEFAULT_EPS, gt=0, le=10.0,
        description="Phase 10 DBSCAN eps parameter. Bounded to prevent a pathologically large "
                    "value from being submitted (the underlying Gower distance scale is [0, 1], "
                    "so any eps above a small multiple of that is already meaningless; 10.0 is a "
                    "generous ceiling, not a tuned value).",
    )
    cluster_min_samples: int = Field(
        DEFAULT_MIN_SAMPLES, ge=1, le=1000,
        description="Phase 10 DBSCAN min_samples parameter. Bounded to prevent a pathologically "
                    "large value from being submitted for a single request.",
    )


class HealthResponse(BaseModel):
    status: str


class CampaignMetadataResponse(BaseModel):
    fuzz_output_dir: str
    campaign_id: Optional[str] = None
    afl_stats: dict
    queue_count: int
    crash_artifact_count: int
    hang_artifact_count: int


class ArtifactErrorResponse(BaseModel):
    artifact_id: str
    stage: str
    error_type: str
    message: str


class StageStatusResponse(BaseModel):
    stage: str
    status: str
    processed_count: int
    error_count: int


class AnalyzeResponse(BaseModel):
    """
    Top-level shape is explicitly typed for OpenAPI clarity. The
    nested findings/deduplication/clustering/priorities payloads stay
    as `dict`/`list[dict]` deliberately -- their real shape is owned
    by Phases 7/9/10/11 respectively (via pipeline_result_to_dict());
    redeclaring their exact structure here as a second set of Pydantic
    models would itself be the kind of "second authoritative
    implementation" this phase is explicitly forbidden from creating.
    """
    campaign: CampaignMetadataResponse
    findings: list
    deduplication: dict
    clustering: dict
    priorities: list
    artifact_errors: list[ArtifactErrorResponse]
    stages: list[StageStatusResponse]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


# Maps security/config error codes to HTTP status codes. A single,
# explicit table -- never an ad hoc if/elif chain scattered through
# the route handler.
_PATH_ERROR_STATUS = {
    "EMPTY_PATH": 422,
    "ABSOLUTE_PATH_REJECTED": 403,
    "PATH_OUTSIDE_ROOT": 403,
    "NOT_FOUND": 404,
    "NOT_A_DIRECTORY": 400,
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    """Cheap liveness check. Never touches the pipeline, filesystem, or database."""
    return {"status": "ok"}


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Malformed request (e.g. path is a file, not a directory)"},
        403: {"model": ErrorResponse, "description": "campaign_path escapes the configured data root"},
        404: {"model": ErrorResponse, "description": "campaign_path does not exist"},
        422: {"model": ErrorResponse, "description": "Request validation failure"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)
def analyze(request: AnalyzeRequest) -> dict:
    """
    Validate `campaign_path`, invoke the existing Phase 12 pipeline,
    and return its real result as JSON. This handler is a plain `def`
    (not `async def`) so FastAPI runs it in a worker thread --
    run_pipeline() is a blocking, synchronous, pure function with no
    shared mutable module-level state, so concurrent requests cannot
    corrupt each other's results.
    """
    try:
        data_root = get_data_root()
    except ConfigError:
        logger.error("FUZZTRIAGE_DATA_ROOT is not configured")
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "SERVER_MISCONFIGURED", "message": "The server is not configured correctly."}},
        )

    try:
        campaign_dir = resolve_campaign_path(data_root, request.campaign_path)
    except PathSecurityError as exc:
        status_code = _PATH_ERROR_STATUS.get(exc.code, 400)
        raise HTTPException(status_code=status_code, detail={"error": {"code": exc.code, "message": exc.message}})

    # The target binary used for reproduction is a SERVER-SIDE
    # configured value ONLY (via the existing Phase 1
    # app.config.load_settings() -> Settings.target_binary), never
    # accepted from the client -- exposing a client-controlled
    # executable path would make this endpoint an arbitrary-binary-
    # execution surface, explicitly forbidden. If the existing config
    # isn't fully set up, analysis proceeds without reproduction
    # (reproduction is an enhancement, not a hard requirement for a
    # valid triage response) rather than failing the request.
    #
    # Uses load_settings() directly (uncached) rather than the module-
    # level get_settings() singleton, so concurrent/successive requests
    # never share cached, potentially-stale configuration state.
    target_command = None
    try:
        settings = load_settings()
        target_command = TargetCommand(binary=settings.target_binary)
    except ConfigError:
        pass  # reproduction simply won't run; findings are still returned honestly

    try:
        result = run_pipeline(
            campaign_dir,
            campaign_id=request.campaign_id,
            target_command=target_command,
            reproduce_hangs=request.reproduce_hangs,
            timeout_seconds=request.timeout_seconds,
            cluster_eps=request.cluster_eps,
            cluster_min_samples=request.cluster_min_samples,
        )
    except (TypeError, ValueError) as exc:
        # Fatal pipeline configuration errors (Phase 10/11's own
        # validation) -- a caller mistake, not a data-quality issue.
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "PIPELINE_CONFIGURATION_ERROR", "message": str(exc)}},
        )
    except Exception:
        # Never leak internal exception text or a traceback to the client.
        logger.exception("Unexpected pipeline failure for campaign_path=%s", request.campaign_path)
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "INTERNAL_ERROR",
                               "message": "An unexpected error occurred while analyzing the campaign."}},
        )

    return pipeline_result_to_dict(result)
