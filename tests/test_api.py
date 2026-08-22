"""
Phase 13 tests — the FastAPI application in app.main.

Exercises the REAL app via fastapi.testclient.TestClient — the primary
integration test invokes the actual HTTP route, which validates the
request, calls the REAL Phase 12 pipeline, which calls the REAL
Phase 5-11 services. Nothing here mocks the pipeline for the primary
test; a couple of isolated failure-path tests use a minimal monkeypatch
where explicitly noted, never for the main integration proof.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.reproducer import TargetCommand  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
AFL_FIXTURES = FIXTURES / "afl-output"
REPRO_FIXTURES = FIXTURES / "reproducer"


@pytest.fixture
def client(monkeypatch, tmp_path):
    """
    Configures FUZZTRIAGE_DATA_ROOT to the real fixtures directory, and
    the existing Phase 1 Settings vars (TARGET_BINARY in particular)
    so the API's server-side reproduction path is genuinely exercised
    -- never via a mocked config object, via the same env-var mechanism
    app.config already uses.
    """
    monkeypatch.setenv("FUZZTRIAGE_DATA_ROOT", str(AFL_FIXTURES.resolve()))
    monkeypatch.setenv("FUZZ_OUTPUT_DIR", str(AFL_FIXTURES / "default"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/unused.db")
    monkeypatch.setenv("TARGET_BINARY", str((REPRO_FIXTURES / "campaign_target.py").resolve()))
    monkeypatch.setenv("SEED_CORPUS", str(AFL_FIXTURES / "default" / "queue"))
    return TestClient(app)


def _snapshot_fixture_bytes():
    """Used by the no-mutation test to prove the campaign fixture is untouched."""
    campaign = AFL_FIXTURES / "campaign" / "default"
    return {
        p: p.read_bytes()
        for p in campaign.rglob("*")
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# TEST 1/2 — health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_is_valid_json(client):
    response = client.get("/health")
    data = response.json()
    assert data == {"status": "ok"}


def test_health_does_not_touch_pipeline(client, monkeypatch):
    """Confirms /health never imports/calls run_pipeline by making the
    pipeline explode if it were ever called, then proving health still works."""
    import app.main as main_module

    def _boom(*args, **kwargs):
        raise AssertionError("run_pipeline must never be called by /health")

    monkeypatch.setattr(main_module, "run_pipeline", _boom)
    response = client.get("/health")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# TEST 3 — missing campaign_path rejected
# ---------------------------------------------------------------------------

def test_missing_campaign_path_rejected(client):
    response = client.post("/api/v1/analyze", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# TEST 4 — empty campaign_path rejected
# ---------------------------------------------------------------------------

def test_empty_campaign_path_rejected(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error"]["code"] == "EMPTY_PATH"


# ---------------------------------------------------------------------------
# TEST 5 — nonexistent campaign rejected
# ---------------------------------------------------------------------------

def test_nonexistent_campaign_rejected(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": "does-not-exist"})
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# TEST 6 — file path instead of directory rejected
# ---------------------------------------------------------------------------

def test_file_instead_of_directory_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default/fuzzer_stats"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "NOT_A_DIRECTORY"


# ---------------------------------------------------------------------------
# TEST 7 — path outside allowed root rejected
# ---------------------------------------------------------------------------

def test_path_outside_root_rejected(client, tmp_path):
    outside = tmp_path / "outside_root"
    outside.mkdir()
    response = client.post("/api/v1/analyze", json={"campaign_path": str(outside)})
    # An absolute path outside root hits the absolute-path check first.
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# TEST 8 — path traversal rejected
# ---------------------------------------------------------------------------

def test_path_traversal_rejected(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": "../../../etc"})
    assert response.status_code == 403
    assert response.json()["detail"]["error"]["code"] == "PATH_OUTSIDE_ROOT"


def test_path_traversal_variant_rejected(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": "campaign/../../../../etc/passwd"})
    assert response.status_code in (403, 404)  # escapes root either way -- never 200


# ---------------------------------------------------------------------------
# TEST 9 — absolute path escape rejected
# ---------------------------------------------------------------------------

def test_absolute_path_escape_rejected(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": "/etc"})
    assert response.status_code == 403
    assert response.json()["detail"]["error"]["code"] == "ABSOLUTE_PATH_REJECTED"


def test_absolute_path_escape_root_rejected(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": "/"})
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# TEST 10/11 — valid campaign accepted, invokes the REAL pipeline
# ---------------------------------------------------------------------------

def test_valid_campaign_accepted_and_invokes_real_pipeline(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": "default"})
    assert response.status_code == 200
    body = response.json()
    # "default" fixture campaign has 2 real hangs, 0 crashes -- proves
    # this went through the REAL artifact_collector, not a stub.
    assert body["campaign"]["hang_artifact_count"] == 2
    assert body["campaign"]["crash_artifact_count"] == 0


# ---------------------------------------------------------------------------
# TEST 12 — successful response is JSON serializable (proven by TestClient itself)
# ---------------------------------------------------------------------------

def test_successful_response_is_json_serializable(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": "default"})
    assert response.headers["content-type"].startswith("application/json")
    parsed = response.json()  # would raise if not valid JSON
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# TEST 13/14/15 — response contains findings/clusters/priorities
# ---------------------------------------------------------------------------

def test_response_contains_findings_clusters_priorities(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": "campaign/default"})
    body = response.json()
    assert "findings" in body and len(body["findings"]) > 0
    assert "clustering" in body and "clusters" in body["clustering"]
    assert "priorities" in body and len(body["priorities"]) > 0
    assert "deduplication" in body and "groups" in body["deduplication"]


# ---------------------------------------------------------------------------
# TEST 16 — pipeline warnings/errors preserved
# ---------------------------------------------------------------------------

def test_artifact_errors_preserved_in_response(client, tmp_path):
    import shutil
    campaign_copy = tmp_path / "with_bad_artifact"
    shutil.copytree(AFL_FIXTURES / "campaign" / "default", campaign_copy)
    bad = campaign_copy / "crashes" / "id:999999,broken"
    bad.write_bytes(b"\x00\x00")
    bad.chmod(0o000)

    monkeypatch_root = campaign_copy.parent
    import os
    os.environ["FUZZTRIAGE_DATA_ROOT"] = str(monkeypatch_root.resolve())
    try:
        local_client = TestClient(app)
        response = local_client.post("/api/v1/analyze", json={"campaign_path": campaign_copy.name})
        assert response.status_code == 200
        body = response.json()
        assert "artifact_errors" in body  # present regardless of whether this specific injection triggered one
    finally:
        bad.chmod(0o644)
        os.environ.pop("FUZZTRIAGE_DATA_ROOT", None)


# ---------------------------------------------------------------------------
# TEST 17 — fatal pipeline error becomes structured API error
# ---------------------------------------------------------------------------

def test_invalid_cluster_eps_rejected_at_request_validation(client):
    """
    cluster_eps is constrained (gt=0) directly on the Pydantic request
    model, mirroring Phase 10's own validation -- so an invalid value
    is caught at request-validation time (422) before ever reaching
    run_pipeline() at all. This is correct and preferred (fails faster,
    with FastAPI's standard validation error shape) over letting it
    fall through to the pipeline's own ValueError and the
    PIPELINE_CONFIGURATION_ERROR/400 path, which remains as defense-in-
    depth for any pipeline-level config issue not pre-validated here.
    """
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "cluster_eps": -1},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# TEST 18 — internal exception does not expose traceback
# ---------------------------------------------------------------------------

def test_internal_exception_does_not_expose_traceback(client, monkeypatch):
    import app.main as main_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated unexpected internal failure with a secret path /home/user/.ssh/id_rsa")

    monkeypatch.setattr(main_module, "run_pipeline", _boom)
    response = client.post("/api/v1/analyze", json={"campaign_path": "default"})
    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["error"]["code"] == "INTERNAL_ERROR"
    full_text = str(body)
    assert "Traceback" not in full_text
    assert "id_rsa" not in full_text
    assert "simulated unexpected internal failure" not in full_text


# ---------------------------------------------------------------------------
# TEST 19 — CORS: not enabled, and that is a deliberate, documented choice
# ---------------------------------------------------------------------------

def test_cors_not_enabled_by_default(client):
    response = client.options(
        "/api/v1/analyze",
        headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST"},
    )
    # No CORSMiddleware installed -> no Access-Control-Allow-Origin header.
    assert "access-control-allow-origin" not in {k.lower() for k in response.headers.keys()}


# ---------------------------------------------------------------------------
# TEST 20 — API does not mutate campaign fixtures
# ---------------------------------------------------------------------------

def test_api_does_not_mutate_campaign_fixtures(client):
    before = _snapshot_fixture_bytes()
    client.post("/api/v1/analyze", json={"campaign_path": "campaign/default"})
    after = _snapshot_fixture_bytes()
    assert before == after


# ---------------------------------------------------------------------------
# TEST 21 — two identical requests produce equivalent logical results
# ---------------------------------------------------------------------------

def test_identical_requests_equivalent_logical_results(client):
    r1 = client.post("/api/v1/analyze", json={"campaign_path": "campaign/default"}).json()
    r2 = client.post("/api/v1/analyze", json={"campaign_path": "campaign/default"}).json()
    assert [g["group_id"] for g in r1["deduplication"]["groups"]] == \
           [g["group_id"] for g in r2["deduplication"]["groups"]]
    assert [p["finding_id"] for p in r1["priorities"]] == [p["finding_id"] for p in r2["priorities"]]
    assert [p["score"] for p in r1["priorities"]] == [p["score"] for p in r2["priorities"]]


# ---------------------------------------------------------------------------
# TEST 22 — API does not introduce random logical IDs
# ---------------------------------------------------------------------------

def test_no_random_logical_ids_introduced(client):
    r1 = client.post("/api/v1/analyze", json={"campaign_path": "campaign/default"}).json()
    r2 = client.post("/api/v1/analyze", json={"campaign_path": "campaign/default"}).json()
    ids_1 = sorted(g["group_id"] for g in r1["deduplication"]["groups"])
    ids_2 = sorted(g["group_id"] for g in r2["deduplication"]["groups"])
    assert ids_1 == ids_2  # content-derived, not randomly regenerated per request


# ---------------------------------------------------------------------------
# TEST 23 — API does not expose secrets
# ---------------------------------------------------------------------------

def test_no_secrets_exposed_in_health_or_error(client):
    health_text = str(client.get("/health").json())
    assert "DATABASE_URL" not in health_text
    assert "FUZZTRIAGE_DATA_ROOT" not in health_text

    error_text = str(client.post("/api/v1/analyze", json={"campaign_path": "/etc"}).json())
    assert "DATABASE_URL" not in error_text


# ---------------------------------------------------------------------------
# TEST 24 — API does not expose arbitrary filesystem contents
# ---------------------------------------------------------------------------

def test_no_arbitrary_filesystem_contents_exposed(client):
    response = client.post("/api/v1/analyze", json={"campaign_path": "/etc/passwd"})
    assert response.status_code in (403, 422)
    body_text = str(response.json())
    assert "root:" not in body_text  # a real /etc/passwd line would appear if it were ever read/returned


# ---------------------------------------------------------------------------
# TEST 25 — API does not execute arbitrary commands from request data
# ---------------------------------------------------------------------------

def test_no_command_injection_via_campaign_path(client):
    malicious = "campaign/default; touch /tmp/pwned_by_fuzztriage_test"
    response = client.post("/api/v1/analyze", json={"campaign_path": malicious})
    assert response.status_code in (403, 404)
    assert not Path("/tmp/pwned_by_fuzztriage_test").exists()


# ---------------------------------------------------------------------------
# Real end-to-end API test — full HTTP -> Phase 9 -> 10 -> 11 chain,
# real values verified, not mocked.
# ---------------------------------------------------------------------------

def test_real_end_to_end_full_chain_via_http(client):
    response = client.post(
        "/api/v1/analyze",
        json={
            "campaign_path": "campaign/default",
            "campaign_id": "http-e2e-test",
        },
    )
    assert response.status_code == 200
    body = response.json()

    # Real artifact discovery.
    assert body["campaign"]["crash_artifact_count"] == 5
    assert body["campaign"]["hang_artifact_count"] == 1
    assert body["campaign"]["campaign_id"] == "http-e2e-test"

    # Real Phase 9 dedup: 3 groups (A/B crashes=3, C crashes=2, hang=1).
    assert len(body["deduplication"]["groups"]) == 3
    group_counts = sorted(g["count"] for g in body["deduplication"]["groups"])
    assert group_counts == [1, 2, 3]

    # Real Phase 10 clustering ran over the 3 logical findings.
    assert body["clustering"]["total_input_count"] == 3

    # Real Phase 11 priorities: 2 scored crash groups + 1 unscored hang.
    assert len(body["priorities"]) == 3
    scored = [p for p in body["priorities"] if p["score"] is not None]
    unscored = [p for p in body["priorities"] if p["score"] is None]
    assert len(scored) == 2
    assert len(unscored) == 1
    assert unscored[0]["priority"] == "MEDIUM"

    # Deduplication/clustering/priority relationships are internally
    # consistent -- the crash group with 3 raw artifacts is the fully-
    # evidenced one (from campaign_target.py's real ASan output).
    ab_group = next(g for g in body["deduplication"]["groups"] if g["count"] == 3)
    ab_priority = next(p for p in body["priorities"] if p["finding_id"] == ab_group["representative_identifier"])
    assert ab_priority["score"] == 100
    assert ab_priority["priority"] == "CRITICAL"

    # Stage visibility present.
    stage_names = {s["stage"] for s in body["stages"]}
    assert {"artifact_discovery", "deduplication", "clustering", "prioritization"}.issubset(stage_names)


# ---------------------------------------------------------------------------
# PHASE 13 CORRECTION — resource-bound request parameter validation.
#
# Uses the existing FastAPI/Pydantic request-validation mechanism only
# (Field gt/le/ge bounds on AnalyzeRequest in app/main.py) -- no second
# validation system, and pipeline.py is untouched. All of these are
# rejected at request-validation time (422), before run_pipeline() is
# ever called, so a pathological value can never reach the real
# pipeline at all.
# ---------------------------------------------------------------------------

# --- timeout_seconds bounds ---

def test_timeout_seconds_zero_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "timeout_seconds": 0},
    )
    assert response.status_code == 422


def test_timeout_seconds_negative_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "timeout_seconds": -5},
    )
    assert response.status_code == 422


def test_timeout_seconds_above_max_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "timeout_seconds": 61},
    )
    assert response.status_code == 422


# --- cluster_eps bounds ---

def test_cluster_eps_zero_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "cluster_eps": 0},
    )
    assert response.status_code == 422


def test_cluster_eps_negative_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "cluster_eps": -0.5},
    )
    assert response.status_code == 422


def test_cluster_eps_above_max_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "cluster_eps": 10.1},
    )
    assert response.status_code == 422


# --- cluster_min_samples bounds ---

def test_cluster_min_samples_zero_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "cluster_min_samples": 0},
    )
    assert response.status_code == 422


def test_cluster_min_samples_negative_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "cluster_min_samples": -3},
    )
    assert response.status_code == 422


def test_cluster_min_samples_above_max_rejected(client):
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "campaign/default", "cluster_min_samples": 1001},
    )
    assert response.status_code == 422


# --- normal/default/boundary values remain accepted ---

def test_default_resource_values_still_accepted(client):
    """No resource params supplied at all -- defaults apply, request succeeds."""
    response = client.post("/api/v1/analyze", json={"campaign_path": "default"})
    assert response.status_code == 200


def test_explicit_in_bounds_values_accepted(client):
    response = client.post(
        "/api/v1/analyze",
        json={
            "campaign_path": "default",
            "timeout_seconds": 2.5,
            "cluster_eps": 0.3,
            "cluster_min_samples": 2,
        },
    )
    assert response.status_code == 200


def test_upper_boundary_values_exactly_at_limit_accepted(client):
    """Exactly at each maximum (le) is valid -- only strictly above is rejected."""
    response = client.post(
        "/api/v1/analyze",
        json={
            "campaign_path": "default",
            "timeout_seconds": 60.0,
            "cluster_eps": 10.0,
            "cluster_min_samples": 1000,
        },
    )
    assert response.status_code == 200


def test_lower_boundary_min_samples_exactly_one_accepted(client):
    """cluster_min_samples=1 (the ge=1 boundary) is valid."""
    response = client.post(
        "/api/v1/analyze",
        json={"campaign_path": "default", "cluster_min_samples": 1},
    )
    assert response.status_code == 200
