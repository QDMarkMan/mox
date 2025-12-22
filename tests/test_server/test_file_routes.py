"""API tests for session file upload and retrieval endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from molx_agent.config import get_settings
from molx_agent.utils import paths as path_utils
from molx_core.memory.factory import reset_store
from molx_server.app import create_app


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """Provide a FastAPI test client backed by the in-memory store."""
    monkeypatch.setenv("MOLX_MEMORY_BACKEND", "memory")
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))
    get_settings.cache_clear()
    path_utils._artifacts_root.cache_clear()
    reset_store()
    app = create_app()
    with TestClient(app) as http_client:
        yield http_client
    reset_store()
    get_settings.cache_clear()
    path_utils._artifacts_root.cache_clear()


def _create_session(client: TestClient) -> str:
    response = client.post("/api/v1/session/create")
    assert response.status_code == 200
    return response.json()["session_id"]


def test_upload_and_download_file_flow(client: TestClient) -> None:
    """Uploading a file should store metadata and allow download."""
    session_id = _create_session(client)

    payload = {
        "uploaded_file": ("dataset.csv", "smiles,activity\nCCO,1", "text/csv"),
    }
    response = client.post(
        f"/api/v1/session/{session_id}/files",
        files=payload,
        data={"description": "primary dataset"},
    )
    assert response.status_code == 200
    body = response.json()
    file_id = body["file"]["file_id"]

    listing = client.get(f"/api/v1/session/{session_id}/files")
    assert listing.status_code == 200
    files = listing.json()
    assert files["uploaded_files"]
    assert files["uploaded_files"][0]["file_name"].endswith("dataset.csv")

    download = client.get(f"/api/v1/session/{session_id}/files/{file_id}")
    assert download.status_code == 200
    assert "text/csv" in download.headers["content-type"]
