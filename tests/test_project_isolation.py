from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("numpy")
pytest.importorskip("cv2")

from test_api import auth_headers, make_client


def test_auth_enabled_project_isolation_and_shared_roles():
    client, repo, _ = make_client(auth_enabled=True)
    repo.users["ben"] = {
        "id": "user-2",
        "username": "ben",
        "display_name": "Ben",
        "is_admin": False,
        "is_active": True,
        "password": "secret",
    }
    repo.memberships[("user-2", "project-2")] = "editor"
    repo.memberships[("user-1", "project-2")] = "viewer"

    project_a = auth_headers(client, username="ada", project_key="default")
    project_b = auth_headers(client, username="ben", project_key="other")
    shared_viewer = auth_headers(client, username="ada", project_key="other")

    assert client.get("/runs", headers=project_a).json()["runs"][0]["id"] == "run-1"
    assert client.get("/runs", headers=project_b).json()["runs"] == []
    assert client.get("/assets/asset-1", headers=project_a).status_code == 200
    assert client.get("/assets/asset-1", headers=project_b).status_code == 404

    cross_project_job = client.post(
        "/jobs",
        headers=project_b,
        json={"stage": "segment", "asset_id": "asset-1"},
    )
    assert cross_project_job.status_code == 404

    viewer_write = client.post(
        "/jobs",
        headers=shared_viewer,
        json={"stage": "segment", "asset_id": "asset-1"},
    )
    assert viewer_write.status_code == 403


def test_auth_enabled_read_requires_session_token():
    client, _, _ = make_client(auth_enabled=True)

    response = client.get("/runs")

    assert response.status_code == 401


def test_auth_enabled_uuid_path_get_requires_session_token():
    client, _, _ = make_client(auth_enabled=True)

    response = client.get("/assets/0b0c65c2-7bdb-40f9-80e4-70a693ffac92")

    assert response.status_code == 401


def test_batch_frame_preprocess_by_asset_is_project_scoped():
    client, repo, _ = make_client(auth_enabled=True)
    repo.users["ben"] = {
        "id": "user-2",
        "username": "ben",
        "display_name": "Ben",
        "is_admin": False,
        "is_active": True,
        "password": "secret",
    }
    repo.memberships[("user-2", "project-2")] = "editor"
    project_b = auth_headers(client, username="ben", project_key="other")

    response = client.post(
        "/frame/preprocess",
        headers=project_b,
        json={"asset_id": "asset-1", "start_frame": 2, "limit": 1, "store": False},
    )

    assert response.status_code == 200
    assert response.json()["frame_count"] == 0
