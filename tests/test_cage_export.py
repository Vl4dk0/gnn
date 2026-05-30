"""Tests for POST /api/cage/export."""

import json
from collections.abc import Iterator
from typing import cast

import pytest
from flask.testing import FlaskClient

from backend.app import create_app


@pytest.fixture()
def client() -> Iterator[FlaskClient]:
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _post(client: FlaskClient, body: dict[str, object]) -> tuple[int, dict[str, str]]:
    resp = client.post(
        "/api/cage/export",
        data=json.dumps(body),
        content_type="application/json",
    )
    return resp.status_code, cast(dict[str, str], resp.get_json())


def test_triangle_g6(client: FlaskClient) -> None:
    status, data = _post(client, {"edge_list": "0 1\n1 2\n0 2", "format": "g6"})
    assert status == 200
    assert data["content"] == "Bw"
    assert data["filename"].endswith(".g6")
    assert data["format"] == "g6"


def test_triangle_g6_with_k_g(client: FlaskClient) -> None:
    status, data = _post(
        client, {"edge_list": "0 1\n1 2\n0 2", "format": "g6", "k": 3, "g": 5}
    )
    assert status == 200
    assert data["content"] == "Bw"
    assert "k3_g5" in data["filename"]
    assert data["filename"].endswith(".g6")


def test_triangle_adjacency(client: FlaskClient) -> None:
    status, data = _post(client, {"edge_list": "0 1\n1 2\n0 2", "format": "adjacency"})
    assert status == 200
    assert data["content"] == "0 1 1\n1 0 1\n1 1 0"
    assert data["filename"].endswith(".txt")
    assert data["format"] == "adjacency"


def test_adjacency_filename_with_k_g(client: FlaskClient) -> None:
    status, data = _post(
        client,
        {"edge_list": "0 1\n1 2\n0 2", "format": "adjacency", "k": 2, "g": 3},
    )
    assert status == 200
    assert "k2_g3" in data["filename"]
    assert data["filename"].endswith(".txt")


def test_invalid_format_returns_400(client: FlaskClient) -> None:
    status, data = _post(client, {"edge_list": "0 1\n1 2", "format": "foo"})
    assert status == 400
    assert "error" in data


def test_empty_edge_list_returns_400(client: FlaskClient) -> None:
    status, data = _post(client, {"edge_list": "", "format": "g6"})
    assert status == 400
    assert "error" in data


def test_missing_fields_returns_400(client: FlaskClient) -> None:
    status, data = _post(client, {"format": "g6"})
    assert status == 400
    assert "error" in data
