"""Tests for POST /api/cage/import."""

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


def _post(
    client: FlaskClient, body: dict[str, object]
) -> tuple[int, dict[str, object]]:
    resp = client.post(
        "/api/cage/import",
        data=json.dumps(body),
        content_type="application/json",
    )
    return resp.status_code, cast(dict[str, object], resp.get_json())


def _parse_edge_list(edge_list: str) -> set[tuple[int, int]]:
    """Parse an edge-list string into a canonical set of (u, v) pairs with u < v."""
    edges: set[tuple[int, int]] = set()
    for line in edge_list.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            u, v = int(parts[0]), int(parts[1])
            edges.add((min(u, v), max(u, v)))
    return edges


# ---- g6 round-trip ----


def test_g6_triangle(client: FlaskClient) -> None:
    # "Bw" = graph6 encoding of K3 (the triangle on nodes 0,1,2)
    status, data = _post(client, {"content": "Bw", "format": "g6"})
    assert status == 200
    edge_list = cast(str, data["edge_list"])
    edges = _parse_edge_list(edge_list)
    assert edges == {(0, 1), (0, 2), (1, 2)}


def test_g6_path_of_two_edges(client: FlaskClient) -> None:
    # "Bg" = graph6 encoding of P3, the path 0-1-2
    status, data = _post(client, {"content": "Bg", "format": "g6"})
    assert status == 200
    edge_list = cast(str, data["edge_list"])
    edges = _parse_edge_list(edge_list)
    assert edges == {(0, 1), (1, 2)}


def test_g6_with_leading_trailing_whitespace(client: FlaskClient) -> None:
    # Should still decode correctly
    status, data = _post(client, {"content": "  Bw\n", "format": "g6"})
    assert status == 200
    edge_list = cast(str, data["edge_list"])
    edges = _parse_edge_list(edge_list)
    assert edges == {(0, 1), (0, 2), (1, 2)}


# ---- adjacency round-trip ----


def test_adjacency_triangle(client: FlaskClient) -> None:
    matrix = "0 1 1\n1 0 1\n1 1 0"
    status, data = _post(client, {"content": matrix, "format": "adjacency"})
    assert status == 200
    edge_list = cast(str, data["edge_list"])
    edges = _parse_edge_list(edge_list)
    assert edges == {(0, 1), (0, 2), (1, 2)}


def test_adjacency_path(client: FlaskClient) -> None:
    matrix = "0 1 0\n1 0 1\n0 1 0"
    status, data = _post(client, {"content": matrix, "format": "adjacency"})
    assert status == 200
    edge_list = cast(str, data["edge_list"])
    edges = _parse_edge_list(edge_list)
    assert edges == {(0, 1), (1, 2)}


def test_adjacency_lower_triangle_only(client: FlaskClient) -> None:
    # Edge 1-2 encoded only in the lower triangle (matrix[2][1]); input is
    # treated as undirected so the edge must still appear.
    matrix = "0 0 0\n0 0 0\n0 1 0"
    status, data = _post(client, {"content": matrix, "format": "adjacency"})
    assert status == 200
    edge_list = cast(str, data["edge_list"])
    edges = _parse_edge_list(edge_list)
    assert edges == {(1, 2)}


def test_adjacency_single_node(client: FlaskClient) -> None:
    status, data = _post(client, {"content": "0", "format": "adjacency"})
    assert status == 200
    edge_list = cast(str, data["edge_list"])
    assert _parse_edge_list(edge_list) == set()


# ---- validation errors ----


def test_missing_content_returns_400(client: FlaskClient) -> None:
    status, data = _post(client, {"format": "g6"})
    assert status == 400
    assert "error" in data


def test_missing_format_returns_400(client: FlaskClient) -> None:
    status, data = _post(client, {"content": "Bw"})
    assert status == 400
    assert "error" in data


def test_invalid_format_returns_400(client: FlaskClient) -> None:
    status, data = _post(client, {"content": "Bw", "format": "dot"})
    assert status == 400
    assert "error" in data


def test_empty_content_returns_400(client: FlaskClient) -> None:
    status, data = _post(client, {"content": "", "format": "g6"})
    assert status == 400
    assert "error" in data


def test_malformed_g6_returns_400(client: FlaskClient) -> None:
    status, data = _post(client, {"content": "not-valid-g6!!!", "format": "g6"})
    assert status == 400
    assert "error" in data


def test_ragged_adjacency_returns_400(client: FlaskClient) -> None:
    # Row 1 has only 2 entries instead of 3
    status, data = _post(
        client, {"content": "0 1 1\n1 0\n1 1 0", "format": "adjacency"}
    )
    assert status == 400
    assert "error" in data


def test_non_binary_adjacency_returns_400(client: FlaskClient) -> None:
    status, data = _post(
        client, {"content": "0 1 2\n1 0 1\n2 1 0", "format": "adjacency"}
    )
    assert status == 400
    assert "error" in data
