"""Error contract and request-ID behaviour, using throwaway routes on a real app instance."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.exceptions import ConflictError, NotFoundError


class Payload(BaseModel):
    value: int


@pytest.fixture
def error_client(app: FastAPI) -> TestClient:
    @app.get("/_test/not-found")
    def not_found() -> None:
        raise NotFoundError("Paper not found", details={"paper_id": "abc"})

    @app.get("/_test/conflict")
    def conflict() -> None:
        raise ConflictError("Duplicate paper")

    @app.get("/_test/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail: password=hunter2")

    @app.post("/_test/validate")
    def validate(payload: Payload) -> Payload:
        return payload

    with TestClient(app) as client:
        yield client


def _assert_error_shape(body: dict, code: str) -> None:
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"]


def test_app_error_maps_to_status_and_schema(error_client: TestClient) -> None:
    resp = error_client.get("/_test/not-found")
    assert resp.status_code == 404
    _assert_error_shape(resp.json(), "not_found")
    assert resp.json()["error"]["details"] == {"paper_id": "abc"}
    assert resp.json()["error"]["request_id"] == resp.headers["X-Request-ID"]


def test_conflict_error(error_client: TestClient) -> None:
    resp = error_client.get("/_test/conflict")
    assert resp.status_code == 409
    _assert_error_shape(resp.json(), "conflict")


def test_validation_error_uses_common_schema(error_client: TestClient) -> None:
    resp = error_client.post("/_test/validate", json={"value": "not-a-number"})
    assert resp.status_code == 422
    body = resp.json()
    _assert_error_shape(body, "validation_error")
    assert body["error"]["details"][0]["loc"] == ["body", "value"]


def test_unknown_route_returns_404_in_common_schema(error_client: TestClient) -> None:
    resp = error_client.get("/does-not-exist")
    assert resp.status_code == 404
    _assert_error_shape(resp.json(), "not_found")


def test_unhandled_exception_is_500_without_leaking_details(error_client: TestClient) -> None:
    resp = error_client.get("/_test/boom")
    assert resp.status_code == 500
    _assert_error_shape(resp.json(), "internal_error")
    assert "hunter2" not in resp.text
    assert resp.json()["error"]["request_id"] == resp.headers["X-Request-ID"]


def test_valid_client_request_id_is_propagated(error_client: TestClient) -> None:
    resp = error_client.get("/_test/conflict", headers={"X-Request-ID": "client-abc_123"})
    assert resp.headers["X-Request-ID"] == "client-abc_123"
    assert resp.json()["error"]["request_id"] == "client-abc_123"


@pytest.mark.parametrize("bad", ["has spaces", "x" * 200, "inject\\nlog"])
def test_unsafe_client_request_id_is_replaced(error_client: TestClient, bad: str) -> None:
    resp = error_client.get("/_test/conflict", headers={"X-Request-ID": bad})
    assert resp.headers["X-Request-ID"] != bad
    assert len(resp.headers["X-Request-ID"]) == 32  # uuid4 hex
