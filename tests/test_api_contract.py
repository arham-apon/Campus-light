from __future__ import annotations

import copy


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_body_is_400(client):
    assert client.post("/optimize-energy", json={"scenario_id": "X"}).status_code == 400


def test_twenty_three_hours_is_400(client, sample_pack):
    payload = copy.deepcopy(sample_pack["cases"][0]["input"])
    payload["hours"] = payload["hours"][:23]
    assert client.post("/optimize-energy", json=payload).status_code == 400


def test_duplicate_hour_is_400(client, sample_pack):
    payload = copy.deepcopy(sample_pack["cases"][0]["input"])
    payload["hours"][5]["hour"] = 6
    assert client.post("/optimize-energy", json=payload).status_code == 400


def test_empty_notes_is_400(client, sample_pack):
    payload = copy.deepcopy(sample_pack["cases"][0]["input"])
    payload["operator_notes"] = []
    assert client.post("/optimize-energy", json=payload).status_code == 400


def test_unknown_extra_fields_are_tolerated(client, sample_pack):
    payload = copy.deepcopy(sample_pack["cases"][0]["input"])
    payload["unexpected_field"] = {"anything": True}
    assert client.post("/optimize-energy", json=payload).status_code == 200


def test_error_body_leaks_nothing(client):
    body = client.post("/optimize-energy", json={"bad": 1}).json()
    text = str(body).lower()
    assert "traceback" not in text and "gemini_api_key" not in text and "aiza" not in text
