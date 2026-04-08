"""
tests/test_server.py
HTTP endpoint tests using FastAPI TestClient.
Validates: POST /reset returns 200, POST /step works, GET /state works.
These are the EXACT checks the hackathon pre-validation script runs.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from server import app


@pytest.mark.asyncio
async def test_post_reset_returns_200():
    """HACKATHON GATE: HF Space validator pings POST /reset, expects HTTP 200."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/reset")
    assert response.status_code == 200, \
        f"POST /reset returned {response.status_code} — HF Space liveness check will FAIL"

@pytest.mark.asyncio
async def test_post_reset_with_task_id_param():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for task in ["object_location", "multi_constraint_query", "movement_prediction"]:
            response = await client.post(f"/reset?task_id={task}")
            assert response.status_code == 200, \
                f"POST /reset?task_id={task} returned {response.status_code}"

@pytest.mark.asyncio
async def test_post_reset_response_has_observation_fields():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/reset")
    data = response.json()
    assert "observation" in data, "reset response must contain 'observation' key"
    obs = data["observation"]
    assert "scene_description" in obs, "observation must have scene_description"
    assert "question"          in obs, "observation must have question"
    assert "task_id"           in obs, "observation must have task_id"
    assert "step_num"          in obs, "observation must have step_num"

@pytest.mark.asyncio
async def test_post_step_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/reset?task_id=object_location")
        response = await client.post("/step", json={"answer": "yes"})
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_post_step_response_has_reward_and_done():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/reset?task_id=object_location")
        response = await client.post("/step", json={"answer": "yes"})
    data = response.json()
    assert "reward" in data, "step response must contain 'reward'"
    assert "done"   in data, "step response must contain 'done'"
    assert isinstance(data["reward"], float)
    assert 0.0 <= data["reward"] <= 1.0

@pytest.mark.asyncio
async def test_get_state_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/reset")
        response = await client.get("/state")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_health_endpoint_returns_healthy():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_metadata_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "description" in data

@pytest.mark.asyncio
async def test_schema_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/schema")
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "observation" in data
    assert "state" in data

@pytest.mark.asyncio
async def test_mcp_endpoint_returns_jsonrpc():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1})
    assert response.status_code == 200
    assert response.json()["jsonrpc"] == "2.0"

@pytest.mark.asyncio
async def test_post_step_without_reset_does_not_crash_server():
    """Server must handle out-of-order calls gracefully — must not return 500."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Do NOT reset first — send step directly
        response = await client.post("/step", json={"answer": "yes"})
    # Acceptable: 400, 422, or handled 200 — NOT 500
    assert response.status_code != 500, \
        "Server must not crash with 500 on out-of-order step() call"

@pytest.mark.asyncio
async def test_post_reset_invalid_task_returns_error():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/reset?task_id=invalid_task")
    assert response.status_code in (400, 422), \
        "Invalid task_id must return 400 or 422, not swallowed silently"
