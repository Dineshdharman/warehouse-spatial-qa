"""
server/app.py — FastAPI entry point for SpatialQAEnv.
Used by Dockerfile (uvicorn server.app:app) and openenv validate.

Exposes:
  POST /reset, POST /step, GET /state, GET /health,
  GET /metadata, GET /schema, POST /mcp, GET /
  GET /run/stream  — RL evaluation with SSE streaming
"""
import asyncio
import json
import os
import sys

# Ensure root package is importable whether run from repo root or server/
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from models import SpatialAction, SpatialObservation, SpatialState
from environment import SpatialQAEnv

app = FastAPI(title="Warehouse Spatial QA", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = SpatialQAEnv()


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.post("/reset")
def reset(task_id: str = "object_location"):
    try:
        obs = _env.reset(task_id=task_id)
        return {"observation": obs, "done": obs.done}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: SpatialAction):
    try:
        obs = _env.step(action)
        return {
            "observation": obs,
            "reward":      obs.reward,
            "done":        obs.done,
            "info":        obs.metadata,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return _env.state()


# ── openenv validate required endpoints ──────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "warehouse-spatial-qa",
        "description": (
            "Spatial reasoning benchmark where an AI agent reads warehouse "
            "floor-plan descriptions and answers spatial questions. Three tasks: "
            "object_location (easy), multi_constraint_query (medium), "
            "movement_prediction (hard)."
        ),
        "version": "1.0.0",
        "tags": ["openenv", "spatial-reasoning", "logistics", "real-world"],
    }


@app.get("/schema")
def schema():
    return {
        "action":      SpatialAction.model_json_schema(),
        "observation": SpatialObservation.model_json_schema(),
        "state":       SpatialState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request):
    """Minimal JSON-RPC 2.0 endpoint for MCP tool discovery."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    method = body.get("method", "")
    req_id = body.get("id", 1)

    if method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Reset the environment for a given task",
                    "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string"}}},
                },
                {
                    "name": "step",
                    "description": "Submit an answer and advance the episode",
                    "inputSchema": SpatialAction.model_json_schema(),
                },
            ]
        }
    else:
        result = {"message": f"method '{method}' not implemented"}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.get("/")
def ui():
    return FileResponse(os.path.join(_BASE_DIR, "index.html"))


# ── RL Auto-Run with SSE streaming ────────────────────────────────────────────

@app.get("/run/stream")
async def run_stream(
    task_ids: str = "object_location,multi_constraint_query,movement_prediction",
    episodes: int = 1,
):
    """
    SSE endpoint: runs N episodes per selected task using the LLM with RL
    feedback loop. Streams one JSON event per step so the UI updates live.

    Event types: episode_start | step | episode_end | done
    """
    from openai import OpenAI
    from inference import run_task_rl, API_BASE_URL, API_KEY

    tasks = [t.strip() for t in task_ids.split(",") if t.strip()]

    async def generate():
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        all_scores: dict = {t: [] for t in tasks}

        for task_id in tasks:
            for ep in range(episodes):
                yield (
                    f"data: {json.dumps({'type': 'episode_start', 'task': task_id, 'episode': ep + 1, 'total': episodes})}\n\n"
                )
                result = await asyncio.to_thread(
                    run_task_rl, client, task_id, ep, False
                )
                for step_data in result["steps"]:
                    yield f"data: {json.dumps({'type': 'step', 'task': task_id, 'episode': ep + 1, **step_data})}\n\n"

                all_scores[task_id].append(result["score"])
                yield (
                    f"data: {json.dumps({'type': 'episode_end', 'task': task_id, 'episode': ep + 1, 'score': result['score'], 'success': result['success'], 'rewards': result['rewards']})}\n\n"
                )

        summary = {}
        for tid, scores in all_scores.items():
            wins = sum(1 for s in scores if s >= 0.5)
            summary[tid] = {
                "avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
                "win_rate":  round(wins / len(scores), 2) if scores else 0.0,
                "episodes":  len(scores),
            }
        overall = (
            round(sum(v["avg_score"] for v in summary.values()) / len(summary), 2)
            if summary else 0.0
        )
        yield f"data: {json.dumps({'type': 'done', 'summary': summary, 'overall': overall})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
