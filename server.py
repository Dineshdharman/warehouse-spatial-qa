"""
server.py — FastAPI HTTP wrapper for SpatialQAEnv.
Exposes all endpoints required by openenv validate:
  POST /reset, POST /step, GET /state, GET /health,
  GET /metadata, GET /schema, POST /mcp, GET /
"""
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
            "reward": obs.reward,
            "done": obs.done,
            "info": obs.metadata,
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
    return FileResponse("index.html")
