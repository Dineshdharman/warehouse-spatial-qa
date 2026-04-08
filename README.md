---
title: Warehouse Spatial QA
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Warehouse Spatial QA — OpenEnv Environment

> A real-world spatial reasoning benchmark where AI agents read natural-language warehouse floor-plan descriptions and answer progressively harder spatial questions. Built for the **Meta × PyTorch × Scaler OpenEnv Hackathon**.

---

## Overview & Motivation

Spatial reasoning is a critical capability for autonomous robots, logistics planners, and warehouse management systems. This environment simulates tasks that human warehouse operators and autonomous systems perform every day:

- Identifying object locations relative to aisles, rows, and operational zones
- Answering compound queries about multiple items simultaneously
- Predicting movement conflicts when items need to be relocated

The environment generates randomized warehouse floor plans (4×10 grid, 4 operational zones) with named items at specific positions, then challenges the agent with spatial questions that require genuine geometric reasoning — not pattern matching.

---

## Observation Space

Each step the agent receives:

| Field | Type | Description |
|---|---|---|
| `scene_description` | `string` | Natural-language description of the full warehouse layout including all item positions, aisles, rows, and zone assignments |
| `question` | `string` | Spatial question the agent must answer this step |
| `task_id` | `string` | Active task: `object_location` / `multi_constraint_query` / `movement_prediction` |
| `step_num` | `int` | Current step within the episode (1-indexed) |
| `hints` | `string` (optional) | Format hint shown for medium/hard tasks |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float \| null` | Reward from the last action (`null` on reset) |

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `answer` | `string` | Agent's free-text answer to the spatial question |

---

## Tasks

### Task 1 — `object_location` (Easy)
- **Max steps**: 1
- **Reward**: Binary — 1.0 (correct) or 0.0 (wrong)
- **Description**: Single spatial question about item positions. Question types include:
  - *West-of*: "Is item A located west of item B? Answer yes or no."
  - *North-of*: "Is item A positioned further north than item B?"
  - *Nearest neighbor*: "Which item is physically closest to item C?"
  - *Zone query*: "Which operational zone is item D assigned to?"
  - *Same zone*: "Are items A and B in the same operational zone?"
- **Anti-gaming**: yes/no questions are guaranteed 50/50 split by construction; answers are computed deterministically from the scene.

---

### Task 2 — `multi_constraint_query` (Medium)
- **Max steps**: 3
- **Reward**: 0.5 per correct sub-answer per step (max 1.0)
- **Description**: Two-part compound query answered in a single response:
  - Part 1: Zone of a specific item
  - Part 2: Nearest item to a different reference item
  - **Format**: `Part1=<ZONE>, Part2=<ID>`
- **Anti-gaming**: Part1 has 4 possible answers (0.25 random baseline); Part2 answer depends on exact Euclidean distances.

---

### Task 3 — `movement_prediction` (Hard)
- **Max steps**: 5
- **Reward**: 1.0 per correctly predicted conflict per step; averaged for final score
- **Description**: Each step presents a warehouse movement event. An item is being relocated. The agent must predict:
  - Which item currently occupying the target slot must be cleared first, OR
  - `"none"` if the target slot is unoccupied
- **Anti-gaming**: ~50% conflict rate by controlled random generation; bounded displacements prevent always-empty-slot gamability.

---

## Reward Function

| Task | Type | Range | Notes |
|---|---|---|---|
| `object_location` | Binary | 0.0 / 1.0 | Exact match required |
| `multi_constraint_query` | Partial | 0.0 – 1.0 | 0.5 per correct part |
| `movement_prediction` | Per-step binary averaged | 0.0 – 1.0 | Averaged across all 5 steps |

All rewards are deterministic and reproducible. Wrong/malformed answers always score 0.0 — never negative.

---

## Baseline Performance

Evaluated with `Qwen/Qwen2.5-72B-Instruct` via Hugging Face router API (`ENV_SEED=42`):

| Task | Score |
|---|---|
| `object_location` | 1.00 |
| `multi_constraint_query` | 0.33 |
| `movement_prediction` | 0.00 |
| **Overall** | **0.44** |

---

## Setup & Usage

### Requirements
```bash
pip install openenv-core
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file:
```env
HF_TOKEN=your_huggingface_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV_SEED=42
```

> **Note**: `HF_TOKEN` has no default and must be set. Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

### 1. Run Locally with Uvicorn
```bash
cd spatial-qa-env
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

### 2. Run via Docker
```bash
docker build -t warehouse-spatial-qa .
docker run -p 7860:7860 --env-file .env warehouse-spatial-qa
```

### 3. Validate with OpenEnv CLI
```bash
# Validate running server
openenv validate --url http://localhost:7860

# Validate live HF Space
openenv validate --url https://dinesh-kumar-26-warehouse-spatial-qa.hf.space
```

### 4. Run Baseline Inference
```bash
python inference.py
```
Output follows the mandatory `[START]` / `[STEP]` / `[END]` format:
```
[START] task=object_location env=warehouse-spatial-qa model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action='yes' reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00
```

### 5. Run Tests
```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset?task_id=<id>` | Start new episode, returns initial observation |
| `POST` | `/step` | Submit `{"answer": "..."}`, returns observation + reward |
| `GET` | `/state` | Full internal environment state |
| `GET` | `/health` | Health check — returns `{"status": "healthy"}` |
| `GET` | `/metadata` | Environment name, description, tags |
| `GET` | `/schema` | JSON schemas for action, observation, state |
| `POST` | `/mcp` | JSON-RPC 2.0 MCP tool discovery |
| `GET` | `/` | Interactive web UI |

---

## Project Structure

```
spatial-qa-env/
├── server.py           # FastAPI HTTP server (all OpenEnv endpoints)
├── environment.py      # SpatialQAEnv(Environment) — core logic
├── models.py           # SpatialAction, SpatialObservation, SpatialState
├── scene_generator.py  # Random warehouse scene generator
├── question_bank.py    # Task question samplers (easy/medium/hard)
├── grader.py           # Deterministic scoring functions
├── inference.py        # Baseline LLM evaluation script
├── index.html          # Interactive web UI
├── openenv.yaml        # OpenEnv metadata
├── pyproject.toml      # Python project config
├── requirements.txt    # Runtime dependencies
├── Dockerfile          # Container definition
└── tests/              # 139 unit tests
```

---

## Links

- **HuggingFace Space**: https://huggingface.co/spaces/Dinesh-Kumar-26/warehouse-spatial-qa
- **GitHub Repository**: https://github.com/Dineshdharman/warehouse-spatial-qa
