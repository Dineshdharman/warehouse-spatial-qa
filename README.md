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

The environment also implements an **RL-style feedback loop**: after each step, the reward and a human-readable feedback message are fed back into the LLM's conversation history so it can self-correct on subsequent steps — exactly like a policy-improvement loop in reinforcement learning.

---

## Observation Space

Each step the agent receives:

| Field | Type | Description |
|---|---|---|
| `scene_description` | `string` | Natural-language description of the full warehouse layout |
| `question` | `string` | Spatial question the agent must answer this step |
| `task_id` | `string` | Active task: `object_location` / `multi_constraint_query` / `movement_prediction` |
| `step_num` | `int` | Current step within the episode (1-indexed) |
| `hints` | `string` (optional) | Format hint for medium/hard tasks |
| `feedback` | `string` (optional) | RL reward feedback from the previous step for self-correction |
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
- **Description**: Single spatial question about item positions. Question types:
  - *West-of*: "Is item A located west of item B? Answer yes or no."
  - *North-of*: "Is item A positioned further north than item B?"
  - *Nearest neighbor*: "Which item is physically closest to item C?"
  - *Zone query*: "Which operational zone is item D assigned to?"
  - *Same zone*: "Are items A and B in the same operational zone?"
- **Anti-gaming**: yes/no questions guaranteed 50/50 split by construction.

---

### Task 2 — `multi_constraint_query` (Medium)
- **Max steps**: 3
- **Reward**: 0.5 per correct sub-answer per step (max 1.0)
- **Description**: Two-part compound query:
  - Part 1: Zone of a specific item
  - Part 2: Nearest item to a different reference item
  - **Required format**: `Part1=<ZONE>, Part2=<ID>`
- **Anti-gaming**: Part1 has 4 possible answers (0.25 random baseline).

---

### Task 3 — `movement_prediction` (Hard)
- **Max steps**: 5
- **Reward**: 1.0 per correctly predicted conflict per step; averaged for final score
- **Description**: Each step presents a movement event. The agent predicts which item must be cleared from the target slot, or `"none"` if unoccupied.
- **Anti-gaming**: ~50% conflict rate; bounded displacements prevent all-empty-slot gamability.

---

## RL Feedback Loop

After each step the environment returns a `feedback` field in the observation:

```
Step 1: Agent answers "no"  →  reward=0.0
Feedback: "Incorrect. Expected 'yes'. Answer exactly 'yes' or 'no'."

Step 2 (multi_constraint): Agent answers "Part1=SHIPPING, Part2=A"  →  reward=0.5
Feedback: "Part1 correct (SHIPPING); Part2 wrong — expected 'C', got 'A'. Use format: Part1=<ZONE>, Part2=<ID>"
```

This feedback is injected into the LLM's conversation history so it can self-correct — just like reward shaping in RL.

---

## Reward Function

| Task | Type | Range | Notes |
|---|---|---|---|
| `object_location` | Binary | 0.0 / 1.0 | Exact match required |
| `multi_constraint_query` | Partial | 0.0 – 1.0 | 0.5 per correct part |
| `movement_prediction` | Per-step binary averaged | 0.0 – 1.0 | Averaged across 5 steps |

All rewards are deterministic and reproducible. Malformed answers score 0.0.

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

### 1. Install dependencies
```bash
pip install openenv-core
pip install -r requirements.txt
```

### 2. Set environment variables
Create a `.env` file:
```env
HF_TOKEN=your_huggingface_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV_SEED=42
```

> **Note**: `HF_TOKEN` has no default and must be set. Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

### 3. Run locally with Uvicorn
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### 4. Run via Docker
```bash
docker build -t warehouse-spatial-qa .
docker run -p 7860:7860 --env-file .env warehouse-spatial-qa
```

### 5. Validate with OpenEnv CLI
```bash
# Validate local server
openenv validate --url http://localhost:7860
```

### 6. Run baseline inference (CLI)
```bash
python inference.py
```

Output follows the mandatory `[START]` / `[STEP]` / `[END]` format:
```
[START] task=object_location env=warehouse-spatial-qa model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action='yes' reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00

[START] task=multi_constraint_query env=warehouse-spatial-qa model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action='Part1=SHIPPING, Part2=A' reward=0.50 done=false error=null
[STEP] step=2 action='Part1=BULK_STORAGE_EAST, Part2=C' reward=1.00 done=false error=null
...

[SUMMARY]
  object_location: 1.00
  multi_constraint_query: 0.33
  movement_prediction: 0.00
  overall: 0.44
```

### 7. Run tests
```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset?task_id=<id>` | Start new episode, returns initial observation |
| `POST` | `/step` | Submit `{"answer": "..."}`, returns observation + reward + feedback |
| `GET` | `/state` | Full internal environment state |
| `GET` | `/health` | Health check — returns `{"status": "healthy"}` |
| `GET` | `/metadata` | Environment name, description, tags |
| `GET` | `/schema` | JSON schemas for action, observation, state |
| `POST` | `/mcp` | JSON-RPC 2.0 MCP tool discovery |
| `GET` | `/run/stream` | SSE stream: runs LLM auto-evaluation with RL feedback loop |
| `GET` | `/` | Interactive web UI (Auto Run + Manual Play) |

---

## Web UI

Run the server locally (`uvicorn server.app:app --port 7860`) and open `http://localhost:7860` in a browser:

**Auto Run tab** — select difficulty (Easy / Medium / Hard), set any number of episodes, click Run. The LLM plays automatically with live step-by-step output and a comprehensive score report.

**Manual Play tab** — interact with the environment yourself, see the warehouse scene, answer questions, and get real-time reward feedback.

---

## Project Structure

```
├── server/
│   ├── app.py          # FastAPI entry point (Dockerfile + openenv validate)
│   └── __init__.py
├── server.py           # FastAPI server (development alias)
├── environment.py      # SpatialQAEnv(Environment) — core logic + RL feedback
├── models.py           # SpatialAction, SpatialObservation (with feedback), SpatialState
├── scene_generator.py  # Random warehouse scene generator
├── question_bank.py    # Task question samplers (easy/medium/hard)
├── grader.py           # Deterministic scoring + RL feedback generator
├── inference.py        # Baseline LLM evaluation script with RL loop
├── index.html          # Interactive web UI
├── openenv.yaml        # OpenEnv metadata
├── pyproject.toml      # Python project config
├── requirements.txt    # Runtime dependencies
├── Dockerfile          # Container definition
└── tests/              # Unit tests
```

---

## Links

- **HuggingFace Space**: [https://huggingface.co/spaces/Dinesh-Kumar-26/warehouse-spatial-qa](https://huggingface.co/spaces/Dinesh-Kumar-26/warehouse-spatial-qa)
- **GitHub Repository**: [https://github.com/Dineshdharman/warehouse-spatial-qa](https://github.com/Dineshdharman/warehouse-spatial-qa)
