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

## Description
A spatial reasoning benchmark for AI agents. The agent reads natural-language
descriptions of warehouse floor plans and answers spatial questions. Models
a task used in robotics navigation, autonomous logistics, and GIS systems.

## Observation Space
| Field | Type | Description |
|---|---|---|
| `scene_description` | string | Natural-language warehouse floor-plan description |
| `question` | string | Spatial question the agent must answer |
| `task_id` | string | Active task: `object_location`, `multi_constraint_query`, `movement_prediction` |
| `step_num` | int | Current step within the episode |
| `hints` | string (optional) | Format hint for medium/hard tasks |

## Action Space
| Field | Type | Description |
|---|---|---|
| `answer` | string | Agent's free-text answer to the spatial question |

## Tasks
| ID | Difficulty | Max Steps | Description |
|---|---|---|---|
| `object_location` | Easy | 1 | Single spatial question (left/right, above/below, nearest, zone) |
| `multi_constraint_query` | Medium | 3 | Compound query with two spatial constraints; partial credit |
| `movement_prediction` | Hard | 5 | Predict post-movement conflicts; per-step reward |

## Reward
- Reward is always `float` in `[0.0, 1.0]`.
- Easy: binary (1.0 correct, 0.0 wrong).
- Medium: 0.5 per correct sub-answer.
- Hard: 1.0 per correctly predicted conflict per step; averaged for final score.

## Setup

```bash
pip install -r requirements.txt
```

## Run Server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

## Run Baseline

```bash
python inference.py
```

## Run Tests

```bash
pytest tests/ -v
```

## API Endpoints
- `POST /reset?task_id=<id>` — Start new episode
- `POST /step` — Submit answer `{"answer": "<text>"}`
- `GET  /state` — Current environment state
