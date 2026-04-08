"""
grader.py — Deterministic scoring for all 3 task types.
All functions return float in [0.0, 1.0].
"""
import re
from typing import Union


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: object_location
# Binary: exact match = 1.0, any mismatch = 0.0
# ─────────────────────────────────────────────────────────────────────────────

def score_easy(agent_answer: str, expected: str) -> float:
    return 1.0 if _normalize(agent_answer) == _normalize(expected) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Medium: multi_constraint_query
# Partial credit: 0.5 per correct part (Part1, Part2).
# Agent format: "Part1=<zone>, Part2=<ID>"
# ─────────────────────────────────────────────────────────────────────────────

def score_medium(agent_answer: str, expected_parts: list) -> float:
    """
    expected_parts = [zone_answer, nearest_id_answer]
    Extracts Part1 and Part2 from agent answer, awards 0.5 per correct part.
    """
    text  = agent_answer.lower()
    score = 0.0

    # Try regex extraction first
    p1_match = re.search(r"part1\s*[=:]\s*([a-z_]+)", text)
    p2_match = re.search(r"part2\s*[=:]\s*([a-z_]+)", text)

    p1 = _normalize(p1_match.group(1)) if p1_match else ""
    p2 = _normalize(p2_match.group(1)) if p2_match else ""

    if p1 == _normalize(expected_parts[0]):
        score += 0.5
    if p2 == _normalize(expected_parts[1]):
        score += 0.5

    # Fallback: if agent didn't use Part1/Part2 format but answer contains the expected tokens
    if score == 0.0:
        if _normalize(expected_parts[0]) in text:
            score += 0.5
        if _normalize(expected_parts[1]) in text:
            score += 0.5

    return round(score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard: movement_prediction
# Per-step binary reward (0.0 or 1.0).
# Partial credit across steps is accumulated by the environment.
# ─────────────────────────────────────────────────────────────────────────────

def score_hard(agent_answer: str, expected: str) -> float:
    return 1.0 if _normalize(agent_answer) == _normalize(expected) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Universal dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def score(task_id: str, agent_answer, expected: Union[str, list]) -> float:
    """
    Routes scoring to the appropriate grader.
    Always returns float in [0.0, 1.0].
    Never raises on bad input — returns 0.0 for malformed answers.
    """
    try:
        if task_id == "object_location":
            return score_easy(agent_answer, expected)
        elif task_id == "multi_constraint_query":
            return score_medium(agent_answer, expected)
        elif task_id == "movement_prediction":
            return score_hard(agent_answer, expected)
        else:
            return 0.0
    except Exception:
        return 0.0


def _normalize(text) -> str:
    """Lowercase, strip punctuation, strip whitespace. Preserves underscores."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text
