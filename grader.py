"""
grader.py — Deterministic scoring for all 3 task types.
All functions return float in [0.0, 1.0].
"""
import re
from typing import Union


# ─────────────────────────────────────────────────────────────────────────────
# Answer extractors — handle verbose LLM responses gracefully
# e.g. "I think it's B" → "b",  "The answer is yes." → "yes"
# ─────────────────────────────────────────────────────────────────────────────

def _extract_yesno(text: str) -> str:
    """Extract first 'yes' or 'no' token from a potentially verbose response."""
    t = text.strip().lower()
    m = re.search(r'\b(yes|no)\b', t)
    return m.group(1) if m else t

def _extract_id(text: str) -> str:
    """
    Extract a single item-letter ID (a–z) or 'none' from a verbose response.
    Priority: explicit 'none' keyword > single-letter match.
    """
    t = text.strip().lower()
    # Explicit 'none'
    if re.search(r'\bnone\b', t):
        return "none"
    # XML tag: <answer>X</answer>
    m = re.search(r'<answer>\s*([a-z])\s*</answer>', t)
    if m:
        return m.group(1)
    # Patterns: "answer is X", "item X", "answer: X"
    m = re.search(r'\b(?:answer(?:\s+is)?|item)[:\s]+([a-z])\b', t)
    if m:
        return m.group(1)
    # Standalone single letter
    m = re.fullmatch(r'([a-z])\.?', t)
    if m:
        return m.group(1)
    # Last resort: first standalone letter a–z
    m = re.search(r'\b([a-z])\b', t)
    if m:
        return m.group(1)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: object_location
# Binary: exact match = 1.0, any mismatch = 0.0
# ─────────────────────────────────────────────────────────────────────────────

def score_easy(agent_answer: str, expected: str) -> float:
    exp = _normalize(expected)
    if exp in ("yes", "no"):
        return 1.0 if _extract_yesno(agent_answer) == exp else 0.0
    # zone or item-ID question
    return 1.0 if _normalize(agent_answer) == exp else 0.0


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
    return 1.0 if _extract_id(agent_answer) == _normalize(expected) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Universal dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def score(task_id: str, agent_answer, expected: Union[str, list]) -> float:
    """
    Routes scoring to the appropriate grader.
    Always returns float strictly in (0.0, 1.0) exclusive — never 0.0 or 1.0.
    Never raises on bad input — returns 0.01 for malformed answers.
    """
    try:
        if task_id == "object_location":
            raw = score_easy(agent_answer, expected)
        elif task_id == "multi_constraint_query":
            raw = score_medium(agent_answer, expected)
        elif task_id == "movement_prediction":
            raw = score_hard(agent_answer, expected)
        else:
            raw = 0.0
        return round(min(max(raw, 0.01), 0.99), 2)
    except Exception:
        return 0.01


def _normalize(text) -> str:
    """Lowercase, strip punctuation, strip whitespace. Preserves underscores."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# RL Feedback generator — human-readable signal fed back to the LLM
# ─────────────────────────────────────────────────────────────────────────────

def generate_feedback(task_id: str, agent_answer: str, expected, reward: float) -> str:
    """
    Returns a natural-language feedback string that is appended to the LLM's
    conversation history so it can self-correct on subsequent steps.
    """
    try:
        if task_id == "object_location":
            if reward >= 0.99:
                return "Correct!"
            return (
                f"Incorrect (reward=0.0). You answered '{agent_answer}', "
                f"but the correct answer is '{expected}'. "
                f"Answer exactly 'yes' or 'no' for direction questions, "
                f"a single letter ID for nearest-item questions, "
                f"or the full zone name for zone questions."
            )

        elif task_id == "multi_constraint_query":
            text = agent_answer.lower()
            p1_match = re.search(r"part1\s*[=:]\s*([a-z_]+)", text)
            p2_match = re.search(r"part2\s*[=:]\s*([a-z_]+)", text)
            p1 = _normalize(p1_match.group(1)) if p1_match else ""
            p2 = _normalize(p2_match.group(1)) if p2_match else ""
            p1_ok = p1 == _normalize(expected[0])
            p2_ok = p2 == _normalize(expected[1])
            if reward >= 0.99:
                return "Both parts correct!"
            parts = []
            if p1_ok:
                parts.append(f"Part1 correct ({expected[0]})")
            else:
                parts.append(f"Part1 wrong — expected '{expected[0]}', got '{p1 or '?'}'")
            if p2_ok:
                parts.append(f"Part2 correct ({expected[1].upper()})")
            else:
                parts.append(f"Part2 wrong — expected '{expected[1].upper()}', got '{p2.upper() or '?'}'")
            return (
                f"Partial reward={reward:.2f}. {'; '.join(parts)}. "
                f"Use format: Part1=<ZONE>, Part2=<ID>"
            )

        elif task_id == "movement_prediction":
            if reward >= 0.99:
                return "Correct! Moving to the next movement event."
            if expected == "none":
                return (
                    f"Incorrect (reward=0.0). The target slot is unoccupied — "
                    f"answer 'none'. You answered '{agent_answer}'."
                )
            return (
                f"Incorrect (reward=0.0). Item {expected.upper()} occupies the target slot "
                f"and must be cleared first. You answered '{agent_answer}'. "
                f"Answer with the single letter ID of the blocking item, or 'none' if empty."
            )
    except Exception:
        pass
    return f"reward={reward:.2f}"
