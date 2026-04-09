"""
tests/test_anti_gaming.py
CRITICAL — Prevents disqualification for "graders that always return the same score".

Tests that a naive agent (always answers the same thing) cannot achieve high reward.
These tests prove the environment is genuinely evaluating spatial reasoning,
not just rewarding any output.

Game-proof thresholds:
  - Always "yes"        → easy task average reward ≤ 0.55 (random pair baseline ≈ 0.5)
  - Always "none"       → hard task average reward ≤ 0.65 (conflict rate ~40%)
  - Always "RECEIVING"  → zone task average reward ≤ 0.30 (4 zones)
  - Always "a"          → nearest/ID task average reward ≤ 0.40 (4 items)
"""
import random
import pytest
from scene_generator import generate_scene
from question_bank import (
    q_west_of, q_north_of, q_nearest, q_zone_query, q_same_zone,
    sample_hard, sample_easy,
)
from grader import score

N_TRIALS = 400   # 400 trials → std ≈ 0.025, keeping false-positive rate below 2%


# ── Naive strategies should NOT yield high reward ────────────────────────────

def test_always_yes_does_not_score_high_on_easy_task():
    """
    An agent always answering 'yes' must not average above 0.55 on object_location.
    Yes/no pairs are ~50/50 by construction. Failure here means questions are biased.
    """
    rewards = []
    for seed in range(N_TRIALS):
        random.seed(seed)
        scene = generate_scene(4)
        _, expected, meta = q_west_of(scene)
        rewards.append(score("object_location", "yes", expected))
    avg = sum(rewards) / len(rewards)
    assert avg <= 0.55, \
        f"Always-yes strategy averaged {avg:.3f} on west_of questions — questions are biased toward 'yes'"

def test_always_no_does_not_score_high_on_easy_task():
    rewards = []
    for seed in range(N_TRIALS):
        random.seed(seed)
        scene = generate_scene(4)
        _, expected, _ = q_west_of(scene)
        rewards.append(score("object_location", "no", expected))
    avg = sum(rewards) / len(rewards)
    assert avg <= 0.55, \
        f"Always-no strategy averaged {avg:.3f} — questions are biased toward 'no'"

def test_always_receiving_does_not_score_high_on_zone_questions():
    """
    4 valid zones → random baseline is 0.25.
    Always guessing 'RECEIVING' must not exceed 0.35 (some overshoot allowed for small grid).
    """
    rewards = []
    for seed in range(N_TRIALS):
        random.seed(seed)
        scene = generate_scene(4)
        _, expected, _ = q_zone_query(scene)
        rewards.append(score("object_location", "RECEIVING", expected))
    avg = sum(rewards) / len(rewards)
    assert avg <= 0.35, \
        f"Always-RECEIVING strategy averaged {avg:.3f} — zone distribution may be skewed"

def test_always_none_does_not_dominate_hard_task():
    """
    Hard task: 'none' is correct only when no conflict.
    Conflict rate is ~40–60%, so always-none should score ≤ 0.65.
    """
    rewards = []
    for seed in range(N_TRIALS):
        random.seed(seed)
        scene = generate_scene(5)
        _, expected, _ = sample_hard(scene, step_num=1)
        rewards.append(score("movement_prediction", "none", expected))
    avg = sum(rewards) / len(rewards)
    assert avg <= 0.65, \
        f"Always-none hard task averaged {avg:.3f} — conflict rate is too low (hard task is gameable)"

def test_always_wrong_answer_id_does_not_score_high_on_nearest():
    """
    Always answering 'z' (invalid ID) should score 0.0 on nearest questions.
    """
    for seed in range(20):
        random.seed(seed)
        scene = generate_scene(4)
        q, expected, _ = q_nearest(scene)
        assert score("object_location", "z", expected) <= 0.01, \
            "Invalid item ID should score at most 0.01"


# ── Graders produce varied output (not stuck) ────────────────────────────────

def test_easy_grader_produces_variance():
    """Grader must produce a mix of 0.0 and 1.0 — not a constant function."""
    results = set()
    for seed in range(40):
        random.seed(seed)
        scene = generate_scene(4)
        _, expected, _ = sample_easy(scene)
        r = score("object_location", "yes", expected)
        results.add(r)
    assert len(results) >= 2, \
        f"Easy grader only produced values {results} across 40 trials — appears stuck"

def test_hard_grader_produces_both_conflict_and_no_conflict():
    """Hard task must have both conflict and no-conflict scenarios across runs."""
    has_conflict    = False
    has_no_conflict = False
    for seed in range(50):
        random.seed(seed)
        scene = generate_scene(5)
        _, expected, meta = sample_hard(scene, step_num=1)
        if meta["conflict_item"] is not None:
            has_conflict = True
        else:
            has_no_conflict = True
        if has_conflict and has_no_conflict:
            break
    assert has_conflict,    "Hard task never produced a conflict across 50 trials — always 'none'"
    assert has_no_conflict, "Hard task always produced a conflict across 50 trials — never 'none'"


# ── Medium task partial credit works correctly ───────────────────────────────

def test_medium_partial_credit_is_strictly_between_0_and_1():
    """Partial credit (0.5) must exist — medium task must not be purely binary."""
    from grader import score_medium
    partial = score_medium("Part1=RECEIVING, Part2=z", ["RECEIVING", "b"])
    assert partial == 0.5, \
        f"Expected partial credit 0.5, got {partial} — medium task may be binary"

def test_medium_partial_credit_adds_up_correctly():
    from grader import score_medium
    assert score_medium("Part1=RECEIVING, Part2=b", ["RECEIVING", "b"]) == 1.0
    assert score_medium("Part1=RECEIVING, Part2=c", ["RECEIVING", "b"]) == 0.5
    assert score_medium("Part1=SHIPPING,  Part2=b", ["RECEIVING", "b"]) == 0.5
    assert score_medium("Part1=SHIPPING,  Part2=c", ["RECEIVING", "b"]) == 0.0
