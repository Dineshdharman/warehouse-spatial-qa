"""
tests/test_grader.py
Validates grader correctness, boundary conditions, and anti-gaming properties.
Catches: graders that always return 0.0 or 1.0, wrong normalization,
         wrong routing, silent exceptions.
"""
import pytest
from grader import score, score_easy, score_medium, score_hard, _normalize


# ── Task 1 Easy ───────────────────────────────────────────────────────────────

def test_easy_exact_match():
    assert score_easy("yes", "yes") == 1.0

def test_easy_case_insensitive():
    assert score_easy("YES", "yes") == 1.0
    assert score_easy("No",  "no")  == 1.0

def test_easy_wrong_answer():
    assert score_easy("no", "yes") == 0.0

def test_easy_id_match():
    assert score_easy("b", "b") == 1.0
    assert score_easy("B", "b") == 1.0
    assert score_easy("c", "b") == 0.0

def test_easy_zone_match():
    assert score_easy("RECEIVING",         "receiving")         == 1.0
    assert score_easy("bulk_storage_west", "BULK_STORAGE_WEST") == 1.0
    assert score_easy("SHIPPING",          "receiving")         == 0.0

def test_easy_returns_float():
    assert isinstance(score_easy("yes", "yes"), float)
    assert isinstance(score_easy("no",  "yes"), float)

def test_easy_whitespace_stripped():
    assert score_easy("  yes  ", "yes") == 1.0


# ── Task 2 Medium ─────────────────────────────────────────────────────────────

def test_medium_both_correct():
    assert score_medium("Part1=RECEIVING, Part2=b", ["RECEIVING", "b"]) == 1.0

def test_medium_part1_correct_only():
    assert score_medium("Part1=RECEIVING, Part2=c", ["RECEIVING", "b"]) == 0.5

def test_medium_part2_correct_only():
    assert score_medium("Part1=SHIPPING, Part2=b", ["RECEIVING", "b"]) == 0.5

def test_medium_both_wrong():
    assert score_medium("Part1=SHIPPING, Part2=z", ["RECEIVING", "b"]) == 0.0

def test_medium_fallback_when_no_part_labels():
    """If agent omits Part1=/Part2= labels, token-matching fallback applies."""
    assert score_medium("RECEIVING b", ["RECEIVING", "b"]) == 1.0

def test_medium_returns_float():
    r = score_medium("Part1=x, Part2=y", ["x", "y"])
    assert isinstance(r, float)

def test_medium_out_of_range_never_exceeds_1():
    r = score_medium("Part1=RECEIVING Part1=RECEIVING Part2=b", ["RECEIVING", "b"])
    assert r <= 1.0

def test_medium_case_insensitive():
    assert score_medium("part1=receiving, part2=B", ["RECEIVING", "b"]) == 1.0


# ── Task 3 Hard ───────────────────────────────────────────────────────────────

def test_hard_correct_id():
    assert score_hard("b", "b") == 1.0

def test_hard_none_answer_correct():
    assert score_hard("none", "none") == 1.0

def test_hard_wrong_id():
    assert score_hard("a", "b") == 0.0

def test_hard_none_when_conflict_expected():
    assert score_hard("none", "b") == 0.0

def test_hard_id_when_none_expected():
    assert score_hard("a", "none") == 0.0

def test_hard_returns_float():
    assert isinstance(score_hard("b", "b"), float)


# ── Universal dispatcher ──────────────────────────────────────────────────────

def test_dispatcher_routes_all_tasks():
    assert score("object_location",        "yes",                        "yes")          == 1.0
    assert score("multi_constraint_query", "Part1=RECEIVING, Part2=b",   ["RECEIVING","b"]) == 1.0
    assert score("movement_prediction",    "b",                          "b")             == 1.0

def test_dispatcher_returns_0_on_unknown_task():
    assert score("unknown_task", "anything", "expected") == 0.0

def test_dispatcher_never_raises_on_none_input():
    assert score("object_location",        None, "yes") == 0.0
    assert score("multi_constraint_query", None, ["x","y"]) == 0.0
    assert score("movement_prediction",    None, "b") == 0.0

def test_dispatcher_never_raises_on_empty_answer():
    assert score("object_location",        "", "yes") == 0.0

def test_dispatcher_never_raises_on_garbage_input():
    assert score("object_location", "!@#$%", "yes") == 0.0
    assert score("object_location", 12345,   "yes") == 0.0

def test_reward_always_float_all_tasks():
    for task in ["object_location", "multi_constraint_query", "movement_prediction"]:
        r = score(task, "yes", "yes")
        assert isinstance(r, float)

def test_reward_always_in_range_all_tasks():
    for task in ["object_location", "multi_constraint_query", "movement_prediction"]:
        for ans in ["yes", "no", "none", "RECEIVING", "b", ""]:
            r = score(task, ans, "yes")
            assert 0.0 <= r <= 1.0, f"Reward {r} out of range for task={task} ans={ans}"


# ── Grader is not trivially stuck on one value ────────────────────────────────

def test_easy_grader_returns_both_0_and_1():
    """DISQUALIFICATION prevention: grader must not always return same value."""
    results = {score_easy("yes", "yes"), score_easy("no", "yes")}
    assert 1.0 in results and 0.0 in results, \
        "Easy grader must return both 0.0 and 1.0 — it appears to be stuck"

def test_medium_grader_returns_multiple_values():
    values = set()
    test_cases = [
        ("Part1=RECEIVING, Part2=b", ["RECEIVING", "b"]),
        ("Part1=SHIPPING,  Part2=b", ["RECEIVING", "b"]),
        ("Part1=RECEIVING, Part2=c", ["RECEIVING", "b"]),
        ("Part1=SHIPPING,  Part2=c", ["RECEIVING", "b"]),
    ]
    for ans, exp in test_cases:
        values.add(score_medium(ans, exp))
    assert len(values) >= 2, \
        f"Medium grader only returns {values} — must produce multiple reward values"

def test_hard_grader_returns_both_0_and_1():
    results = {score_hard("b", "b"), score_hard("a", "b")}
    assert 1.0 in results and 0.0 in results


# ── Normalize helper ──────────────────────────────────────────────────────────

def test_normalize_lowercases():
    assert _normalize("YES") == "yes"

def test_normalize_strips_whitespace():
    assert _normalize("  yes  ") == "yes"

def test_normalize_strips_punctuation():
    assert _normalize("yes!") == "yes"

def test_normalize_preserves_underscores():
    assert _normalize("BULK_STORAGE_WEST") == "bulk_storage_west"

def test_normalize_handles_none():
    assert isinstance(_normalize(None), str)
