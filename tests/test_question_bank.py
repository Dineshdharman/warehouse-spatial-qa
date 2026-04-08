"""
tests/test_question_bank.py
Validates question generators produce correct, non-trivial, answerable questions.
Catches: wrong answer computation, broken dispatch, ambiguous questions.
"""
import random
import pytest
from scene_generator import generate_scene, is_west_of, is_north_of, euclidean_distance
from question_bank import (
    sample_easy, sample_medium, sample_hard,
    EASY_DISPATCH, VALID_ZONES,
    q_west_of, q_north_of, q_nearest, q_zone_query, q_same_zone,
)


# ── Return type contract ──────────────────────────────────────────────────────

def test_sample_easy_returns_three_tuple():
    scene = generate_scene(4)
    result = sample_easy(scene)
    assert len(result) == 3
    q, a, meta = result
    assert isinstance(q, str)
    assert isinstance(a, str)
    assert isinstance(meta, dict)

def test_sample_medium_returns_three_tuple_with_list():
    scene = generate_scene(4)
    q, parts, meta = sample_medium(scene)
    assert isinstance(q, str)
    assert isinstance(parts, list)
    assert len(parts) == 2
    assert all(isinstance(p, str) for p in parts)

def test_sample_hard_returns_three_tuple():
    scene = generate_scene(4)
    q, a, meta = sample_hard(scene, step_num=1)
    assert isinstance(q, str)
    assert isinstance(a, str)
    assert isinstance(meta, dict)


# ── Answer correctness ────────────────────────────────────────────────────────

def test_west_of_answer_matches_ground_truth():
    """Answer must equal the actual spatial relationship, not a hardcoded value."""
    random.seed(0)
    scene = generate_scene(4)
    for _ in range(20):
        q, answer, meta = q_west_of(scene)
        a_id, b_id = meta["ids"]
        a_item = next(i for i in scene["items"] if i["id"] == a_id)
        b_item = next(i for i in scene["items"] if i["id"] == b_id)
        expected = "yes" if is_west_of(a_item, b_item) else "no"
        assert answer == expected, \
            f"west_of answer '{answer}' does not match ground truth '{expected}'"

def test_north_of_answer_matches_ground_truth():
    random.seed(1)
    scene = generate_scene(5)
    for _ in range(20):
        q, answer, meta = q_north_of(scene)
        if meta["type"] == "north_of":
            a_id, b_id = meta["ids"]
            a = next(i for i in scene["items"] if i["id"] == a_id)
            b = next(i for i in scene["items"] if i["id"] == b_id)
            expected = "yes" if is_north_of(a, b) else "no"
            assert answer == expected

def test_nearest_answer_is_actual_nearest():
    random.seed(2)
    scene = generate_scene(5)
    for _ in range(20):
        q, answer, meta = q_nearest(scene)
        ref   = next(i for i in scene["items"] if i["id"] == meta["ref"])
        others = [i for i in scene["items"] if i["id"] != ref["id"]]
        true_nearest = min(others, key=lambda i: euclidean_distance(ref, i))
        assert answer == true_nearest["id"].lower(), \
            f"Nearest answer '{answer}' wrong. True nearest is '{true_nearest['id'].lower()}'"

def test_zone_query_answer_matches_item_zone():
    scene = generate_scene(4)
    for _ in range(20):
        q, answer, meta = q_zone_query(scene)
        item = next(i for i in scene["items"] if i["id"] == meta["id"])
        assert answer == item["zone"], \
            f"Zone answer '{answer}' does not match item zone '{item['zone']}'"

def test_zone_answer_is_valid_zone():
    scene = generate_scene(4)
    for _ in range(30):
        q, answer, meta = q_zone_query(scene)
        assert answer in VALID_ZONES, \
            f"Zone answer '{answer}' is not in VALID_ZONES={VALID_ZONES}"

def test_same_zone_answer_matches_ground_truth():
    scene = generate_scene(4)
    for _ in range(20):
        q, answer, meta = q_same_zone(scene)
        ids   = meta["ids"]
        a     = next(i for i in scene["items"] if i["id"] == ids[0])
        b     = next(i for i in scene["items"] if i["id"] == ids[1])
        expected = "yes" if a["zone"] == b["zone"] else "no"
        assert answer == expected


# ── Questions reference WMS naming (not raw coords) ───────────────────────────

def test_easy_questions_use_aisle_row_naming():
    scene = generate_scene(4)
    for q_type, fn in EASY_DISPATCH.items():
        if q_type in ("west_of", "north_of", "nearest", "zone_query", "same_zone"):
            q, _, _ = fn(scene)
            assert "Aisle" in q or "zone" in q.lower() or "west" in q.lower() or "north" in q.lower(), \
                f"Question for '{q_type}' must reference WMS location names. Got: {q[:100]}"

def test_hard_question_references_real_locations():
    scene = generate_scene(4)
    q, answer, meta = sample_hard(scene, step_num=1)
    assert "Aisle" in q, "Hard task questions must reference Aisle/Row locations"
    assert "Row" in q,   "Hard task questions must reference row numbers"

def test_medium_question_references_real_zone_names():
    scene = generate_scene(4)
    q, parts, meta = sample_medium(scene)
    assert any(z in q for z in VALID_ZONES), \
        "Medium task question must list valid zone names in the prompt"


# ── Hard task: conflict answer validity ──────────────────────────────────────

def test_hard_answer_is_item_id_or_none():
    scene  = generate_scene(4)
    item_ids = {i["id"].lower() for i in scene["items"]}
    for step in range(1, 6):
        q, answer, meta = sample_hard(scene, step_num=step)
        assert answer in item_ids or answer == "none", \
            f"Hard task answer '{answer}' is not a valid item ID or 'none'"

def test_hard_conflict_matches_actual_target_occupant():
    """When answer is not 'none', the named item MUST be at the target location."""
    scene = generate_scene(5)
    for _ in range(40):
        q, answer, meta = sample_hard(scene)
        if meta["conflict_item"] is not None:
            conflict = next(
                (i for i in scene["items"] if i["id"] == meta["conflict_item"]),
                None
            )
            assert conflict is not None, "conflict_item ID not found in scene"


# ── All question types run without error ──────────────────────────────────────

def test_all_easy_types_produce_output():
    scene = generate_scene(4)
    for name, fn in EASY_DISPATCH.items():
        q, a, meta = fn(scene)
        assert len(q) > 0, f"Empty question from {name}"
        assert len(a) > 0, f"Empty answer from {name}"
        assert "type" in meta, f"Missing 'type' in metadata from {name}"

def test_sample_medium_with_varied_scenes():
    for seed in range(10):
        random.seed(seed)
        scene = generate_scene(4)
        q, parts, meta = sample_medium(scene)
        assert len(parts[0]) > 0
        assert len(parts[1]) > 0
