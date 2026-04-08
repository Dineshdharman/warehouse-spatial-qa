"""
tests/test_scene_generator.py
Validates that scenes are realistic, correctly structured, and deterministic.
Catches: abstract/toy descriptions, duplicate positions, wrong zone logic.
"""
import os
import random
import pytest
from scene_generator import (
    generate_scene, get_item, _zone_for, _describe,
    AISLES, MAX_ROWS, ITEM_CATALOGUE,
    is_west_of, is_north_of, euclidean_distance,
)


# ── Structure ─────────────────────────────────────────────────────────────────

def test_scene_has_correct_n_items():
    for n in [3, 4, 5, 6]:
        scene = generate_scene(n)
        assert len(scene["items"]) == n, f"Expected {n} items, got {len(scene['items'])}"

def test_scene_rejects_invalid_n():
    with pytest.raises(AssertionError):
        generate_scene(2)
    with pytest.raises(AssertionError):
        generate_scene(7)

def test_items_have_required_fields():
    scene = generate_scene(4)
    required = {"id", "type", "aisle", "row", "zone", "x", "y", "location"}
    for item in scene["items"]:
        missing = required - set(item.keys())
        assert not missing, f"Item missing fields: {missing}"

def test_item_ids_are_sequential_uppercase():
    scene = generate_scene(4)
    ids = [i["id"] for i in scene["items"]]
    assert ids == ["A", "B", "C", "D"]

def test_item_types_are_from_catalogue():
    scene = generate_scene(5)
    for item in scene["items"]:
        assert item["type"] in ITEM_CATALOGUE, \
            f"Unknown item type: {item['type']}"

def test_positions_are_unique():
    for _ in range(30):
        scene = generate_scene(5)
        positions = [(i["aisle"], i["row"]) for i in scene["items"]]
        assert len(positions) == len(set(positions)), \
            "Two items share the same warehouse slot — positions must be unique"

def test_aisle_values_valid():
    scene = generate_scene(4)
    for item in scene["items"]:
        assert item["aisle"] in AISLES, f"Invalid aisle: {item['aisle']}"

def test_row_values_in_range():
    scene = generate_scene(4)
    for item in scene["items"]:
        assert 1 <= item["row"] <= MAX_ROWS, \
            f"Row {item['row']} is outside valid range 1–{MAX_ROWS}"

def test_x_matches_aisle_index():
    scene = generate_scene(4)
    for item in scene["items"]:
        assert item["x"] == AISLES.index(item["aisle"]), \
            "x must equal the aisle index"

def test_y_matches_row():
    scene = generate_scene(4)
    for item in scene["items"]:
        assert item["y"] == item["row"], "y must equal the row number"


# ── Real-world framing ────────────────────────────────────────────────────────

def test_description_contains_aisle_row_naming():
    scene = generate_scene(4)
    desc  = scene["description"]
    assert "Aisle" in desc, "Description must use real WMS aisle naming"
    assert "Row"   in desc, "Description must use real WMS row naming"

def test_description_contains_wms_keywords():
    """Description must read like WMS output, not a game."""
    scene = generate_scene(4)
    desc  = scene["description"]
    wms_keywords = ["RECEIVING", "SHIPPING", "BULK_STORAGE", "zone"]
    assert any(kw in desc for kw in wms_keywords), \
        f"Description lacks WMS zone vocabulary. Got: {desc[:200]}"

def test_description_does_not_expose_raw_coordinates():
    """Raw (x, y) tuples must NOT appear in description — use named locations."""
    scene = generate_scene(4)
    desc  = scene["description"]
    import re
    raw_coord_pattern = r"\(\d+,\s*\d+\)"
    assert not re.search(raw_coord_pattern, desc), \
        "Description must not expose raw (x, y) coordinates to the agent"

def test_location_string_format():
    scene = generate_scene(4)
    for item in scene["items"]:
        loc = item["location"]
        assert "Aisle" in loc and "Row" in loc, \
            f"Location string '{loc}' must be in format 'Aisle X, Row N'"

def test_description_length_reasonable():
    """Description must be long enough to be informative, not a one-liner."""
    scene = generate_scene(4)
    assert len(scene["description"]) >= 200, \
        "Description too short — must include layout reference for agents with no prior knowledge"


# ── Zone logic correctness ────────────────────────────────────────────────────

def test_zone_receiving_rows_1_2():
    for aisle in AISLES:
        for row in [1, 2]:
            assert _zone_for(aisle, row) == "RECEIVING", \
                f"Rows 1–2 must always be RECEIVING zone, got {_zone_for(aisle, row)}"

def test_zone_shipping_rows_9_10():
    for aisle in AISLES:
        for row in [9, 10]:
            assert _zone_for(aisle, row) == "SHIPPING"

def test_zone_bulk_storage_west_aisles_AB():
    for aisle in ["A", "B"]:
        for row in range(3, 9):
            assert _zone_for(aisle, row) == "BULK_STORAGE_WEST"

def test_zone_bulk_storage_east_aisles_CD():
    for aisle in ["C", "D"]:
        for row in range(3, 9):
            assert _zone_for(aisle, row) == "BULK_STORAGE_EAST"


# ── Spatial helpers ───────────────────────────────────────────────────────────

def test_is_west_of():
    scene = generate_scene(4)
    # Find items in different aisles; skip test if scene lacks them
    a_item = next((i for i in scene["items"] if i["aisle"] == "A"), None)
    d_item = next((i for i in scene["items"] if i["aisle"] == "D"), None)
    if a_item and d_item:
        assert is_west_of(a_item, d_item) is True

def test_euclidean_distance_same_item():
    scene = generate_scene(3)
    item  = scene["items"][0]
    assert euclidean_distance(item, item) == 0.0


# ── Determinism ───────────────────────────────────────────────────────────────

def test_same_seed_produces_same_scene():
    """Critical: same ENV_SEED → same scene for reproducible baselines."""
    random.seed(42)
    scene_a = generate_scene(4)

    random.seed(42)
    scene_b = generate_scene(4)

    for ia, ib in zip(scene_a["items"], scene_b["items"]):
        assert ia["aisle"] == ib["aisle"]
        assert ia["row"]   == ib["row"]
        assert ia["type"]  == ib["type"]

def test_different_seeds_usually_produce_different_scenes():
    random.seed(1)
    scene_a = generate_scene(4)
    random.seed(9999)
    scene_b = generate_scene(4)
    positions_a = [(i["aisle"], i["row"]) for i in scene_a["items"]]
    positions_b = [(i["aisle"], i["row"]) for i in scene_b["items"]]
    assert positions_a != positions_b, \
        "Different seeds should produce different scenes"


# ── get_item ──────────────────────────────────────────────────────────────────

def test_get_item_returns_correct_item():
    scene = generate_scene(3)
    for item in scene["items"]:
        found = get_item(scene, item["id"])
        assert found["id"] == item["id"]

def test_get_item_raises_on_missing():
    scene = generate_scene(3)
    with pytest.raises(KeyError):
        get_item(scene, "Z")
