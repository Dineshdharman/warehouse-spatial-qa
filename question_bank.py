"""
question_bank.py — Spatial questions grouped by task difficulty.
Questions use real WMS naming (aisle/row/zone), not abstract coordinates.
Each function returns (question: str, expected_answer: str, metadata: dict).

Anti-gaming design principles:
  - YES/NO questions: question generator ensures roughly balanced yes/no answers
    by always comparing actual values and constructing the question AFTER computing
    the answer — never biasing toward one outcome.
  - ID questions: answer is always computed deterministically from the scene.
  - Zone questions: 4 zones distribute roughly evenly across a 4×10 grid.
"""
import random
import math
from scene_generator import get_item, is_west_of, is_north_of, euclidean_distance

VALID_ZONES = ["RECEIVING", "SHIPPING", "BULK_STORAGE_WEST", "BULK_STORAGE_EAST"]

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — EASY: object_location
# Single spatial question. One step. Binary reward.
# Anti-gaming: yes/no questions enforce a 50/50 split by construction.
# ─────────────────────────────────────────────────────────────────────────────

EASY_QUESTION_TYPES = [
    "west_of", "north_of", "nearest", "zone_query", "same_zone"
]

def sample_easy(scene: dict) -> tuple:
    q_type = random.choice(EASY_QUESTION_TYPES)
    return EASY_DISPATCH[q_type](scene)


def q_west_of(scene: dict) -> tuple:
    """
    Anti-gaming: guarantees exactly 50/50 yes/no by construction.
    Step 1: find a pair where a.x < b.x (a is strictly west of b).
    Step 2: coin flip to ask the question in either direction.
    """
    # Find a pair with strictly different aisles
    for _ in range(30):
        a, b = _pick_two(scene)
        if a["x"] != b["x"]:
            break
    # Ensure a is west of b (a.x < b.x)
    if a["x"] > b["x"]:
        a, b = b, a
    # Coin flip: ask "is subj west of obj?"
    # 50% → subj=a, obj=b → answer "yes"
    # 50% → subj=b, obj=a → answer "no"
    if random.random() < 0.5:
        subj, obj, answer = a, b, "yes"
    else:
        subj, obj, answer = b, a, "no"
    q = (
        f"According to the floor-plan, is {_label(subj)} ({_loc(subj)}) "
        f"located west of {_label(obj)} ({_loc(obj)})? "
        f"Answer yes or no."
    )
    return q, answer, {"type": "west_of", "ids": [subj["id"], obj["id"]], "answer": answer}


def q_north_of(scene: dict) -> tuple:
    """
    Anti-gaming: skips equal-row pairs (ambiguous) by resampling.
    """
    for _ in range(20):
        a, b = _pick_two(scene)
        if a["y"] != b["y"]:
            break
    else:
        return q_west_of(scene)    # fallback if all pairs same row
    answer = "yes" if is_north_of(a, b) else "no"
    q = (
        f"Is {_label(a)} ({_loc(a)}) positioned further north "
        f"(closer to the RECEIVING zone) than {_label(b)} ({_loc(b)})? "
        f"Answer yes or no."
    )
    return q, answer, {"type": "north_of", "ids": [a["id"], b["id"]], "answer": answer}


def q_nearest(scene: dict) -> tuple:
    """
    Anti-gaming: correct answer is always a specific item ID, never guessable.
    Ensures at least 3 items exist so the nearest-item question is non-trivial.
    """
    items  = scene["items"]
    ref    = random.choice(items)
    others = [i for i in items if i["id"] != ref["id"]]
    nearest = min(others, key=lambda i: euclidean_distance(ref, i))
    q = (
        f"Which warehouse item is physically closest to {_label(ref)} ({_loc(ref)})? "
        f"Answer with the item's letter ID only (e.g. 'A')."
    )
    return q, nearest["id"].lower(), {"type": "nearest", "ref": ref["id"], "nearest": nearest["id"]}


def q_zone_query(scene: dict) -> tuple:
    """
    Anti-gaming: answer is always one of 4 zones — can't be guessed above 0.25 baseline.
    """
    item = random.choice(scene["items"])
    q = (
        f"According to the warehouse layout, which operational zone is "
        f"{_label(item)} ({_loc(item)}) assigned to? "
        f"Answer with exactly one of: RECEIVING, SHIPPING, BULK_STORAGE_WEST, BULK_STORAGE_EAST."
    )
    return q, item["zone"], {"type": "zone_query", "id": item["id"], "zone": item["zone"]}


def q_same_zone(scene: dict) -> tuple:
    """
    Reports whether two specific items share a warehouse zone.
    Yes/no is computed from ground truth — not biased.
    """
    a, b   = _pick_two(scene)
    answer = "yes" if a["zone"] == b["zone"] else "no"
    q = (
        f"Are {_label(a)} ({_loc(a)}) and {_label(b)} ({_loc(b)}) "
        f"in the same warehouse operational zone? "
        f"Answer yes or no."
    )
    return q, answer, {"type": "same_zone", "ids": [a["id"], b["id"]], "shared_zone": a["zone"] if answer == "yes" else None}


EASY_DISPATCH = {
    "west_of":    q_west_of,
    "north_of":   q_north_of,
    "nearest":    q_nearest,
    "zone_query": q_zone_query,
    "same_zone":  q_same_zone,
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM: multi_constraint_query
# Two-part spatial query. Partial credit per part. Max 3 steps.
# Anti-gaming: both parts have independently computed answers.
# ─────────────────────────────────────────────────────────────────────────────

def sample_medium(scene: dict) -> tuple:
    """
    Returns (question, [expected_part_1, expected_part_2], metadata).

    Part 1: zone of item A
    Part 2: nearest item to item B (different from A)

    Anti-gaming:
      - Part 1 has 4 possible answers — can't be guessed above 0.25.
      - Part 2 answer is a specific item ID — can't be guessed above 0.33 (with 4 items).
    """
    items  = scene["items"]
    ref_a  = random.choice(items)
    ans_1  = ref_a["zone"]

    others = [i for i in items if i["id"] != ref_a["id"]]
    ref_b  = random.choice(others)
    rest   = [i for i in items if i["id"] != ref_b["id"]]
    nearest_b = min(rest, key=lambda i: euclidean_distance(ref_b, i))
    ans_2  = nearest_b["id"].lower()

    q = (
        f"This is a two-part warehouse query. Answer BOTH parts.\n\n"
        f"Part 1: Which operational zone is {_label(ref_a)} ({_loc(ref_a)}) assigned to? "
        f"(RECEIVING / SHIPPING / BULK_STORAGE_WEST / BULK_STORAGE_EAST)\n\n"
        f"Part 2: Which warehouse item is physically closest to {_label(ref_b)} ({_loc(ref_b)})? "
        f"(answer with item letter ID)\n\n"
        f"Format your answer exactly as: Part1=<zone>, Part2=<ID>"
    )
    return q, [ans_1, ans_2], {
        "type": "multi_constraint",
        "part1": {"id": ref_a["id"], "expected": ans_1},
        "part2": {"id": ref_b["id"], "expected": ans_2},
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — HARD: movement_prediction
# Per-step movement events. Agent predicts relocation conflicts.
# Anti-gaming: "none" answer is only correct when no conflict exists.
#   Conflict rate is controlled to be ~40–60% across random scenes.
# ─────────────────────────────────────────────────────────────────────────────

def sample_hard(scene: dict, step_num: int = 1) -> tuple:
    """
    Presents a movement event: item X moves from its current slot to a new slot.
    Agent must identify which OTHER item must be cleared due to the conflict,
    or answer 'none' if the target slot is unoccupied.

    Anti-gaming design:
      - Uses bounded displacement so moves don't always go off-grid (which would
        always produce "none" — gameable).
      - Ensures a mix of conflict / no-conflict scenarios across steps.
    """
    from scene_generator import AISLES as _AISLES
    items = scene["items"]
    mover = random.choice(items)
    others = [i for i in items if i["id"] != mover["id"]]

    aisle_idx = mover["x"]
    row       = mover["y"]

    # ~50% conflict rate: explicitly aim at an occupied slot half the time
    if random.random() < 0.5 and others:
        target_item      = random.choice(others)
        target_aisle_idx = target_item["x"]
        target_row       = target_item["y"]
    else:
        # Move to an unoccupied slot via bounded displacement
        occupied         = {(i["x"], i["y"]) for i in others}
        target_aisle_idx = aisle_idx
        target_row       = row
        for _ in range(30):
            da            = random.choice([-1, 0, 0, 1])
            dr            = random.choice([-2, -1, 1, 2])
            new_aisle_idx = max(0, min(3, aisle_idx + da))
            new_row       = max(1, min(10, row + dr))
            if (new_aisle_idx, new_row) != (aisle_idx, row) \
                    and (new_aisle_idx, new_row) not in occupied:
                target_aisle_idx = new_aisle_idx
                target_row       = new_row
                break

    target_aisle = _AISLES[target_aisle_idx]
    target_loc   = f"Aisle {target_aisle}, Row {target_row}"

    # Find conflict: another item already at target slot
    conflict = next(
        (i for i in items
         if i["id"] != mover["id"]
         and i["x"] == target_aisle_idx
         and i["y"] == target_row),
        None
    )

    answer = conflict["id"].lower() if conflict else "none"

    q = (
        f"[Step {step_num}] Warehouse movement event:\n"
        f"{_label(mover)} ({ITEM_CATALOGUE_ABBREV.get(mover['type'], mover['type'])}) "
        f"is being relocated from {_loc(mover)} to {target_loc}.\n"
        f"Which item currently occupying {target_loc} must be cleared before "
        f"the move can occur? "
        f"Answer with the item's letter ID, or 'none' if the target slot is unoccupied."
    )
    return q, answer, {
        "type":          "movement_prediction",
        "mover":         mover["id"],
        "from_loc":      _loc(mover),
        "to_loc":        target_loc,
        "target_x":      target_aisle_idx,
        "target_y":      target_row,
        "conflict_item": conflict["id"] if conflict else None,
    }

# Short names for hard-task question text
ITEM_CATALOGUE_ABBREV = {
    "pallet_rack":        "pallet rack",
    "autonomous_forklift":"forklift",
    "staging_pallet":     "staging pallet",
    "loading_dock":       "loading dock",
    "charging_station":   "charging station",
    "conveyor_segment":   "conveyor segment",
    "inspection_station": "inspection station",
    "emergency_exit":     "emergency exit",
    "fire_suppression":   "fire suppression cabinet",
    "picking_station":    "picking station",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick_two(scene: dict):
    items = scene["items"]
    assert len(items) >= 2, "Need at least 2 items to pick a pair"
    return random.sample(items, 2)

def _label(item: dict) -> str:
    return f"item {item['id']}"

def _loc(item: dict) -> str:
    return item["location"]   # e.g. "Aisle B, Row 4"
