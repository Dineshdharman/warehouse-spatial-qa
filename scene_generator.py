"""
scene_generator.py — Generates warehouse floor-plan scenes using real-world
WMS naming conventions (aisles, rows, zone classifications).

Real-world basis:
  - Warehouses are organized into aisles (A, B, C, D) running north-south.
  - Each aisle has numbered rows (Row 1 = northmost, Row 10 = southmost).
  - Zones: RECEIVING (north end), SHIPPING (south end), BULK_STORAGE (center),
           STAGING (east), CHARGING (west corner).
  - Items have functional roles (not abstract labels).
"""
import random
from typing import List, Dict, Optional

# ── Real warehouse naming ─────────────────────────────────────────────────────

AISLES   = ["A", "B", "C", "D"]   # 4 aisles, west (A) to east (D)
MAX_ROWS = 10                       # rows 1–10, north (1) to south (10)

# item_type → human-readable role (used in WMS-style descriptions)
ITEM_CATALOGUE = {
    "pallet_rack":        "a fixed pallet rack holding inventory",
    "autonomous_forklift":"an autonomous forklift vehicle",
    "staging_pallet":     "a staging pallet awaiting shipment",
    "loading_dock":       "a truck loading/unloading dock bay",
    "charging_station":   "a forklift battery charging station",
    "conveyor_segment":   "a segment of the automated conveyor belt",
    "inspection_station": "a quality control inspection station",
    "emergency_exit":     "an emergency exit door",
    "fire_suppression":   "a fire suppression equipment cabinet",
    "picking_station":    "an order-picking workstation",
}

# Warehouse zone boundaries (based on row number)
def _zone_for(aisle: str, row: int) -> str:
    """
    Zone classification matching real warehouse layout:
      Rows 1–2  → RECEIVING
      Rows 9–10 → SHIPPING
      Rows 3–8, Aisles A–B → BULK_STORAGE_WEST
      Rows 3–8, Aisles C–D → BULK_STORAGE_EAST
    """
    if row <= 2:
        return "RECEIVING"
    if row >= 9:
        return "SHIPPING"
    if aisle in ("A", "B"):
        return "BULK_STORAGE_WEST"
    return "BULK_STORAGE_EAST"


# ── Scene generator ──────────────────────────────────────────────────────────

def generate_scene(n_items: int = 4) -> Dict:
    """
    Returns a warehouse scene dict with WMS-style named locations.

    Each item has:
        id      : single uppercase letter (A, B, C, ...)
        type    : item type key from ITEM_CATALOGUE
        aisle   : "A" | "B" | "C" | "D"
        row     : int 1–10
        zone    : warehouse zone name
        x       : int  (aisle index 0–3, used for left/right comparisons)
        y       : int  (row number 1–10, used for north/south comparisons)
        location: human-readable location string (e.g. "Aisle B, Row 4")
    description: full WMS-style paragraph describing the floor plan

    n_items: must be 3–6 to ensure sufficient items for all question types.
    """
    assert 3 <= n_items <= 6, "n_items must be 3–6"

    labels    = [chr(65 + i) for i in range(n_items)]
    used      = set()
    items     = []
    item_keys = list(ITEM_CATALOGUE.keys())

    for label in labels:
        # Guarantee unique (aisle, row) pair — real warehouse: one item per slot
        for _ in range(200):                     # safety limit, not infinite loop
            aisle = random.choice(AISLES)
            row   = random.randint(1, MAX_ROWS)
            key   = (aisle, row)
            if key not in used:
                used.add(key)
                break
        else:
            raise RuntimeError("Could not place all items — scene too crowded.")

        x    = AISLES.index(aisle)               # 0=westmost, 3=eastmost
        zone = _zone_for(aisle, row)
        items.append({
            "id":       label,
            "type":     random.choice(item_keys),
            "aisle":    aisle,
            "row":      row,
            "zone":     zone,
            "x":        x,
            "y":        row,
            "location": f"Aisle {aisle}, Row {row}",
        })

    return {
        "items":        items,
        "description":  _describe(items),
        "layout_dims":  {"aisles": AISLES, "max_row": MAX_ROWS},
    }


def _describe(items: List[Dict]) -> str:
    """
    Produce a WMS-style natural-language floor-plan description.
    Reads like real logistics software output, not a game.
    """
    lines = [
        "Warehouse floor-plan report — automated WMS snapshot:",
        "",
    ]
    for item in items:
        role = ITEM_CATALOGUE[item["type"]]
        lines.append(
            f"  [{item['id']}] {item['location']} ({item['zone']} zone): "
            f"{role}."
        )
    lines.append("")
    lines.append(
        "Layout reference: Aisles A–D run west to east. "
        "Rows 1–10 run north (Row 1) to south (Row 10). "
        "Rows 1–2 = RECEIVING, Rows 9–10 = SHIPPING, "
        "Rows 3–8 Aisles A–B = BULK_STORAGE_WEST, "
        "Rows 3–8 Aisles C–D = BULK_STORAGE_EAST."
    )
    return "\n".join(lines)


def get_item(scene: Dict, item_id: str) -> Dict:
    """Retrieve an item dict by its label ID. Raises KeyError if not found."""
    for item in scene["items"]:
        if item["id"] == item_id:
            return item
    raise KeyError(f"Item '{item_id}' not found in scene.")


def describe_scene(items: List[Dict]) -> str:
    """Public wrapper — regenerate a WMS-style description from the current item list."""
    return _describe(items)


def zone_for(aisle: str, row: int) -> str:
    """Public wrapper — return zone classification for a given aisle/row."""
    return _zone_for(aisle, row)


# ── Spatial helpers (used by question_bank and tests) ────────────────────────

def is_west_of(a: Dict, b: Dict) -> bool:
    """Returns True if item a is in a physically west-of (lower aisle index) position than b."""
    return a["x"] < b["x"]

def is_north_of(a: Dict, b: Dict) -> bool:
    """Returns True if item a has a lower row number (closer to RECEIVING) than b."""
    return a["y"] < b["y"]

def euclidean_distance(a: Dict, b: Dict) -> float:
    import math
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)
