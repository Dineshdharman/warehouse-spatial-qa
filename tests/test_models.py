"""
tests/test_models.py
Verifies Pydantic model contracts. If models are wrong, everything downstream breaks.
"""
import pytest
from pydantic import ValidationError
from models import SpatialObservation, SpatialAction, SpatialState


# ── SpatialObservation ────────────────────────────────────────────────────────

def test_observation_requires_all_mandatory_fields():
    with pytest.raises(ValidationError):
        SpatialObservation()   # missing all fields

def test_observation_valid_minimal():
    obs = SpatialObservation(
        scene_description="Warehouse snapshot: [A] Aisle B, Row 4.",
        question="Is A west of B?",
        task_id="object_location",
        step_num=1,
    )
    assert obs.task_id == "object_location"
    assert obs.step_num == 1
    assert obs.hints is None
    assert obs.done is False        # inherited from Observation base
    assert obs.reward is None       # inherited from Observation base

def test_observation_accepts_hints():
    obs = SpatialObservation(
        scene_description="desc", question="q",
        task_id="multi_constraint_query", step_num=2,
        hints="Use format: Part1=<zone>, Part2=<ID>"
    )
    assert obs.hints is not None

def test_observation_scene_description_is_string():
    obs = SpatialObservation(
        scene_description="text", question="q",
        task_id="object_location", step_num=0,
    )
    assert isinstance(obs.scene_description, str)

def test_observation_done_reward_settable():
    obs = SpatialObservation(
        scene_description=".", question="?",
        task_id="object_location", step_num=1,
    )
    obs.done   = True
    obs.reward = 1.0
    assert obs.done is True
    assert obs.reward == 1.0


# ── SpatialAction ─────────────────────────────────────────────────────────────

def test_action_requires_answer():
    with pytest.raises(ValidationError):
        SpatialAction()

def test_action_valid():
    assert SpatialAction(answer="yes").answer == "yes"

def test_action_accepts_empty_string():
    assert SpatialAction(answer="").answer == ""


# ── SpatialState ──────────────────────────────────────────────────────────────

def test_state_defaults():
    s = SpatialState()
    assert s.task_id is None
    assert s.step_rewards == []
    assert s.step_count == 0     # inherited from State base

def test_state_with_values():
    s = SpatialState(task_id="object_location", step_count=2, step_rewards=[1.0, 0.5])
    assert s.task_id == "object_location"
    assert s.step_count == 2
    assert s.step_rewards == [1.0, 0.5]
