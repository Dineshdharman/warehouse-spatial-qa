"""
tests/test_environment.py
Full contract and integration tests for SpatialQAEnv.
"""
import pytest
from models import SpatialAction
from environment import SpatialQAEnv, VALID_TASKS, MAX_STEPS


# ── reset() contract ──────────────────────────────────────────────────────────

def test_reset_returns_observation_for_all_tasks():
    env = SpatialQAEnv()
    for task in VALID_TASKS:
        obs = env.reset(task_id=task)
        assert hasattr(obs, "scene_description"), "reset() must return SpatialObservation"
        assert hasattr(obs, "done"),              "observation must have .done"

def test_reset_done_is_always_false():
    env = SpatialQAEnv()
    for task in VALID_TASKS:
        obs = env.reset(task_id=task)
        assert obs.done is False, \
            f"reset() returned done=True for task '{task}' — must be False"

def test_reset_observation_has_all_required_fields():
    env = SpatialQAEnv()
    for task in VALID_TASKS:
        obs = env.reset(task_id=task)
        assert obs.scene_description != "", f"scene_description empty for task {task}"
        assert obs.question          != "", f"question empty for task {task}"
        assert obs.task_id           == task
        assert obs.step_num          == 0

def test_reset_scene_description_uses_wms_naming():
    env = SpatialQAEnv()
    obs = env.reset(task_id="object_location")
    assert "Aisle" in obs.scene_description, \
        "scene_description must contain WMS 'Aisle' naming"
    assert "Row" in obs.scene_description, \
        "scene_description must contain WMS 'Row' naming"

def test_reset_raises_on_invalid_task():
    env = SpatialQAEnv()
    with pytest.raises(ValueError, match="Unknown task"):
        env.reset(task_id="fake_task")

def test_reset_clears_previous_episode():
    env = SpatialQAEnv()
    env.reset(task_id="object_location")
    env.step(SpatialAction(answer="yes"))
    env.reset(task_id="object_location")
    state = env.state()
    assert state["step"] == 0,         "reset() must clear step counter to 0"
    assert state["done"] is False,     "reset() must clear done flag"
    assert state["step_rewards"] == [], "reset() must clear reward history"


# ── step() contract ───────────────────────────────────────────────────────────

def test_step_requires_reset_first():
    env = SpatialQAEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(SpatialAction(answer="yes"))

def test_step_returns_required_fields():
    env = SpatialQAEnv()
    env.reset(task_id="object_location")
    obs = env.step(SpatialAction(answer="yes"))
    assert hasattr(obs, "scene_description"), "step() must return SpatialObservation"
    assert hasattr(obs, "reward"),            "observation must have .reward"
    assert hasattr(obs, "done"),              "observation must have .done"
    assert hasattr(obs, "metadata"),          "observation must have .metadata"

def test_step_reward_is_float_in_range():
    env = SpatialQAEnv()
    for task in VALID_TASKS:
        env.reset(task_id=task)
        obs = env.step(SpatialAction(answer="yes"))
        assert isinstance(obs.reward, float), \
            f"reward must be float, got {type(obs.reward)}"
        assert 0.0 <= obs.reward <= 1.0, \
            f"reward {obs.reward} out of range [0,1] for task {task}"

def test_step_done_is_bool():
    env = SpatialQAEnv()
    env.reset(task_id="object_location")
    obs = env.step(SpatialAction(answer="yes"))
    assert isinstance(obs.done, bool), "done must be a bool"

def test_step_info_contains_correct_answer():
    env = SpatialQAEnv()
    env.reset(task_id="object_location")
    obs = env.step(SpatialAction(answer="yes"))
    assert "correct_answer" in obs.metadata, \
        "step() metadata must contain 'correct_answer' for easy task"

def test_step_raises_after_done():
    env = SpatialQAEnv()
    env.reset(task_id="object_location")
    env.step(SpatialAction(answer="yes"))   # done=True after 1 step
    with pytest.raises(RuntimeError, match="done"):
        env.step(SpatialAction(answer="yes"))


# ── Episode boundary: exact step counts ──────────────────────────────────────

def test_easy_task_done_after_exactly_1_step():
    env = SpatialQAEnv()
    env.reset(task_id="object_location")
    obs = env.step(SpatialAction(answer="yes"))
    assert obs.done is True, \
        "object_location must be done=True after exactly 1 step"

def test_medium_task_done_after_max_steps():
    env   = SpatialQAEnv()
    env.reset(task_id="multi_constraint_query")
    max_s = MAX_STEPS["multi_constraint_query"]
    last  = None
    for i in range(max_s):
        last = env.step(SpatialAction(answer="Part1=RECEIVING, Part2=a"))
        if last.done:
            assert i + 1 == max_s, \
                f"Medium task ended at step {i+1}, expected step {max_s}"
            break
    assert last.done is True, "Medium task must be done after max_steps"

def test_hard_task_done_after_max_steps():
    env   = SpatialQAEnv()
    env.reset(task_id="movement_prediction")
    max_s = MAX_STEPS["movement_prediction"]
    last  = None
    for _ in range(max_s):
        last = env.step(SpatialAction(answer="none"))
    assert last.done is True, \
        f"Hard task must be done after {max_s} steps"

@pytest.mark.parametrize("task,max_s", MAX_STEPS.items())
def test_episode_never_exceeds_max_steps(task, max_s):
    env   = SpatialQAEnv()
    env.reset(task_id=task)
    steps = 0
    done  = False
    while not done and steps < max_s + 5:
        obs  = env.step(SpatialAction(answer="none"))
        done = obs.done
        steps += 1
        assert steps <= max_s, \
            f"Task '{task}' is still running at step {steps}, max is {max_s}"
    assert done is True


# ── state() contract ──────────────────────────────────────────────────────────

def test_state_returns_dict_with_required_keys():
    env = SpatialQAEnv()
    env.reset(task_id="object_location")
    state    = env.state()
    required = {"task_id", "scene", "step", "done", "step_rewards", "expected"}
    missing  = required - set(state.keys())
    assert not missing, f"state() dict missing keys: {missing}"

def test_state_task_id_matches_reset():
    env = SpatialQAEnv()
    for task in VALID_TASKS:
        env.reset(task_id=task)
        assert env.state()["task_id"] == task

def test_state_step_increments():
    env = SpatialQAEnv()
    env.reset(task_id="multi_constraint_query")
    assert env.state()["step"] == 0
    env.step(SpatialAction(answer="Part1=RECEIVING, Part2=a"))
    assert env.state()["step"] == 1

def test_state_scene_has_wms_description():
    env = SpatialQAEnv()
    env.reset(task_id="object_location")
    state = env.state()
    assert "description" in state["scene"], "scene must have 'description' key"
    assert "Aisle" in state["scene"]["description"], \
        "scene description must use WMS naming"


# ── close() ───────────────────────────────────────────────────────────────────

def test_close_does_not_raise():
    env = SpatialQAEnv()
    env.reset(task_id="object_location")
    env.close()   # must not raise


# ── Multiple resets ───────────────────────────────────────────────────────────

def test_can_reset_multiple_times_in_sequence():
    env = SpatialQAEnv()
    for task in VALID_TASKS:
        obs = env.reset(task_id=task)
        assert obs.done is False

def test_full_episode_across_all_tasks():
    env = SpatialQAEnv()
    for task in VALID_TASKS:
        env.reset(task_id=task)
        done  = False
        steps = 0
        while not done:
            obs  = env.step(SpatialAction(answer="none"))
            done = obs.done
            steps += 1
            assert 0.0 <= obs.reward <= 1.0
        assert done is True
        assert steps <= MAX_STEPS[task]
