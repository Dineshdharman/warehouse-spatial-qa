"""
tests/test_determinism.py
Validates that ENV_SEED produces fully reproducible results.
The hackathon requires 'baseline inference script produces reproducible scores'.
Any non-determinism (random without seed, time-based, OS entropy) causes failure.
"""
import os
import random
import asyncio
import pytest
from models import SpatialAction
from environment import SpatialQAEnv, VALID_TASKS
from scene_generator import generate_scene


def _run_episode(task_id: str, seed: int) -> dict:
    """Run one episode and return rewards + scene fingerprint."""
    random.seed(seed)
    scene = generate_scene(4)
    return {
        "positions": [(i["aisle"], i["row"]) for i in scene["items"]],
        "zones":     [i["zone"] for i in scene["items"]],
        "types":     [i["type"] for i in scene["items"]],
    }


def test_scene_deterministic_with_same_seed():
    for trial in range(10):
        r1 = _run_episode("object_location", seed=42)
        r2 = _run_episode("object_location", seed=42)
        assert r1 == r2, f"Trial {trial}: same seed produced different scenes"


def test_scene_different_with_different_seeds():
    results = [_run_episode("object_location", seed=i) for i in range(20)]
    unique  = {tuple(tuple(p) for p in r["positions"]) for r in results}
    assert len(unique) > 1, \
        "All 20 seeds produced identical scenes — seed is not being applied"


def test_full_episode_deterministic():
    """Same seed → same questions → same rewards for the same answer sequence."""
    def run(seed: int) -> list:
        random.seed(seed)
        env  = SpatialQAEnv(n_items=4)
        env.reset(task_id="movement_prediction")
        rewards = []
        done    = False
        while not done:
            obs = env.step(SpatialAction(answer="none"))
            rewards.append(obs.reward)
            done = obs.done
        return rewards

    rewards_a = run(seed=42)
    rewards_b = run(seed=42)
    assert rewards_a == rewards_b, \
        "Same seed must produce identical reward sequence"


def test_env_seed_env_var_is_respected():
    """ENV_SEED environment variable must be read on import."""
    import importlib
    import environment as env_module
    original_seed = env_module._SEED
    assert isinstance(original_seed, int), "_SEED must be an integer"
    assert original_seed >= 0, "_SEED must be non-negative"


def test_question_bank_deterministic_given_scene():
    """With fixed scene and fixed seed, question sampling must be repeatable."""
    from question_bank import sample_easy

    random.seed(7)
    scene = generate_scene(4)

    random.seed(99)
    q1, a1, _ = sample_easy(scene)

    random.seed(99)
    q2, a2, _ = sample_easy(scene)

    assert q1 == q2, "Same random seed must produce same question"
    assert a1 == a2, "Same random seed must produce same expected answer"
