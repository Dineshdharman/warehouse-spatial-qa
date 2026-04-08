"""
environment.py — SpatialQAEnv implementing the OpenEnv Environment interface.
Three tasks: object_location (easy), multi_constraint_query (medium),
             movement_prediction (hard).
"""
import os
import random
import numpy as np
from typing import Optional
from openenv.core import Environment
from models import SpatialObservation, SpatialAction, SpatialState
from scene_generator import generate_scene
from question_bank import sample_easy, sample_medium, sample_hard
from grader import score as grade, generate_feedback

_SEED = int(os.environ.get("ENV_SEED", 42))
random.seed(_SEED)
np.random.seed(_SEED)

MAX_STEPS = {
    "object_location":        1,
    "multi_constraint_query": 3,
    "movement_prediction":    5,
}

VALID_TASKS = list(MAX_STEPS.keys())


class SpatialQAEnv(Environment[SpatialAction, SpatialObservation, SpatialState]):
    """
    OpenEnv environment for warehouse spatial reasoning.

    Sync usage (openenv-core compatible):
        obs = env.reset(task_id="object_location")
        obs = env.step(SpatialAction(answer="yes"))

    Async usage (via inherited wrappers):
        obs = await env.reset_async(task_id="object_location")
        obs = await env.step_async(SpatialAction(answer="yes"))
    """

    def __init__(self, n_items: int = 4):
        super().__init__()
        self.n_items           = n_items
        self._task_id          = None
        self._scene            = None
        self._step             = 0
        self._done             = False
        self._current_expected = None
        self._step_rewards     = []

    # ── Core sync API (openenv-core abstract method implementation) ───────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "object_location",
        **kwargs,
    ) -> SpatialObservation:
        """Generate a fresh scene and return the first observation."""
        if task_id not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task_id}'. Valid: {VALID_TASKS}")

        self._task_id      = task_id
        self._scene        = generate_scene(self.n_items)
        self._step         = 0
        self._done         = False
        self._step_rewards = []
        self._current_expected = None

        obs        = self._make_observation()
        obs.done   = False
        obs.reward = None
        return obs

    def step(
        self,
        action: SpatialAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SpatialObservation:
        """Score the agent's answer, advance state, return next observation."""
        if self._task_id is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        reward = 0.0
        info   = {}

        expected = self._current_expected

        if self._task_id == "object_location":
            reward     = grade("object_location", action.answer, expected)
            self._done = True
            info       = {"correct_answer": expected}

        elif self._task_id == "multi_constraint_query":
            reward     = grade("multi_constraint_query", action.answer, expected)
            self._done = self._step >= MAX_STEPS["multi_constraint_query"]
            info       = {"correct_parts": expected}

        elif self._task_id == "movement_prediction":
            reward     = grade("movement_prediction", action.answer, expected)
            self._done = self._step >= MAX_STEPS["movement_prediction"]
            info       = {"correct_answer": expected}

        self._step_rewards.append(reward)

        fb           = generate_feedback(self._task_id, action.answer, expected, reward)
        obs          = self._make_observation()
        obs.done     = self._done
        obs.reward   = float(reward)
        obs.metadata = info
        obs.feedback = fb
        return obs

    def state(self) -> dict:
        """Return full internal state dict for debugging / GET /state."""
        return {
            "task_id":      self._task_id,
            "scene":        self._scene,
            "step":         self._step,
            "done":         self._done,
            "step_rewards": self._step_rewards,
            "expected":     self._current_expected,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_observation(self) -> SpatialObservation:
        """Build the observation the agent sees. Caches expected answer for step()."""
        hint = None
        if self._task_id == "object_location":
            q, expected, _ = sample_easy(self._scene)
            self._current_expected = expected
        elif self._task_id == "multi_constraint_query":
            q, expected, _ = sample_medium(self._scene)
            self._current_expected = expected
            hint = "Answer both parts using the format: Part1=<zone>, Part2=<ID>"
        else:  # movement_prediction
            q, expected, _ = sample_hard(self._scene, step_num=self._step + 1)
            self._current_expected = expected
            hint = "Answer with the item ID that must be relocated, or 'none'."

        return SpatialObservation(
            scene_description=self._scene["description"],
            question=q,
            task_id=self._task_id or "",
            step_num=self._step,
            hints=hint,
        )
