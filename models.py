"""
models.py — Typed Pydantic models for the Warehouse Spatial QA environment.
Inherits from openenv-core base classes for proper OpenEnv spec compliance.
"""
from typing import Optional, List
from pydantic import Field
from openenv.core import Action, Observation, State


class SpatialObservation(Observation):
    """What the agent sees each step. Inherits done/reward/metadata from Observation."""
    scene_description: str = Field(
        description="Natural-language description of the warehouse floor plan"
    )
    question: str = Field(
        description="Spatial question the agent must answer"
    )
    task_id: str = Field(
        description="Active task: object_location | multi_constraint_query | movement_prediction"
    )
    step_num: int = Field(
        default=1,
        description="Current step number within the episode (1-indexed)"
    )
    hints: Optional[str] = Field(
        default=None,
        description="Optional contextual hint for hard tasks"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="RL reward feedback from the previous step so the LLM can self-correct"
    )


class SpatialAction(Action):
    """Action the agent submits. Inherits metadata from Action."""
    answer: str = Field(
        description="The agent's free-text answer to the spatial question"
    )


class SpatialState(State):
    """Internal episode state. Inherits episode_id/step_count from State."""
    task_id: Optional[str] = Field(default=None)
    step_rewards: List[float] = Field(default_factory=list)
