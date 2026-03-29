"""
Pydantic models for the AI Code Review OpenEnv environment.

Defines the structured Action, Observation, StepResult, and EnvironmentState
models used across the entire system.
"""

from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Possible actions the agent can take during code review."""
    FIX_BUG = "fix_bug"
    OPTIMIZE_CODE = "optimize_code"
    FLAG_ISSUE = "flag_issue"
    LEAVE_AS_IS = "leave_as_is"


class Action(BaseModel):
    """Structured action submitted by the agent each step."""
    action_type: ActionType = Field(
        ..., description="The type of action to perform."
    )
    explanation: str = Field(
        default="",
        description="Free-text explanation of why this action was chosen.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this action (0-1).",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees after each step (or after reset)."""
    code_snippet: str = Field(
        ..., description="Current version of the code under review."
    )
    issue_type: str = Field(
        ..., description="Primary issue type currently surfaced."
    )
    context: str = Field(
        ..., description="Contextual information about the code / task."
    )
    task_id: str = Field(
        ..., description="Identifier of the current task."
    )
    step_number: int = Field(
        ..., description="Current step within the episode."
    )
    remaining_issues: List[str] = Field(
        default_factory=list,
        description="IDs of issues still unresolved.",
    )
    max_steps: int = Field(
        ..., description="Maximum steps allowed for this task."
    )
    noise_applied: bool = Field(
        default=False,
        description="Whether deterministic noise was injected into this observation's hint.",
    )


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Returned by environment.step()."""
    observation: Observation
    reward: float = Field(
        ..., description="Reward received for the action taken."
    )
    done: bool = Field(
        ..., description="Whether the episode has ended."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary diagnostic information.",
    )


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Full snapshot of the environment's internal state."""
    current_task_id: Optional[str] = None
    step_count: int = 0
    max_steps: int = 0
    done: bool = True
    total_reward: float = 0.0
    resolved_issues: List[str] = Field(default_factory=list)
    remaining_issues: List[str] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
