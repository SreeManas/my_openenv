"""
Agent strategies for the AI Code Review environment.

Defines a standardized agent interface and multiple agent implementations
with different strategies:

  - BaselineAgent:    keyword matching, moderate confidence (0.9)
  - AggressiveAgent:  fix-everything approach, high confidence (0.95)
  - SafeAgent:        flag-first approach, cautious confidence (0.6–0.75)

All agents are deterministic and rule-based.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from models import Action, ActionType, Observation
from environment import CodeReviewEnv


# ══════════════════════════════════════════════════════════════════════════════
# Standardized agent interface
# ══════════════════════════════════════════════════════════════════════════════

class Agent(ABC):
    """Abstract base class for all agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the agent strategy."""
        ...

    @abstractmethod
    def decide(self, observation: Observation) -> Action:
        """Choose an action given the current observation."""
        ...

    def run_task(self, env: CodeReviewEnv, task_id: str) -> Dict[str, Any]:
        """
        Run a full episode on a single task.

        Returns a standardized result dict with:
          - agent, task_id, steps, total_reward, score
          - grade breakdown, actions, confidence_scores
          - resolved_issues, unresolved_issues
        """
        obs = env.reset(task_id)
        actions_taken: List[str] = []
        confidence_scores: List[float] = []

        while True:
            action = self.decide(obs)
            actions_taken.append(action.action_type.value)
            confidence_scores.append(action.confidence)

            result = env.step(action)
            obs = result.observation

            if result.done:
                break

        grade = env.grade()
        state = env.state()

        return {
            "agent": self.name,
            "task_id": task_id,
            "steps": grade["steps_used"],
            "total_reward": round(state.total_reward, 4),
            "score": grade["score"],
            "grade": grade,
            "actions": actions_taken,
            "confidence_scores": confidence_scores,
            "resolved_issues": len(state.resolved_issues),
            "total_issues": grade["total_issues"],
            "unresolved_issues": list(state.remaining_issues),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1: Baseline (existing logic, wrapped in standard interface)
# ══════════════════════════════════════════════════════════════════════════════

# Keyword map — same as the original baseline.py
_BASELINE_HINTS = [
    ("parse", ActionType.FIX_BUG),
    ("control-flow", ActionType.FIX_BUG),
    ("unexpected behavior", ActionType.FIX_BUG),
    ("loop bounds", ActionType.FIX_BUG),
    ("crash", ActionType.FIX_BUG),
    ("downstream", ActionType.FIX_BUG),
    ("return type", ActionType.FIX_BUG),
    ("return paths", ActionType.FIX_BUG),
    ("o(n", ActionType.OPTIMIZE_CODE),
    ("hashing", ActionType.OPTIMIZE_CODE),
    ("efficient", ActionType.OPTIMIZE_CODE),
    ("scale", ActionType.OPTIMIZE_CODE),
    ("concise", ActionType.OPTIMIZE_CODE),
    ("sanitiz", ActionType.FLAG_ISSUE),
    ("sensitive operation", ActionType.FLAG_ISSUE),
]


class BaselineAgent(Agent):
    """
    Keyword-matching agent with fixed 0.9 confidence.

    Strengths: handles clear hints well.
    Weaknesses: misses hidden issues, overconfident, no ordering awareness.
    """

    @property
    def name(self) -> str:
        return "baseline"

    def decide(self, obs: Observation) -> Action:
        hint = obs.issue_type.lower()
        for keyword, action_type in _BASELINE_HINTS:
            if keyword in hint:
                return Action(
                    action_type=action_type,
                    explanation=f"Baseline: '{keyword}' → {action_type.value}",
                    confidence=0.9,
                )
        return Action(
            action_type=ActionType.LEAVE_AS_IS,
            explanation="Baseline: no keyword matched.",
            confidence=0.5,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2: Aggressive — fix everything, fast, high confidence
# ══════════════════════════════════════════════════════════════════════════════

_AGGRESSIVE_HINTS = [
    # Aggressively maps EVERYTHING to fix_bug or optimize_code.
    # Rarely flags — prefers direct action over escalation.
    ("parse", ActionType.FIX_BUG),
    ("control-flow", ActionType.FIX_BUG),
    ("unexpected", ActionType.FIX_BUG),
    ("loop", ActionType.FIX_BUG),
    ("crash", ActionType.FIX_BUG),
    ("downstream", ActionType.FIX_BUG),
    ("return", ActionType.FIX_BUG),
    ("exception", ActionType.FIX_BUG),       # catches resource leak as fix
    ("resource", ActionType.FIX_BUG),         # catches resource leak as fix
    ("defensive", ActionType.FIX_BUG),        # catches edge-case hint
    ("none", ActionType.FIX_BUG),             # catches None-input hint
    # NOTE: maps security hints to fix_bug instead of flag_issue — WRONG
    ("sanitiz", ActionType.FIX_BUG),
    ("sensitive", ActionType.FIX_BUG),
    ("o(n", ActionType.OPTIMIZE_CODE),
    ("hashing", ActionType.OPTIMIZE_CODE),
    ("efficient", ActionType.OPTIMIZE_CODE),
    ("scale", ActionType.OPTIMIZE_CODE),
    ("concise", ActionType.OPTIMIZE_CODE),
    ("idiomatic", ActionType.OPTIMIZE_CODE),
]


class AggressiveAgent(Agent):
    """
    Fix-everything agent with high confidence (0.95).

    Strategy: always tries to fix or optimize directly. Never flags.
    Strengths: catches hidden issues, efficient step usage.
    Weaknesses: wrong action type for security (fix instead of flag),
    overconfident on wrong decisions, violates ordering constraints.
    """

    @property
    def name(self) -> str:
        return "aggressive_agent"

    def decide(self, obs: Observation) -> Action:
        hint = obs.issue_type.lower()
        for keyword, action_type in _AGGRESSIVE_HINTS:
            if keyword in hint:
                return Action(
                    action_type=action_type,
                    explanation=f"Aggressive: '{keyword}' → {action_type.value}",
                    confidence=0.95,
                )
        return Action(
            action_type=ActionType.FIX_BUG,  # Even default is aggressive
            explanation="Aggressive: no keyword matched, attempting fix anyway.",
            confidence=0.85,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3: Safe — flag first, cautious, lower confidence
# ══════════════════════════════════════════════════════════════════════════════

_SAFE_HINTS = [
    # Safety-first: flags anything that looks risky before fixing
    ("sanitiz", ActionType.FLAG_ISSUE),
    ("sensitive", ActionType.FLAG_ISSUE),
    ("exception", ActionType.FLAG_ISSUE),       # catches resource leak
    ("resource", ActionType.FLAG_ISSUE),         # catches resource leak
    ("released", ActionType.FLAG_ISSUE),         # catches resource leak hint
    ("parse", ActionType.FIX_BUG),
    ("control-flow", ActionType.FIX_BUG),
    ("unexpected", ActionType.FIX_BUG),
    ("loop", ActionType.FIX_BUG),
    ("crash", ActionType.FIX_BUG),
    ("downstream", ActionType.FIX_BUG),
    ("return", ActionType.FIX_BUG),
    ("defensive", ActionType.FIX_BUG),          # catches edge-case
    ("none", ActionType.FIX_BUG),               # catches None-input
    ("o(n", ActionType.OPTIMIZE_CODE),
    ("hashing", ActionType.OPTIMIZE_CODE),
    ("efficient", ActionType.OPTIMIZE_CODE),
    ("scale", ActionType.OPTIMIZE_CODE),
    ("concise", ActionType.OPTIMIZE_CODE),
    ("idiomatic", ActionType.OPTIMIZE_CODE),
]

# Confidence levels: lower for ambiguous hints, higher for clear ones
_SAFE_CONFIDENCE = {
    ActionType.FLAG_ISSUE: 0.75,      # Cautious flagging
    ActionType.FIX_BUG: 0.7,          # Moderate confidence on fixes
    ActionType.OPTIMIZE_CODE: 0.65,   # Lower on optimization
    ActionType.LEAVE_AS_IS: 0.6,      # Unsure
}


class SafeAgent(Agent):
    """
    Safety-first agent with calibrated confidence (0.6–0.75).

    Strategy: flags risky issues first, uses lower confidence, catches
    hidden issues via broader keyword coverage.
    Strengths: correct action types for security/resource issues, better
    calibration, respects ordering.
    Weaknesses: uses more steps, lower efficiency bonus, may be
    overly cautious on straightforward bugs.
    """

    @property
    def name(self) -> str:
        return "safe_agent"

    def decide(self, obs: Observation) -> Action:
        hint = obs.issue_type.lower()
        for keyword, action_type in _SAFE_HINTS:
            if keyword in hint:
                conf = _SAFE_CONFIDENCE[action_type]
                return Action(
                    action_type=action_type,
                    explanation=f"Safe: '{keyword}' → {action_type.value}",
                    confidence=conf,
                )
        return Action(
            action_type=ActionType.LEAVE_AS_IS,
            explanation="Safe: uncertain, leaving as-is.",
            confidence=0.6,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

ALL_AGENTS = [BaselineAgent, AggressiveAgent, SafeAgent]

def get_all_agents() -> List[Agent]:
    """Instantiate and return all registered agents."""
    return [cls() for cls in ALL_AGENTS]
