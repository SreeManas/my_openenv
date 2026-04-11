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
    # OPTIMIZE keywords first to prevent false FIX_BUG matches
    ("string assembly", ActionType.OPTIMIZE_CODE),
    ("temporary object", ActionType.OPTIMIZE_CODE),
    ("concatenat", ActionType.OPTIMIZE_CODE),
    ("sort", ActionType.OPTIMIZE_CODE),
    ("lazy", ActionType.OPTIMIZE_CODE),
    ("intermediate", ActionType.OPTIMIZE_CODE),
    ("o(n", ActionType.OPTIMIZE_CODE),
    ("hashing", ActionType.OPTIMIZE_CODE),
    ("efficient", ActionType.OPTIMIZE_CODE),
    ("scale", ActionType.OPTIMIZE_CODE),
    ("concise", ActionType.OPTIMIZE_CODE),
    ("scales poorly", ActionType.OPTIMIZE_CODE),
    # FIX_BUG keywords
    ("parse", ActionType.FIX_BUG),
    ("control-flow", ActionType.FIX_BUG),
    ("unexpected behavior", ActionType.FIX_BUG),
    ("logic defect", ActionType.FIX_BUG),
    ("incorrect loop", ActionType.FIX_BUG),
    ("loop bounds", ActionType.FIX_BUG),
    ("well-formed", ActionType.FIX_BUG),
    ("malformed", ActionType.FIX_BUG),
    ("empty", ActionType.FIX_BUG),
    ("crash", ActionType.FIX_BUG),
    ("downstream", ActionType.FIX_BUG),
    ("return type", ActionType.FIX_BUG),
    ("return paths", ActionType.FIX_BUG),
    ("credential", ActionType.FIX_BUG),
    ("side effect", ActionType.FIX_BUG),
    ("original list", ActionType.FIX_BUG),
    ("caller", ActionType.FIX_BUG),
    ("calculation", ActionType.FIX_BUG),
    ("boundary", ActionType.FIX_BUG),
    ("payload", ActionType.FIX_BUG),
    ("missing", ActionType.FIX_BUG),
    ("dict key", ActionType.FIX_BUG),
    ("cast", ActionType.FIX_BUG),
    ("age field", ActionType.FIX_BUG),
    ("default parameter", ActionType.FIX_BUG),
    ("shared", ActionType.FIX_BUG),
    ("leaking", ActionType.FIX_BUG),
    ("counter", ActionType.FIX_BUG),
    ("contamination", ActionType.FIX_BUG),
    ("batch", ActionType.FIX_BUG),
    ("unbound", ActionType.FIX_BUG),
    ("undefined", ActionType.FIX_BUG),
    # FLAG_ISSUE keywords
    ("sanitiz", ActionType.FLAG_ISSUE),
    ("sensitive operation", ActionType.FLAG_ISSUE),
    ("predictable", ActionType.FLAG_ISSUE),
    ("security concern", ActionType.FLAG_ISSUE),
    ("email", ActionType.FLAG_ISSUE),
    ("contact", ActionType.FLAG_ISSUE),
    ("verification", ActionType.FLAG_ISSUE),
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
    ("logic defect", ActionType.FIX_BUG),
    ("incorrect loop", ActionType.FIX_BUG),
    ("loop", ActionType.FIX_BUG),
    ("crash", ActionType.FIX_BUG),
    ("downstream", ActionType.FIX_BUG),
    ("return", ActionType.FIX_BUG),
    ("exception", ActionType.FIX_BUG),       # catches resource leak as fix
    ("resource", ActionType.FIX_BUG),         # catches resource leak as fix
    ("defensive", ActionType.FIX_BUG),        # catches edge-case hint
    ("none", ActionType.FIX_BUG),             # catches None-input hint
    ("credential", ActionType.FIX_BUG),       # catches credential hint as fix
    ("side effect", ActionType.FIX_BUG),
    ("original list", ActionType.FIX_BUG),
    ("caller", ActionType.FIX_BUG),
    ("calculation", ActionType.FIX_BUG),
    ("boundary", ActionType.FIX_BUG),
    ("payload", ActionType.FIX_BUG),
    ("missing", ActionType.FIX_BUG),
    ("cast", ActionType.FIX_BUG),
    ("default parameter", ActionType.FIX_BUG),
    ("shared", ActionType.FIX_BUG),
    ("leaking", ActionType.FIX_BUG),
    ("counter", ActionType.FIX_BUG),
    ("batch", ActionType.FIX_BUG),
    ("unbound", ActionType.FIX_BUG),
    ("undefined", ActionType.FIX_BUG),
    ("email", ActionType.FIX_BUG),            # aggressive: fix, not flag
    ("contact", ActionType.FIX_BUG),
    # NOTE: maps security hints to fix_bug instead of flag_issue — WRONG
    ("sanitiz", ActionType.FIX_BUG),
    ("sensitive", ActionType.FIX_BUG),
    ("predictable", ActionType.FIX_BUG),      # aggressive: fix, not flag
    ("security concern", ActionType.FIX_BUG), # aggressive: fix, not flag
    ("o(n", ActionType.OPTIMIZE_CODE),
    ("hashing", ActionType.OPTIMIZE_CODE),
    ("efficient", ActionType.OPTIMIZE_CODE),
    ("scale", ActionType.OPTIMIZE_CODE),
    ("concise", ActionType.OPTIMIZE_CODE),
    ("idiomatic", ActionType.OPTIMIZE_CODE),
    ("sort", ActionType.OPTIMIZE_CODE),
    ("concatenat", ActionType.OPTIMIZE_CODE),
    ("lazy", ActionType.OPTIMIZE_CODE),
    ("intermediate", ActionType.OPTIMIZE_CODE),
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
    # ── Priority 0: Specific multi-word phrases (highest priority) ──
    ("credential exposure", ActionType.FIX_BUG),   # catches "should be fixed" logging issue
    ("should be fixed", ActionType.FIX_BUG),        # catches explicit fix instructions

    # ── Priority 1: FLAG security/risk issues first ──
    ("sanitiz", ActionType.FLAG_ISSUE),
    ("sensitive", ActionType.FLAG_ISSUE),
    ("predictable", ActionType.FLAG_ISSUE),      # catches token predictability
    ("security concern", ActionType.FLAG_ISSUE),  # catches security flag hint
    ("exception", ActionType.FLAG_ISSUE),       # catches resource leak
    ("resource", ActionType.FLAG_ISSUE),         # catches resource leak
    ("released", ActionType.FLAG_ISSUE),         # catches resource leak hint
    ("email", ActionType.FLAG_ISSUE),            # catches email validation
    ("contact", ActionType.FLAG_ISSUE),          # catches contact info flag
    ("verification", ActionType.FLAG_ISSUE),     # catches verification flag

    # ── Priority 2: OPTIMIZE — specific perf keywords BEFORE generic ones ──
    # These must come before "loop" to prevent false FIX_BUG matches
    ("string assembly", ActionType.OPTIMIZE_CODE),
    ("temporary object", ActionType.OPTIMIZE_CODE),
    ("concatenat", ActionType.OPTIMIZE_CODE),    # catches string concat
    ("sort", ActionType.OPTIMIZE_CODE),          # catches sort optimization
    ("lazy", ActionType.OPTIMIZE_CODE),          # catches lazy eval
    ("intermediate", ActionType.OPTIMIZE_CODE),  # catches intermediate lists
    ("o(n", ActionType.OPTIMIZE_CODE),
    ("hashing", ActionType.OPTIMIZE_CODE),
    ("efficient", ActionType.OPTIMIZE_CODE),
    ("scale", ActionType.OPTIMIZE_CODE),
    ("concise", ActionType.OPTIMIZE_CODE),
    ("idiomatic", ActionType.OPTIMIZE_CODE),
    ("scales poorly", ActionType.OPTIMIZE_CODE),

    # ── Priority 3: FIX bugs — specific then generic ──
    ("parse", ActionType.FIX_BUG),
    ("control-flow", ActionType.FIX_BUG),
    ("unexpected", ActionType.FIX_BUG),
    ("logic defect", ActionType.FIX_BUG),        # catches improved logic hint
    ("incorrect loop", ActionType.FIX_BUG),      # catches improved loop hint
    ("well-formed", ActionType.FIX_BUG),         # catches validation hints
    ("malformed", ActionType.FIX_BUG),           # catches malformed input
    ("empty", ActionType.FIX_BUG),               # catches empty input edge cases
    ("loop", ActionType.FIX_BUG),
    ("crash", ActionType.FIX_BUG),
    ("downstream", ActionType.FIX_BUG),
    ("return", ActionType.FIX_BUG),
    ("credential", ActionType.FIX_BUG),          # catches credential exposure
    ("defensive", ActionType.FIX_BUG),          # catches edge-case
    ("none", ActionType.FIX_BUG),               # catches None-input
    ("side effect", ActionType.FIX_BUG),        # catches mutation bugs
    ("original list", ActionType.FIX_BUG),      # catches mutation bugs
    ("caller", ActionType.FIX_BUG),             # catches side-effect on caller
    ("calculation", ActionType.FIX_BUG),        # catches computation bugs
    ("boundary", ActionType.FIX_BUG),           # catches edge-case
    ("averaging", ActionType.FIX_BUG),          # catches averaging bugs
    ("payload", ActionType.FIX_BUG),            # catches validation bugs
    ("missing", ActionType.FIX_BUG),            # catches missing field bugs
    ("dict key", ActionType.FIX_BUG),           # catches key access bugs
    ("cast", ActionType.FIX_BUG),               # catches type cast bugs
    ("age field", ActionType.FIX_BUG),          # catches age validation
    ("default parameter", ActionType.FIX_BUG),  # catches mutable default
    ("shared", ActionType.FIX_BUG),             # catches shared state
    ("leaking", ActionType.FIX_BUG),            # catches state leaking
    ("counter", ActionType.FIX_BUG),            # catches counter bugs
    ("contamination", ActionType.FIX_BUG),      # catches data contamination
    ("batch", ActionType.FIX_BUG),              # catches batch processing bugs
    ("unbound", ActionType.FIX_BUG),            # catches unbound vars
    ("undefined", ActionType.FIX_BUG),          # catches undefined vars
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
# Agent 4: Adaptive — reward-based learning with anti-repetition
# ══════════════════════════════════════════════════════════════════════════════

# Task-based strategy biases: prioritized action types per task category
_TASK_STRATEGY = {
    "security": [ActionType.FLAG_ISSUE, ActionType.FIX_BUG],
    "performance": [ActionType.OPTIMIZE_CODE, ActionType.FIX_BUG],
    "syntax": [ActionType.FIX_BUG, ActionType.OPTIMIZE_CODE],
    "validation": [ActionType.FIX_BUG, ActionType.FLAG_ISSUE],
    "concurrency": [ActionType.FIX_BUG, ActionType.FLAG_ISSUE],
    "edge": [ActionType.FIX_BUG, ActionType.FLAG_ISSUE],
}

# Maps hint keywords to task categories for strategy selection
_TASK_CATEGORY_HINTS = [
    ("sanitiz", "security"), ("sensitive", "security"), ("predictable", "security"),
    ("security", "security"), ("credential", "security"),
    ("o(n", "performance"), ("efficient", "performance"), ("scale", "performance"),
    ("sort", "performance"), ("concatenat", "performance"), ("lazy", "performance"),
    ("parse", "syntax"), ("control-flow", "syntax"), ("well-formed", "syntax"),
    ("malformed", "syntax"),
    ("email", "validation"), ("cast", "validation"), ("payload", "validation"),
    ("boundary", "edge"), ("none", "edge"), ("crash", "edge"), ("empty", "edge"),
    ("mutable", "concurrency"), ("shared", "concurrency"), ("counter", "concurrency"),
    ("leaking", "concurrency"), ("batch", "concurrency"),
]


class AdaptiveAgent(Agent):
    """
    Reward-adaptive agent with RL-like behavior.

    Implements:
      - Anti-repetition: forces action switch after consecutive failure
      - Reward-based avoidance: avoids action types that produced negative reward
      - Task-based strategy: biases action selection by inferred task category
      - Calibrated confidence: adjusts based on past success rate

    This agent maintains a step-by-step memory and adapts its behavior
    within each episode — the core observe → act → reward → adapt loop.
    """

    def __init__(self) -> None:
        self._avoid_actions: set = set()
        self._last_action: str = ""
        self._last_reward: float = 0.0
        self._step_history: List[Dict[str, Any]] = []
        self._task_category: str = ""

    @property
    def name(self) -> str:
        return "adaptive_agent"

    def reset_memory(self) -> None:
        """Clear episode memory for a new task."""
        self._avoid_actions = set()
        self._last_action = ""
        self._last_reward = 0.0
        self._step_history = []
        self._task_category = ""

    def update(self, action_type: str, reward: float) -> None:
        """Update internal state after receiving reward (the 'learn' step)."""
        self._step_history.append({"action": action_type, "reward": reward})
        self._last_action = action_type
        self._last_reward = reward

        # If negative reward → avoid this action type in future steps
        if reward < 0:
            self._avoid_actions.add(action_type)

        # Safety valve: if all non-leave actions are blocked, reset
        non_leave = {at.value for at in ActionType if at != ActionType.LEAVE_AS_IS}
        if self._avoid_actions >= non_leave:
            self._avoid_actions.clear()

    def _infer_category(self, hint: str) -> str:
        """Infer task category from the observation hint."""
        hint_lower = hint.lower()
        for keyword, category in _TASK_CATEGORY_HINTS:
            if keyword in hint_lower:
                return category
        return ""

    def _get_preferred_actions(self) -> List[ActionType]:
        """Return action types ordered by task-based preference, filtered by avoidance."""
        if self._task_category and self._task_category in _TASK_STRATEGY:
            preferred = list(_TASK_STRATEGY[self._task_category])
        else:
            preferred = [ActionType.FIX_BUG, ActionType.FLAG_ISSUE, ActionType.OPTIMIZE_CODE]

        # All action types as fallback
        all_types = [ActionType.FIX_BUG, ActionType.FLAG_ISSUE,
                     ActionType.OPTIMIZE_CODE, ActionType.LEAVE_AS_IS]

        # Filter out avoided actions (but keep alternatives)
        filtered = [a for a in preferred if a.value not in self._avoid_actions]
        if not filtered:
            # Fallback: include all non-avoided actions
            filtered = [a for a in all_types if a.value not in self._avoid_actions]
        if not filtered:
            # Ultimate fallback: use all actions
            filtered = all_types

        return filtered

    def _compute_confidence(self) -> float:
        """Compute confidence based on recent success rate."""
        if not self._step_history:
            return 0.75  # Initial moderate confidence

        recent = self._step_history[-3:]  # Last 3 steps
        positive = sum(1 for s in recent if s["reward"] > 0)
        rate = positive / len(recent)

        # Map success rate to confidence: [0.55, 0.85]
        return round(0.55 + 0.30 * rate, 2)

    def decide(self, obs: Observation) -> Action:
        # Infer task category on first step
        if not self._task_category:
            self._task_category = self._infer_category(obs.issue_type)

        # Anti-repetition: if same action repeated AND last reward negative → force switch
        hint = obs.issue_type.lower()
        preferred = self._get_preferred_actions()

        # If repeating a failed action, remove it from consideration
        if (self._last_action and self._last_reward < 0
                and preferred[0].value == self._last_action):
            alternatives = [a for a in preferred if a.value != self._last_action]
            if alternatives:
                preferred = alternatives

        # Use Safe agent's keyword matching as the base intelligence
        chosen_type = None
        for keyword, action_type in _SAFE_HINTS:
            if keyword in hint:
                if action_type in preferred:
                    chosen_type = action_type
                    break
                # If matched keyword's action is avoided, use first preferred
                chosen_type = preferred[0]
                break

        if chosen_type is None:
            chosen_type = preferred[0] if preferred else ActionType.LEAVE_AS_IS

        confidence = self._compute_confidence()

        # Build contextual explanation
        explanation = self._build_explanation(chosen_type, hint)

        return Action(
            action_type=chosen_type,
            explanation=explanation,
            confidence=confidence,
        )

    def _build_explanation(self, action_type: ActionType, hint: str) -> str:
        """Generate specific explanation tied to the action and context."""
        explanations = {
            ActionType.FIX_BUG: {
                "loop": "Fixing incorrect loop bounds causing off-by-one or redundant iterations",
                "parse": "Fixing parsing logic that fails on malformed input structures",
                "crash": "Fixing unhandled edge case that causes runtime crash",
                "return": "Fixing incorrect return path that produces wrong output type",
                "cast": "Fixing unsafe type cast that fails on non-numeric input",
                "unbound": "Fixing unbound variable reference on incomplete code path",
                "default": "Fixing identified logic defect based on observation analysis",
            },
            ActionType.FLAG_ISSUE: {
                "sensitive": "Flagging sensitive operation without proper input validation",
                "sanitiz": "Flagging unsanitized input used in privileged operation",
                "predictable": "Flagging predictable token generation as security vulnerability",
                "email": "Flagging email/contact field requiring validation before use",
                "resource": "Flagging resource that may leak without proper cleanup",
                "default": "Flagging potential security or resource concern for review",
            },
            ActionType.OPTIMIZE_CODE: {
                "o(n": "Optimizing quadratic algorithm with hash-based lookup for O(n)",
                "concatenat": "Replacing string concatenation loop with join for efficiency",
                "sort": "Optimizing sort by using more efficient comparison strategy",
                "scale": "Refactoring to improve scalability under high load",
                "default": "Applying performance optimization based on detected inefficiency",
            },
            ActionType.LEAVE_AS_IS: {
                "default": "No further issues identified — all detected problems addressed",
            },
        }

        type_map = explanations.get(action_type, {"default": "Taking action based on analysis"})
        for keyword, explanation in type_map.items():
            if keyword != "default" and keyword in hint:
                return f"Adaptive: {explanation}"
        return f"Adaptive: {type_map['default']}"

    def run_task(self, env: CodeReviewEnv, task_id: str) -> Dict[str, Any]:
        """
        Run a full episode with reward-based adaptation.

        Overrides base run_task to inject the observe→act→reward→adapt loop.
        """
        self.reset_memory()
        obs = env.reset(task_id)
        actions_taken: List[str] = []
        confidence_scores: List[float] = []

        while True:
            action = self.decide(obs)
            actions_taken.append(action.action_type.value)
            confidence_scores.append(action.confidence)

            result = env.step(action)
            obs = result.observation

            # RL adaptation step: update memory with reward signal
            self.update(action.action_type.value, result.reward)

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
# Registry
# ══════════════════════════════════════════════════════════════════════════════

ALL_AGENTS = [BaselineAgent, AggressiveAgent, SafeAgent, AdaptiveAgent]

def get_all_agents() -> List[Agent]:
    """Instantiate and return all registered agents."""
    return [cls() for cls in ALL_AGENTS]

