"""
Core environment for the AI Code Review OpenEnv.

Implements the OpenEnv API:
  - reset(task_id) → Observation
  - step(action)   → StepResult  (observation, reward, done, info)
  - state()        → EnvironmentState

Design features (v2 — upgraded):
  - Multi-step episodes with dynamic code evolution
  - Confidence-modulated rewards (calibration)
  - Order-constraint penalties (hard task: security-first gate)
  - Hidden-issue reveal (issues appear after initial steps)
  - Repeated-action and over-action penalties
  - Ambiguous hints in observations (NOT exact issue types)
"""

import copy
import logging
from typing import Dict, Any, List, Optional, Set

from models import (
    Action,
    ActionType,
    Observation,
    StepResult,
    EnvironmentState,
)
from tasks import get_task, TASK_REGISTRY
from grader import grade_trajectory
from noise import inject_noise

logger = logging.getLogger("code_review_env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


class CodeReviewEnv:
    """Multi-step code-review environment following the OpenEnv spec."""

    def __init__(self) -> None:
        self._task: Optional[Dict[str, Any]] = None
        self._resolved: List[str] = []
        self._remaining: List[Dict[str, Any]] = []
        self._visible: List[Dict[str, Any]] = []   # Issues the agent can see
        self._hidden: List[Dict[str, Any]] = []     # Issues not yet revealed
        self._step_count: int = 0
        self._max_steps: int = 0
        self._total_reward: float = 0.0
        self._done: bool = True
        self._action_history: List[Dict[str, Any]] = []
        self._expected_idx: int = 0  # pointer into expected_sequence
        self._past_action_types: List[str] = []  # for repeat detection
        self._last_action_wrong: bool = False  # for observation feedback hint

    # ──────────────────────────────────────────────────────────────────────
    # reset
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, task_id: str) -> Observation:
        """Load a task and return the initial observation."""
        self._task = copy.deepcopy(get_task(task_id))
        self._resolved = []

        # Separate visible vs hidden issues
        all_issues = list(self._task["issues"])
        self._visible = [i for i in all_issues if not i.get("hidden", False)]
        self._hidden = [i for i in all_issues if i.get("hidden", False)]
        self._remaining = list(self._visible)  # Agent starts seeing only visible

        self._step_count = 0
        self._max_steps = self._task["max_steps"]
        self._total_reward = 0.0
        self._done = False
        self._action_history = []
        self._expected_idx = 0
        self._past_action_types = []
        self._last_action_wrong = False

        logger.info(
            "Environment reset — task=%s  visible_issues=%d  "
            "hidden_issues=%d  max_steps=%d",
            task_id,
            len(self._visible),
            len(self._hidden),
            self._max_steps,
        )
        return self._make_observation()

    # ──────────────────────────────────────────────────────────────────────
    # step
    # ──────────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """Process one agent action and return the step result."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._task is None:
            raise RuntimeError("No task loaded. Call reset() first.")

        self._step_count += 1
        reward = 0.0
        info: Dict[str, Any] = {
            "action": action.action_type.value,
            "confidence": action.confidence,
            "step": self._step_count,
        }
        matched_issue: Optional[str] = None

        # ── Check for repeated action ─────────────────────────────────
        repeat_penalty = 0.0
        if (
            self._past_action_types
            and action.action_type.value == self._past_action_types[-1]
            and action.action_type != ActionType.LEAVE_AS_IS
        ):
            repeat_penalty = -0.15
            info["repeat_penalty"] = repeat_penalty
            logger.info(
                "Step %d: REPEAT of %s → penalty %.2f",
                self._step_count,
                action.action_type.value,
                repeat_penalty,
            )
        self._past_action_types.append(action.action_type.value)

        # ── Try to match action to an unresolved issue ────────────────
        best_match = self._find_matching_issue(action)

        if best_match is not None:
            issue = best_match
            matched_issue = issue["id"]

            if action.action_type.value == issue["expected_action"]:
                # ✅ Correct action for this issue
                base_reward = 1.0

                # Severity weighting: critical issues worth more
                severity = issue.get("severity", 1.0)
                base_reward *= severity

                # ── Confidence calibration ────────────────────────
                # Correct + high confidence → bonus
                # Correct + low confidence → reduced reward
                conf = action.confidence
                calibration = 0.5 + 0.5 * conf  # range [0.5, 1.0]
                base_reward *= calibration
                info["calibration_factor"] = round(calibration, 4)

                # ── Sequence bonus ────────────────────────────────
                expected_seq = self._task["expected_sequence"]
                if (
                    self._expected_idx < len(expected_seq)
                    and action.action_type.value
                    == expected_seq[self._expected_idx]
                ):
                    base_reward += 0.5
                    self._expected_idx += 1
                    info["sequence_bonus"] = True

                # ── Order-constraint check ────────────────────────
                order_penalty = self._check_order_constraint(issue["id"])
                if order_penalty < 0:
                    base_reward += order_penalty
                    info["order_violation_penalty"] = order_penalty

                reward = base_reward

                # Resolve the issue
                self._resolved.append(issue["id"])
                self._remaining = [
                    i for i in self._remaining if i["id"] != issue["id"]
                ]
                info["resolved_issue"] = issue["id"]
                self._last_action_wrong = False
                logger.info(
                    "Step %d: CORRECT %s resolved %s  "
                    "severity=%.1f  conf=%.2f  reward=%.2f",
                    self._step_count,
                    action.action_type.value,
                    issue["id"],
                    severity,
                    conf,
                    reward,
                )
            else:
                # ⚠️ Action targets an issue but wrong action type
                # Wrong + high confidence → harsher penalty
                conf = action.confidence
                base_penalty = -0.5
                penalty_factor = 0.5 + 0.5 * conf  # [0.5, 1.0]
                reward = base_penalty * penalty_factor
                self._last_action_wrong = True
                info["wrong_action_for_issue"] = issue["id"]
                info["calibration_factor"] = round(penalty_factor, 4)
                logger.info(
                    "Step %d: WRONG action %s for issue %s  "
                    "conf=%.2f  reward=%.2f",
                    self._step_count,
                    action.action_type.value,
                    issue["id"],
                    conf,
                    reward,
                )
        else:
            # No matching issue found for this action
            if action.action_type == ActionType.LEAVE_AS_IS:
                if self._remaining:
                    reward = -0.1
                    self._last_action_wrong = True
                    info["note"] = "Issues remain but agent chose to leave as-is"
                else:
                    reward = 0.0
                    self._last_action_wrong = False
                    info["note"] = "All issues resolved; leave_as_is is acceptable"
            else:
                # Unnecessary action on no matching issue
                reward = -0.2
                self._last_action_wrong = True
                info["note"] = "Action did not match any remaining issue"
                logger.info(
                    "Step %d: UNMATCHED action %s  reward=%.2f",
                    self._step_count,
                    action.action_type.value,
                    reward,
                )

        # ── Apply repeat penalty ──────────────────────────────────────
        reward += repeat_penalty

        # ── Record history ────────────────────────────────────────────
        self._total_reward += reward
        self._action_history.append(
            {
                "action_type": action.action_type.value,
                "explanation": action.explanation,
                "confidence": action.confidence,
                "step": self._step_count,
                "reward": round(reward, 4),
                "matched_issue": matched_issue,
            }
        )

        # ── Reveal hidden issues after first resolution ───────────────
        self._maybe_reveal_hidden(info)

        # ── Check termination ─────────────────────────────────────────
        if not self._remaining:
            self._done = True
            info["termination_reason"] = "all_issues_resolved"
            # Efficiency bonus for finishing under max steps
            remaining_budget = self._max_steps - self._step_count
            if remaining_budget > 0:
                efficiency_bonus = 0.5 * (remaining_budget / self._max_steps)
                reward += efficiency_bonus
                self._total_reward += efficiency_bonus
                info["efficiency_bonus"] = round(efficiency_bonus, 4)
            logger.info(
                "Episode DONE — all issues resolved in %d steps. "
                "Total reward=%.2f",
                self._step_count,
                self._total_reward,
            )
        elif self._step_count >= self._max_steps:
            self._done = True
            info["termination_reason"] = "max_steps_reached"
            logger.info(
                "Episode DONE — max steps reached. "
                "Unresolved: %s  Total reward=%.2f",
                [i["id"] for i in self._remaining],
                self._total_reward,
            )

        return StepResult(
            observation=self._make_observation(),
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    # ──────────────────────────────────────────────────────────────────────
    # state
    # ──────────────────────────────────────────────────────────────────────

    def state(self) -> EnvironmentState:
        """Return a snapshot of the current environment state."""
        return EnvironmentState(
            current_task_id=self._task["task_id"] if self._task else None,
            step_count=self._step_count,
            max_steps=self._max_steps,
            done=self._done,
            total_reward=round(self._total_reward, 4),
            resolved_issues=list(self._resolved),
            remaining_issues=[i["id"] for i in self._remaining],
            action_history=list(self._action_history),
        )

    # ──────────────────────────────────────────────────────────────────────
    # grade (convenience wrapper)
    # ──────────────────────────────────────────────────────────────────────

    def grade(self) -> Dict[str, Any]:
        """Grade the current (or just-finished) trajectory."""
        if self._task is None:
            raise RuntimeError("No task loaded.")
        return grade_trajectory(
            self._task, self._action_history, self._resolved
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        """
        Build an Observation from current state.

        Uses the ambiguous 'hint' field instead of the raw issue type
        to introduce uncertainty in what the agent perceives.
        """
        assert self._task is not None

        # Determine current code version based on resolved issues
        resolved_key = frozenset(self._resolved)
        code_versions = self._task["code_versions"]
        code = code_versions.get(resolved_key)
        if code is None:
            # Fallback: version with the most matching resolved issues
            best_key = max(
                code_versions.keys(),
                key=lambda k: len(k & resolved_key),
            )
            code = code_versions[best_key]

        # Surface the HINT (ambiguous) for the most relevant remaining issue
        if self._remaining:
            issue = self._remaining[0]
            raw_hint = issue.get("hint", issue["type"])
            # Append deterministic feedback hints based on agent state.
            # These guide the LLM without revealing ground truth.
            if self._last_action_wrong:
                raw_hint = (
                    raw_hint
                    + " — Previous action may have been incorrect or "
                    "incomplete. Re-evaluate assumptions about the issue type."
                )
            # Detect repeated actions: if last 2+ actions are the same type
            if (
                len(self._past_action_types) >= 2
                and self._past_action_types[-1] == self._past_action_types[-2]
                and self._past_action_types[-1] != "leave_as_is"
            ):
                raw_hint = (
                    raw_hint
                    + " — Repeated actions are not yielding progress. "
                    "Try a different approach or action type."
                )
            issue_type, noise_applied = inject_noise(raw_hint, self._step_count)
        else:
            issue_type = "No remaining issues detected."
            noise_applied = False

        return Observation(
            code_snippet=code,
            issue_type=issue_type,
            context=self._task["context"],
            task_id=self._task["task_id"],
            step_number=self._step_count,
            remaining_issues=[i["id"] for i in self._remaining],
            max_steps=self._max_steps,
            noise_applied=noise_applied,
        )

    def _find_matching_issue(
        self, action: Action
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best-matching unresolved issue for the given action.

        Prefers exact expected_action match → then semantic type match.
        """
        if not self._remaining:
            return None

        # First pass: exact match (action matches expected_action)
        for issue in self._remaining:
            if action.action_type.value == issue["expected_action"]:
                return issue

        # Second pass: semantic match based on issue type
        action_to_types = {
            ActionType.FIX_BUG: {
                "syntax_error", "logic_error", "edge_case",
            },
            ActionType.OPTIMIZE_CODE: {
                "performance", "style",
            },
            ActionType.FLAG_ISSUE: {
                "security_vulnerability", "resource_leak",
            },
        }
        target_types = action_to_types.get(action.action_type, set())
        for issue in self._remaining:
            if issue["type"] in target_types:
                return issue

        return None

    def _check_order_constraint(self, resolved_id: str) -> float:
        """
        Check if resolving `resolved_id` violates an order constraint.

        Some tasks require certain issues (e.g., security) to be resolved
        before others. Violations incur a penalty.
        """
        if self._task is None:
            return 0.0

        order_constraints = self._task.get("order_constraints", {})
        penalty = 0.0

        for gate_id, constraint in order_constraints.items():
            must_before = constraint.get("must_before", [])
            if resolved_id in must_before and gate_id not in self._resolved:
                # Agent resolved an issue that SHOULD come after gate_id
                penalty += constraint.get("violation_penalty", -0.3)
                logger.warning(
                    "ORDER VIOLATION: %s resolved before %s — penalty %.2f "
                    "(%s)",
                    resolved_id,
                    gate_id,
                    penalty,
                    constraint.get("reason", ""),
                )

        return penalty

    def _maybe_reveal_hidden(self, info: Dict[str, Any]) -> None:
        """
        Reveal hidden issues after the first issue is resolved.

        This simulates real-world reviews where fixing one problem
        exposes another.
        """
        if not self._hidden or not self._resolved:
            return

        # Reveal all hidden issues once the agent has resolved at least 1
        newly_revealed = []
        for issue in self._hidden:
            self._remaining.append(issue)
            newly_revealed.append(issue["id"])
            logger.info(
                "REVEALED hidden issue: %s (%s)",
                issue["id"],
                issue.get("hint", issue["type"]),
            )

        if newly_revealed:
            info["revealed_issues"] = newly_revealed
            self._hidden = []  # All revealed
