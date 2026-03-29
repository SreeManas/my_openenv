"""
Baseline agent for the AI Code Review environment.

A SIMPLE rule-based agent that demonstrates reasonable but imperfect
behavior. The baseline is intentionally limited:

  - It uses keyword matching on the hint text (NOT perfect labels)
  - It uses a fixed confidence of 0.9 (slightly overconfident)
  - It does NOT know about hidden issues until they appear
  - It has NO concept of action ordering priorities
  - It cannot distinguish "edge_case" from other issue types

This produces realistic, non-perfect scores:
  Easy  ≈ 0.85–1.0
  Medium ≈ 0.7–0.85 (misses hidden edge-case, suboptimal confidence)
  Hard   ≈ 0.5–0.7  (doesn't prioritize security, misses resource leak)
"""

import logging
from typing import Dict, Any

from models import Action, ActionType
from environment import CodeReviewEnv
from tasks import list_tasks

logger = logging.getLogger("baseline_agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Decision logic — deliberately imperfect keyword matching
# ──────────────────────────────────────────────────────────────────────────────

# Maps hint-text keywords → action type.
# This is intentionally imperfect: the hints are ambiguous.
HINT_ACTION_MAP = [
    # These keywords will match some hints but miss others
    ("parse", ActionType.FIX_BUG),
    ("control-flow", ActionType.FIX_BUG),
    ("unexpected behavior", ActionType.FIX_BUG),
    ("loop bounds", ActionType.FIX_BUG),
    ("crash", ActionType.FIX_BUG),
    ("downstream", ActionType.FIX_BUG),  # catches the hard logic bug
    ("return type", ActionType.FIX_BUG),
    ("return paths", ActionType.FIX_BUG),
    ("o(n", ActionType.OPTIMIZE_CODE),
    ("hashing", ActionType.OPTIMIZE_CODE),
    ("efficient", ActionType.OPTIMIZE_CODE),
    ("scale", ActionType.OPTIMIZE_CODE),
    ("concise", ActionType.OPTIMIZE_CODE),
    ("sanitiz", ActionType.FLAG_ISSUE),
    ("sensitive operation", ActionType.FLAG_ISSUE),
    # NOTE: "exception" and "resource" keywords for resource_leak are
    # NOT included — the baseline will miss the hidden resource-leak issue
]


def decide_action(hint: str) -> Action:
    """
    Deterministic policy: keyword-match on the hint text.

    The baseline always uses confidence=0.9, which is slightly over-
    confident and will hurt its calibration score when it's wrong.
    """
    hint_lower = hint.lower()
    for keyword, action_type in HINT_ACTION_MAP:
        if keyword in hint_lower:
            return Action(
                action_type=action_type,
                explanation=(
                    f"Baseline rule: keyword '{keyword}' matched "
                    f"→ {action_type.value}"
                ),
                confidence=0.9,  # Fixed, slightly overconfident
            )
    # Fallback: no keyword match → leave as-is (this is often wrong)
    return Action(
        action_type=ActionType.LEAVE_AS_IS,
        explanation="Baseline rule: no keyword matched; leaving as-is.",
        confidence=0.5,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline() -> Dict[str, Any]:
    """
    Run the baseline agent on every registered task and return a summary.
    """
    env = CodeReviewEnv()
    all_results: Dict[str, Any] = {}

    for task_summary in list_tasks():
        task_id = task_summary["task_id"]
        print(f"\n{'═' * 70}")
        print(f"  TASK: {task_id}  (difficulty={task_summary['difficulty']})")
        print(f"{'═' * 70}")

        obs = env.reset(task_id)

        step = 0
        while True:
            step += 1
            action = decide_action(obs.issue_type)
            print(
                f"  Step {step}:\n"
                f"    hint     = {obs.issue_type!r}\n"
                f"    action   = {action.action_type.value}\n"
                f"    conf     = {action.confidence}\n"
                f"    remaining= {obs.remaining_issues}"
            )

            result = env.step(action)
            obs = result.observation

            # Log reward details
            reward_str = f"{result.reward:+.4f}"
            extras = []
            if "sequence_bonus" in result.info:
                extras.append("SEQ_BONUS")
            if "order_violation_penalty" in result.info:
                extras.append(
                    f"ORDER_PENALTY={result.info['order_violation_penalty']}"
                )
            if "repeat_penalty" in result.info:
                extras.append(
                    f"REPEAT_PENALTY={result.info['repeat_penalty']}"
                )
            if "revealed_issues" in result.info:
                extras.append(
                    f"REVEALED={result.info['revealed_issues']}"
                )
            if "efficiency_bonus" in result.info:
                extras.append(
                    f"EFFICIENCY_BONUS={result.info['efficiency_bonus']}"
                )

            extra_str = "  ".join(extras)
            print(f"    reward   = {reward_str}  done={result.done}")
            if extra_str:
                print(f"    flags    = {extra_str}")

            if result.done:
                break

        # Grade the full trajectory
        grade = env.grade()
        state = env.state()

        print(f"\n  ── FINAL GRADE ──")
        print(f"  Score:       {grade['score']:.4f}")
        print(f"  Completion:  {grade['completion']:.4f}")
        print(f"  Efficiency:  {grade['efficiency']:.4f}")
        print(f"  Safety:      {grade['safety']:.4f}")
        print(f"  Sequence:    {grade['sequence']:.4f}")
        print(f"  Calibration: {grade['calibration']:.4f}")
        print(f"  Steps used:  {grade['steps_used']}")
        print(f"  Resolved:    {state.resolved_issues}")
        print(f"  Unresolved:  {state.remaining_issues}")
        print(f"  Total reward: {state.total_reward:.4f}")

        all_results[task_id] = {
            "grade": grade,
            "total_reward": state.total_reward,
            "steps": grade["steps_used"],
            "resolved": state.resolved_issues,
            "unresolved": state.remaining_issues,
        }

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  BASELINE SUMMARY")
    print(f"{'═' * 70}")
    for tid, res in all_results.items():
        print(
            f"  {tid:25s}  score={res['grade']['score']:.4f}  "
            f"reward={res['total_reward']:+.4f}  "
            f"steps={res['steps']}  "
            f"resolved={len(res['resolved'])}/{res['grade']['total_issues']}"
        )
    avg_score = sum(
        r["grade"]["score"] for r in all_results.values()
    ) / len(all_results)
    print(f"\n  Average score: {avg_score:.4f}")

    return all_results


if __name__ == "__main__":
    run_baseline()
