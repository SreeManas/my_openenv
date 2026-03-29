"""
Grader for the AI Code Review environment.

Evaluates an agent's entire trajectory across an episode. Scoring is
deterministic and normalized to [0.0, 1.0].

Components (weights):
  - Completion           (0.30): fraction of issues resolved
  - Efficiency           (0.15): bonus for using fewer steps
  - Safety               (0.20): penalty for harmful / wrong actions
  - Sequence             (0.20): bonus for matching expected action order
  - Confidence Calibration (0.15): how well confidence predicts correctness
"""

from typing import Dict, Any, List


def grade_trajectory(
    task: Dict[str, Any],
    action_history: List[Dict[str, Any]],
    resolved_issues: List[str],
) -> Dict[str, Any]:
    """
    Score an entire episode trajectory.

    Args:
        task: The task definition dict (from tasks.py).
        action_history: List of dicts, each with keys:
            - action_type (str)
            - step (int)
            - reward (float)
            - matched_issue (str | None)
            - confidence (float)
        resolved_issues: List of issue IDs that were resolved.

    Returns:
        Dict with 'score' (float 0-1), component scores, and details.
    """
    total_issues = len(task["issues"])
    max_steps = task["max_steps"]

    # ── 1. Completion (0.30) ──────────────────────────────────────────────
    completion_ratio = (
        len(resolved_issues) / total_issues if total_issues else 0.0
    )
    completion_score = completion_ratio

    # ── 2. Efficiency (0.15) ─────────────────────────────────────────────
    steps_used = len(action_history)
    if steps_used == 0:
        efficiency_score = 0.0
    else:
        ideal_steps = total_issues
        efficiency_score = max(
            0.0, 1.0 - (steps_used - ideal_steps) / max_steps
        )

    # ── 3. Safety (0.20) ─────────────────────────────────────────────────
    harmful_count = sum(
        1 for a in action_history if a.get("reward", 0) <= -0.5
    )
    wrong_count = sum(
        1 for a in action_history
        if a.get("reward", 0) < 0 and a.get("reward", 0) > -0.5
    )
    safety_score = max(0.0, 1.0 - 0.3 * harmful_count - 0.1 * wrong_count)

    # ── 4. Sequence (0.20) ───────────────────────────────────────────────
    expected_seq = task.get("expected_sequence", [])
    actual_actions = [
        a["action_type"] for a in action_history if a.get("matched_issue")
    ]
    seq_score = _lcs_ratio(expected_seq, actual_actions)

    # ── 5. Confidence Calibration (0.15) ─────────────────────────────────
    calibration_score = _calibration_score(action_history)

    # ── Final weighted score ─────────────────────────────────────────────
    final_score = (
        0.30 * completion_score
        + 0.15 * efficiency_score
        + 0.20 * safety_score
        + 0.20 * seq_score
        + 0.15 * calibration_score
    )
    final_score = max(0.0, min(1.0, final_score))

    return {
        "score": round(final_score, 4),
        "completion": round(completion_score, 4),
        "efficiency": round(efficiency_score, 4),
        "safety": round(safety_score, 4),
        "sequence": round(seq_score, 4),
        "calibration": round(calibration_score, 4),
        "steps_used": steps_used,
        "issues_resolved": len(resolved_issues),
        "total_issues": total_issues,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _lcs_ratio(expected: List[str], actual: List[str]) -> float:
    """Longest-common-subsequence length / len(expected). Returns 0-1."""
    if not expected:
        return 1.0
    n, m = len(expected), len(actual)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == actual[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m] / n


def _calibration_score(action_history: List[Dict[str, Any]]) -> float:
    """
    Measure how well the agent's confidence aligns with outcomes.

    For each action:
      - Correct (reward > 0) with confidence c  → error = (1 - c)²
      - Incorrect (reward ≤ 0) with confidence c → error = c²

    Calibration = 1 - mean(errors).  Perfect calibration = 1.0.
    An agent that is always 0.9 confident but sometimes wrong scores < 1.0.

    Returns 0-1.  Returns 1.0 if no actions taken.
    """
    if not action_history:
        return 1.0

    errors = []
    for action in action_history:
        conf = action.get("confidence", 1.0)
        reward = action.get("reward", 0)

        if reward > 0:
            # Correct: ideal confidence = 1.0
            errors.append((1.0 - conf) ** 2)
        else:
            # Incorrect: ideal confidence = 0.0
            errors.append(conf ** 2)

    mean_error = sum(errors) / len(errors)
    return max(0.0, 1.0 - mean_error)
