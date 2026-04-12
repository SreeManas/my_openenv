"""
Grader for the AI Code Review environment.

Evaluates an agent's entire trajectory across an episode.
Scores are strictly within (0, 1) — never 0.0 or 1.0.
"""

from typing import Dict, Any, List


def _safe_score(score: float) -> float:
    """Enforce strict (0, 1) AFTER rounding.

    This is the SINGLE gatekeeper for all score values.
    No score leaves the grader without passing through this.
    """
    if score >= 0.999:
        return 0.999
    if score <= 0.001:
        return 0.001
    return score


def grade_trajectory(
    task: Dict[str, Any],
    action_history: List[Dict[str, Any]],
    resolved_issues: List[str],
) -> Dict[str, Any]:

    total_issues = len(task["issues"])
    max_steps = task["max_steps"]

    # ── 1. Completion (0.30)
    completion_ratio = (
        len(resolved_issues) / total_issues if total_issues else 0.001
    )
    completion_score = completion_ratio

    # ── 2. Efficiency (0.15)
    steps_used = len(action_history)
    if steps_used == 0:
        efficiency_score = 0.001
    else:
        ideal_steps = total_issues
        efficiency_score = max(
            0.001, 1.0 - (steps_used - ideal_steps) / max_steps
        )

    # ── 3. Safety (0.20)
    harmful_count = sum(
        1 for a in action_history if a.get("reward", 0) <= -0.5
    )
    wrong_count = sum(
        1 for a in action_history
        if -0.5 < a.get("reward", 0) < 0
    )
    safety_score = max(0.001, 1.0 - 0.3 * harmful_count - 0.1 * wrong_count)

    # ── 4. Sequence (0.20)
    expected_seq = task.get("expected_sequence", [])
    actual_actions = [
        a["action_type"] for a in action_history if a.get("matched_issue")
    ]
    seq_score = _lcs_ratio(expected_seq, actual_actions)

    # ── 5. Calibration (0.15)
    calibration_score = _calibration_score(action_history)

    # ── Final weighted score
    raw_score = (
        0.30 * completion_score
        + 0.15 * efficiency_score
        + 0.20 * safety_score
        + 0.20 * seq_score
        + 0.15 * calibration_score
    )

    # Apply _safe_score AFTER rounding to every single metric
    final_score = _safe_score(round(raw_score, 4))
    completion_score = _safe_score(round(completion_score, 4))
    efficiency_score = _safe_score(round(efficiency_score, 4))
    safety_score = _safe_score(round(safety_score, 4))
    seq_score = _safe_score(round(seq_score, 4))
    calibration_score = _safe_score(round(calibration_score, 4))

    return {
        "score": final_score,
        "completion": completion_score,
        "efficiency": efficiency_score,
        "safety": safety_score,
        "sequence": seq_score,
        "calibration": calibration_score,
        "steps_used": steps_used,
        "issues_resolved": len(resolved_issues),
        "total_issues": total_issues,
    }


# ─────────────────────────────────────────────

def _lcs_ratio(expected: List[str], actual: List[str]) -> float:
    if not expected:
        return 0.999  # No sequence expected → near-perfect

    n, m = len(expected), len(actual)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == actual[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    ratio = dp[n][m] / n
    # Clamp: ratio can be exactly 0.0 (no match) or 1.0 (perfect match)
    return _safe_score(ratio)


def _calibration_score(action_history: List[Dict[str, Any]]) -> float:
    if not action_history:
        return 0.999  # No history → near-perfect calibration

    errors = []
    for action in action_history:
        conf = action.get("confidence", 1.0)
        reward = action.get("reward", 0)

        if reward > 0:
            errors.append((1.0 - conf) ** 2)
        else:
            errors.append(conf ** 2)

    mean_error = sum(errors) / len(errors)
    result = 1.0 - mean_error
    # Clamp: result can be exactly 0.0 (all wrong) or 1.0 (perfect calibration)
    return _safe_score(result)