"""
Grader for the AI Code Review environment.

Evaluates an agent's entire trajectory across an episode.
Scores are strictly within (0, 1) — never 0.0 or 1.0.

Strategy: round first, clamp second. The clamp is the final gatekeeper.
"""

from typing import Dict, Any, List

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clamp(value: float) -> float:
    """Clamp into strict open interval (0, 1). Always called AFTER rounding."""
    if value >= _SCORE_MAX:
        return _SCORE_MAX
    if value <= _SCORE_MIN:
        return _SCORE_MIN
    return value


def grade_trajectory(
    task: Dict[str, Any],
    action_history: List[Dict[str, Any]],
    resolved_issues: List[str],
) -> Dict[str, Any]:

    total_issues = len(task["issues"])
    max_steps = task["max_steps"]

    # ── 1. Completion (weight 0.30)
    completion_ratio = (
        len(resolved_issues) / total_issues if total_issues else 0.0
    )
    completion_score = _clamp(round(completion_ratio, 4))

    # ── 2. Efficiency (weight 0.15)
    steps_used = len(action_history)
    if steps_used == 0:
        raw_efficiency = 0.0
    else:
        ideal_steps = total_issues
        raw_efficiency = 1.0 - (steps_used - ideal_steps) / max_steps
    efficiency_score = _clamp(round(raw_efficiency, 4))

    # ── 3. Safety (weight 0.20)
    harmful_count = sum(
        1 for a in action_history if a.get("reward", 0) <= -0.5
    )
    wrong_count = sum(
        1 for a in action_history
        if -0.5 < a.get("reward", 0) < 0
    )
    raw_safety = 1.0 - 0.3 * harmful_count - 0.1 * wrong_count
    safety_score = _clamp(round(raw_safety, 4))

    # ── 4. Sequence (weight 0.20)
    expected_seq = task.get("expected_sequence", [])
    actual_actions = [
        a["action_type"] for a in action_history if a.get("matched_issue")
    ]
    seq_score = _clamp(round(_lcs_ratio(expected_seq, actual_actions), 4))

    # ── 5. Calibration (weight 0.15)
    calibration_score = _clamp(round(_calibration_score(action_history), 4))

    # ── Weighted final score
    raw_score = (
        0.30 * completion_score
        + 0.15 * efficiency_score
        + 0.20 * safety_score
        + 0.20 * seq_score
        + 0.15 * calibration_score
    )
    final_score = _clamp(round(raw_score, 4))

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lcs_ratio(expected: List[str], actual: List[str]) -> float:
    if not expected:
        return _SCORE_MAX

    n, m = len(expected), len(actual)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == actual[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    ratio = dp[n][m] / n
    return _clamp(round(ratio, 4))


def _calibration_score(action_history: List[Dict[str, Any]]) -> float:
    if not action_history:
        return _SCORE_MAX

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
    return _clamp(round(result, 4))