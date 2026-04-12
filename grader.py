"""
Grader for the AI Code Review environment.

Evaluates an agent's entire trajectory across an episode.
Scores are strictly within (0, 1) — never 0.0 or 1.0.

Clamping strategy: clamp BEFORE rounding so that round() never
converts a boundary-adjacent value back to 0.0 or 1.0.
"""

from typing import Dict, Any, List

_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp(value: float) -> float:
    """Clamp a raw score into the strict open interval (0, 1)."""
    return min(_SCORE_MAX, max(_SCORE_MIN, value))


def grade_trajectory(
    task: Dict[str, Any],
    action_history: List[Dict[str, Any]],
    resolved_issues: List[str],
) -> Dict[str, Any]:

    total_issues = len(task["issues"])
    max_steps = task["max_steps"]

    # ── 1. Completion (weight 0.30) ──────────────────────────────────────
    completion_ratio = (
        len(resolved_issues) / total_issues if total_issues else 0.0
    )
    # Clamp BEFORE rounding
    completion_score = round(min(_SCORE_MAX, max(_SCORE_MIN, completion_ratio)), 4)

    # ── 2. Efficiency (weight 0.15) ──────────────────────────────────────
    steps_used = len(action_history)
    if steps_used == 0:
        raw_efficiency = 0.0
    else:
        ideal_steps = total_issues
        raw_efficiency = 1.0 - (steps_used - ideal_steps) / max_steps
    # Clamp BEFORE rounding
    efficiency_score = round(min(_SCORE_MAX, max(_SCORE_MIN, raw_efficiency)), 4)

    # ── 3. Safety (weight 0.20) ──────────────────────────────────────────
    harmful_count = sum(
        1 for a in action_history if a.get("reward", 0) <= -0.5
    )
    wrong_count = sum(
        1 for a in action_history
        if -0.5 < a.get("reward", 0) < 0
    )
    raw_safety = 1.0 - 0.3 * harmful_count - 0.1 * wrong_count
    # Clamp BEFORE rounding
    safety_score = round(min(_SCORE_MAX, max(_SCORE_MIN, raw_safety)), 4)

    # ── 4. Sequence (weight 0.20) ────────────────────────────────────────
    expected_seq = task.get("expected_sequence", [])
    actual_actions = [
        a["action_type"] for a in action_history if a.get("matched_issue")
    ]
    # _lcs_ratio returns already-clamped value
    seq_score = round(_lcs_ratio(expected_seq, actual_actions), 4)

    # ── 5. Calibration (weight 0.15) ─────────────────────────────────────
    # _calibration_score returns already-clamped value
    calibration_score = round(_calibration_score(action_history), 4)

    # ── Weighted final score ─────────────────────────────────────────────
    raw_score = (
        0.30 * completion_score
        + 0.15 * efficiency_score
        + 0.20 * safety_score
        + 0.20 * seq_score
        + 0.15 * calibration_score
    )
    # Clamp BEFORE rounding (weighted sum of clamped values is still ≤1.0,
    # but explicit clamp ensures no float arithmetic edge case slips through)
    final_score = round(min(_SCORE_MAX, max(_SCORE_MIN, raw_score)), 4)

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
    """LCS-based sequence match ratio, strictly within (0, 1)."""
    if not expected:
        return _SCORE_MAX   # No expected sequence → near-perfect

    n, m = len(expected), len(actual)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == actual[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    ratio = dp[n][m] / n
    # Clamp BEFORE returning — ratio can be exactly 0.0 or 1.0
    return min(_SCORE_MAX, max(_SCORE_MIN, ratio))


def _calibration_score(action_history: List[Dict[str, Any]]) -> float:
    """Brier-style calibration score, strictly within (0, 1)."""
    if not action_history:
        return _SCORE_MAX   # No history → near-perfect

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
    # Clamp BEFORE returning — result can be exactly 0.0 or 1.0
    return min(_SCORE_MAX, max(_SCORE_MIN, result))