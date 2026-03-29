"""
Analysis engine for CodeReviewBench.

Provides three capabilities:
  1. Failure Analysis  — detect WHY an agent fails (missed issues,
     overconfidence, repetition, ordering violations)
  2. Impact Modeling   — map unresolved issues to real-world consequences
  3. Insight Generation — produce natural-language comparative insights

All logic is deterministic and rule-based (no ML/NLP).
"""

from typing import Dict, Any, List, Optional

from tasks import get_task


# ══════════════════════════════════════════════════════════════════════════════
# 1. Failure Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_run(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single agent run and return structured failure diagnostics.

    Args:
        run_data: Output from Agent.run_task() — must include:
            actions, confidence_scores, resolved_issues, total_issues,
            steps, grade, unresolved_issues, task_id

    Returns:
        Dict with failure_modes (list of strings), component scores,
        and a decision_quality summary.
    """
    actions = run_data.get("actions", [])
    confs = run_data.get("confidence_scores", [])
    resolved = run_data.get("resolved_issues", 0)
    total = run_data.get("total_issues", 0)
    grade = run_data.get("grade", {})
    unresolved = run_data.get("unresolved_issues", [])
    task_id = run_data.get("task_id", "")

    failure_modes: List[str] = []

    # ── Missed issues ─────────────────────────────────────────────────────
    if resolved < total:
        missed = total - resolved
        failure_modes.append(
            f"Missed {missed} issue(s): {unresolved}"
        )

        # Check if any missed issues were hidden
        try:
            task = get_task(task_id)
            hidden_ids = {
                i["id"] for i in task["issues"] if i.get("hidden")
            }
            missed_hidden = set(unresolved) & hidden_ids
            if missed_hidden:
                failure_modes.append(
                    "Failed to detect hidden issue(s) revealed mid-episode "
                    f"({', '.join(missed_hidden)}) — indicates poor partial "
                    "observability handling"
                )
        except KeyError:
            pass

    # ── Overconfidence ────────────────────────────────────────────────────
    # High confidence actions that didn't match any issue (likely wrong)
    overconf_count = 0
    for i, (action, conf) in enumerate(zip(actions, confs)):
        if conf > 0.8 and action == "leave_as_is":
            # High confidence leave-as-is when issues remain
            if resolved < total:
                overconf_count += 1
        elif conf > 0.8 and action != "leave_as_is":
            # Check if the action was correct by looking at the grade
            pass  # Handled via calibration score below

    if grade.get("calibration", 1.0) < 0.85:
        avg_conf = sum(confs) / len(confs) if confs else 0
        failure_modes.append(
            f"Overconfident decisions: average confidence {avg_conf:.2f} "
            f"but calibration score only {grade.get('calibration', 0):.2f} — "
            "agent is more confident than warranted"
        )

    # ── Repetition ────────────────────────────────────────────────────────
    repeat_count = 0
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1] and actions[i] != "leave_as_is":
            repeat_count += 1

    if repeat_count > 0:
        failure_modes.append(
            f"Repeated same action type {repeat_count} time(s) consecutively "
            "— inefficient action selection"
        )

    # ── Excessive leave_as_is ─────────────────────────────────────────────
    leave_count = actions.count("leave_as_is")
    if leave_count >= 3 and resolved < total:
        failure_modes.append(
            f"Used leave_as_is {leave_count} times with issues remaining — "
            "wasted step budget on inaction"
        )

    # ── Ordering violations ───────────────────────────────────────────────
    if grade.get("sequence", 1.0) < 0.7:
        failure_modes.append(
            f"Poor action ordering: sequence score {grade.get('sequence', 0):.2f} — "
            "actions were not in the expected priority order"
        )

    # ── Wrong action types ────────────────────────────────────────────────
    if grade.get("safety", 1.0) < 0.7:
        failure_modes.append(
            f"Safety violations: safety score {grade.get('safety', 0):.2f} — "
            "agent took harmful or incorrect actions"
        )

    # ── Compute summary scores ────────────────────────────────────────────
    efficiency_score = grade.get("efficiency", 0.0)
    safety_score = grade.get("safety", 0.0)

    # Decision quality: composite of calibration, sequence, and completion
    decision_quality = (
        0.4 * grade.get("completion", 0)
        + 0.3 * grade.get("sequence", 0)
        + 0.3 * grade.get("calibration", 0)
    )

    if not failure_modes:
        failure_modes.append("No significant failure modes detected")

    return {
        "agent": run_data.get("agent", "unknown"),
        "task_id": task_id,
        "score": grade.get("score", 0),
        "failure_modes": failure_modes,
        "efficiency_score": round(efficiency_score, 4),
        "safety_score": round(safety_score, 4),
        "decision_quality": round(decision_quality, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Impact Modeling — "What if you ignore this?"
# ══════════════════════════════════════════════════════════════════════════════

def generate_impact_report(
    task_id: str, unresolved_issue_ids: List[str]
) -> Dict[str, Any]:
    """
    Generate a real-world impact report for unresolved issues.

    Maps each unresolved issue to its potential real-world consequence
    using the 'impact' field in the task definition.

    Returns:
        Dict with unresolved_issues list and real_world_impacts list.
    """
    if not unresolved_issue_ids:
        return {
            "unresolved_issues": [],
            "real_world_impacts": [],
            "risk_level": "none",
            "summary": "All issues resolved. No outstanding risks.",
        }

    try:
        task = get_task(task_id)
    except KeyError:
        return {
            "unresolved_issues": unresolved_issue_ids,
            "real_world_impacts": ["Unable to assess — unknown task"],
            "risk_level": "unknown",
            "summary": "Task not found.",
        }

    issue_map = {i["id"]: i for i in task["issues"]}
    impacts: List[str] = []
    max_severity = 0.0

    for issue_id in unresolved_issue_ids:
        issue = issue_map.get(issue_id)
        if not issue:
            continue

        impact_text = issue.get("impact", "Unknown impact")
        desc = issue.get("description", issue_id)
        impacts.append(f"{desc} → {impact_text}")
        max_severity = max(max_severity, issue.get("severity", 0))

    # Risk level based on highest unresolved severity
    if max_severity >= 0.9:
        risk_level = "critical"
    elif max_severity >= 0.6:
        risk_level = "high"
    elif max_severity >= 0.3:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "unresolved_issues": unresolved_issue_ids,
        "real_world_impacts": impacts,
        "risk_level": risk_level,
        "summary": (
            f"{len(unresolved_issue_ids)} unresolved issue(s) with "
            f"{risk_level} risk level. "
            f"Highest unresolved severity: {max_severity:.1f}."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Insight Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_insights(comparison_results: Dict[str, Any]) -> List[str]:
    """
    Generate natural-language insights from a multi-agent comparison.

    Analyzes the comparison data and produces 3–6 meaningful,
    deterministic observations about relative agent performance.

    Args:
        comparison_results: Output from run_all_agents() — must include
            'comparison' (per-task) and 'summary' (average scores, best).

    Returns:
        List of insight strings.
    """
    insights: List[str] = []
    summary = comparison_results.get("summary", {})
    comparison = comparison_results.get("comparison", [])
    avg_scores = summary.get("average_scores", {})
    best = summary.get("best_agent", "")

    if not avg_scores:
        return ["Insufficient data for insight generation."]

    # ── Best agent insight ────────────────────────────────────────────────
    if best:
        best_score = avg_scores.get(best, 0)
        insights.append(
            f"{_fmt(best)} achieves the highest average score ({best_score:.3f}), "
            f"demonstrating superior overall strategy."
        )

    # ── Per-agent diagnostics ─────────────────────────────────────────────
    for agent_name, avg in sorted(avg_scores.items(), key=lambda x: -x[1]):
        if agent_name == best:
            continue

        gap = avg_scores.get(best, 0) - avg
        if gap > 0.15:
            insights.append(
                f"{_fmt(agent_name)} trails by {gap:.3f} points — "
                f"significant capability gap exists."
            )

    # ── Task-level analysis ───────────────────────────────────────────────
    for task_data in comparison:
        task_id = task_data.get("task_id", "")
        agents = task_data.get("agents", [])
        if not agents:
            continue

        # Find best and worst for this task
        sorted_agents = sorted(agents, key=lambda a: a["score"], reverse=True)
        task_best = sorted_agents[0]
        task_worst = sorted_agents[-1]

        # Large gap on a single task
        gap = task_best["score"] - task_worst["score"]
        if gap > 0.3:
            insights.append(
                f"On {task_id}, {_fmt(task_best['agent'])} scores "
                f"{task_best['score']:.3f} vs {_fmt(task_worst['agent'])} "
                f"at {task_worst['score']:.3f} — a {gap:.3f}-point gap "
                f"revealing divergent strategies under task pressure."
            )

        # Hidden issue detection
        for agent_data in agents:
            if agent_data["resolved_issues"] < agent_data["total_issues"]:
                missed = (
                    agent_data["total_issues"]
                    - agent_data["resolved_issues"]
                )
                insights.append(
                    f"{_fmt(agent_data['agent'])} missed {missed} issue(s) "
                    f"on {task_id} — struggles with partial observability "
                    f"or hidden issue detection."
                )

    # ── Confidence calibration insight ────────────────────────────────────
    for task_data in comparison:
        agents = task_data.get("agents", [])
        for agent_data in agents:
            cal = agent_data.get("grade", {}).get("calibration", 1.0)
            if cal < 0.8:
                insights.append(
                    f"{_fmt(agent_data['agent'])} shows poor confidence "
                    f"calibration ({cal:.2f}) on {task_data['task_id']} — "
                    f"overconfident on incorrect decisions."
                )

    # ── Efficiency insight ────────────────────────────────────────────────
    for task_data in comparison:
        agents = task_data.get("agents", [])
        for agent_data in agents:
            steps = agent_data.get("steps", 0)
            total = agent_data.get("total_issues", 0)
            if total > 0 and steps > total * 2:
                insights.append(
                    f"{_fmt(agent_data['agent'])} used {steps} steps to "
                    f"address {total} issues on {task_data['task_id']} — "
                    f"significant step waste reduces efficiency score."
                )

    # Deduplicate while preserving order
    seen = set()
    unique_insights = []
    for ins in insights:
        if ins not in seen:
            seen.add(ins)
            unique_insights.append(ins)

    return unique_insights[:8]  # Cap at 8 insights


def _fmt(agent_name: str) -> str:
    """Format agent name for display."""
    return agent_name.replace("_", " ").title()
