"""
Multi-agent evaluation runner for CodeReviewBench.

Provides comparative evaluation of multiple agent strategies
on the same tasks using fresh environment instances.

Usage:
    python3 multi_agent.py

All runs are deterministic — identical agents always produce identical results.
"""

from typing import Dict, Any, List, Optional

from environment import CodeReviewEnv
from tasks import list_tasks
from agents import get_all_agents, Agent
from analysis import analyze_run, generate_impact_report, generate_insights


def run_agent_on_task(agent: Agent, task_id: str) -> Dict[str, Any]:
    """
    Run a single agent on a single task with a FRESH environment.

    This ensures no shared state between agents or tasks.
    """
    env = CodeReviewEnv()
    return agent.run_task(env, task_id)


def run_all_agents(
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run all registered agents on one or all tasks and return
    a structured comparison with insights and analysis.

    Args:
        task_id: If provided, run only on this task.
                 If None, run on all tasks.

    Returns:
        Dict with 'comparison', 'summary', 'ranking',
        'insights', and 'analysis'.
    """
    agents = get_all_agents()

    # Determine which tasks to run
    if task_id:
        task_ids = [task_id]
    else:
        task_ids = [t["task_id"] for t in list_tasks()]

    # ── Run all agents on all tasks ───────────────────────────────────────
    comparison: List[Dict[str, Any]] = []
    all_run_data: List[Dict[str, Any]] = []  # For analysis

    for tid in task_ids:
        task_results: List[Dict[str, Any]] = []

        print(f"\n{'═' * 70}")
        print(f"  TASK: {tid}")
        print(f"{'═' * 70}")

        for agent in agents:
            result = run_agent_on_task(agent, tid)
            all_run_data.append(result)

            # Print lightweight summary
            print(
                f"  {agent.name:20s}  "
                f"score={result['score']:.4f}  "
                f"reward={result['total_reward']:+.4f}  "
                f"steps={result['steps']}  "
                f"resolved={result['resolved_issues']}/{result['total_issues']}  "
                f"actions={result['actions']}"
            )

            task_results.append({
                "agent": agent.name,
                "score": result["score"],
                "steps": result["steps"],
                "total_reward": result["total_reward"],
                "resolved_issues": result["resolved_issues"],
                "total_issues": result["total_issues"],
                "actions": result["actions"],
                "confidence_scores": result["confidence_scores"],
                "grade": result["grade"],
            })

        comparison.append({
            "task_id": tid,
            "agents": task_results,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    avg_scores: Dict[str, float] = {}
    for agent in agents:
        scores = [
            entry["score"]
            for task_comp in comparison
            for entry in task_comp["agents"]
            if entry["agent"] == agent.name
        ]
        avg_scores[agent.name] = round(
            sum(scores) / len(scores), 4
        ) if scores else 0.0

    best_agent = max(avg_scores, key=avg_scores.get)  # type: ignore

    summary = {
        "best_agent": best_agent,
        "average_scores": avg_scores,
    }

    # ── Ranking ───────────────────────────────────────────────────────────
    ranking = sorted(
        [{"agent": name, "avg_score": score}
         for name, score in avg_scores.items()],
        key=lambda x: -x["avg_score"],
    )

    # ── Build result so far (needed for insight generation) ───────────────
    result = {
        "comparison": comparison,
        "summary": summary,
        "ranking": ranking,
    }

    # ── Insights ──────────────────────────────────────────────────────────
    result["insights"] = generate_insights(result)

    # ── Per-agent analysis ────────────────────────────────────────────────
    per_agent_analysis: List[Dict[str, Any]] = []
    for run_data in all_run_data:
        analysis = analyze_run(run_data)
        impact = generate_impact_report(
            run_data.get("task_id", ""),
            run_data.get("unresolved_issues", []),
        )
        analysis["impact"] = impact
        per_agent_analysis.append(analysis)

    result["analysis"] = per_agent_analysis

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  MULTI-AGENT COMPARISON SUMMARY")
    print(f"{'═' * 70}")
    for r in ranking:
        marker = " ◀ BEST" if r["agent"] == best_agent else ""
        print(f"  {r['agent']:20s}  avg_score={r['avg_score']:.4f}{marker}")

    print(f"\n  INSIGHTS:")
    for i, ins in enumerate(result["insights"], 1):
        print(f"    {i}. {ins}")

    return result


if __name__ == "__main__":
    results = run_all_agents()
