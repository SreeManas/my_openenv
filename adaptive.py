"""
Adaptive difficulty system for CodeReviewBench.

Provides dynamic task selection based on agent performance,
simulating progressive evaluation where task difficulty adjusts
to the agent's capability level.

All behavior is deterministic — identical agents produce identical
adaptive trajectories.
"""

from typing import Dict, Any, List, Optional

from environment import CodeReviewEnv
from agents import Agent, get_all_agents
from analysis import analyze_run, generate_impact_report
from tasks import list_tasks

# ══════════════════════════════════════════════════════════════════════════════
# Difficulty progression
# ══════════════════════════════════════════════════════════════════════════════

# Task ordering by difficulty level
DIFFICULTY_MAP = {
    "easy": "easy_syntax_bug",
    "medium": "medium_logic_bug",
    "hard": "hard_multi_issue",
}

# Progression rules
LEVEL_ORDER = ["easy", "medium", "hard"]


def select_next_level(current_level: str, score: float) -> str:
    """
    Select the next difficulty level based on agent performance.

    Rules:
        score > 0.85  → promote to harder level
        0.6 ≤ score ≤ 0.85 → stay at same level
        score < 0.6   → demote to easier level

    Caps at 'hard' (ceiling) and 'easy' (floor).
    """
    idx = LEVEL_ORDER.index(current_level)

    if score > 0.85:
        idx = min(idx + 1, len(LEVEL_ORDER) - 1)  # Promote
    elif score < 0.6:
        idx = max(idx - 1, 0)  # Demote

    return LEVEL_ORDER[idx]


def run_adaptive(
    agent: Agent,
    num_rounds: int = 3,
    start_level: str = "easy",
) -> Dict[str, Any]:
    """
    Run an adaptive evaluation sequence for a single agent.

    The agent starts at `start_level` and the difficulty adjusts
    after each round based on its score.

    Args:
        agent: Agent to evaluate.
        num_rounds: Number of consecutive tasks (default 3).
        start_level: Starting difficulty level.

    Returns:
        Dict with trajectory, final_level, and agent summary.
    """
    current_level = start_level
    trajectory: List[Dict[str, Any]] = []

    print(f"\n{'═' * 70}")
    print(f"  ADAPTIVE RUN: {agent.name}  (start={start_level}, rounds={num_rounds})")
    print(f"{'═' * 70}")

    for round_num in range(1, num_rounds + 1):
        task_id = DIFFICULTY_MAP[current_level]
        env = CodeReviewEnv()
        run_data = agent.run_task(env, task_id)

        score = run_data["score"]
        next_level = select_next_level(current_level, score)

        # Build analysis for this round
        analysis = analyze_run(run_data)
        impact = generate_impact_report(
            task_id, run_data.get("unresolved_issues", [])
        )

        round_result = {
            "round": round_num,
            "task_id": task_id,
            "difficulty": current_level,
            "score": score,
            "steps": run_data["steps"],
            "resolved": f"{run_data['resolved_issues']}/{run_data['total_issues']}",
            "next_level": next_level,
            "key_failures": _extract_key_failures(analysis),
            "risk_level": impact.get("risk_level", "none"),
        }

        trajectory.append(round_result)

        print(
            f"  Round {round_num}: {current_level:6s} → "
            f"score={score:.4f}  "
            f"resolved={round_result['resolved']}  "
            f"next={next_level}"
        )

        current_level = next_level

    # Summary
    avg_score = sum(r["score"] for r in trajectory) / len(trajectory)
    print(f"\n  Final level: {current_level}  avg_score={avg_score:.4f}")

    return {
        "agent": agent.name,
        "start_level": start_level,
        "num_rounds": num_rounds,
        "trajectory": trajectory,
        "final_level": current_level,
        "average_score": round(avg_score, 4),
    }


def _extract_key_failures(analysis: Dict[str, Any]) -> List[str]:
    """Extract concise failure labels from a full analysis."""
    failures = analysis.get("failure_modes", [])
    key = []
    for f in failures:
        if "No significant" in f:
            continue
        # Shorten failure descriptions to key phrases
        if "Missed" in f:
            key.append("Missed issues")
        elif "hidden" in f.lower():
            key.append("Hidden issue blindness")
        elif "Overconfident" in f:
            key.append("Overconfidence")
        elif "Repeated" in f:
            key.append("Action repetition")
        elif "ordering" in f.lower():
            key.append("Ordering violation")
        elif "Safety" in f:
            key.append("Safety violation")
        elif "leave_as_is" in f:
            key.append("Step waste")
        else:
            key.append(f[:50])
    return key if key else ["None"]


if __name__ == "__main__":
    for agent in get_all_agents():
        run_adaptive(agent)
