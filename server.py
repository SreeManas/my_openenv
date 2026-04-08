"""
FastAPI server for the AI Code Review OpenEnv environment.

Exposes the following endpoints:
  POST /reset            — Start a new episode
  POST /step             — Take an action
  GET  /state            — Get current environment state
  GET  /tasks            — List available tasks
  POST /grader           — Grade a completed trajectory
  POST /baseline         — Run the baseline agent
  POST /compare_agents   — Run all agents and compare results
  POST /analysis         — Run a specific agent and return failure analysis
  POST /adaptive_run     — Run adaptive difficulty evaluation
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from models import Action, Observation, StepResult, EnvironmentState
from environment import CodeReviewEnv
from tasks import list_tasks as _list_tasks
from baseline import run_baseline
from multi_agent import run_all_agents, run_agent_on_task
from agents import get_all_agents
from analysis import analyze_run, generate_impact_report
from adaptive import run_adaptive

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CodeReviewBench — OpenEnv Evaluation Framework",
    description=(
        "A multi-step, self-explaining evaluation framework for AI agents "
        "performing code review, bug fixing, and optimization tasks. "
        "Features adaptive difficulty, deterministic noise injection, "
        "failure analysis, and real-world impact modeling."
    ),
    version="2.1.0",
)

# Singleton environment instance
env = CodeReviewEnv()


# ──────────────────────────────────────────────────────────────────────────────
# Request / response models for endpoints
# ──────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field(default="easy_syntax_bug", description="ID of the task to start.")


class GraderRequest(BaseModel):
    """No body needed — grades the current (or just-finished) trajectory."""
    pass


class CompareRequest(BaseModel):
    task_id: Optional[str] = Field(
        None,
        description=(
            "Task ID to compare agents on. "
            "If not provided, all tasks are used."
        ),
    )


class AnalysisRequest(BaseModel):
    task_id: str = Field(..., description="Task ID to run analysis on.")
    agent: str = Field(
        "baseline",
        description=(
            "Agent name: 'baseline', 'aggressive_agent', or 'safe_agent'."
        ),
    )


class AdaptiveRequest(BaseModel):
    agent: str = Field(
        "baseline",
        description="Agent name to run adaptive evaluation on.",
    )
    num_rounds: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of adaptive rounds (1–10).",
    )
    start_level: str = Field(
        "easy",
        description="Starting difficulty level: 'easy', 'medium', or 'hard'.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build clean agent summary
# ──────────────────────────────────────────────────────────────────────────────

def _make_agent_summary(run_data: dict, analysis: dict, impact: dict) -> dict:
    """Build a clean, scannable summary for a single agent run."""
    key_failures = []
    for f in analysis.get("failure_modes", []):
        if "No significant" in f:
            continue
        if "Missed" in f:
            key_failures.append("Missed issues")
        elif "hidden" in f.lower():
            key_failures.append("Hidden issue blindness")
        elif "Overconfident" in f:
            key_failures.append("Overconfidence")
        elif "Repeated" in f:
            key_failures.append("Action repetition")
        elif "ordering" in f.lower():
            key_failures.append("Ordering violation")
        elif "Safety" in f:
            key_failures.append("Safety violation")
        elif "leave_as_is" in f:
            key_failures.append("Step waste")
        else:
            key_failures.append(f[:50])

    return {
        "agent": run_data.get("agent", analysis.get("agent", "unknown")),
        "score": run_data.get("score", analysis.get("score", 0)),
        "risk_level": impact.get("risk_level", "none"),
        "key_failures": key_failures if key_failures else ["None"],
        "performance": {
            "steps": run_data.get("steps", 0),
            "resolved": (
                f"{run_data.get('resolved_issues', 0)}"
                f"/{run_data.get('total_issues', 0)}"
            ),
            "efficiency": analysis.get("efficiency_score", 0),
            "safety": analysis.get("safety_score", 0),
            "decision_quality": analysis.get("decision_quality", 0),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = Body(default=ResetRequest())):
    """Start a new episode for the given task."""
    try:
        obs = env.reset(req.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """Submit an action and receive the next observation + reward."""
    try:
        result = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/state", response_model=EnvironmentState)
def state():
    """Return the full current environment state."""
    return env.state()


@app.get("/tasks")
def tasks():
    """List all available tasks with metadata."""
    return _list_tasks()


@app.post("/grader")
def grader():
    """Grade the current trajectory and return the score breakdown."""
    try:
        return env.grade()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/baseline")
def baseline():
    """Run the baseline agent across all tasks and return results."""
    return run_baseline()


@app.get("/compare_agents")
def compare_agents_info():
    """Inform the user that this endpoint requires a POST request."""
    return {
        "message": "This endpoint requires a POST request.",
        "usage": "Use /docs (Swagger UI) or send a POST request to /compare_agents",
    }


@app.post("/compare_agents")
def compare_agents(req: CompareRequest = CompareRequest()):
    """
    Run all agents on one or all tasks and return a structured comparison.

    Includes per-task breakdowns, ranked agent list, natural-language
    insights, per-agent failure analysis with impact reports, and
    clean agent summaries.
    """
    try:
        result = run_all_agents(task_id=req.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Add clean summaries per analysis entry
    summaries = []
    for entry in result.get("analysis", []):
        impact = entry.get("impact", {})
        # Build a minimal run_data dict from the analysis entry
        run_stub = {
            "agent": entry.get("agent"),
            "score": entry.get("score"),
            "steps": 0,
            "resolved_issues": 0,
            "total_issues": 0,
        }
        summaries.append(_make_agent_summary(run_stub, entry, impact))

    result["agent_summaries"] = summaries
    return result


@app.post("/analysis")
def analysis(req: AnalysisRequest):
    """
    Run a specific agent on a task and return detailed failure analysis
    with real-world impact report and a clean summary.
    """
    # Find the requested agent
    agents = get_all_agents()
    agent = None
    for a in agents:
        if a.name == req.agent:
            agent = a
            break

    if agent is None:
        available = [a.name for a in agents]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown agent '{req.agent}'. "
                f"Available: {available}"
            ),
        )

    # Run the agent on a fresh environment
    try:
        run_data = run_agent_on_task(agent, req.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Analyze the run
    failure_analysis = analyze_run(run_data)
    impact = generate_impact_report(
        req.task_id,
        run_data.get("unresolved_issues", []),
    )

    return {
        # Clean summary at the top
        "summary": _make_agent_summary(run_data, failure_analysis, impact),
        # Full details
        "details": {
            "agent": req.agent,
            "task_id": req.task_id,
            "score": run_data["score"],
            "steps": run_data["steps"],
            "total_reward": run_data["total_reward"],
            "actions": run_data["actions"],
            "confidence_scores": run_data["confidence_scores"],
            "resolved_issues": run_data["resolved_issues"],
            "total_issues": run_data["total_issues"],
            "failure_modes": failure_analysis["failure_modes"],
            "efficiency_score": failure_analysis["efficiency_score"],
            "safety_score": failure_analysis["safety_score"],
            "decision_quality": failure_analysis["decision_quality"],
            "impact": impact,
        },
    }


@app.post("/adaptive_run")
def adaptive_run(req: AdaptiveRequest):
    """
    Run an adaptive difficulty evaluation for a specific agent.

    The agent starts at the specified difficulty level. After each round,
    the system promotes, holds, or demotes the difficulty based on the
    agent's score. Returns the full trajectory and final difficulty level.
    """
    # Find the requested agent
    agents = get_all_agents()
    agent = None
    for a in agents:
        if a.name == req.agent:
            agent = a
            break

    if agent is None:
        available = [a.name for a in agents]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown agent '{req.agent}'. "
                f"Available: {available}"
            ),
        )

    if req.start_level not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid start_level '{req.start_level}'. Use: easy, medium, hard.",
        )

    return run_adaptive(
        agent,
        num_rounds=req.num_rounds,
        start_level=req.start_level,
    )
