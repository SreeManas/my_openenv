"""
LLM-based inference agent for CodeReviewBench.

Connects to the OpenEnv API and uses an LLM to decide actions
for each task. Requires a running server (uvicorn server:app).

Environment variables:
    API_BASE_URL  — LLM API base URL (e.g. https://api-inference.huggingface.co/v1)
    MODEL_NAME    — Model identifier (e.g. meta-llama/Llama-3-70B-Instruct)
    HF_TOKEN      — HuggingFace API token
"""

import os
import json
import requests
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ENV_URL = os.environ.get("ENV_URL", "")

MAX_STEPS = 8
MAX_REASON_LEN = 160  # max chars shown for explanation in step logs

SYSTEM_PROMPT = """You are an AI code reviewer. Based on the given issue description, choose one action:
- fix_bug: Fix a bug or logic error in the code
- flag_issue: Flag a security vulnerability or resource leak for review
- optimize_code: Improve performance or code style
- leave_as_is: No action needed

Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{
  "action_type": "fix_bug | flag_issue | optimize_code | leave_as_is",
  "explanation": "Brief reason for your choice",
  "confidence": 0.0 to 1.0
}"""


# ──────────────────────────────────────────────────────────────────────────────
# LLM client
# ──────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def ask_llm(code_snippet: str, issue_type: str, context: str, step: int, max_steps: int) -> dict:
    """
    Send the current observation to the LLM and parse its action.
    """
    user_prompt = (
        f"## Code Under Review\n```python\n{code_snippet}\n```\n\n"
        f"## Issue Description\n{issue_type}\n\n"
        f"## Context\n{context}\n\n"
        f"Step {step}/{max_steps}. Choose the best action."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        action = json.loads(raw)

        # Validate fields
        valid_actions = {"fix_bug", "flag_issue", "optimize_code", "leave_as_is"}
        if action.get("action_type") not in valid_actions:
            action["action_type"] = "leave_as_is"

        action.setdefault("explanation", "LLM decision")
        action["confidence"] = max(0.0, min(1.0, float(action.get("confidence", 0.7))))

        return action

    except Exception as e:
        print(f"    [LLM ERROR] {e} — falling back to leave_as_is")
        return {
            "action_type": "leave_as_is",
            "explanation": f"LLM error: {e}",
            "confidence": 0.5,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Environment interaction
# ──────────────────────────────────────────────────────────────────────────────

def get_tasks() -> list:
    """Fetch available tasks from the environment."""
    resp = requests.get(f"{ENV_URL}/tasks", timeout=10)
    resp.raise_for_status()
    return resp.json()


def reset_task(task_id: str) -> dict:
    """Reset the environment for a specific task."""
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def step_action(action: dict) -> dict:
    """Send an action to the environment."""
    resp = requests.post(f"{ENV_URL}/step", json=action, timeout=10)
    resp.raise_for_status()
    return resp.json()


def grade() -> dict:
    """Grade the current trajectory."""
    resp = requests.post(f"{ENV_URL}/grader", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    """Run the LLM agent on a single task."""
    print(f"\n{'═' * 60}")
    print(f"  Task: {task_id}")
    print(f"{'═' * 60}")

    obs = reset_task(task_id)
    total_reward = 0.0

    for step_num in range(1, MAX_STEPS + 1):
        # Ask the LLM
        action = ask_llm(
            code_snippet=obs.get("code_snippet", ""),
            issue_type=obs.get("issue_type", ""),
            context=obs.get("context", ""),
            step=step_num,
            max_steps=obs.get("max_steps", MAX_STEPS),
        )

        clean_reason = action["explanation"].strip().replace("\n", " ")
        if len(clean_reason) > MAX_REASON_LEN:
            clean_reason = clean_reason[:MAX_REASON_LEN] + "..."
        print(
            f"  Step {step_num}: {action['action_type']:<15}  "
            f"conf={action['confidence']:.2f}  "
            f"reason={clean_reason}"
        )

        # Send action to environment
        result = step_action(action)
        obs = result["observation"]
        total_reward += result["reward"]

        if result["done"]:
            break

    # Grade the trajectory
    score = grade()
    final_score = score.get("score", 0.0)
    steps_used = score.get("steps_used", 0)

    print(f"\n  Score:  {final_score:.4f}")
    print(f"  Steps:  {steps_used}")
    print(f"  Reward: {total_reward:+.4f}")

    return {
        "task_id": task_id,
        "score": final_score,
        "steps": steps_used,
        "total_reward": round(total_reward, 4),
        "grade": score,
    }


def main():
    """Run the LLM agent on all available tasks."""
    global ENV_URL

    # Validate ENV_URL here (not at import time)
    ENV_URL = os.environ.get("ENV_URL", ENV_URL)
    if not ENV_URL:
        raise ValueError(
            "ENV_URL must be set (e.g. http://localhost:8000 or your HF Space URL)"
        )

    print(f"Model:  {MODEL_NAME}")
    print(f"Server: {ENV_URL}")
    print(f"Token:  {'set' if HF_TOKEN else 'NOT SET'}")

    tasks = get_tasks()
    results = []

    for task in tasks:
        # Support both list-of-dicts and list-of-strings
        task_id = task["task_id"] if isinstance(task, dict) else task
        result = run_task(task_id)
        results.append(result)

    # Summary
    print(f"\n{'═' * 60}")
    print("  LLM AGENT SUMMARY")
    print(f"{'═' * 60}")
    for r in results:
        print(f"  {r['task_id']:25s}  score={r['score']:.4f}  steps={r['steps']}")

    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"\n  Average score: {avg:.4f}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"[CONFIG ERROR] {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}")
        raise SystemExit(1)
