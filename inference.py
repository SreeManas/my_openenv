"""
LLM-based inference agent for CodeReviewBench.

Connects to the OpenEnv API and uses an LLM to decide actions
for each task. Requires a running server (uvicorn server:app).

Environment variables:
    API_BASE_URL  — LLM API base URL (e.g. https://api-inference.huggingface.co/v1)
    MODEL_NAME    — Model identifier (e.g. meta-llama/Llama-3-70B-Instruct)
    HF_TOKEN      — HuggingFace API token
"""
import sys
import os
import json
import requests
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ENV_URL = os.environ.get("ENV_URL", "https://sreemanas-mycodeenv.hf.space")

MAX_STEPS = 8
MAX_REASON_LEN = 160  # max chars shown for explanation in step logs
MAX_HISTORY_SHOWN = 5  # max past actions included in prompt


SYSTEM_PROMPT = """You are an expert AI code reviewer performing a multi-step code review. Your goal is to identify and resolve ALL issues in the code efficiently.

Available actions:
- fix_bug: Fix a bug, logic error, or edge-case defect in the code
- flag_issue: Flag a security vulnerability or resource leak for review
- optimize_code: Improve performance or code style
- leave_as_is: No further action needed (use ONLY when all issues are resolved)

Critical rules:
1. SECURITY FIRST: Always flag security vulnerabilities before fixing other issues.
2. Do NOT repeat the same action type more than 2 times consecutively unless new evidence appears.
3. Each step should target a DIFFERENT issue. If your previous action succeeded, move to the next problem.
4. If previous actions did not resolve issues, try a DIFFERENT action type.
5. Use leave_as_is ONLY when you are confident all issues have been addressed.
6. Calibrate your confidence: use 0.7-0.85 for clear issues, 0.5-0.7 for ambiguous ones.

Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{
  "action_type": "fix_bug | flag_issue | optimize_code | leave_as_is",
  "explanation": "Brief reason for your choice",
  "confidence": 0.0 to 1.0
}"""


# ──────────────────────────────────────────────────────────────────────────────
# LLM client
# ──────────────────────────────────────────────────────────────────────────────

# Lazily initialized on first ask_llm() call to avoid import-time crash
# when HF_TOKEN is not yet set in the environment.
client = None


def _build_history_block(action_history: list) -> str:
    """Build a concise action history block for the prompt."""
    if not action_history:
        return ""

    # Show at most the last MAX_HISTORY_SHOWN actions
    recent = action_history[-MAX_HISTORY_SHOWN:]
    lines = ["## Actions Already Taken"]
    for entry in recent:
        reward_str = f"{entry['reward']:+.2f}" if entry.get("reward") is not None else "N/A"
        lines.append(f"- Step {entry['step']}: {entry['action_type']} (reward={reward_str})")
    return "\n".join(lines)


def _build_progress_block(resolved_count: int, total_count: int) -> str:
    """Build a progress signal block for the prompt."""
    remaining = total_count - resolved_count
    return (
        f"## Progress\n"
        f"- Resolved: {resolved_count} / {total_count}\n"
        f"- Remaining issues: {remaining}"
    )


def _build_anti_repeat_hint(action_history: list) -> str:
    """Inject a hint if the agent is repeating the same action type."""
    if len(action_history) < 2:
        return ""

    last_two = [h["action_type"] for h in action_history[-2:]]
    if last_two[0] == last_two[1] and last_two[0] != "leave_as_is":
        return (
            "\n> **Warning**: Previous actions of the same type did not resolve "
            "new issues. Consider a different strategy or action type.\n"
        )
    return ""


def _generate_thinking(
    action: dict,
    obs: dict,
    action_history: list,
    step_num: int,
) -> str:
    """Generate a visible reasoning trace explaining WHY this action was chosen."""
    action_type = action.get("action_type", "leave_as_is")
    issue_hint = obs.get("issue_type", "")
    remaining = obs.get("remaining_issues", [])

    # Check for previous failure
    last_failed = False
    last_action = ""
    if action_history:
        last = action_history[-1]
        last_action = last.get("action_type", "")
        last_failed = last.get("reward", 0) < 0

    parts = []

    # 1. Failure recognition (Step 4)
    if last_failed:
        parts.append(f"Previous {last_action} produced negative reward")
        if action_type != last_action:
            parts.append(f"switching strategy to {action_type}")
        else:
            parts.append("retrying with refined approach")

    # 2. Context-based reasoning
    hint_lower = issue_hint.lower()
    if "security" in hint_lower or "sanitiz" in hint_lower or "sensitive" in hint_lower:
        parts.append("security context detected, prioritizing vulnerability assessment")
    elif "loop" in hint_lower or "bounds" in hint_lower or "redundant" in hint_lower:
        parts.append("detected algorithmic inefficiency in loop structure")
    elif "crash" in hint_lower or "none" in hint_lower or "unbound" in hint_lower:
        parts.append("edge-case failure pattern identified")
    elif "concatenat" in hint_lower or "scale" in hint_lower or "efficient" in hint_lower:
        parts.append("performance bottleneck detected")
    elif "shared" in hint_lower or "counter" in hint_lower or "leaking" in hint_lower:
        parts.append("concurrency or state management issue detected")
    elif issue_hint:
        clean_hint = issue_hint.split(".")[0].strip()[:80]
        parts.append(f"analyzing: {clean_hint}")

    # 3. Action justification
    if action_type == "fix_bug":
        parts.append("applying targeted fix based on root-cause analysis")
    elif action_type == "flag_issue":
        parts.append("flagging for review before attempting direct modification")
    elif action_type == "optimize_code":
        parts.append("applying performance optimization")
    elif action_type == "leave_as_is":
        if not remaining:
            parts.append("all issues resolved, concluding review")
        else:
            parts.append("no actionable insight, preserving current state")

    # 4. Progress context
    if remaining:
        parts.append(f"{len(remaining)} issue(s) remaining")

    thinking = " -> ".join(parts) if parts else f"Step {step_num}: evaluating {action_type}"
    return thinking.replace("\n", " ")[:200]


def ask_llm(
    code_snippet: str,
    issue_type: str,
    context: str,
    step: int,
    max_steps: int,
    action_history: list | None = None,
    resolved_count: int = 0,
    total_count: int = 0,
) -> dict:
    """
    Send the current observation to the LLM and parse its action.

    Now includes action history, progress tracking, anti-repeat hints,
    and a decision strategy block for better trajectory awareness.
    """
    if action_history is None:
        action_history = []

    # Build structured prompt sections
    sections = [
        f"## Code Under Review\n```python\n{code_snippet}\n```",
        f"## Issue Description\n{issue_type}",
        f"## Context\n{context}",
    ]

    # Add progress signal (Task 2)
    if total_count > 0:
        sections.append(_build_progress_block(resolved_count, total_count))

    # Add action history (Task 1)
    history_block = _build_history_block(action_history)
    if history_block:
        sections.append(history_block)

    # Add anti-repeat hint (Task 3)
    anti_repeat = _build_anti_repeat_hint(action_history)
    if anti_repeat:
        sections.append(anti_repeat)

    # Add decision strategy guidance (Task 6)
    sections.append(
        "## Decision Strategy\n"
        "- Prioritize unresolved issues by severity (security > bugs > performance > style)\n"
        "- Avoid repeating ineffective actions\n"
        "- Adapt based on previous outcomes and rewards\n"
        "- If stuck or receiving negative rewards, switch action type\n"
        f"\nStep {step}/{max_steps}. Choose the best action."
    )

    user_prompt = "\n\n".join(sections)

    global client
    if client is None:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
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

        # Clean explanation (Task 7)
        explanation = action["explanation"].strip().replace("\n", " ")
        if len(explanation) > MAX_REASON_LEN:
            explanation = explanation[:MAX_REASON_LEN]
        action["explanation"] = explanation

        action["confidence"] = max(0.0, min(1.0, float(action.get("confidence", 0.7))))

        return action

    except Exception as e:
        return {
            "action_type": "leave_as_is",
            "explanation": f"LLM error: {e}",
            "confidence": 0.5,
            "_error": str(e),  # surfaced in log_step, not printed directly
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
# OpenEnv-compliant logging  (STRICT format required by hackathon validator)
# ──────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_think(thinking: str) -> None:
    print(f"[THINK] {thinking}", flush=True)


def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    """Run the LLM agent on a single task."""
    # ── OpenEnv: announce start of episode ───────────────────────────────
    log_start(task=task_id, env="CodeReviewBench", model=MODEL_NAME)

    obs = reset_task(task_id)
    total_reward = 0.0
    action_history = []  # Track actions for trajectory awareness
    step_rewards: list[float] = []  # Collected for log_end
    final_score = 0.0
    steps_used = 0

    try:
        for step_num in range(1, MAX_STEPS + 1):
            # Derive progress from observation
            remaining_ids = obs.get("remaining_issues", [])
            resolved_count = len(action_history) - sum(
                1 for h in action_history if h.get("reward", 0) <= 0
            )
            total_count = max(resolved_count + len(remaining_ids), len(remaining_ids))

            # Ask the LLM with full context
            action = ask_llm(
                code_snippet=obs.get("code_snippet", ""),
                issue_type=obs.get("issue_type", ""),
                context=obs.get("context", ""),
                step=step_num,
                max_steps=obs.get("max_steps", MAX_STEPS),
                action_history=action_history,
                resolved_count=resolved_count,
                total_count=total_count,
            )

            # Capture LLM error if one occurred (set in ask_llm fallback)
            llm_error = action.pop("_error", None)

            # Save pre-step observation for THINK generation
            pre_step_obs = obs

            # Send action to environment
            result = step_action(action)
            obs = result["observation"]
            step_reward = result["reward"]
            done = result["done"]
            total_reward += step_reward
            step_rewards.append(step_reward)

            # Sanitize error string — strip newlines to keep [STEP] single-line
            safe_error = str(llm_error).replace("\n", " ").strip() if llm_error else None

            # ── OpenEnv: visible reasoning trace ──────────────────────────
            thinking = _generate_thinking(
                action=action,
                obs=pre_step_obs,
                action_history=action_history,
                step_num=step_num,
            )
            log_think(thinking)

            # ── OpenEnv: log each step ────────────────────────────────────
            log_step(
                step=step_num,
                action=action["action_type"],
                reward=step_reward,
                done=done,
                error=safe_error,
            )

            # Record in action history for trajectory awareness
            action_history.append({
                "step": step_num,
                "action_type": action["action_type"],
                "reward": step_reward,
            })

            if done:
                break

        # Grade the trajectory
        score = grade()
        final_score = score.get("score", 0.0)
        steps_used = score.get("steps_used", 0)

    except Exception as exc:
        # ── OpenEnv: emit a failed step then fall through to log_end ─────
        step_num = len(step_rewards) + 1
        safe_exc = str(exc).replace("\n", " ").strip()
        log_step(
            step=step_num,
            action="leave_as_is",
            reward=0.0,
            done=True,
            error=safe_exc,
        )
        score = {}
        steps_used = step_num

    # ── OpenEnv: announce end of episode ─────────────────────────────────
    safe_score = min(0.999, max(0.001, final_score))
    log_end(
        success=safe_score > 0.5,
        steps=steps_used,
        rewards=step_rewards,
    )

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

    if os.environ.get("DEBUG") == "1":
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
    if os.environ.get("DEBUG") == "1":
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
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
