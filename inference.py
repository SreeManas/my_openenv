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
import random
import requests
from openai import OpenAI


def sanitize_text(text: str) -> str:
    """Strip all non-ASCII characters to prevent encode errors in API calls."""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("ascii", "ignore").decode("ascii")


def sanitize_dict(obj):
    """Recursively sanitize all string values in a parsed JSON object."""
    if isinstance(obj, dict):
        return {k: sanitize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_dict(v) for v in obj]
    elif isinstance(obj, str):
        return sanitize_text(obj)
    return obj


def analyze_context(context: str, issue_type: str = "") -> str:
    """Map observation text to the most likely correct action (no output, internal only).

    Returns one of: fix_bug | flag_issue | optimize_code
    This acts as a grounded prior that biases the LLM prompt toward the right action
    before the LLM makes its own decision.  It does NOT override the LLM — it informs it.
    """
    combined = (context + " " + issue_type).lower()

    # Memory / OOM signals — these need a code fix (not just flagging)
    if any(k in combined for k in (
        "memory leak", "memory usage", "oom", "out of memory",
        "unbounded cache", "unbounded dict", "never evicted",
        "grows without bound", "heap",
    )):
        return "fix_bug"

    # Security / compliance signals → flag first before touching code
    if any(k in combined for k in (
        "security", "token", "secret", "password", "credential", "pii",
        "gdpr", "sensitive", "sanitiz", "injection", "xss", "csrf",
        "exploit", "vulnerab", "auth", "permission", "data breach",
        "private key", "plaintext",
    )):
        return "flag_issue"

    # Concurrency / shared state → flag before fixing (root cause first)
    if any(k in combined for k in (
        "race condition", "thread", "concurr", "deadlock", "atomic",
        "shared state", "mutex", "lock", "synchroni",
        "global variable", "request_count",
    )):
        return "flag_issue"

    # Performance / efficiency signals → optimize
    if any(k in combined for k in (
        "slow", "performance", "latency", "inefficien", "n+1", "n + 1",
        "o(n", "o(n^2", "quadratic", "nested loop", "redundant",
        "cache miss", "bottleneck", "timeout",
        "connection pool", "saturated",
    )):
        return "optimize_code"

    # Logic / crash / edge-case signals → fix
    if any(k in combined for k in (
        "logic", "loop bound", "off-by-one", "duplicate", "incorrect",
        "crash", "exception", "unhandled", "none", "null", "undefined",
        "edge case", "unexpected", "wrong", "fail", "syntax", "parse",
        "missing colon", "unboundlocal", "keyerror", "typeerror",
        "leak", "resource leak",
    )):
        return "fix_bug"

    # Default
    return "fix_bug"



def get_all_available_tasks() -> list:
    """Dynamically discover all task IDs.

    Merges results from the live /tasks endpoint AND the local TASK_REGISTRY
    so that newly added tasks always appear, even if the server image is stale.
    """
    api_ids = []
    try:
        resp = requests.get(f"{ENV_URL}/tasks", timeout=15)
        resp.raise_for_status()
        tasks = sanitize_dict(resp.json())
        if tasks and isinstance(tasks, list):
            api_ids = [
                t["task_id"] if isinstance(t, dict) else t
                for t in tasks
            ]
    except Exception:
        pass

    # Local registry (picks up tasks added to tasks.py but not yet deployed)
    local_ids = []
    try:
        from tasks import TASK_REGISTRY  # type: ignore
        local_ids = list(TASK_REGISTRY.keys())
    except Exception:
        pass

    # Merge: API order first, then any local-only tasks appended at the end
    merged = list(api_ids)
    for tid in local_ids:
        if tid not in merged:
            merged.append(tid)

    if merged:
        return merged

    # Last resort
    return [
        "easy_syntax_bug", "medium_logic_bug", "hard_multi_issue",
        "security_review", "data_validation_pipeline", "concurrency_bug",
        "production_incident_response",
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

MAX_STEPS = 8
MAX_REASON_LEN = 160  # max chars shown for explanation in step logs
MAX_HISTORY_SHOWN = 5  # max past actions included in prompt
LLM_TIMEOUT = 60       # seconds before giving up on a slow LLM response


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
    memory: dict = None,
) -> str:
    """Generate a visible reasoning trace explaining WHY this action was chosen."""
    action_type = action.get("action_type", "fix_bug")
    issue_hint = obs.get("issue_type", "")
    remaining = obs.get("remaining_issues", [])
    override_reason = (memory or {}).get("_override_reason", "")

    # Check actual last reward
    last_reward = None
    last_action = ""
    if action_history:
        last = action_history[-1]
        last_action = last.get("action_type", "")
        last_reward = last.get("reward", None)

    parts = []

    # 1. Previous step outcome
    if last_reward is not None:
        if last_reward < 0:
            parts.append(f"previous {last_action} failed ({last_reward:+.2f})")
            if action_type != last_action:
                parts.append(f"switching to {action_type}")
            else:
                parts.append("retrying with adjusted approach")
        elif last_reward > 0:
            parts.append(f"previous {last_action} succeeded ({last_reward:+.2f})")

    # 2. Override explanation (if policy overrode the LLM)
    if override_reason:
        parts.append(override_reason)

    # 3. Context-based reasoning from current observation
    hint_lower = issue_hint.lower()
    if "security" in hint_lower or "sanitiz" in hint_lower or "sensitive" in hint_lower:
        parts.append("security context detected")
    elif "loop" in hint_lower or "bounds" in hint_lower or "redundant" in hint_lower:
        parts.append("algorithmic inefficiency detected")
    elif "crash" in hint_lower or "unbound" in hint_lower:
        parts.append("edge-case failure pattern")
    elif "concatenat" in hint_lower or "scale" in hint_lower or "efficient" in hint_lower:
        parts.append("performance bottleneck")
    elif "shared" in hint_lower or "counter" in hint_lower or "leaking" in hint_lower:
        parts.append("concurrency issue detected")
    elif "parse" in hint_lower or "control-flow" in hint_lower or "syntax" in hint_lower:
        parts.append("syntax defect detected")
    elif issue_hint:
        clean_hint = issue_hint.split(".")[0].strip()[:60]
        parts.append(f"analyzing: {sanitize_text(clean_hint)}")

    # 4. Action justification
    if action_type == "fix_bug":
        parts.append("applying targeted fix")
    elif action_type == "flag_issue":
        parts.append("flagging for review first")
    elif action_type == "optimize_code":
        parts.append("optimizing code")
    elif action_type == "leave_as_is":
        parts.append("concluding review")

    # 5. Progress
    if remaining:
        parts.append(f"{len(remaining)} remaining")

    thinking = " -> ".join(parts) if parts else f"step {step_num}: {action_type}"
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

    Includes: action history, progress, anti-repeat hints, decision strategy,
    and a context-analysis hint that biases the LLM toward the correct action
    before it chooses (analyze_context runs internally — no stdout output).
    """
    if action_history is None:
        action_history = []

    # ── Internal context analysis (no stdout) ────────────────────────────────
    # Runs keyword matching over context + issue_type to recommend an action.
    # Injected into the prompt as a grounded prior — does NOT override LLM.
    recommended_action = analyze_context(context=context, issue_type=issue_type)

    # Build structured prompt sections
    sections = [
        f"## Code Under Review\n```python\n{code_snippet}\n```",
        f"## Issue Description\n{issue_type}",
        f"## Context\n{context}",
    ]

    # Inject context-analysis recommendation (grounded prior)
    sections.append(
        f"## Recommended First Action (from context analysis)\n"
        f"Based on the issue description and context, the most appropriate action "
        f"is likely: **{recommended_action}**\n"
        f"Use this as a starting point, but apply your own reasoning."
    )

    # Add progress signal
    if total_count > 0:
        sections.append(_build_progress_block(resolved_count, total_count))

    # Add action history
    history_block = _build_history_block(action_history)
    if history_block:
        sections.append(history_block)

    # Add anti-repeat hint
    anti_repeat = _build_anti_repeat_hint(action_history)
    if anti_repeat:
        sections.append(anti_repeat)

    # Add decision strategy guidance
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
            timeout=LLM_TIMEOUT,
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

        # Sanitize + clean explanation
        explanation = sanitize_text(action["explanation"]).strip().replace("\n", " ")
        if len(explanation) > MAX_REASON_LEN:
            explanation = explanation[:MAX_REASON_LEN]
        action["explanation"] = explanation

        # Sanitize action_type (safety)
        action["action_type"] = sanitize_text(action.get("action_type", "leave_as_is"))

        action["confidence"] = max(0.0, min(1.0, float(action.get("confidence", 0.7))))

        return action

    except Exception:
        # Fallback: use context-analysis recommendation, never leave_as_is
        fallback = recommended_action if recommended_action != "leave_as_is" else "fix_bug"
        return {
            "action_type": fallback,
            "explanation": f"LLM fallback: context suggests {fallback}",
            "confidence": 0.6,
            "_error": None,
        }



# ──────────────────────────────────────────────────────────────────────────────
# Environment interaction
# ──────────────────────────────────────────────────────────────────────────────

def get_tasks() -> list:
    """Fetch available tasks from the environment."""
    resp = requests.get(f"{ENV_URL}/tasks", timeout=30)
    resp.raise_for_status()
    return resp.json()


def reset_task(task_id: str) -> dict:
    """Reset the environment for a specific task."""
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return sanitize_dict(resp.json())  # sanitize ALL incoming strings (incl. code_snippet)


def step_action(action: dict) -> dict:
    """Send an action to the environment. Sanitizes outgoing and incoming data."""
    safe_action = {
        "action_type": sanitize_text(action.get("action_type", "fix_bug")),
        "explanation": sanitize_text(action.get("explanation", "")),
        "confidence": float(action.get("confidence", 0.7)),
    }
    resp = requests.post(f"{ENV_URL}/step", json=safe_action, timeout=30)
    resp.raise_for_status()
    return sanitize_dict(resp.json())  # sanitize ALL incoming strings (incl. code_snippet)


def grade() -> dict:
    """Grade the current trajectory."""
    resp = requests.post(f"{ENV_URL}/grader", timeout=30)
    resp.raise_for_status()
    return sanitize_dict(resp.json())


# ──────────────────────────────────────────────────────────────────────────────
# OpenEnv-compliant logging  (STRICT format required by hackathon validator)
# ──────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_val = sanitize_text(error) if error else "null"  # sanitize error string
    print(
        f"[STEP] step={step} action={sanitize_text(action)} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_think(thinking: str) -> None:
    """Internal reasoning trace — written to stderr under DEBUG only.

    Deliberately NOT printed to stdout: the OpenEnv validator accepts
    only [START], [STEP], and [END] lines on stdout.
    """
    if os.environ.get("DEBUG") == "1":
        print(f"[THINK] {sanitize_text(thinking)}", file=sys.stderr, flush=True)


def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

# ── Hard task step strategy: flag first, then fix, then optimize ──────────
_HARD_TASK_STRATEGY = ["flag_issue", "fix_bug", "fix_bug", "optimize_code",
                       "fix_bug", "flag_issue", "optimize_code", "fix_bug"]

EPSILON = 0.15  # exploration rate for epsilon-greedy


def _select_adaptive_action(
    llm_action: str,
    memory: dict,
    remaining_ids: list,
    step_num: int,
    task_id: str,
) -> tuple:
    """
    Adaptive policy that overrides the LLM when evidence shows it's failing.

    Returns (action_type, override_reason_or_empty_string).
    """
    all_actions = ["fix_bug", "flag_issue", "optimize_code"]
    failed = memory["failed_actions"]
    banned = memory["banned_actions"]
    last_action = memory["last_action"]
    last_reward = memory["last_reward"]

    # Build available actions (exclude banned)
    available = [a for a in all_actions if a not in banned]
    if not available:
        # all banned — reset and try again
        memory["banned_actions"] = set()
        memory["failed_actions"] = set()
        available = all_actions

    override_reason = ""

    # ── Rule 0: no issues remaining → leave_as_is ──────────────────────
    if not remaining_ids:
        return "leave_as_is", "all issues resolved"

    # ── Rule 1: hard task strategy ────────────────────────────────────
    is_hard = "hard" in task_id or "concurrency" in task_id
    if is_hard and step_num <= len(_HARD_TASK_STRATEGY):
        strategy_action = _HARD_TASK_STRATEGY[step_num - 1]
        if strategy_action in available:
            return strategy_action, f"hard-task strategy step {step_num}"

    # 🔥 Rule 2: FORCE SWITCH after ANY negative reward (MOST IMPORTANT)
    if last_reward is not None and last_reward < 0:
        alternatives = [a for a in available if a != last_action]
        if alternatives:
            pick = random.choice(alternatives)
            return pick, f"forcing strategy switch after failure: {last_action} → {pick}"

    # ── Rule 3: prevent repeating failed action ───────────────────────
    if last_reward is not None and last_reward < 0 and last_action == llm_action:
        alternatives = [a for a in available if a != last_action]
        if alternatives:
            pick = random.choice(alternatives)
            return pick, f"overriding repeated failure {llm_action} → {pick}"

    # ── Rule 4: banned action ─────────────────────────────────────────
    if llm_action in banned:
        alternatives = [a for a in available if a != llm_action]
        if alternatives:
            pick = random.choice(alternatives)
            return pick, f"{llm_action} banned → {pick}"

    # ── Rule 5: epsilon exploration (bias toward higher Q-value actions) ─
    if random.random() < EPSILON and len(available) > 1:
        alternatives = [a for a in available if a != llm_action]
        if alternatives:
            # Prefer the alternative with highest cumulative reward
            scores = memory.get("action_scores", {})
            pick = max(alternatives, key=lambda a: scores.get(a, 0.0))
            return pick, f"exploring {pick} (epsilon={EPSILON})"

    # ── Default: trust LLM ────────────────────────────────────────────
    if llm_action in available:
        return llm_action, ""

    # fallback
    return available[0], f"{llm_action} unavailable, using {available[0]}"


# ── Cross-episode memory ─────────────────────────────────────────────────────
# Persists Q-values and bans across tasks so that feedback from episode N
# actually influences action selection in episode N+1.
_global_memory = {
    "action_scores": {},     # action -> cumulative Q-value (persists)
    "banned_actions": set(), # actions banned due to poor episode scores
}


def run_task(task_id: str) -> dict:
    """Run the LLM agent on a single task with adaptive episode memory."""
    # ── OpenEnv: announce start of episode ───────────────────────────────
    log_start(task=task_id, env="CodeReviewBench", model=MODEL_NAME)

    obs = reset_task(task_id)
    total_reward = 0.0
    action_history = []  # Track actions for trajectory awareness
    step_rewards: list[float] = []  # Collected for log_end
    final_score = 0.0
    steps_used = 0

    # ── Episode memory (seeded from cross-episode global memory) ──────
    memory = {
        "failed_actions": set(),
        "banned_actions": set(_global_memory["banned_actions"]),  # inherit bans
        "successful_actions": set(),
        "last_action": None,
        "last_reward": None,
        "fail_counts": {},
        "_override_reason": "",
        "action_scores": dict(_global_memory["action_scores"]),   # inherit Q-values
    }


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
                code_snippet=sanitize_text(obs.get("code_snippet", "")),
                issue_type=sanitize_text(obs.get("issue_type", "")),
                context=sanitize_text(obs.get("context", "")),
                step=step_num,
                max_steps=obs.get("max_steps", MAX_STEPS),
                action_history=action_history,
                resolved_count=resolved_count,
                total_count=total_count,
            )

            # Remove internal error field
            action.pop("_error", None)

            # ── Adaptive policy: override LLM if needed ───────────────
            llm_action = action["action_type"]
            chosen_action, override_reason = _select_adaptive_action(
                llm_action=llm_action,
                memory=memory,
                remaining_ids=remaining_ids,
                step_num=step_num,
                task_id=task_id,
            )
            action["action_type"] = chosen_action
            if override_reason:
                action["explanation"] = sanitize_text(override_reason)
            memory["_override_reason"] = override_reason

            # Save pre-step observation for THINK generation
            pre_step_obs = obs

            # Send action to environment
            result = step_action(action)
            obs = result["observation"]
            step_reward = result["reward"]
            done = result["done"]
            total_reward += step_reward
            step_rewards.append(step_reward)

            # ── OpenEnv: visible reasoning THINK (before STEP) ────────
            thinking = _generate_thinking(
                action=action,
                obs=pre_step_obs,
                action_history=action_history,
                step_num=step_num,
                memory=memory,
            )
            log_think(sanitize_text(thinking))

            # ── OpenEnv: log each step ────────────────────────────────
            log_step(
                step=step_num,
                action=sanitize_text(action["action_type"]),
                reward=step_reward,
                done=done,
                error=None,
            )

            # ── Update episode memory ─────────────────────────────────
            action_history.append({
                "step": step_num,
                "action_type": chosen_action,
                "reward": step_reward,
            })
            memory["last_action"] = chosen_action
            memory["last_reward"] = step_reward

            if step_reward < 0:
                memory["failed_actions"].add(chosen_action)
                memory["fail_counts"][chosen_action] = \
                    memory["fail_counts"].get(chosen_action, 0) + 1
                # Ban after 2 failures
                if memory["fail_counts"][chosen_action] >= 2:
                    memory["banned_actions"].add(chosen_action)
            elif step_reward > 0:
                memory["successful_actions"].add(chosen_action)
                # Un-fail if it worked this time
                memory["failed_actions"].discard(chosen_action)

            # Update lightweight Q-value (action_scores)
            memory["action_scores"][chosen_action] = (
                memory["action_scores"].get(chosen_action, 0.0) + step_reward
            )

            if done:
                break

        # Grade the trajectory
        score = grade()
        final_score = min(0.999, max(0.001, score.get("score", 0.001)))
        steps_used = score.get("steps_used", 0)

        # Runtime safety net — catch boundary violations immediately
        assert 0 < final_score < 1, f"INVALID SCORE: {final_score}"

        # ── RL feedback: blend episode score into Q-values ────────────────
        # Closes the loop: poor episode scores reduce Q-value of actions taken
        # and soft-ban the last action so epsilon-greedy avoids repeating it.
        if action_history:
            for entry in action_history:
                act = entry["action_type"]
                # 70% step reward already recorded; add 30% episode-grade signal
                memory["action_scores"][act] = (
                    memory["action_scores"].get(act, 0.0) + 0.3 * final_score
                )
            # Soft-ban last action if episode quality is poor
            if final_score < 0.6 and memory["last_action"]:
                memory["banned_actions"].add(memory["last_action"])

        # ── Write back to cross-episode global memory ─────────────────
        _global_memory["action_scores"].update(memory["action_scores"])
        _global_memory["banned_actions"] = set(memory["banned_actions"])

    except Exception as exc:
        step_num = len(step_rewards) + 1
        safe_exc = sanitize_text(str(exc).replace("\n", " ").strip())
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
    # final_score is already clamped (0.001, 0.999) above; clamp again for safety
    safe_score = min(0.999, max(0.001, final_score))
    log_end(
        success=safe_score > 0.5,
        steps=steps_used,
        rewards=step_rewards,
    )

    return {
        "task_id": task_id,
        "score": safe_score,
        "steps": steps_used,
        "total_reward": round(total_reward, 4),
        "grade": score,
    }


def main():
    """Run the LLM agent on all available tasks (discovered dynamically)."""
    global ENV_URL

    # Validate ENV_URL here (not at import time)
    ENV_URL = os.environ.get("ENV_URL", ENV_URL)
    if not ENV_URL:
        raise ValueError(
            "ENV_URL must be set (e.g. http://localhost:8000 or your HF Space URL)"
        )

    if os.environ.get("DEBUG") == "1":
        print(f"Model:  {MODEL_NAME}", file=sys.stderr)
        print(f"Server: {ENV_URL}", file=sys.stderr)
        print(f"Token:  {'set' if HF_TOKEN else 'NOT SET'}", file=sys.stderr)

    # Dynamic task discovery — picks up new tasks without any code changes
    task_ids = get_all_available_tasks()

    if os.environ.get("DEBUG") == "1":
        print(f"Tasks ({len(task_ids)}): {task_ids}", file=sys.stderr)

    results = []
    for task_id in task_ids:
        result = run_task(task_id)
        results.append(result)

    # Debug summary (stderr only — never touches OpenEnv stdout)
    if os.environ.get("DEBUG") == "1":
        print(f"\n{'=' * 60}", file=sys.stderr)
        print("  LLM AGENT SUMMARY", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        for r in results:
            print(f"  {r['task_id']:30s}  score={r['score']:.4f}  steps={r['steps']}",
                  file=sys.stderr)
        avg = sum(r["score"] for r in results) / len(results) if results else 0
        print(f"\n  Average score: {avg:.4f}", file=sys.stderr)



if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
