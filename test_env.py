"""
Basic integration tests for CodeReviewBench environment.

Run with:
    python3 test_env.py

Tests:
    1. reset() returns a valid Observation
    2. step() returns a valid StepResult structure
    3. A full easy episode runs without crashing
    4. All 8 tasks reset and run with score > 0.5
    5. Grader is deterministic across identical runs
    6. Error conditions raise appropriate exceptions
    7. Hidden issue reveal mechanic works correctly
"""

import sys
import logging

logging.disable(logging.CRITICAL)

from environment import CodeReviewEnv
from models import Action, ActionType, Observation, StepResult
from tasks import TASK_REGISTRY


def _pass(name: str) -> None:
    print(f"  ✅ PASS: {name}")


def _fail(name: str, reason: str) -> None:
    print(f"  ❌ FAIL: {name} — {reason}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — reset() returns a valid Observation
# ──────────────────────────────────────────────────────────────────────────────

def test_reset_returns_observation() -> None:
    env = CodeReviewEnv()
    obs = env.reset("easy_syntax_bug")

    if not isinstance(obs, Observation):
        _fail("reset returns Observation", f"got {type(obs)}")
    if not obs.task_id:
        _fail("reset returns Observation", "task_id is empty")
    if obs.step_number != 0:
        _fail("reset returns Observation", f"expected step_number=0, got {obs.step_number}")
    if obs.max_steps <= 0:
        _fail("reset returns Observation", "max_steps must be positive")
    if not obs.code_snippet:
        _fail("reset returns Observation", "code_snippet is empty")

    _pass("reset returns valid Observation")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — step() returns a valid StepResult
# ──────────────────────────────────────────────────────────────────────────────

def test_step_returns_valid_structure() -> None:
    env = CodeReviewEnv()
    env.reset("easy_syntax_bug")

    action = Action(
        action_type=ActionType.FIX_BUG,
        explanation="Fixing the missing colon after the if-condition.",
        confidence=0.9,
    )
    result = env.step(action)

    if not isinstance(result, StepResult):
        _fail("step returns StepResult", f"got {type(result)}")
    if not isinstance(result.observation, Observation):
        _fail("step returns StepResult", "observation field is not an Observation")
    if not isinstance(result.reward, float):
        _fail("step returns StepResult", f"reward is not float: {type(result.reward)}")
    if not isinstance(result.done, bool):
        _fail("step returns StepResult", f"done is not bool: {type(result.done)}")
    if not isinstance(result.info, dict):
        _fail("step returns StepResult", f"info is not dict: {type(result.info)}")

    _pass("step returns valid StepResult")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Full easy episode runs without crashing
# ──────────────────────────────────────────────────────────────────────────────

def test_full_easy_episode() -> None:
    env = CodeReviewEnv()
    env.reset("easy_syntax_bug")

    actions = [
        Action(action_type=ActionType.FIX_BUG, explanation="Fix syntax error.", confidence=0.95),
        Action(action_type=ActionType.OPTIMIZE_CODE, explanation="Simplify return.", confidence=0.8),
    ]

    total_reward = 0.0
    done = False

    for action in actions:
        if done:
            break
        result = env.step(action)
        total_reward += result.reward
        done = result.done

    grade = env.grade()

    if "score" not in grade:
        _fail("full easy episode", "'score' key missing from grade()")
    if not (0.0 <= grade["score"] <= 1.0):
        _fail("full easy episode", f"score out of range: {grade['score']}")
    if grade["issues_resolved"] < 1:
        _fail("full easy episode", "expected at least 1 issue resolved")

    _pass(f"full easy episode (score={grade['score']:.3f}, resolved={grade['issues_resolved']}/{grade['total_issues']})")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — All 8 tasks: reset + run expected sequence + grade > 0.5
# ──────────────────────────────────────────────────────────────────────────────

def test_all_tasks_run() -> None:
    action_map = {
        "fix_bug": ActionType.FIX_BUG,
        "optimize_code": ActionType.OPTIMIZE_CODE,
        "flag_issue": ActionType.FLAG_ISSUE,
        "leave_as_is": ActionType.LEAVE_AS_IS,
    }

    for task_id, task_def in TASK_REGISTRY.items():
        env = CodeReviewEnv()
        env.reset(task_id)

        seq = task_def.get("expected_sequence", [])
        done = False
        for action_str in seq:
            if done:
                break
            atype = action_map.get(action_str, ActionType.FIX_BUG)
            result = env.step(Action(action_type=atype, explanation="test", confidence=0.8))
            done = result.done

        grade = env.grade()

        if "score" not in grade:
            _fail(f"all tasks — {task_id}", "'score' missing from grade()")
        if not (0.0 <= grade["score"] <= 1.0):
            _fail(f"all tasks — {task_id}", f"score out of range: {grade['score']}")
        if grade["score"] < 0.5:
            _fail(f"all tasks — {task_id}", f"expected score > 0.5, got {grade['score']:.3f}")

    _pass(f"all {len(TASK_REGISTRY)} tasks reset, run, and graded > 0.5")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — Grader is deterministic across identical runs
# ──────────────────────────────────────────────────────────────────────────────

def test_determinism() -> None:
    def _run_and_grade() -> float:
        env = CodeReviewEnv()
        env.reset("hard_multi_issue")
        env.step(Action(action_type=ActionType.FLAG_ISSUE, explanation="sec", confidence=0.85))
        env.step(Action(action_type=ActionType.FIX_BUG, explanation="logic", confidence=0.75))
        env.step(Action(action_type=ActionType.OPTIMIZE_CODE, explanation="perf", confidence=0.7))
        return env.grade()["score"]

    scores = [_run_and_grade() for _ in range(3)]
    if len(set(scores)) != 1:
        _fail("determinism", f"3 identical runs gave different scores: {scores}")

    _pass(f"determinism verified (score={scores[0]:.4f} across 3 identical runs)")


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — Error conditions raise appropriate exceptions
# ──────────────────────────────────────────────────────────────────────────────

def test_error_conditions() -> None:
    # 6a: step() before reset → RuntimeError
    env = CodeReviewEnv()
    try:
        env.step(Action(action_type=ActionType.FIX_BUG, explanation="x", confidence=0.5))
        _fail("error: step before reset", "expected RuntimeError, got no exception")
    except RuntimeError:
        pass

    # 6b: step() after episode done → RuntimeError
    env2 = CodeReviewEnv()
    env2.reset("easy_syntax_bug")
    env2.step(Action(action_type=ActionType.FIX_BUG, explanation="x", confidence=0.9))
    env2.step(Action(action_type=ActionType.OPTIMIZE_CODE, explanation="x", confidence=0.8))
    try:
        env2.step(Action(action_type=ActionType.LEAVE_AS_IS, explanation="x", confidence=0.5))
        _fail("error: step after done", "expected RuntimeError, got no exception")
    except RuntimeError:
        pass

    # 6c: reset with unknown task_id → KeyError
    env3 = CodeReviewEnv()
    try:
        env3.reset("this_task_does_not_exist")
        _fail("error: invalid task_id", "expected KeyError, got no exception")
    except KeyError:
        pass

    _pass("error conditions raise correct exceptions (step-before-reset, step-after-done, bad-task-id)")


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 — Hidden issue reveal mechanic works correctly
# ──────────────────────────────────────────────────────────────────────────────

def test_hidden_issue_reveal() -> None:
    # medium_logic_bug: 2 visible issues, 1 hidden (med_edge_01)
    env = CodeReviewEnv()
    env.reset("medium_logic_bug")

    state_before = env.state()
    if len(state_before.remaining_issues) != 2:
        _fail(
            "hidden issue reveal",
            f"expected 2 visible issues before any action, got {len(state_before.remaining_issues)}"
        )

    # First correct action → triggers hidden issue reveal
    result = env.step(Action(action_type=ActionType.FIX_BUG, explanation="fix logic", confidence=0.8))

    if "revealed_issues" not in result.info:
        _fail("hidden issue reveal", "revealed_issues key not present in info after first correct action")
    if len(result.info["revealed_issues"]) < 1:
        _fail("hidden issue reveal", "expected at least 1 revealed issue, got 0")

    _pass(f"hidden issue reveal: {result.info['revealed_issues']} revealed after first action")


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nCodeReviewBench — Integration Tests\n" + "─" * 40)

    test_reset_returns_observation()
    test_step_returns_valid_structure()
    test_full_easy_episode()
    test_all_tasks_run()
    test_determinism()
    test_error_conditions()
    test_hidden_issue_reveal()

    print("\n" + "─" * 40)
    print("All tests passed ✅\n")
