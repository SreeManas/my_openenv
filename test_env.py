"""
Basic integration tests for CodeReviewBench environment.

Run with:
    python3 test_env.py

Tests:
    1. reset() returns a valid Observation
    2. step() returns a valid StepResult structure
    3. A full easy episode runs without crashing
"""

import sys

from environment import CodeReviewEnv
from models import Action, ActionType, Observation, StepResult


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

    # Grading should work after episode
    grade = env.grade()

    if "score" not in grade:
        _fail("full easy episode", "'score' key missing from grade()")
    if not (0.0 <= grade["score"] <= 1.0):
        _fail("full easy episode", f"score out of range: {grade['score']}")
    if grade["issues_resolved"] < 1:
        _fail("full easy episode", "expected at least 1 issue resolved")

    _pass(f"full easy episode (score={grade['score']:.3f}, resolved={grade['issues_resolved']}/{grade['total_issues']})")


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nCodeReviewBench — Integration Tests\n" + "─" * 40)

    test_reset_returns_observation()
    test_step_returns_valid_structure()
    test_full_easy_episode()

    print("\n" + "─" * 40)
    print("All tests passed ✅\n")
