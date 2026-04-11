---
title: CodeReviewBench
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
# CodeReviewBench: A Multi-Step RL Evaluation Framework for Code Intelligence Agents

> **A multi-step reinforcement learning environment that evaluates AI agents on sequential code review reasoning — with trajectory-based scoring, partial observability, and confidence-calibrated rewards. Designed to expose meaningful capability differences between weak and strong models across 8 tasks of increasing difficulty.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-v1.0-blue)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## What Makes This Unique

Most AI benchmarks are single-step. CodeReviewBench is not.

- **Multi-step code evolution** — the codebase changes dynamically as issues are resolved
- **Hidden issues revealed mid-episode** — partial observability forces adaptive replanning
- **Confidence-aware reward shaping** — calibration scoring penalizes overconfident wrong decisions
- **Order constraints** — security-first gating with penalties for unsafe sequencing
- **Deterministic noise injection** — ~40% of hints are perturbed to test semantic robustness
- **Multi-agent benchmarking** — safe vs. aggressive vs. baseline strategies compared head-to-head
- **Failure analysis engine** — explains *why* agents fail, not just *that* they failed
- **Real-world impact modeling** — maps each unresolved bug to a concrete consequence

---

## Agent Performance (8 tasks)

| Agent            | Easy  | Med (logic) | Med (security) | Hard (multi) | Hard (edge) | Med (perf) | Med (validation) | Hard (conc) | **Avg** |
| ---------------- | ----- | ----------- | -------------- | ------------ | ----------- | ---------- | ---------------- | ----------- | ------- |
| `adaptive_agent` | 0.994 | 0.995       | **0.995**      | 0.833        | 0.822       | 0.995      | 0.865            | 0.822       | **0.915** |
| `safe_agent`     | 0.984 | 0.985       | 0.988          | 0.211        | 0.481       | 0.982      | 0.921            | 0.481       | 0.754   |
| `baseline`       | 0.929 | 0.653       | 0.999          | 0.228        | 0.448       | 0.999      | 0.597            | 0.448       | 0.662   |
| `aggressive`     | 0.998 | 0.998       | 0.588          | 0.315        | 0.438       | 0.379      | 0.588            | 0.438       | 0.588   |

Key insight: **the adaptive agent significantly outperforms static strategies** — it achieves 0.915 avg vs 0.754 for the next best agent. The gap is widest on hard tasks (0.82 vs 0.48), proving that reward-based adaptation is critical for multi-step code review. Easy tasks remain non-differentiating.

---

## Model Performance

| Model | Average Score | Notes |
|-------|--------------|-------|
| Qwen-72B-Instruct | ~0.85 | Strong multi-step reasoning, correct prioritization |
| GPT-4o-mini | ~0.72 | Good on simple tasks, struggles with hidden issues and hard tasks |
| Llama-3-8B | ~0.50 | Frequent action loops, poor ordering on hard tasks |

*Approximate values based on local inference runs. Exact scores vary with prompt formatting and API endpoint.*

- Smaller models struggle with **hidden issues** and **multi-step planning** — they repeat actions and miss newly revealed defects
- Larger models achieve higher scores through **better prioritization** and **adaptive reasoning** after failures
- The score gap between 8B and 72B models (**+0.35**) demonstrates meaningful difficulty scaling

---

## Key Challenges

- **Hidden issues revealed mid-episode** — agents cannot plan a complete solution upfront
- **Ambiguous hints requiring inference** — symptom descriptions, not root-cause labels
- **Order constraints** — security must be resolved before optimization (violations penalized −0.3)
- **Penalties for repeated actions** — looping on the same action type costs −0.15 per repeat
- **Confidence calibration** — overconfident wrong answers penalized more than cautious ones
- **Multi-step trajectories with recovery** — suboptimal early actions don't end the episode; agents can recover

---

## Why This Environment is Non-Trivial

- **Keyword matching alone is insufficient for high scores** — hints describe symptoms ("intermittent failures when no records match"), not causes ("returns None instead of []"). Hard tasks require mixed action types that simple heuristics cannot determine
- **Multiple valid trajectories exist** with different reward profiles — there is no single optimal path
- **Suboptimal actions are penalized but recovery is possible** — the environment rewards agents that adapt after mistakes
- **Stronger models significantly outperform smaller ones** — the 8B→72B gap (+0.35) proves the tasks test genuine reasoning beyond pattern matching
- **Hidden issues force replanning** — agents that pre-commit to a fixed strategy fail on medium and hard tasks
- **Explanations matter** — agents that provide empty or trivially short explanations receive a per-step penalty
- **Reward-based adaptation is measurably superior** — the adaptive agent's 0.915 avg vs 0.754 (safe) proves that observe→act→reward→adapt behavior is rewarded by the environment

---

## Agent Behavior: RL Characteristics

CodeReviewBench's reward structure incentivizes RL-like behavior. The `AdaptiveAgent` demonstrates this:

**Observe → Act → Reward → Adapt loop:**
1. **Observe**: receive code snippet, issue hint, remaining issue count
2. **Act**: select action type + explanation + calibrated confidence
3. **Reward**: receive dense per-step reward (positive for correct, negative for wrong)
4. **Adapt**: avoid action types that produced negative reward; switch strategy on failure

**Concrete mechanisms:**

| Mechanism | Implementation | Impact |
|-----------|---------------|--------|
| Anti-repetition | If last action failed → force different action type | Prevents `fix_bug` spam on hard tasks |
| Reward-based avoidance | Track `avoid_actions` set from negative rewards | Agent avoids repeating failed strategies |
| Task-based strategy | Infer task category → bias action preference | Security tasks prioritize `flag_issue` first |
| Dynamic confidence | Confidence = f(recent success rate) | Starts 0.75, adapts to [0.55, 0.85] based on outcomes |

**Reproducibility**: optional `seed` parameter in `/reset` enables deterministic-but-varied evaluation runs.

---

## RL Formulation

CodeReviewBench is formalized as a finite-horizon MDP:

| Component | Definition |
|-----------|-----------|
| **State** *s* | `(code_snippet, issue_hint, context, remaining_issues, step_number)` — evolves after each action |
| **Action** *a* | `{fix_bug, flag_issue, optimize_code, leave_as_is}` × explanation × confidence ∈ [0,1] |
| **Transition** *T(s,a)* | Deterministic: resolves matched issue, reveals hidden issues, updates code version |
| **Reward** *R(s,a)* | Dense per-step: `base × severity × calibration + sequence_bonus + repeat_penalty + explanation_penalty` |
| **Horizon** | 3–8 steps depending on task difficulty |
| **Termination** | All issues resolved OR max steps reached |

**Key RL properties:**
- **Non-trivial state transitions**: code evolves after each fix (code_versions keyed by resolved set)
- **Partial observability**: hidden issues not visible until prerequisites resolved
- **Credit assignment challenge**: some actions unlock future rewards (sequence bonuses)
- **Exploration vs exploitation**: trying new action types risks penalty but may reveal better strategies

---

## Environment Dynamics

The environment state evolves at each step — this is **not** a static classification task:

1. **Code evolution**: after resolving an issue, the code snippet physically changes to reflect the fix (via `code_versions` lookup keyed by the set of resolved issue IDs)
2. **Context evolution**: the observation context updates with resolution progress (e.g., `[Progress: 2 issue(s) resolved, 1 remaining]`)
3. **Issue reveal**: hidden issues (e.g., edge-case bugs) become visible only after the agent resolves prerequisite problems
4. **Hint degradation**: after wrong actions, the hint appends adversarial guidance ("re-evaluate assumptions about the issue type")

This means two agents taking different action sequences will observe **different states** — the environment is path-dependent.

---

## Visible Reasoning (`[THINK]` Traces)

The inference agent emits a visible reasoning trace **before every action**:

```
[START] task=hard_multi_issue env=CodeReviewBench model=Qwen/Qwen2.5-72B-Instruct
[THINK] security context detected, prioritizing vulnerability assessment -> flagging for review -> 5 issue(s) remaining
[STEP] step=1 action=flag_issue reward=1.22 done=false error=null
[THINK] Previous flag_issue produced negative reward -> switching strategy to fix_bug -> 4 issue(s) remaining
[STEP] step=2 action=fix_bug reward=0.85 done=false error=null
[END] success=true steps=4 rewards=1.22,-0.35,0.85,1.09
```

Each `[THINK]` line includes:
- **Failure recognition**: references negative reward from previous step
- **Strategy adaptation**: explains why the action type changed
- **Context analysis**: identifies issue category (security, performance, edge-case)
- **Progress tracking**: remaining issue count

---

## Scalability

While the current version ships with 8 hand-crafted tasks, the architecture supports scalable extension:

- **Seed-based reproducibility**: `/reset` accepts an optional `seed` parameter — same seed produces identical episode
- **Task template structure**: each task is a self-contained dict with `issues`, `code_versions`, `expected_sequence` — new tasks require no code changes
- **Difficulty spectrum**: easy (1–2 issues, 3 steps) → hard (4–6 issues, 8 steps, hidden dependencies)
- **Agent-agnostic API**: any agent (LLM-based, rule-based, RL-trained) can interact via the standard `/reset` → `/step` → `/grader` loop

---

## Abstract

**CodeReviewBench** is a fully OpenEnv-compliant, deployment-ready reinforcement-learning environment that provides a structured framework for evaluating AI agents on sequential code analysis, bug resolution, and optimization tasks across **8 diverse tasks**. Unlike single-step classification benchmarks, CodeReviewBench requires agents to operate under **partial observability**, **ambiguous feedback signals**, and **inter-dependent action constraints** — properties characteristic of real-world software engineering workflows.

The environment features dynamic state evolution, hidden defect discovery, confidence-calibrated reward shaping, adaptive difficulty adjustment, deterministic noise injection, trajectory-aware observation feedback, and trajectory-based grading across five evaluation dimensions.

**CodeReviewBench not only evaluates what decisions an agent makes, but why those decisions succeed or fail, and what their real-world consequences would be.**

A rule-based baseline agent achieves an average score of **0.662** across 8 tasks — demonstrating that the environment is neither trivially solvable nor intractably difficult. The adaptive agent with reward-based learning achieves **0.915**, while static agents plateau at **0.754**, proving that the environment meaningfully rewards adaptive behavior.

---

## What This Environment Evaluates

CodeReviewBench measures six core capabilities underrepresented in existing benchmarks:

- **Sequential decision-making** — early choices constrain later options
- **Reasoning under partial observability** — agents must act on incomplete knowledge
- **Hidden information handling** — new defects revealed mid-episode require adaptive replanning
- **Trade-off management** — genuine safety-vs-efficiency tensions with no single dominant strategy
- **Confidence calibration** — overconfidence on wrong decisions is penalized more than cautious uncertainty
- **Robustness to noise** — deterministic perturbations add controlled ambiguity to observations

---

## 1. Motivation

### The Evaluation Gap

| Property | Toy Environments | Real Developer Workflows |
|----------|-----------------|--------------------------|
| Decisions per episode | 1 | 5–50+ |
| Information availability | Complete | Partial, evolving |
| Feedback signals | Unambiguous | Noisy, contextual |
| Action dependencies | Independent | Sequence-sensitive |
| Risk management | Absent | Critical |
| Confidence awareness | Irrelevant | Essential |

A developer performing code review does not simply classify a snippet — they must *read*, *hypothesize*, *prioritize*, *act*, *observe the consequences*, and *adapt*. Errors in early steps cascade. Overconfidence leads to regressions.

CodeReviewBench reintroduces these dynamics in a controlled, deterministic setting where agent capabilities can be measured with diagnostic precision.

### Design Objective

CodeReviewBench was designed to close this gap. It provides a deterministic, reproducible, and scalable environment in which agents must demonstrate:

1. **Diagnostic reasoning** — inferring issue types from ambiguous hints
2. **Strategic prioritization** — resolving critical issues before functional ones
3. **Calibrated confidence** — expressing appropriate certainty in each decision
4. **Adaptive planning** — responding to newly revealed information mid-episode
5. **Efficient execution** — achieving objectives in minimal steps

---

## 2. Environment Overview

### Interaction Model

CodeReviewBench follows the standard OpenEnv protocol:

```
observation₀ ← env.reset(task_id)

for t = 1, 2, ..., T:
    actionₜ ← agent.decide(observationₜ₋₁)
    observationₜ, rewardₜ, doneₜ, infoₜ ← env.step(actionₜ)
    if doneₜ: break

trajectory_score ← env.grade()
```

### Real-World Analogy

Each episode simulates a developer reviewing a pull request:

1. **Initial review** — reads the code and sees reviewer comments (ambiguous hints)
2. **Triage** — decides what to address first based on context
3. **Iterative fixes** — each fix modifies the codebase; may reveal hidden issues
4. **Completion** — ends when all issues resolved or step budget exhausted
5. **Post-hoc evaluation** — entire trajectory scored on multiple quality dimensions

### Action Space

| Action | Semantics | Typical Targets |
|--------|-----------|-----------------|
| `fix_bug` | Correct a functional defect | Syntax errors, logic bugs, edge-case failures |
| `optimize_code` | Improve non-functional quality | Performance bottlenecks, style improvements |
| `flag_issue` | Escalate a systemic concern | Security vulnerabilities, resource leaks |
| `leave_as_is` | Decline to act | Used when no issues remain |

Each action includes a **free-text explanation** and a **confidence score** ∈ [0, 1] that directly influences reward.

### Observation Space

| Field | Description |
|-------|-------------|
| `code_snippet` | Current version of the code (evolves dynamically) |
| `issue_type` | An **ambiguous hint** — *not* a ground-truth label |
| `context` | Task-level background information |
| `remaining_issues` | IDs of *visible* unresolved issues (hidden issues excluded) |
| `step_number` | Current step index |
| `max_steps` | Episode budget |

> **Key design choice**: `issue_type` presents *hints* rather than labels. A logic error appears as *"Unexpected behavior: the function produces correct output but performs redundant comparisons"* — not *"logic_error"*. Agents must reason about symptom-to-cause mappings.

---

## 3. Key Design Features

### 3.1 Partial Observability

**Hidden issues** are revealed only after the agent resolves at least one visible issue — modeling how fixing one bug frequently exposes another.

- **Medium task**: a `None`-input crash hidden until the loop-bounds bug is fixed
- **Hard task**: a database connection resource leak surfaces only after SQL injection is patched

This prevents agents from planning a complete solution upfront and forces adaptive replanning.

### 3.2 Ambiguous Observations

Issue hints are deliberately vague — describing **symptoms**, not causes:

| Ground Truth | Agent Sees |
|-------------|------------|
| `syntax_error` | *"Code fails to parse; check control-flow statements."* |
| `logic_error` | *"The function behaves unexpectedly when no records match the query."* |
| `security_vulnerability` | *"An external value is incorporated into a privileged operation without validation."* |
| `resource_leak` | *"Monitoring shows slow exhaustion of a finite system resource over time."* |

### 3.3 Sequential Action Dependencies

The hard task enforces **order constraints**: fixing functional bugs before patching a security vulnerability incurs a penalty (−0.3 per violation). Correctly ordered actions receive a **sequence bonus** (+0.5), evaluated via longest common subsequence (LCS) matching.

### 3.4 Safety–Efficiency Trade-offs

Agents face genuine trade-offs with no single dominant strategy:

- **Flagging security** is correct but delays functional fixes, risking step budget exhaustion
- **Optimizing first** may implicitly resolve a logic bug but forfeits explicit resolution credit
- **Skipping minor issues** preserves steps for critical bugs but reduces completion score

### 3.5 Confidence-Calibrated Rewards

```
Correct actions:   reward = base × severity × (0.5 + 0.5 × confidence)
Incorrect actions: penalty = base × (0.5 + 0.5 × confidence)
```

Verified impact (same correct action, different confidence on `easy_syntax_bug`):

| Confidence | Reward | Δ from baseline |
|------------|--------|-----------------|
| 0.95 | **+1.475** | +0.200 |
| 0.55 | **+1.275** | — (baseline) |

An agent that is *correctly confident* earns up to **15.7% more reward**. An agent that is *incorrectly confident* loses proportionally more. This incentivizes well-calibrated probability estimates — critical for trustworthy AI systems.

### 3.6 Dynamic State Evolution

After each action, the environment updates `code_snippet` to reflect the applied fix. The agent observes *changed* code in subsequent steps, requiring reasoning about a moving target. Code versions are pre-computed for all combinations of resolved issues, ensuring full determinism.

---

## Design Philosophy

- **Realism over simplicity** — models properties of real developer workflows (ambiguity, hidden information, cascading consequences)
- **No trivial solutions** — ambiguous hints and hidden issues prevent keyword matching and fixed heuristics from scoring highly
- **Confidence as a first-class signal** — evaluates metacognitive capability, not just task performance
- **Trade-offs are genuine** — no universally optimal action sequence; agents balance competing objectives under a limited step budget
- **Determinism and reproducibility** — identical agent trajectories always receive identical scores

---

## 4. Task Design

### Difficulty Progression

| Property | Easy | Medium | Hard |
|----------|------|--------|------|
| Total issues | 2 | 3 | 3–4 |
| *Hidden* issues | 0 | 1 | 1 |
| Max steps | 4 | 6 | 8 |
| Order constraints | No | Some | Yes (security-first) |
| Trade-offs | Minimal | Moderate | Significant |
| Issue types | Syntax, style | Logic, performance, edge-case, security | Security, logic, performance, resource leak |

### Task 1: Easy — Syntax Error (`easy_syntax_bug`)

A function with a missing colon after an `if` statement, plus a minor style improvement. Straightforward for any agent capable of reading Python syntax.

### Task 2: Medium — Logic + Performance + Hidden Edge Case (`medium_logic_bug`)

A duplicate-finder function with:
- An off-by-one loop (described as *"incorrect loop bounds"*, not *"logic error"*)
- An O(n²) algorithm (clearly signaled)
- A **hidden** `None`-input crash, revealed only after the first fix

Trade-off: optimizing first with a set-based approach implicitly fixes the loop bug, but the agent receives no explicit resolution credit and misses the sequence bonus.

### Task 3: Hard — Multi-Issue with Ordering (`hard_multi_issue`)

A database query function with four defects:
1. **SQL injection** — must be flagged *first* (security-first gate; violations penalized −0.3)
2. **Wrong return type** — returns `None` instead of `[]` on empty results
3. **Bubble sort** — O(n²) when `sorted()` is available
4. **Resource leak** — connection not wrapped in `try/finally` (**hidden**, revealed after first fix)

The ideal agent flags security, fixes the return type, flags the resource leak, then optimizes the sort.

### Task 4: Medium — Security Variant (`medium_security_variant`)

A password reset token module with:
1. **Predictable tokens** — timestamp-based generation (must be flagged, security-first)
2. **Credential leakage** — token printed to logs (**hidden**, revealed after flagging)
3. **Missing input validation** — no guard on malformed tokens

Tests whether agents can distinguish "flag for review" from "fix directly" on security issues.

### Task 5: Hard — Edge Case Cascade (`hard_edge_case`)

A running-average utility with:
1. **Mutable default** — `values.sort()` mutates caller's list (data corruption)
2. **Loop index error** — off-by-one in window calculation
3. **Empty input crash** — no guard for empty lists (**hidden**)

Order matters: fixing the mutation before the logic bug prevents misdiagnosis.

### Task 6: Medium — Performance Heavy (`performance_heavy`)

A report-generation function with three independent performance issues:
1. **Selection sort** — O(n²) when `sorted()` is available
2. **String concatenation** — quadratic allocations in loop
3. **Intermediate list** — unnecessary materialization (**hidden**)

All three expected actions are `optimize_code` — tests whether agents avoid defaulting to `fix_bug`.

### Task 7: Medium — Data Validation Pipeline (`data_validation_pipeline`)

A user registration handler with:
1. **Missing field guard** — `KeyError` on absent fields (priority gate)
2. **Unsafe type cast** — `int()` on non-numeric input (**hidden**)
3. **No email validation** — arbitrary strings stored (flag required)

### Task 8: Hard — Concurrency Bug (`concurrency_bug`)

A batch processing module with:
1. **Mutable default argument** — shared state across calls (priority gate)
2. **Counter undercount** — only last batch reported
3. **Unbound variable** — crash on empty input (**hidden**)

---

## 5. Reward Function

| Event | Reward | Modifiers |
|-------|--------|-----------|
| Correct action for issue | `+1.0` | `× severity × calibration(conf)` |
| Action in expected sequence | `+0.5` | Added on top of base |
| Wrong action for matching issue | `−0.5` | `× calibration(conf)` — overconfidence hurts more |
| Unmatched action (no issue fits) | `−0.2` | — |
| `leave_as_is` with unresolved issues | `−0.1` | — |
| Repeated consecutive action type | `−0.15` | Discourages action repetition |
| Order-constraint violation | `−0.3` | Per violated constraint |
| Empty/trivial explanation (<10 chars) | `−0.05` | Incentivizes meaningful reasoning |
| Efficiency bonus (early finish) | `+0.5 × (budget remaining / max steps)` | Awarded once at episode end |

---

## 6. Grading Methodology

### Five Evaluation Components

| Component | Weight | Methodology | Range |
|-----------|--------|-------------|-------|
| **Completion** | 30% | `|resolved| / |total_issues|` | [0, 1] |
| **Efficiency** | 15% | `max(0, 1 − (steps − ideal) / max_steps)` | [0, 1] |
| **Safety** | 20% | `max(0, 1 − 0.3 × harmful − 0.1 × minor_wrong)` | [0, 1] |
| **Sequence** | 20% | LCS(expected, actual) / len(expected) | [0, 1] |
| **Calibration** | 15% | `1 − mean(cal_error²)` where cal_error is confidence-correctness mismatch | [0, 1] |

**Final score** = weighted sum, strictly clamped to **(0.001, 0.999)** to comply with OpenEnv evaluation constraints — scores of exactly 0.0 or 1.0 are never returned.

Properties:
- **Deterministic** — identical trajectories always produce identical scores
- **Sensitive** — different strategies yield meaningfully different grades
- **Decomposable** — each component diagnoses a specific capability gap

---

## 7. Baseline Agent Analysis

### Design

The baseline is a **rule-based keyword matcher** that:
- Scans the hint text for keywords (e.g., *"parse"* → `fix_bug`, *"sanitiz"* → `flag_issue`)
- Uses a fixed confidence of 0.9 for all matched actions
- Falls back to `leave_as_is` (confidence 0.5) when no keyword matches

### Results

| Task | Score | Completed | Failure Mode |
|------|-------|-----------|--------------|
| Easy | **0.929** | 2/2 | Near-perfect; simple keyword matching suffices |
| Medium (logic) | **0.653** | 2/3 | Misses hidden edge-case; wastes steps on `leave_as_is` |
| Medium (security) | **0.999** | 3/3 | Catches all keywords after hint improvement |
| Hard (multi) | **0.228** | 1/4 | Fails to match rewritten symptom-based hints; wrong action types |
| Hard (edge) | **0.448** | 2/3 | Misses flag_issue requirement for data corruption |
| Medium (perf) | **0.999** | 3/3 | Optimize keywords matched correctly |
| Medium (validation) | **0.597** | 2/3 | Misses email flag — requires flag_issue |
| Hard (concurrency) | **0.448** | 2/3 | Misses flag_issue requirement for reporting defect |
| **Average** | **0.662** | | |

### Failure Analysis

1. **Hidden issues** — no mechanism to anticipate newly revealed issues; falls back to `leave_as_is`, burning step budget
2. **Ambiguity** — the edge-case hint (*"What happens if the function receives None?"*) contains no keyword the baseline recognizes
3. **Overconfidence** — fixed 0.9 confidence means calibration loss when wrong
4. **Inefficiency** — repeated `leave_as_is` actions penalized; excess steps reduce efficiency component

These are genuine capability gaps, not implementation bugs — a more sophisticated agent can address them through reasoning, planning, and confidence estimation.

---

## 8. Why This Environment Is Challenging

A perfect-scoring agent must simultaneously:

- **Read and comprehend code** to infer issue types from ambiguous textual hints
- **Prioritize safety concerns** over functional improvements
- **Anticipate hidden defects** and allocate step budget accordingly
- **Calibrate confidence** — expressing certainty only when warranted
- **Minimize unnecessary actions** to maximize the efficiency score
- **Follow the optimal resolution sequence** to earn ordering bonuses
- **Navigate trade-offs** where no single action is unambiguously best

No single heuristic achieves this. The environment requires integrated reasoning across multiple competencies — precisely what distinguishes frontier AI agents from simple classifiers.

---

## 9. Setup & Usage

### Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: `fastapi`, `uvicorn`, `pydantic`, `pyyaml`, `openai`

### Run Baseline Agent

```bash
python3 baseline.py
```

### Start API Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t codereviewbench .
docker run -p 8000:8000 codereviewbench
```

### API Endpoints

| Method | Path | Request Body | Description |
|--------|------|-------------|-------------|
| `POST` | `/reset` | `{"task_id": "easy_syntax_bug"}` | Initialize episode |
| `POST` | `/step` | `{"action_type": "fix_bug", "explanation": "...", "confidence": 0.9}` | Submit action |
| `GET` | `/state` | — | Current environment state |
| `GET` | `/tasks` | — | List available tasks |
| `POST` | `/grader` | — | Grade current trajectory |
| `POST` | `/baseline` | — | Execute baseline across all tasks |
| `POST` | `/compare_agents` | `{"task_id": "hard_multi_issue"}` (optional) | Compare all agents |
| `POST` | `/analysis` | `{"task_id": "...", "agent": "baseline"}` | Failure analysis + impact report |
| `POST` | `/adaptive_run` | `{"agent": "safe_agent", "num_rounds": 3}` | Adaptive difficulty evaluation |

### Example Interaction

```bash
# Start episode
curl -X POST localhost:8000/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "hard_multi_issue"}'

# Take action (flag SQL injection with high confidence)
curl -X POST localhost:8000/step \
  -H 'Content-Type: application/json' \
  -d '{"action_type": "flag_issue", "explanation": "SQL injection via string interpolation", "confidence": 0.95}'

# Check trajectory score
curl -X POST localhost:8000/grader
```

---

## Example Trajectories (Verified)

All traces below were generated from actual `environment.py` runs with deterministic outputs.

### Successful Episode — Safe Agent on `medium_logic_bug` (score: 0.990)

```
[START] task=medium_logic_bug env=CodeReviewBench agent=safe_agent
[STEP] step=1 action=fix_bug reward=1.22 done=false error=null
[STEP] step=2 action=optimize_code reward=1.09 done=false error=null
[STEP] step=3 action=fix_bug reward=1.19 done=true error=null
[END] success=true steps=3 score=0.9904 rewards=1.22,1.09,1.19
```

Grade breakdown: completion=1.0, efficiency=1.0, safety=1.0, sequence=1.0, calibration=0.936

### Failed Episode — Aggressive Agent on `hard_multi_issue` (score: 0.463)

```
[START] task=hard_multi_issue env=CodeReviewBench agent=aggressive_agent
[STEP] step=1 action=fix_bug reward=0.38 done=false error=null   # order violation: −0.3
[STEP] step=2 action=fix_bug reward=-0.35 done=false error=null  # repeat penalty: −0.15
[STEP] step=3 action=fix_bug reward=-0.35 done=false error=null  # repeat penalty: −0.15
[STEP] step=4 action=fix_bug reward=-0.35 done=false error=null  # repeat penalty: −0.15
[END] success=false steps=4 score=0.4634 rewards=0.38,-0.35,-0.35,-0.35
```

Grade breakdown: completion=0.25, efficiency=1.0, safety=0.7, sequence=0.25, calibration=0.322

**Why it failed:** The agent used `fix_bug` for all 4 steps. Step 1 triggered an **order-constraint violation** (−0.3) because it resolved an issue before flagging the security vulnerability. Steps 2–4 incurred **repeat penalties** (−0.15 each) and no longer matched any expected action type. Only 1 of 4 issues was resolved. Calibration score collapsed to 0.322 because the agent reported 0.95 confidence on every wrong decision.

This trajectory demonstrates three distinct failure modes detected by the 5-component grading system — poor ordering, action repetition, and miscalibrated confidence — none of which would be visible in a single-number accuracy metric.

---

## Example Analysis Output

The `/analysis` endpoint returns structured failure diagnostics and real-world impact reports. Below is representative output for the aggressive agent on the hard task:

```json
{
  "agent": "aggressive_agent",
  "task_id": "hard_multi_issue",
  "score": 0.292,
  "failure_modes": [
    "Missed 3 issue(s): hard_sec_01, hard_perf_01, hard_resource_01",
    "Failed to detect hidden issue(s) revealed mid-episode",
    "Repeated same action 7 times consecutively",
    "Actions were not in the expected priority order"
  ],
  "impact_report": [
    {
      "issue_id": "hard_sec_01",
      "impact": "Data breach risk — attacker can exfiltrate or modify database contents via crafted input",
      "risk_level": "critical"
    },
    {
      "issue_id": "hard_resource_01",
      "impact": "System instability — database connections leak on exceptions, causing resource exhaustion",
      "risk_level": "high"
    }
  ]
}
```



---

## Robustness & Anti-Exploitation (Verified)

The environment is hardened against degenerate strategies. All results below are from actual runs:

### Single-Action Spam

A trivial agent that sends `fix_bug` every step:

| Task | Spam Score | Why It Fails |
|------|-----------|------|
| `hard_multi_issue` | **0.303** | Security issue requires `flag_issue`, not `fix_bug` — 3 of 4 actions unmatched |
| `hard_edge_case` | **0.448** | Data corruption requires `flag_issue` — wrong action type for first issue |
| `concurrency_bug` | **0.448** | Reporting defect requires `flag_issue` — wrong action type for second issue |

Single-action spam scores **0.30–0.45** on hard tasks. No rule-based agent exceeds **0.50** on these tasks either — hard tasks require **mixed action types** (`flag_issue` + `fix_bug`) that cannot be solved by defaulting to one action.

### Explanation Quality

| Explanation | Reward | Penalty |
|-------------|--------|---------|
| `""` (empty) | 1.400 | −0.05 per step |
| `"Fix the missing colon after the if-condition"` | 1.450 | none |

Agents that provide empty or trivially short explanations (<10 characters) incur a deterministic penalty on every step.

### Active Penalties Summary

| Anti-Exploitation Mechanism | Penalty | Verified |
|----------------------------|---------|----------|
| Repeated consecutive action type | −0.15 per step | ✅ |
| Order-constraint violation (security-first) | −0.30 per violation | ✅ |
| Empty/short explanation (<10 chars) | −0.05 per step | ✅ |
| Wrong action type for matched issue | −0.50 × calibration | ✅ |
| `leave_as_is` with unresolved issues | −0.10 per step | ✅ |

---

## What This Benchmark Reveals (Empirical)

Based on observed agent trajectories across 8 tasks:

**1. Overconfidence is measurably destructive.** The aggressive agent reports 0.95 confidence on every action. On `hard_multi_issue`, this produces a calibration score of **0.322** (vs. the safe agent's 0.936 on `medium_logic_bug` with calibrated 0.70–0.80 confidence). A 15.7% reward difference between confidence levels on correct actions makes calibration a genuine differentiator.

**2. Action repetition is the primary failure mode for simple agents.** The baseline agent falls back to `leave_as_is` when no keyword matches, burning up to 4 steps at −0.10 each on `medium_logic_bug`. The aggressive agent loops `fix_bug` on `hard_multi_issue`, accumulating −0.15 repeat penalties on steps 2–4.

**3. Hidden issues defeat fixed strategies.** On `medium_logic_bug`, a hidden `None`-input crash is revealed only after the first fix. The baseline agent has no mechanism to detect this, defaulting to `leave_as_is` for 4 remaining steps. The safe agent's trajectory-aware strategy resolves it with score **0.985** vs. baseline's **0.653**.

**4. Hard tasks expose the limits of all rule-based strategies.** All three agents score below **0.50** on hard tasks. The safe agent (0.754 overall average) leads only because it performs well on medium tasks — it still fails on hard tasks (0.211–0.481). This leaves a clear gap for LLM-based agents with genuine code reasoning.

---

## Multi-Agent Evaluation

CodeReviewBench includes a **comparative evaluation mode** running multiple agent strategies on identical tasks.

### Agent Strategies

| Agent | Strategy | Confidence | Key Weakness |
|-------|----------|------------|--------------|
| `adaptive_agent` | Reward-based learning with failure avoidance | 0.55–0.85 (dynamic) | Requires multi-step episodes to learn |
| `baseline` | Keyword matching on hints | 0.9 (fixed) | Misses hidden issues, overconfident |
| `aggressive_agent` | Fix everything directly | 0.95 (high) | Wrong action types for security, order violations |
| `safe_agent` | Flag risks first, then fix | 0.6–0.75 (calibrated) | Slightly slower, cautious on clear bugs |

### Results

| Agent | Easy | Med† | Med† | Hard | Hard | Med | Med | Hard | **Average** |
|-------|------|------|------|------|------|-----|-----|------|-------------|
| `adaptive_agent` | 0.994 | 0.995 | **0.995** | 0.833 | 0.822 | 0.995 | 0.865 | 0.822 | **0.915** ◀ BEST |
| `safe_agent` | 0.984 | 0.985 | 0.988 | 0.211 | 0.481 | 0.982 | 0.921 | 0.481 | 0.754 |
| `baseline` | 0.929 | 0.653 | 0.999 | 0.228 | 0.448 | 0.999 | 0.597 | 0.448 | 0.662 |
| `aggressive_agent` | 0.998 | 0.998 | 0.588 | 0.315 | 0.438 | 0.379 | 0.588 | 0.438 | 0.588 |

*†Med = medium difficulty tasks with different focus areas*

### Usage

```bash
# Run all agents on all tasks
python3 multi_agent.py

# Via API — compare on a specific task
curl -X POST localhost:8000/compare_agents \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "hard_multi_issue"}'
```

---

## Failure Analysis Engine

Beyond scoring, CodeReviewBench explains **why** agents fail:

| Failure Mode | Detection Rule | Example |
|-------------|----------------|---------|
| Missed issues | `resolved < total` | "Missed 3 issue(s): hard_sec_01, hard_perf_01, hard_resource_01" |
| Hidden issue blindness | Unresolved issue has `hidden: true` | "Failed to detect hidden issue(s) revealed mid-episode" |
| Overconfidence | High confidence + low calibration score | "Average confidence 0.95 but calibration score 0.21" |
| Action repetition | Same action type used consecutively | "Repeated same action 7 times consecutively" |
| Poor ordering | Sequence score < 0.7 | "Actions were not in the expected priority order" |
| Safety violations | Safety score < 0.7 | "Agent took harmful or incorrect actions" |
| Step waste | ≥3 `leave_as_is` with issues remaining | "Used leave_as_is 5 times with issues remaining" |

```bash
curl -X POST localhost:8000/analysis \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "hard_multi_issue", "agent": "aggressive_agent"}'
```

---

## Real-World Impact Modeling

Each issue carries an `impact` field describing the real-world consequence of leaving it unresolved:

| Issue Type | Impact |
|-----------|--------|
| Syntax error | Runtime failure — code cannot execute |
| Logic error | Incorrect results — duplicates missed or double-counted |
| Performance issue | Slow execution — O(n²) runtime causes timeouts |
| Security vulnerability | Data breach risk — SQL injection enables unauthorized access |
| Resource leak | System instability — connections leak, causing exhaustion |
| Edge case | Application crash — unhandled None input causes TypeError |

---

## Adaptive Evaluation

The dynamic difficulty adjustment system progressively tests agent capabilities:

| Agent Score | Difficulty Transition |
|------------|----------------------|
| > 0.85 | Promote to harder task |
| 0.60 – 0.85 | Stay at same level |
| < 0.60 | Demote to easier task |

### Adaptive Results

| Agent | Round 1 | Round 2 | Round 3 | Final Level |
|-------|---------|---------|---------|-------------|
| `safe_agent` | easy → 0.984 | medium → 0.985 | hard → 0.211 | **medium** (demoted) |
| `aggressive_agent` | easy → 0.998 | medium → 0.998 | hard → 0.315 | **medium** (demoted) |
| `baseline` | easy → 0.929 | medium → 0.653 | medium → 0.653 | **medium** (plateaued) |

Hard tasks now genuinely challenge all agent strategies — no rule-based agent sustains performance at the highest difficulty level.

```bash
curl -X POST localhost:8000/adaptive_run \
  -H 'Content-Type: application/json' \
  -d '{"agent": "safe_agent", "num_rounds": 3}'
```

---

## Robustness via Noise

CodeReviewBench applies **deterministic perturbations** to observation hints, simulating imperfect signals real developers encounter:

- Perturbations use hash-based selection (`seed=42 + step_number`) for full reproducibility
- ~40% of hints receive word-level substitutions that preserve meaning but add ambiguity
- Each observation includes a `noise_applied` transparency flag
- Toggle via `ENABLE_NOISE = True/False` in `noise.py`

| Original | Perturbed |
|----------|-----------|
| "fails to parse" | "encounters a structural issue" |
| "Unexpected behavior" | "Possible anomaly detected" |
| "sensitive operation" | "privileged operation" |
| "loop bounds" | "iteration boundaries" |

This tests whether agents rely on brittle keyword matching or robust semantic understanding.

---

## 10. Project Structure

```
├── models.py          Pydantic models (Action, Observation, StepResult, EnvironmentState)
├── tasks.py           8 task definitions with hints, hidden issues, impacts, and order constraints
├── grader.py          Five-component trajectory grader with calibration scoring
├── environment.py     Multi-step environment with state evolution, reward shaping, feedback hints, and noise
├── noise.py           Deterministic noise injection engine (toggleable)
├── agents.py          Agent ABC + 3 strategy implementations (baseline, aggressive, safe)
├── inference.py       LLM inference agent with trajectory awareness and anti-repeat guidance
├── analysis.py        Failure analysis, impact modeling, and insight generation
├── adaptive.py        Adaptive difficulty system with progressive task selection
├── multi_agent.py     Comparative evaluation runner with integrated analysis
├── baseline.py        Legacy baseline runner (standalone)
├── server.py          FastAPI server exposing the OpenEnv API (9 endpoints)
├── test_env.py        Integration tests (7 tests: all tasks, determinism, errors, hidden reveal)
├── openenv.yaml       Environment metadata and schema definitions
├── requirements.txt   Python dependencies
├── Dockerfile         Container build specification
└── README.md          This document
```

---

## OpenEnv Evaluation Criteria Mapping

How CodeReviewBench addresses each official evaluation criterion:

### Real-World Utility (30%)
- Models a **genuine software engineering workflow** — code review is a high-demand, high-stakes activity performed daily by millions of developers
- Evaluates capabilities directly relevant to AI coding assistants (Copilot, Cursor, CodeRabbit): bug identification, fix prioritization, confidence calibration
- Partial observability and hidden issue mechanics mirror real review scenarios where fixing one bug reveals another
- 5-component grading provides **actionable diagnostic feedback**, not just a pass/fail score

### Task & Grader Quality (25%)
- **8 tasks** across 3 difficulty levels (easy, medium, hard) with verified difficulty progression
- Grader outputs strictly in **(0.001, 0.999)** — verified range: [0.118, 0.994]; final scores are clamped to avoid boundary values
- Fully **deterministic** — verified: identical trajectories produce score 0.7811 across 3 runs
- Hard tasks genuinely challenge all strategies — best rule-based agent scores **0.211–0.481** on hard tasks

### Environment Design (20%)
- Clean `reset()` with task isolation; `RuntimeError` on invalid state transitions
- 4-action space with typed Pydantic models and documented enum values
- **Dense reward shaping**: 8 distinct reward signals (correct/wrong action, sequence bonus, order penalty, repeat penalty, explanation penalty, calibration, efficiency bonus)
- Proper episode termination: all-resolved OR max-steps, with `done` flag and termination reason

### Code Quality & Spec Compliance (15%)
- Follows OpenEnv spec: `/reset`, `/step`, `/grader`, `/tasks`, `/state` endpoints
- Docker builds on `python:3.11-slim`, port 7860, healthcheck included
- `inference.py` outputs **only** `[START]`, `[STEP]`, `[END]` to stdout (all other prints DEBUG-guarded or stderr)
- 7 integration tests covering all 8 tasks, determinism, error handling, and hidden issue reveal
- Pydantic models with typed enums throughout

### Creativity & Novelty (10%)
- **Confidence calibration as a scoring component** — novel in OpenEnv environments
- **Hidden issue reveal mechanic** — mid-episode partial observability evolution
- **Explanation quality penalty** — incentivizes reasoning, not just action selection
- **Trajectory-aware observation feedback** — hints adapt based on agent's mistakes
- **Multi-agent failure diagnosis** — explains *why* agents fail, not just *that* they fail

---

## Limitations & Future Work

We acknowledge the following limitations honestly:

1. **No real code execution.** The environment pattern-matches action types against expected actions — it does not compile, run, or verify code changes. A future version could integrate a sandboxed execution engine to validate fixes.

2. **Fixed task set (8 tasks).** All tasks are hand-authored with static code snippets. There is no procedural generation, which limits generalization evaluation. Future work could template-generate tasks from real GitHub PRs.

3. **Action-type level matching.** Issue resolution matches on `action_type` (e.g., `fix_bug`), not on the specific fix applied. Two different bugs both expecting `fix_bug` are resolved in first-match order. Finer-grained action targeting would strengthen the evaluation signal.

4. **Toy-scale code snippets.** Code under review is 5–15 lines. Real code review operates on files with imports, classes, and cross-module dependencies. Scaling to realistic file sizes is a natural next step.

5. **Explanation evaluation is shallow.** The current penalty checks only explanation length (<10 characters), not semantic quality. A future version could use keyword overlap with issue descriptions for a lightweight quality signal without adding NLP dependencies.

These limitations are structural design choices made to prioritize **determinism, reproducibility, and deployment simplicity** within the hackathon scope. Each is addressable in future iterations without changing the core API contract.

---

## 11. Conclusion

CodeReviewBench provides a **structured, deterministic, extensible, and diagnostically rich** evaluation framework for AI agents in sequential decision-making settings. By grounding evaluation in realistic software engineering workflows, it measures capabilities that matter in practice — diagnostic reasoning, strategic prioritization, calibrated decision-making, and adaptive planning under partial observability.

The framework occupies the productive region of the difficulty spectrum — easy enough that progress is measurable, hard enough that no simple heuristic achieves a perfect score. Its five-component grading methodology provides actionable diagnostic feedback, identifying *specific* capability gaps rather than returning a single opaque metric.

CodeReviewBench is fully compatible with the OpenEnv specification and ready for integration into agent evaluation pipelines.

---

## 🧠 What This Environment Evaluates

This environment evaluates an agent's ability to:

- **Prioritize critical issues** in complex multi-failure systems (memory leaks, race conditions, query inefficiencies)
- **Adapt strategies based on reward feedback** — switching action types after negative outcomes
- **Handle hidden and dynamically revealed issues** — partial observability forces replanning mid-episode
- **Avoid repeated failure patterns** — recognizing that repeating the same failing action degrades the score
- **Balance multiple action types** — fixing bugs, flagging vulnerabilities, and optimizing performance in the right order

---

## 🔥 Why This Is NOT a Toy Benchmark

Unlike simple code-review environments, this system includes:

- **Production-style failure scenarios** — memory leaks causing OOM crashes, race conditions corrupting metrics, N+1 query patterns exhausting connection pools, and GDPR violations from PII logging
- **Hidden issues revealed only after partial resolution** — agents must earn access to deeper problems by first solving surface-level ones (`prod_errors_01`, `med_edge_01`)
- **Order-sensitive grading with penalties** — fixing a race condition before eliminating the unbounded cache that masks it is penalized as addressing a symptom before the root cause (-0.3 in score)
- **Multi-step dependencies between actions** — resolving issue A changes the code version and reveals the next layer of issue B, making each episode a genuine sequential decision problem

---

## ⚙️ RL Compatibility

This environment is designed to simulate reinforcement learning conditions:

- Each action produces a **reward signal** (positive for correct resolution, negative for wrong action, additional penalty for dangerous ordering)
- **State evolves dynamically** after each step — the code version changes as issues are resolved
- **Optimal behavior requires sequential decision-making** — a greedy single-step policy cannot achieve high scores on hard tasks

The environment is compatible with RL algorithms such as:

- **PPO** (Proximal Policy Optimization)
- **DQN** (Deep Q-Networks)
- **A2C** (Advantage Actor-Critic)

---

## 🤖 Agent Behavior

The current inference agent (`inference.py`) demonstrates RL-inspired behavior:

- **Epsilon-greedy exploration with Q-value biasing** — explores alternative actions, preferring those with higher cumulative episode reward
- **Strategy switching after negative rewards** — a failed action on step N guarantees a different action on step N+1
- **Action banning** — actions that fail twice in an episode are excluded from future selection within that episode
- **Hard-task structured strategy** — production incident and concurrency tasks follow a `flag_issue → fix_bug → optimize_code` sequence enforcing correct diagnostic order
- **Leave-as-is guard** — the agent never selects `leave_as_is` while unresolved issues remain

---

## 📊 Observed Performance

Measured across all 9 tasks using the adaptive LLM agent (`Qwen/Qwen2.5-72B-Instruct`):

| Difficulty | Example Tasks | Typical Score |
|------------|---------------|---------------|
| Easy | `easy_syntax_bug` | ~0.99 |
| Medium | `medium_logic_bug`, `security_review` | ~0.93–0.99 |
| Hard | `hard_multi_issue`, `concurrency_bug`, `production_incident_response` | ~0.70–0.88 |

Key behavioral observations:
- **Correct action ordering** (critical issues first) scores ~0.24 higher than wrong-order trajectories
- **Epsilon-greedy exploration** improves hard-task scores by surfacing non-obvious action sequences
- **Hidden issue reveal** works correctly — `prod_errors_01` and `med_edge_01` appear only after the first issue is resolved

---

## License

MIT
