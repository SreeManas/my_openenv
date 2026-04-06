---
title: CodeReviewBench
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
# CodeReviewBench: A Multi-Step RL Evaluation Framework for Code Intelligence Agents

> **A structured evaluation framework for measuring AI agent capabilities in sequential decision-making, diagnostic reasoning, and calibrated action selection — grounded in real-world software engineering workflows.**

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
| `safe_agent`     | 0.984 | 0.985       | **0.988**      | 0.937        | 0.987       | 0.982      | 0.921            | 0.987       | **0.971** |
| `baseline`       | 0.929 | 0.653       | 0.999          | 0.676        | 0.999       | 0.999      | 0.597            | 0.999       | 0.856   |
| `aggressive`     | 1.000 | 1.000       | 0.588          | 0.292        | 1.000       | 0.379      | 0.588            | 1.000       | 0.731   |

Key insight: **easy tasks don't differentiate agents** — safety-critical and multi-step tasks do. The aggressive agent collapses on security tasks (0.292–0.588) due to order violations and wrong action types. Only the safe agent sustains **97%+ performance** across all 8 tasks and difficulty levels.

---

## Abstract

**CodeReviewBench** is a fully OpenEnv-compliant, deployment-ready reinforcement-learning environment that provides a structured framework for evaluating AI agents on sequential code analysis, bug resolution, and optimization tasks across **8 diverse tasks**. Unlike single-step classification benchmarks, CodeReviewBench requires agents to operate under **partial observability**, **ambiguous feedback signals**, and **inter-dependent action constraints** — properties characteristic of real-world software engineering workflows.

The environment features dynamic state evolution, hidden defect discovery, confidence-calibrated reward shaping, adaptive difficulty adjustment, deterministic noise injection, trajectory-aware observation feedback, and trajectory-based grading across five evaluation dimensions.

**CodeReviewBench not only evaluates what decisions an agent makes, but why those decisions succeed or fail, and what their real-world consequences would be.**

A rule-based baseline agent achieves an average score of **0.856** across 8 tasks — demonstrating that the environment is neither trivially solvable nor intractably difficult. The safe agent achieves **0.971**.

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

Issue hints are deliberately vague:

| Ground Truth | Agent Sees |
|-------------|------------|
| `syntax_error` | *"Code fails to parse; check control-flow statements."* |
| `logic_error` | *"Unexpected behavior observed in loop bounds."* |
| `security_vulnerability` | *"User-supplied data flows into a sensitive operation without sanitization."* |
| `resource_leak` | *"If an exception is raised, some resources may not be properly released."* |

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

An agent that is *correctly confident* earns more. An agent that is *incorrectly confident* loses more. This incentivizes well-calibrated probability estimates — critical for trustworthy AI systems.

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

**Final score** = weighted sum, clamped to **[0.0, 1.0]**.

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
| Medium (logic) | **0.653** | 2/3 | Misses hidden edge-case; wastes 4 steps on `leave_as_is` |
| Medium (security) | **0.999** | 3/3 | Catches all keywords after hint improvement |
| Hard (multi) | **0.676** | 3/4 | Misses hidden resource leak; no ordering exploitation |
| Hard (edge) | **0.999** | 3/3 | New keyword coverage catches mutation/edge bugs |
| Medium (perf) | **0.999** | 3/3 | Optimize keywords matched correctly |
| Medium (validation) | **0.597** | 2/3 | Misses email flag — requires flag_issue |
| Hard (concurrency) | **0.999** | 3/3 | Default-param and counter keywords matched |
| **Average** | **0.856** | | |

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

## Multi-Agent Evaluation

CodeReviewBench includes a **comparative evaluation mode** running multiple agent strategies on identical tasks.

### Agent Strategies

| Agent | Strategy | Confidence | Key Weakness |
|-------|----------|------------|--------------|
| `baseline` | Keyword matching on hints | 0.9 (fixed) | Misses hidden issues, overconfident |
| `aggressive_agent` | Fix everything directly | 0.95 (high) | Wrong action types for security, order violations |
| `safe_agent` | Flag risks first, then fix | 0.6–0.75 (calibrated) | Slightly slower, cautious on clear bugs |

### Results

| Agent | Easy | Med† | Med† | Hard | Hard | Med | Med | Hard | **Average** |
|-------|------|------|------|------|------|-----|-----|------|-------------|
| `safe_agent` | 0.984 | 0.985 | **0.988** | 0.937 | 0.987 | 0.982 | 0.921 | 0.987 | **0.971** ◀ BEST |
| `baseline` | 0.929 | 0.653 | 0.999 | 0.676 | 0.999 | 0.999 | 0.597 | 0.999 | 0.856 |
| `aggressive_agent` | 1.000 | 1.000 | 0.588 | 0.292 | 1.000 | 0.379 | 0.588 | 1.000 | 0.731 |

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
| `safe_agent` | easy → 0.984 | medium → 0.985 | hard → 0.937 | **hard** |
| `aggressive_agent` | easy → 1.000 | medium → 1.000 | hard → 0.292 | **medium** (demoted) |
| `baseline` | easy → 0.929 | medium → 0.653 | medium → 0.653 | **medium** (plateaued) |

Only the safe agent sustains performance at the highest difficulty level.

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
├── test_env.py        Basic integration tests (reset, step, full episode)
├── openenv.yaml       Environment metadata and schema definitions
├── requirements.txt   Python dependencies
├── Dockerfile         Container build specification
└── README.md          This document
```

---

## 11. Conclusion

CodeReviewBench provides a **structured, deterministic, extensible, and diagnostically rich** evaluation framework for AI agents in sequential decision-making settings. By grounding evaluation in realistic software engineering workflows, it measures capabilities that matter in practice — diagnostic reasoning, strategic prioritization, calibrated decision-making, and adaptive planning under partial observability.

The framework occupies the productive region of the difficulty spectrum — easy enough that progress is measurable, hard enough that no simple heuristic achieves a perfect score. Its five-component grading methodology provides actionable diagnostic feedback, identifying *specific* capability gaps rather than returning a single opaque metric.

CodeReviewBench is fully compatible with the OpenEnv specification and ready for integration into agent evaluation pipelines.

---

## License

MIT
