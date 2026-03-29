# CodeReviewBench: A Multi-Step Evaluation Environment for AI Agents in Sequential Code Analysis Under Uncertainty

> **A structured evaluation framework for measuring AI agent capabilities in sequential decision-making, diagnostic reasoning, and calibrated action selection — grounded in real-world software engineering workflows.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-v1.0-blue)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Abstract

**CodeReviewBench** is a fully OpenEnv-compliant, deployment-ready reinforcement-learning environment that provides a structured framework for evaluating AI agents on sequential code analysis, bug resolution, and optimization tasks. Unlike single-step classification benchmarks, CodeReviewBench requires agents to operate under **partial observability**, **ambiguous feedback signals**, and **inter-dependent action constraints** — properties characteristic of real-world software engineering workflows. The environment features dynamic state evolution, hidden defect discovery, confidence-calibrated reward shaping, adaptive difficulty adjustment, deterministic noise injection, and trajectory-based grading across five evaluation dimensions.

**CodeReviewBench not only evaluates what decisions an agent makes, but why those decisions succeed or fail, and what their real-world consequences would be.**

A rule-based baseline agent achieves an average score of **0.776**, demonstrating that the environment is neither trivially solvable nor intractably difficult — it occupies the challenging but informative region of the evaluation spectrum where meaningful capability differences between agents can be measured.

---

## What This Environment Evaluates

CodeReviewBench is designed to measure six core agent capabilities that are essential in professional knowledge work but underrepresented in existing benchmarks:

- **Sequential decision-making** — agents must plan and execute multi-step action sequences where early choices constrain later options
- **Reasoning under partial observability** — not all information is available upfront; agents must act on incomplete knowledge
- **Handling of hidden information** — new defects are revealed mid-episode, requiring adaptive replanning rather than fixed strategies
- **Trade-off management** — agents face genuine safety-vs-efficiency tensions with no single dominant strategy
- **Confidence calibration** — agents must express well-calibrated certainty; overconfidence on wrong decisions is penalized more than cautious uncertainty
- **Adaptive planning** — agents must revise their approach when the environment state changes unexpectedly
- **Robustness to noise** — deterministic perturbations add controlled ambiguity to observations, simulating imperfect real-world signals

---

## 1. Motivation

### The Evaluation Gap

Current AI agent benchmarks fall broadly into two categories: *knowledge assessments* (static question answering) and *interactive environments* (grid-worlds, Atari games, simplified simulations). Both fail to capture the characteristics of professional knowledge work:

| Property | Toy Environments | Real Developer Workflows |
|----------|-----------------|-------------------------|
| Decisions per episode | 1 | 5–50+ |
| Information availability | Complete | Partial, evolving |
| Feedback signals | Unambiguous | Noisy, contextual |
| Action dependencies | Independent | Sequence-sensitive |
| Risk management | Absent | Critical |
| Confidence awareness | Irrelevant | Essential |

A developer performing code review does not simply classify a snippet — they must *read*, *hypothesize*, *prioritize*, *act*, *observe the consequences*, and *adapt*. Errors in early steps cascade. Overconfidence leads to regressions. Ignoring security concerns in favor of quick fixes introduces systemic risk.

Existing evaluation frameworks — from fully observable grid-worlds to single-step code classification tasks — strip away precisely these dynamics. CodeReviewBench reintroduces them in a controlled, deterministic setting where agent capabilities can be measured with diagnostic precision.

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

1. **Initial review** — the developer reads the code and sees an initial set of reviewer comments (observations with ambiguous hints)
2. **Triage** — based on the hints and code context, they decide what to address first
3. **Iterative fixes** — each fix modifies the codebase; the updated code may reveal new issues (hidden defects surfacing)
4. **Completion** — the review ends when all issues are addressed or the review budget (max steps) is exhausted
5. **Post-hoc evaluation** — the entire review trajectory is scored on multiple quality dimensions

### Action Space

Agents select from four structured actions per step:

| Action | Semantics | Typical Targets |
|--------|-----------|-----------------|
| `fix_bug` | Correct a functional defect | Syntax errors, logic bugs, edge-case failures |
| `optimize_code` | Improve non-functional quality | Performance bottlenecks, style improvements |
| `flag_issue` | Escalate a systemic concern | Security vulnerabilities, resource leaks |
| `leave_as_is` | Decline to act | Used when no issues remain |

Each action includes a **free-text explanation** and a **confidence score** ∈ [0, 1], which directly influences the reward received.

### Observation Space

Each observation contains:

| Field | Description |
|-------|-------------|
| `code_snippet` | Current version of the code (evolves dynamically) |
| `issue_type` | An **ambiguous hint** about the primary issue — *not* a ground-truth label |
| `context` | Task-level background information |
| `remaining_issues` | IDs of *visible* unresolved issues (hidden issues excluded) |
| `step_number` | Current step index |
| `max_steps` | Episode budget |

> **Key design choice**: the `issue_type` field deliberately presents *hints* rather than labels. For example, a logic error is described as *"Unexpected behavior: the function produces correct output but performs redundant comparisons"* rather than *"logic_error"*. This forces agents to reason about symptom-to-cause mappings.

---

## 3. Key Design Features

### 3.1 Partial Observability

Not all defects are visible at episode start. **Hidden issues** are revealed only after the agent resolves at least one visible issue — modeling the real-world phenomenon where fixing one bug frequently exposes another.

- **Medium task**: a `None`-input crash is hidden until the loop-bounds bug is fixed
- **Hard task**: a database connection resource leak surfaces only after the SQL injection is patched

This prevents agents from planning a complete solution upfront and forces adaptive replanning.

### 3.2 Ambiguous Observations

Issue hints are deliberately vague. A rule-based agent cannot reliably map hints to actions without deeper contextual reasoning:

| Ground Truth | Agent Sees |
|-------------|------------|
| `syntax_error` | *"Code fails to parse; check control-flow statements."* |
| `logic_error` | *"Unexpected behavior observed in loop bounds."* |
| `security_vulnerability` | *"User-supplied data flows into a sensitive operation without sanitization."* |
| `resource_leak` | *"If an exception is raised, some resources may not be properly released."* |

### 3.3 Sequential Action Dependencies

The hard task enforces **order constraints**: fixing functional bugs before patching a security vulnerability incurs a penalty (−0.3 per violation). This reflects the principle that *unsafe code should not be executed*, and models the real-world priority of security over functionality during review.

Agents that resolve issues in the expected order receive a **sequence bonus** (+0.5 per correctly ordered action), evaluated via longest common subsequence (LCS) matching against the ideal sequence.

### 3.4 Safety–Efficiency Trade-offs

Agents face genuine trade-offs:

- **Flagging a security issue** is correct but delays functional fixes, potentially causing the agent to exceed the step budget
- **Optimizing code first** may implicitly resolve a logic bug (the set-based O(n) approach subsumes the redundant-comparison fix) but forfeits explicit resolution credit
- **Skipping minor issues** (e.g., style) preserves steps for critical bugs but reduces the completion score

There is no single dominant strategy — optimal behavior depends on the task structure and remaining budget.

### 3.5 Confidence-Calibrated Rewards

The `confidence` field is not decorative. It directly modulates rewards:

```
For correct actions:   reward = base × severity × (0.5 + 0.5 × confidence)
For incorrect actions: penalty = base × (0.5 + 0.5 × confidence)
```

An agent that is *correctly confident* earns more. An agent that is *incorrectly confident* loses more. An agent that expresses low confidence on uncertain decisions is penalized less for mistakes. This incentivizes well-calibrated probability estimates — a critical capability for trustworthy AI systems.

### 3.6 Dynamic State Evolution

After each action, the environment updates the `code_snippet` to reflect the fix that was applied. The agent observes the *changed* code in subsequent steps, requiring it to reason about a moving target. Code versions are pre-computed for all combinations of resolved issues, ensuring determinism.

---

## Design Philosophy

The following principles guided the design of CodeReviewBench:

- **Realism over simplicity** — the environment models properties of real developer workflows (ambiguity, hidden information, cascading consequences) even when this increases complexity
- **No trivial solutions** — ambiguous hints and hidden issues ensure that keyword matching, fixed heuristics, and single-step reasoning are insufficient for high scores
- **Confidence as a first-class signal** — agents must not only choose the right action but express appropriate certainty; this evaluates metacognitive capability, not just task performance
- **Trade-offs are genuine** — there is no universally optimal action sequence; agents must balance competing objectives (safety vs. speed, thoroughness vs. efficiency) under a limited step budget
- **Determinism and reproducibility** — despite the complexity, all grading is fully deterministic; identical agent trajectories always receive identical scores, enabling fair and repeatable evaluation

---

## 4. Task Design

### Difficulty Progression

| Property | Easy | Medium | Hard |
|----------|------|--------|------|
| Total issues | 2 | 3 | 4 |
| *Hidden* issues | 0 | 1 | 1 |
| Max steps | 4 | 6 | 8 |
| Order constraints | No | No | Yes (security-first) |
| Trade-offs | Minimal | Moderate | Significant |
| Issue types | Syntax, style | Logic, performance, edge-case | Security, logic, performance, resource leak |

### Easy — Syntax Error (`easy_syntax_bug`)

A function with a missing colon after an `if` statement, plus a minor style improvement. Straightforward for any agent capable of reading Python syntax.

### Medium — Logic + Performance + Hidden Edge Case (`medium_logic_bug`)

A duplicate-finder function with:
- An off-by-one loop (described as *"unexpected behavior"*, not *"logic error"*)
- An O(n²) algorithm (clearly signaled)
- A **hidden** `None`-input crash, revealed only after the first fix

The trade-off: optimizing first with a set-based approach implicitly fixes the loop bug, but the agent receives no explicit resolution credit and misses the sequence bonus.

### Hard — Multi-Issue with Ordering (`hard_multi_issue`)

A database query function with four defects:
1. **SQL injection** — must be flagged *first* (security-first gate; violations penalized at −0.3)
2. **Wrong return type** — returns `None` instead of `[]` on empty results
3. **Bubble sort** — O(n²) when `sorted()` is available
4. **Resource leak** — connection not wrapped in `try/finally` (**hidden**, revealed after first fix)

The ideal agent flags security, fixes the return type, flags the resource leak, then optimizes the sort — requiring four correctly ordered actions with appropriate confidence.

---

## 5. Reward Function

Rewards are computed per-step and accumulate across the trajectory:

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

This reward structure ensures that:
- Partial solutions receive partial credit
- Harmful decisions (wrong fixes with high confidence) are strongly penalized
- Efficient, well-ordered trajectories are consistently preferred over wasteful ones

---

## 6. Grading Methodology

### Trajectory-Based Scoring

The grader evaluates the **entire episode trajectory**, not individual actions. This captures the cumulative quality of the agent's decision-making process.

### Five Evaluation Components

| Component | Weight | Methodology | Range |
|-----------|--------|-------------|-------|
| **Completion** | 30% | `|resolved| / |total_issues|` | [0, 1] |
| **Efficiency** | 15% | `max(0, 1 − (steps − ideal) / max_steps)` | [0, 1] |
| **Safety** | 20% | `max(0, 1 − 0.3 × harmful − 0.1 × minor_wrong)` | [0, 1] |
| **Sequence** | 20% | LCS(expected, actual) / len(expected) | [0, 1] |
| **Calibration** | 15% | `1 − mean(cal_error²)` where cal_error is confidence-correctness mismatch | [0, 1] |

**Final score** = weighted sum, clamped to **[0.0, 1.0]**.

### Properties

- **Deterministic** — identical trajectories always produce identical scores
- **Sensitive** — different strategies yield meaningfully different grades
- **Decomposable** — each component diagnoses a specific capability

---

## 7. Baseline Agent Analysis

### Design

The baseline is a **rule-based keyword matcher** that:
- Scans the hint text for keywords (e.g., *"parse"* → `fix_bug`, *"sanitiz"* → `flag_issue`)
- Uses a fixed confidence of 0.9 for all matched actions
- Falls back to `leave_as_is` (confidence 0.5) when no keyword matches

### Results

| Task | Score | Completed | Failure Mode |
|------|-------|-----------|-------------|
| Easy | **0.999** | 2/2 | Near-perfect; simple keyword matching suffices |
| Medium | **0.653** | 2/3 | Misses hidden edge-case; wastes 4 steps on `leave_as_is`; calibration lost due to overconfidence |
| Hard | **0.676** | 3/4 | Misses hidden resource leak; wastes 5 steps; no exploitation of ordering |
| **Average** | **0.776** | | |

### Failure Analysis

1. **Hidden issues**: the baseline has no mechanism to anticipate or respond to newly revealed issues — it falls back to `leave_as_is` repeatedly, burning step budget
2. **Ambiguity**: the edge-case hint (*"What happens if the function receives None?"*) contains no keyword the baseline recognizes, so it is ignored
3. **Overconfidence**: a fixed 0.9 confidence means calibration loss when the baseline is wrong — a well-calibrated agent expressing 0.5 confidence on uncertain steps would score higher
4. **Inefficiency**: repeated `leave_as_is` actions are penalized, and excess steps reduce the efficiency component

These are not implementation bugs — they represent genuine capability gaps that a more sophisticated agent could address through reasoning, planning, and confidence estimation.

---

## 8. Why This Environment Is Challenging

A perfect-scoring agent must simultaneously:

- **Read and comprehend code** to infer true issue types from ambiguous textual hints
- **Prioritize safety concerns** over functional improvements, even when functional fixes are easier
- **Anticipate hidden defects** and allocate step budget accordingly
- **Calibrate confidence** — expressing certainty only when warranted
- **Minimize unnecessary actions** to maximize the efficiency score
- **Follow the optimal resolution sequence** to earn ordering bonuses and avoid penalties
- **Navigate trade-offs** where no single action is unambiguously best

No single heuristic achieves this. The environment requires integrated reasoning across multiple competencies — precisely the kind of capability that distinguishes frontier AI agents from simple classifiers.

---

## 9. Setup & Usage

### Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: `fastapi`, `uvicorn`, `pydantic`, `pyyaml`

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
| `POST` | `/compare_agents` | `{"task_id": "hard_multi_issue"}` (optional) | Compare all agents on one or all tasks |
| `POST` | `/analysis` | `{"task_id": "...", "agent": "baseline"}` | Run agent and return failure analysis + impact report |
| `POST` | `/adaptive_run` | `{"agent": "safe_agent", "num_rounds": 3}` | Run adaptive difficulty evaluation |

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

## Multi-Agent Evaluation

CodeReviewBench includes a **comparative evaluation mode** that runs multiple agent strategies on the same tasks under identical conditions.

### Agent Strategies

| Agent | Strategy | Confidence | Key Weakness |
|-------|----------|------------|--------------|
| `baseline` | Keyword matching on hints | 0.9 (fixed) | Misses hidden issues, overconfident |
| `aggressive_agent` | Fix everything directly | 0.95 (high) | Wrong action types for security, order violations |
| `safe_agent` | Flag risks first, then fix | 0.6–0.75 (calibrated) | Slightly slower, cautious on clear bugs |

### Results

| Agent | Easy | Medium | Hard | **Average** |
|-------|------|--------|------|-------------|
| `safe_agent` | 0.992 | 0.985 | **0.937** | **0.969** ◀ BEST |
| `baseline` | 0.999 | 0.653 | 0.676 | 0.776 |
| `aggressive_agent` | 1.000 | 1.000 | **0.292** | 0.764 |

### Insights

- **Easy tasks don't differentiate agents** — all three score ≥ 0.99
- **Medium tasks reveal hidden-issue handling** — the aggressive and safe agents detect the edge-case while the baseline cannot
- **Hard tasks expose strategic failures** — the aggressive agent uses `fix_bug` on a security vulnerability (wrong action type), triggers order-violation penalties, then loops `fix_bug` for 8 steps with repeat penalties, resolving only 1/4 issues
- **Calibration matters** — the safe agent's lower but honest confidence (0.6–0.75) produces better calibration scores than the baseline's fixed 0.9

### Usage

```bash
# Run all agents on all tasks
python3 multi_agent.py

# Via API — compare on a specific task
curl -X POST localhost:8000/compare_agents \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "hard_multi_issue"}'

# Via API — compare on all tasks
curl -X POST localhost:8000/compare_agents
```

---

## Failure Analysis Engine

Beyond scoring, CodeReviewBench explains **why** agents fail through automated failure mode detection:

| Failure Mode | Detection Rule | Example |
|-------------|----------------|--------|
| Missed issues | `resolved < total` | "Missed 3 issue(s): hard_sec_01, hard_perf_01, hard_resource_01" |
| Hidden issue blindness | Unresolved issue has `hidden: true` | "Failed to detect hidden issue(s) revealed mid-episode" |
| Overconfidence | High confidence + low calibration score | "Average confidence 0.95 but calibration score 0.21" |
| Action repetition | Same action type used consecutively | "Repeated same action 7 times consecutively" |
| Poor ordering | Sequence score < 0.7 | "Actions were not in the expected priority order" |
| Safety violations | Safety score < 0.7 | "Agent took harmful or incorrect actions" |
| Step waste | ≥3 `leave_as_is` with issues remaining | "Used leave_as_is 5 times with issues remaining" |

This provides **actionable diagnostics** — not just a number, but an explanation of what went wrong.

```bash
# Get failure analysis for a specific agent×task
curl -X POST localhost:8000/analysis \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "hard_multi_issue", "agent": "aggressive_agent"}'
```

---

## Real-World Impact Modeling

Each issue in the environment carries an `impact` field describing the real-world consequence of leaving it unresolved:

| Issue Type | Impact |
|-----------|--------|
| Syntax error | Runtime failure — code cannot execute |
| Logic error | Incorrect results — duplicates missed or double-counted |
| Performance issue | Slow execution — O(n²) runtime causes timeouts |
| Security vulnerability | Data breach risk — SQL injection enables unauthorized access |
| Resource leak | System instability — connections leak, causing exhaustion |
| Edge case | Application crash — unhandled None input causes TypeError |

The `/analysis` endpoint returns a full impact report with risk level assessment (`none`, `low`, `moderate`, `high`, `critical`).

---

## Example Insights

The `/compare_agents` endpoint generates natural-language insights automatically:

> 1. *Safe Agent achieves the highest average score (0.969), demonstrating superior overall strategy.*
> 2. *Baseline trails by 0.193 points — significant capability gap exists.*
> 3. *On medium_logic_bug, Aggressive Agent scores 1.000 vs Baseline at 0.653 — a 0.347-point gap revealing divergent strategies under task pressure.*
> 4. *Baseline missed 1 issue(s) on hard_multi_issue — struggles with partial observability or hidden issue detection.*
> 5. *Aggressive Agent missed 3 issue(s) on hard_multi_issue — struggles with partial observability or hidden issue detection.*

All insights are deterministic and rule-based — identical runs produce identical explanations.

---

## Adaptive Evaluation

CodeReviewBench includes a **dynamic difficulty adjustment** system that progressively tests agent capabilities:

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

CodeReviewBench applies **deterministic perturbations** to observation hints, simulating the imperfect signals real developers encounter:

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
├── tasks.py           Task definitions with hints, hidden issues, impacts, and order constraints
├── grader.py          Five-component trajectory grader with calibration scoring
├── environment.py     Multi-step environment with state evolution, reward shaping, and noise
├── noise.py           Deterministic noise injection engine (toggleable)
├── agents.py          Agent ABC + 3 strategy implementations (baseline, aggressive, safe)
├── analysis.py        Failure analysis, impact modeling, and insight generation
├── adaptive.py        Adaptive difficulty system with progressive task selection
├── multi_agent.py     Comparative evaluation runner with integrated analysis
├── baseline.py        Legacy baseline runner (standalone)
├── server.py          FastAPI server exposing the OpenEnv API (9 endpoints)
├── openenv.yaml       Environment metadata and schema definitions
├── requirements.txt   Python dependencies
├── Dockerfile         Container build specification
└── README.md          This document
```

---

## 11. Conclusion

CodeReviewBench provides a **structured, deterministic, extensible, and diagnostically rich** evaluation framework for AI agents operating in sequential decision-making settings. By grounding evaluation in a realistic software engineering workflow, it measures capabilities that matter in practice — diagnostic reasoning, strategic prioritization, calibrated decision-making, and adaptive planning under partial observability — capabilities that single-step benchmarks and fully observable environments fundamentally cannot assess.

The framework is designed to occupy the productive region of the difficulty spectrum — easy enough that progress is measurable, hard enough that no simple heuristic achieves a perfect score. Its five-component grading methodology provides actionable diagnostic feedback, identifying *specific* capability gaps (e.g., poor calibration, inefficient sequencing) rather than returning a single opaque metric. CodeReviewBench is fully compatible with the OpenEnv specification and ready for integration into agent evaluation pipelines. This makes it a practical foundation for evaluating next-generation AI agents in real-world decision-making scenarios.

---

## License

MIT
