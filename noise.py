"""
Deterministic noise injection for CodeReviewBench.

Adds controlled ambiguity to observation hints, simulating the imperfect
signals real developers encounter during code review. All perturbations
are fully deterministic — the same (seed, step_number, hint) triple
always produces the same output.

Toggle: set ENABLE_NOISE = False to disable entirely.
"""

import hashlib
from typing import Tuple

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

ENABLE_NOISE = True
NOISE_SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# Word-level perturbation table
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (original_fragment, noisy_replacement)
# Perturbations preserve semantic meaning but add ambiguity.
_PERTURBATION_TABLE = [
    ("fails to parse", "encounters a structural issue"),
    ("control-flow statements", "branching logic"),
    ("Unexpected behavior", "Possible anomaly detected"),
    ("unexpected behavior", "possible anomaly detected"),
    ("redundant comparisons", "suboptimal traversal pattern"),
    ("loop bounds", "iteration boundaries"),
    ("O(n²)", "quadratic complexity"),
    ("hashing", "alternative data structures"),
    ("more concisely", "with less overhead"),
    ("idiomatic Python", "cleaner patterns"),
    ("sensitive operation", "privileged operation"),
    ("sanitization", "input validation"),
    ("not be properly released", "remain allocated"),
    ("exception-safety", "error-handling"),
    ("defensive coding", "guard-clause"),
    ("downstream crashes", "downstream failures"),
    ("built-in language features", "standard library utilities"),
]


def _should_perturb(seed: int, step_number: int, hint: str) -> bool:
    """
    Deterministically decide whether to apply noise to this hint.

    Uses a hash of (seed, step, hint) to produce a stable boolean.
    Approximately 40% of hints get perturbed.
    """
    raw = f"{seed}:{step_number}:{hint[:30]}"
    h = int(hashlib.md5(raw.encode()).hexdigest(), 16)
    return (h % 10) < 4  # 40% perturbation rate


def inject_noise(hint: str, step_number: int) -> Tuple[str, bool]:
    """
    Apply deterministic noise to an observation hint.

    Args:
        hint: The original hint text.
        step_number: Current step number (used for determinism).

    Returns:
        Tuple of (possibly_modified_hint, was_noise_applied).
    """
    if not ENABLE_NOISE:
        return hint, False

    if not _should_perturb(NOISE_SEED, step_number, hint):
        return hint, False

    # Apply the FIRST matching perturbation (deterministic choice)
    for original, replacement in _PERTURBATION_TABLE:
        if original in hint:
            return hint.replace(original, replacement, 1), True

    # No matching perturbation — return unchanged
    return hint, False
