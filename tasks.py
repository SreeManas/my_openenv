"""
Task definitions for the AI Code Review environment.

Each task specifies:
  - A list of issues embedded in the code (some may be hidden / not obvious)
  - Multiple code versions keyed by the *set of resolved issues*
  - An expected resolution sequence (for ordering bonuses)
  - Ambiguous "hint" strings (NOT exact labels) for observations
  - Trade-off metadata controlling inter-issue dependencies

All tasks are fully deterministic.
"""

from typing import Dict, List, Any

# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a frozen-set key from a list of resolved issue IDs
# ──────────────────────────────────────────────────────────────────────────────

def _key(*resolved_ids: str) -> frozenset:
    return frozenset(resolved_ids)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — EASY: Simple syntax bug  (1 main issue + 1 minor style issue)
# Baseline should score ≈ 1.0  (simple enough for keyword matching)
# ══════════════════════════════════════════════════════════════════════════════

EASY_CODE_VERSIONS: Dict[frozenset, str] = {
    _key(): (
        "def is_even(n):\n"
        "    if n % 2 == 0\n"
        "        return True\n"
        "    return False\n"
    ),
    _key("easy_syntax_01"): (
        "def is_even(n):\n"
        "    if n % 2 == 0:\n"
        "        return True\n"
        "    return False\n"
    ),
    # Both syntax + style fixed (ideal)
    _key("easy_syntax_01", "easy_style_01"): (
        "def is_even(n):\n"
        "    return n % 2 == 0\n"
    ),
    # Only style fixed (doesn't even parse, so same as original)
    _key("easy_style_01"): (
        "def is_even(n):\n"
        "    if n % 2 == 0\n"
        "        return True\n"
        "    return False\n"
    ),
}

EASY_TASK: Dict[str, Any] = {
    "task_id": "easy_syntax_bug",
    "difficulty": "easy",
    "context": (
        "This Python function checks whether a number is even. "
        "It was written by a junior developer and may contain simple mistakes."
    ),
    "issues": [
        {
            "id": "easy_syntax_01",
            "type": "syntax_error",
            # Ambiguous hint — agent sees this, not the explicit type
            "hint": "Code fails to parse; check control-flow statements.",
            "description": "Missing colon after the if-condition.",
            "expected_action": "fix_bug",
            "severity": 1.0,   # Critical — code won't run
            "impact": "Runtime failure — code cannot execute at all",
        },
        {
            "id": "easy_style_01",
            "type": "style",
            "hint": (
                "The function works but could be expressed more concisely. "
                "Consider whether the structure is idiomatic Python."
            ),
            "description": "Verbose boolean return; can simplify to `return n % 2 == 0`.",
            "expected_action": "optimize_code",
            "severity": 0.3,   # Low — cosmetic / optional
            "impact": "Reduced code readability and maintainability",
        },
    ],
    "expected_sequence": ["fix_bug", "optimize_code"],
    "code_versions": EASY_CODE_VERSIONS,
    "max_steps": 4,
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — MEDIUM: Logic bug + performance + hidden edge-case
# The baseline should score ≈ 0.7–0.85 because:
#   - The hint is ambiguous ("unexpected behavior" not "logic error")
#   - There's a hidden issue the baseline won't detect
#   - Trade-off: fixing perf first changes the code structure,
#     making the logic bug description less clear
# ══════════════════════════════════════════════════════════════════════════════

MEDIUM_CODE_VERSIONS: Dict[frozenset, str] = {
    # Initial: off-by-one + O(n²) + silent None-handling bug
    _key(): (
        "def find_duplicates(items):\n"
        '    """Return list of duplicate values."""\n'
        "    duplicates = []\n"
        "    for i in range(len(items)):\n"
        "        for j in range(len(items)):  # redundant comparisons\n"
        "            if i != j and items[i] == items[j]:\n"
        "                if items[i] not in duplicates:\n"
        "                    duplicates.append(items[i])\n"
        "    return duplicates\n"
    ),
    # Logic fixed only
    _key("med_logic_01"): (
        "def find_duplicates(items):\n"
        '    """Return list of duplicate values."""\n'
        "    duplicates = []\n"
        "    for i in range(len(items)):\n"
        "        for j in range(i + 1, len(items)):\n"
        "            if items[i] == items[j]:\n"
        "                if items[i] not in duplicates:\n"
        "                    duplicates.append(items[i])\n"
        "    return duplicates\n"
    ),
    # Performance fixed only
    _key("med_perf_01"): (
        "def find_duplicates(items):\n"
        '    """Return list of duplicate values."""\n'
        "    seen = set()\n"
        "    duplicates = set()\n"
        "    for item in items:\n"
        "        if item in seen:\n"
        "            duplicates.add(item)\n"
        "        seen.add(item)\n"
        "    return list(duplicates)\n"
    ),
    # Edge-case fixed only (null guard)
    _key("med_edge_01"): (
        "def find_duplicates(items):\n"
        '    """Return list of duplicate values."""\n'
        "    if items is None:\n"
        "        return []\n"
        "    duplicates = []\n"
        "    for i in range(len(items)):\n"
        "        for j in range(len(items)):\n"
        "            if i != j and items[i] == items[j]:\n"
        "                if items[i] not in duplicates:\n"
        "                    duplicates.append(items[i])\n"
        "    return duplicates\n"
    ),
    # Logic + perf
    _key("med_logic_01", "med_perf_01"): (
        "def find_duplicates(items):\n"
        '    """Return list of duplicate values."""\n'
        "    seen = set()\n"
        "    duplicates = set()\n"
        "    for item in items:\n"
        "        if item in seen:\n"
        "            duplicates.add(item)\n"
        "        seen.add(item)\n"
        "    return list(duplicates)\n"
    ),
    # Logic + edge
    _key("med_logic_01", "med_edge_01"): (
        "def find_duplicates(items):\n"
        '    """Return list of duplicate values."""\n'
        "    if items is None:\n"
        "        return []\n"
        "    duplicates = []\n"
        "    for i in range(len(items)):\n"
        "        for j in range(i + 1, len(items)):\n"
        "            if items[i] == items[j]:\n"
        "                if items[i] not in duplicates:\n"
        "                    duplicates.append(items[i])\n"
        "    return duplicates\n"
    ),
    # Perf + edge
    _key("med_perf_01", "med_edge_01"): (
        "def find_duplicates(items):\n"
        '    """Return list of duplicate values."""\n'
        "    if items is None:\n"
        "        return []\n"
        "    seen = set()\n"
        "    duplicates = set()\n"
        "    for item in items:\n"
        "        if item in seen:\n"
        "            duplicates.add(item)\n"
        "        seen.add(item)\n"
        "    return list(duplicates)\n"
    ),
    # ALL resolved (ideal)
    _key("med_logic_01", "med_perf_01", "med_edge_01"): (
        "def find_duplicates(items):\n"
        '    """Return list of duplicate values."""\n'
        "    if items is None:\n"
        "        return []\n"
        "    seen = set()\n"
        "    duplicates = set()\n"
        "    for item in items:\n"
        "        if item in seen:\n"
        "            duplicates.add(item)\n"
        "        seen.add(item)\n"
        "    return list(duplicates)\n"
    ),
}

MEDIUM_TASK: Dict[str, Any] = {
    "task_id": "medium_logic_bug",
    "difficulty": "medium",
    "context": (
        "This function is supposed to find duplicate values in a list. "
        "During testing, unexpected behavior was observed: it runs slowly "
        "on large inputs and reviewers noticed redundant comparisons "
        "suggesting incorrect loop bounds. There may also be robustness "
        "concerns with unexpected input types."
    ),
    "issues": [
        {
            "id": "med_logic_01",
            "type": "logic_error",
            # Ambiguous hint — doesn't say "logic error"
            "hint": (
                "Unexpected behavior: the function produces correct output "
                "but performs redundant comparisons due to incorrect loop "
                "bounds. The inner loop iterates more than necessary — "
                "this is a logic defect, not just a performance issue."
            ),
            "description": (
                "Inner loop starts at 0 instead of i+1, "
                "causing redundant comparisons."
            ),
            "expected_action": "fix_bug",
            "severity": 0.8,
            "impact": "Incorrect results — duplicates may be missed or double-counted in downstream processing",
        },
        {
            "id": "med_perf_01",
            "type": "performance",
            # Somewhat clear but the fix conflicts with logic-bug fix
            "hint": (
                "Runs in O(n²) time. A more efficient approach using "
                "hashing could bring this down to O(n)."
            ),
            "description": (
                "O(n²) nested-loop approach; should use a set-based O(n) "
                "algorithm."
            ),
            "expected_action": "optimize_code",
            "severity": 0.7,
            "impact": "Slow execution — O(n²) runtime causes timeouts on large datasets",
        },
        {
            "id": "med_edge_01",
            "type": "edge_case",
            # Hidden issue — hint is vague, baseline won't match it
            "hint": (
                "What happens if the function receives None instead of a "
                "list? Consider defensive coding practices."
            ),
            "description": "Crashes with TypeError on None input; needs a null guard.",
            "expected_action": "fix_bug",
            "severity": 0.5,
            "hidden": True,  # Only revealed after step 1 resolves something
            "impact": "Application crash — unhandled None input causes TypeError in production",
        },
    ],
    # Ideal order: fix logic first, then optimize, then handle edge case
    "expected_sequence": ["fix_bug", "optimize_code", "fix_bug"],
    "code_versions": MEDIUM_CODE_VERSIONS,
    "max_steps": 6,
    # Trade-off metadata: if you optimize first, the logic bug description
    # becomes less relevant (the optimization subsumes it), but you miss
    # sequence bonus
    "tradeoffs": {
        "optimize_before_fix": (
            "Optimizing first (set-based approach) implicitly fixes the "
            "redundant-comparison issue but the agent doesn't get credit "
            "for explicitly fixing the logic bug."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — HARD: Security + performance + logic + resource leak
# Baseline should score ≈ 0.5–0.7 because:
#   - Issue hints are deliberately ambiguous
#   - There are 4 issues (not 3) — the 4th is hidden
#   - WRONG ORDER incurs an order-violation penalty
#     (fixing logic before security is penalized because the code is
#      unsafe to even run until the injection is patched)
#   - Conflicting actions: flagging security is correct but delays
#     functional fixes
# ══════════════════════════════════════════════════════════════════════════════

HARD_CODE_VERSIONS: Dict[frozenset, str] = {
    # Initial
    _key(): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    query = f\"SELECT * FROM orders WHERE user = '{username}'\"\n"
        "    cursor = conn.execute(query)\n"
        "    rows = cursor.fetchall()\n"
        "    conn.close()\n"
        "\n"
        "    # Sort by date column (index 2)\n"
        "    for i in range(len(rows)):\n"
        "        for j in range(len(rows) - 1):\n"
        "            if rows[j][2] > rows[j + 1][2]:\n"
        "                rows[j], rows[j + 1] = rows[j + 1], rows[j]\n"
        "\n"
        "    return rows if rows else None\n"
    ),
    # Sec only
    _key("hard_sec_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    query = \"SELECT * FROM orders WHERE user = ?\"\n"
        "    cursor = conn.execute(query, (username,))\n"
        "    rows = cursor.fetchall()\n"
        "    conn.close()\n"
        "\n"
        "    for i in range(len(rows)):\n"
        "        for j in range(len(rows) - 1):\n"
        "            if rows[j][2] > rows[j + 1][2]:\n"
        "                rows[j], rows[j + 1] = rows[j + 1], rows[j]\n"
        "\n"
        "    return rows if rows else None\n"
    ),
    # Logic only
    _key("hard_logic_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    query = f\"SELECT * FROM orders WHERE user = '{username}'\"\n"
        "    cursor = conn.execute(query)\n"
        "    rows = cursor.fetchall()\n"
        "    conn.close()\n"
        "\n"
        "    for i in range(len(rows)):\n"
        "        for j in range(len(rows) - 1):\n"
        "            if rows[j][2] > rows[j + 1][2]:\n"
        "                rows[j], rows[j + 1] = rows[j + 1], rows[j]\n"
        "\n"
        "    return rows if rows else []\n"
    ),
    # Perf only
    _key("hard_perf_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    query = f\"SELECT * FROM orders WHERE user = '{username}'\"\n"
        "    cursor = conn.execute(query)\n"
        "    rows = cursor.fetchall()\n"
        "    conn.close()\n"
        "\n"
        "    rows = sorted(rows, key=lambda r: r[2])\n"
        "\n"
        "    return rows if rows else None\n"
    ),
    # Resource leak only
    _key("hard_resource_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    try:\n"
        "        query = f\"SELECT * FROM orders WHERE user = '{username}'\"\n"
        "        cursor = conn.execute(query)\n"
        "        rows = cursor.fetchall()\n"
        "    finally:\n"
        "        conn.close()\n"
        "\n"
        "    for i in range(len(rows)):\n"
        "        for j in range(len(rows) - 1):\n"
        "            if rows[j][2] > rows[j + 1][2]:\n"
        "                rows[j], rows[j + 1] = rows[j + 1], rows[j]\n"
        "\n"
        "    return rows if rows else None\n"
    ),
    # Sec + logic
    _key("hard_sec_01", "hard_logic_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    query = \"SELECT * FROM orders WHERE user = ?\"\n"
        "    cursor = conn.execute(query, (username,))\n"
        "    rows = cursor.fetchall()\n"
        "    conn.close()\n"
        "\n"
        "    for i in range(len(rows)):\n"
        "        for j in range(len(rows) - 1):\n"
        "            if rows[j][2] > rows[j + 1][2]:\n"
        "                rows[j], rows[j + 1] = rows[j + 1], rows[j]\n"
        "\n"
        "    return rows if rows else []\n"
    ),
    # Sec + perf
    _key("hard_sec_01", "hard_perf_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    query = \"SELECT * FROM orders WHERE user = ?\"\n"
        "    cursor = conn.execute(query, (username,))\n"
        "    rows = cursor.fetchall()\n"
        "    conn.close()\n"
        "\n"
        "    rows = sorted(rows, key=lambda r: r[2])\n"
        "\n"
        "    return rows if rows else None\n"
    ),
    # Sec + resource
    _key("hard_sec_01", "hard_resource_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    try:\n"
        "        query = \"SELECT * FROM orders WHERE user = ?\"\n"
        "        cursor = conn.execute(query, (username,))\n"
        "        rows = cursor.fetchall()\n"
        "    finally:\n"
        "        conn.close()\n"
        "\n"
        "    for i in range(len(rows)):\n"
        "        for j in range(len(rows) - 1):\n"
        "            if rows[j][2] > rows[j + 1][2]:\n"
        "                rows[j], rows[j + 1] = rows[j + 1], rows[j]\n"
        "\n"
        "    return rows if rows else None\n"
    ),
    # Sec + logic + perf
    _key("hard_sec_01", "hard_logic_01", "hard_perf_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    query = \"SELECT * FROM orders WHERE user = ?\"\n"
        "    cursor = conn.execute(query, (username,))\n"
        "    rows = cursor.fetchall()\n"
        "    conn.close()\n"
        "\n"
        "    rows = sorted(rows, key=lambda r: r[2])\n"
        "\n"
        "    return rows if rows else []\n"
    ),
    # Sec + logic + resource
    _key("hard_sec_01", "hard_logic_01", "hard_resource_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    try:\n"
        "        query = \"SELECT * FROM orders WHERE user = ?\"\n"
        "        cursor = conn.execute(query, (username,))\n"
        "        rows = cursor.fetchall()\n"
        "    finally:\n"
        "        conn.close()\n"
        "\n"
        "    for i in range(len(rows)):\n"
        "        for j in range(len(rows) - 1):\n"
        "            if rows[j][2] > rows[j + 1][2]:\n"
        "                rows[j], rows[j + 1] = rows[j + 1], rows[j]\n"
        "\n"
        "    return rows if rows else []\n"
    ),
    # Sec + perf + resource
    _key("hard_sec_01", "hard_perf_01", "hard_resource_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    try:\n"
        "        query = \"SELECT * FROM orders WHERE user = ?\"\n"
        "        cursor = conn.execute(query, (username,))\n"
        "        rows = cursor.fetchall()\n"
        "    finally:\n"
        "        conn.close()\n"
        "\n"
        "    rows = sorted(rows, key=lambda r: r[2])\n"
        "\n"
        "    return rows if rows else None\n"
    ),
    # Logic + perf
    _key("hard_logic_01", "hard_perf_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    query = f\"SELECT * FROM orders WHERE user = '{username}'\"\n"
        "    cursor = conn.execute(query)\n"
        "    rows = cursor.fetchall()\n"
        "    conn.close()\n"
        "\n"
        "    rows = sorted(rows, key=lambda r: r[2])\n"
        "\n"
        "    return rows if rows else []\n"
    ),
    # Logic + resource
    _key("hard_logic_01", "hard_resource_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    try:\n"
        "        query = f\"SELECT * FROM orders WHERE user = '{username}'\"\n"
        "        cursor = conn.execute(query)\n"
        "        rows = cursor.fetchall()\n"
        "    finally:\n"
        "        conn.close()\n"
        "\n"
        "    for i in range(len(rows)):\n"
        "        for j in range(len(rows) - 1):\n"
        "            if rows[j][2] > rows[j + 1][2]:\n"
        "                rows[j], rows[j + 1] = rows[j + 1], rows[j]\n"
        "\n"
        "    return rows if rows else []\n"
    ),
    # Perf + resource
    _key("hard_perf_01", "hard_resource_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    try:\n"
        "        query = f\"SELECT * FROM orders WHERE user = '{username}'\"\n"
        "        cursor = conn.execute(query)\n"
        "        rows = cursor.fetchall()\n"
        "    finally:\n"
        "        conn.close()\n"
        "\n"
        "    rows = sorted(rows, key=lambda r: r[2])\n"
        "\n"
        "    return rows if rows else None\n"
    ),
    # Logic + perf + resource
    _key("hard_logic_01", "hard_perf_01", "hard_resource_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    try:\n"
        "        query = f\"SELECT * FROM orders WHERE user = '{username}'\"\n"
        "        cursor = conn.execute(query)\n"
        "        rows = cursor.fetchall()\n"
        "    finally:\n"
        "        conn.close()\n"
        "\n"
        "    rows = sorted(rows, key=lambda r: r[2])\n"
        "\n"
        "    return rows if rows else []\n"
    ),
    # Sec + logic + perf + resource (ALL — ideal)
    _key("hard_sec_01", "hard_logic_01", "hard_perf_01", "hard_resource_01"): (
        "import sqlite3\n"
        "\n"
        "def get_user_orders(db_path, username):\n"
        '    """Fetch orders for a given user, sorted by date."""\n'
        "    conn = sqlite3.connect(db_path)\n"
        "    try:\n"
        "        query = \"SELECT * FROM orders WHERE user = ?\"\n"
        "        cursor = conn.execute(query, (username,))\n"
        "        rows = cursor.fetchall()\n"
        "    finally:\n"
        "        conn.close()\n"
        "\n"
        "    rows = sorted(rows, key=lambda r: r[2])\n"
        "\n"
        "    return rows if rows else []\n"
    ),
}

HARD_TASK: Dict[str, Any] = {
    "task_id": "hard_multi_issue",
    "difficulty": "hard",
    "context": (
        "This function queries a SQLite database for a user's orders and "
        "sorts them by date. Reviewers have raised concerns about "
        "robustness, safety, and performance. Some issues may interact "
        "with each other — prioritize critical risks first."
    ),
    "issues": [
        {
            "id": "hard_sec_01",
            "type": "security_vulnerability",
            "hint": (
                "User-supplied data appears to flow into a sensitive "
                "operation without sanitization. Investigate the query "
                "construction."
            ),
            "description": (
                "SQL injection: user input is interpolated directly "
                "into the query string."
            ),
            "expected_action": "flag_issue",
            "severity": 1.0,
            # Order constraint: MUST be resolved before other fixes
            # to avoid running unsafe code
            "priority_gate": True,
            "impact": "Data breach risk — attacker can exfiltrate or modify database contents via crafted input",
        },
        {
            "id": "hard_logic_01",
            "type": "logic_error",
            "hint": (
                "Callers of this function sometimes receive an unexpected "
                "return type that causes downstream crashes. Check the "
                "return paths."
            ),
            "description": (
                "Returns None instead of an empty list when no orders "
                "are found, which breaks downstream callers."
            ),
            "expected_action": "fix_bug",
            "severity": 0.7,
            "impact": "Downstream crashes — callers receive None instead of empty list, causing AttributeError",
        },
        {
            "id": "hard_perf_01",
            "type": "performance",
            "hint": (
                "The sorting step uses a pattern that doesn't scale well. "
                "Consider using built-in language features."
            ),
            "description": (
                "Uses O(n²) bubble sort instead of Python's built-in "
                "sorted() which is O(n log n)."
            ),
            "expected_action": "optimize_code",
            "severity": 0.5,
            "impact": "Performance degradation — O(n²) sorting causes latency spikes on large order histories",
        },
        {
            "id": "hard_resource_01",
            "type": "resource_leak",
            "hint": (
                "If an exception is raised during execution, some "
                "resources may not be properly released. Consider "
                "exception-safety patterns."
            ),
            "description": (
                "Database connection is not wrapped in try/finally; "
                "connection leaks on exceptions."
            ),
            "expected_action": "flag_issue",
            "severity": 0.6,
            "hidden": True,  # Revealed after the first issue is resolved
            "impact": "System instability — database connections leak on exceptions, causing resource exhaustion over time",
        },
    ],
    # Ideal order: security first, then logic, then resource, then perf
    "expected_sequence": ["flag_issue", "fix_bug", "flag_issue", "optimize_code"],
    "code_versions": HARD_CODE_VERSIONS,
    "max_steps": 8,
    # Wrong-order penalty: resolving non-security issues before security
    # incurs a penalty because the code is unsafe to run
    "order_constraints": {
        "hard_sec_01": {
            "must_before": ["hard_logic_01", "hard_perf_01", "hard_resource_01"],
            "violation_penalty": -0.3,
            "reason": (
                "Fixing functional issues on code with a SQL injection "
                "vulnerability is unsafe — the code shouldn't be run."
            ),
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "easy_syntax_bug": EASY_TASK,
    "medium_logic_bug": MEDIUM_TASK,
    "hard_multi_issue": HARD_TASK,
}


def get_task(task_id: str) -> Dict[str, Any]:
    """Retrieve a task definition by ID. Raises KeyError if not found."""
    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[Dict[str, str]]:
    """Return lightweight summaries of all available tasks."""
    return [
        {
            "task_id": t["task_id"],
            "difficulty": t["difficulty"],
            "num_issues": str(len(t["issues"])),
            "max_steps": str(t["max_steps"]),
            "context": t["context"],
        }
        for t in TASK_REGISTRY.values()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — MEDIUM: Security variant (unsafe password reset + logging leak)
# Baseline should score ≈ 0.70–0.80 because:
#   - Main issue is a security flag (not a keyword-accessible fix_bug)
#   - Hidden logging issue revealed after security flag
#   - Hints are deliberately phrased as performance/UX concerns
# ══════════════════════════════════════════════════════════════════════════════

SEC_CODE_VERSIONS: Dict[frozenset, str] = {
    # Initial: predictable token + password logged in plaintext
    _key(): (
        "import time\n"
        "\n"
        "def generate_reset_token(user_email):\n"
        '    """Generate a password reset token for the user."""\n'
        "    token = str(int(time.time()))  # Sequential, predictable\n"
        "    print(f'Reset token for {user_email}: {token}')  # Logged\n"
        "    return token\n"
        "\n"
        "def validate_reset_token(token, max_age_seconds=3600):\n"
        '    """Return True if the token is still valid."""\n'
        "    issued_at = int(token)  # Assumes token is a timestamp\n"
        "    return (time.time() - issued_at) < max_age_seconds\n"
    ),
    # Security flagged only (predictable token issue acknowledged)
    _key("sec_token_01"): (
        "import time\n"
        "import secrets\n"
        "\n"
        "def generate_reset_token(user_email):\n"
        '    """Generate a password reset token for the user."""\n'
        "    token = secrets.token_urlsafe(32)\n"
        "    print(f'Reset token for {user_email}: {token}')  # Still logged\n"
        "    return token\n"
        "\n"
        "def validate_reset_token(token, expiry_time):\n"
        '    """Return True if the token is still valid."""\n'
        "    return time.time() < expiry_time\n"
    ),
    # Logging issue fixed only
    _key("sec_log_01"): (
        "import time\n"
        "\n"
        "def generate_reset_token(user_email):\n"
        '    """Generate a password reset token for the user."""\n'
        "    token = str(int(time.time()))\n"
        "    # Sensitive token not logged\n"
        "    return token\n"
        "\n"
        "def validate_reset_token(token, max_age_seconds=3600):\n"
        '    """Return True if the token is still valid."""\n'
        "    issued_at = int(token)\n"
        "    return (time.time() - issued_at) < max_age_seconds\n"
    ),
    # Validation gap fixed only
    _key("sec_validate_01"): (
        "import time\n"
        "\n"
        "def generate_reset_token(user_email):\n"
        '    """Generate a password reset token for the user."""\n'
        "    token = str(int(time.time()))\n"
        "    print(f'Reset token for {user_email}: {token}')\n"
        "    return token\n"
        "\n"
        "def validate_reset_token(token, max_age_seconds=3600):\n"
        '    """Return True if the token is still valid."""\n'
        "    try:\n"
        "        issued_at = int(token)\n"
        "    except (ValueError, TypeError):\n"
        "        return False\n"
        "    return (time.time() - issued_at) < max_age_seconds\n"
    ),
    # Security + logging fixed
    _key("sec_token_01", "sec_log_01"): (
        "import time\n"
        "import secrets\n"
        "\n"
        "def generate_reset_token(user_email):\n"
        '    """Generate a password reset token for the user."""\n'
        "    token = secrets.token_urlsafe(32)\n"
        "    return token\n"
        "\n"
        "def validate_reset_token(token, expiry_time):\n"
        '    """Return True if the token is still valid."""\n'
        "    return time.time() < expiry_time\n"
    ),
    # Security + validation fixed
    _key("sec_token_01", "sec_validate_01"): (
        "import time\n"
        "import secrets\n"
        "\n"
        "def generate_reset_token(user_email):\n"
        '    """Generate a password reset token for the user."""\n'
        "    token = secrets.token_urlsafe(32)\n"
        "    print(f'Reset token for {user_email}: {token}')\n"
        "    return token\n"
        "\n"
        "def validate_reset_token(token, expiry_time):\n"
        '    """Return True if the token is still valid."""\n'
        "    if not isinstance(token, str) or not token:\n"
        "        return False\n"
        "    return time.time() < expiry_time\n"
    ),
    # All resolved (ideal)
    _key("sec_token_01", "sec_log_01", "sec_validate_01"): (
        "import time\n"
        "import secrets\n"
        "\n"
        "def generate_reset_token(user_email):\n"
        '    """Generate a password reset token for the user."""\n'
        "    token = secrets.token_urlsafe(32)\n"
        "    return token\n"
        "\n"
        "def validate_reset_token(token, expiry_time):\n"
        '    """Return True if the token is still valid."""\n'
        "    if not isinstance(token, str) or not token:\n"
        "        return False\n"
        "    return time.time() < expiry_time\n"
    ),
}

SEC_TASK: Dict[str, Any] = {
    "task_id": "medium_security_variant",
    "difficulty": "medium",
    "context": (
        "This module handles password reset token generation and validation. "
        "A security review flagged concerns about the token lifecycle. "
        "Reviewers noted the tokens may be predictable and some sensitive "
        "information may be exposed through application logs."
    ),
    "issues": [
        {
            "id": "sec_token_01",
            "type": "security_vulnerability",
            "hint": (
                "The token generation method may be predictable or reused — "
                "an attacker could guess valid tokens. This is a security "
                "concern that should be flagged for review."
            ),
            "description": (
                "Token is derived from the current Unix timestamp — "
                "predictable and trivially guessable by an attacker."
            ),
            "expected_action": "flag_issue",
            "severity": 1.0,
            "priority_gate": True,
            "impact": (
                "Account takeover — attacker can forge valid reset tokens "
                "and gain unauthorised access to any account"
            ),
        },
        {
            "id": "sec_log_01",
            "type": "security_vulnerability",
            "hint": (
                "Sensitive token data is printed to application logs during "
                "normal operation. This credential exposure should be fixed "
                "by removing the print statement."
            ),
            "description": (
                "Reset token is printed to stdout, which lands in "
                "application logs and is visible to anyone with log access."
            ),
            "expected_action": "fix_bug",
            "severity": 0.8,
            "hidden": True,  # Revealed after sec_token_01 is resolved
            "impact": (
                "Credential leakage — tokens visible in logs allow "
                "internal attackers to hijack reset flows"
            ),
        },
        {
            "id": "sec_validate_01",
            "type": "edge_case",
            "hint": (
                "The validation function assumes well-formed input. "
                "What happens if the token is empty or malformed?"
            ),
            "description": (
                "validate_reset_token raises ValueError/TypeError on "
                "non-numeric token strings; no input guard present."
            ),
            "expected_action": "fix_bug",
            "severity": 0.5,
            "impact": (
                "Application crash — malformed tokens cause unhandled "
                "exceptions that expose internal stack traces"
            ),
        },
    ],
    # Security must be flagged before the logging fix (unsafe to keep tokens)
    "expected_sequence": ["flag_issue", "fix_bug", "fix_bug"],
    "code_versions": SEC_CODE_VERSIONS,
    "max_steps": 6,
    "order_constraints": {
        "sec_token_01": {
            "must_before": ["sec_log_01", "sec_validate_01"],
            "violation_penalty": -0.3,
            "reason": (
                "Fixing secondary issues on a broken token scheme entrenches "
                "the insecure design — the root cause must be flagged first."
            ),
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — HARD: Tricky logic bug + silent data corruption + hidden edge case
# Baseline should score ≈ 0.50–0.65 because:
#   - Off-by-one is subtle and the hint describes it as a calculation issue
#   - Data corruption is a silent wrong-value bug (no crash)
#   - Hidden edge case only visible after the logic is corrected
#   - Order matters: fixing data mutation before the logic bug makes it worse
# ══════════════════════════════════════════════════════════════════════════════

EDGE_CODE_VERSIONS: Dict[frozenset, str] = {
    # Initial: off-by-one + mutates caller's list + no empty-list guard
    _key(): (
        "def running_average(values):\n"
        '    """Return list of running averages."""\n'
        "    averages = []\n"
        "    values.sort()  # Mutates the caller's list!\n"
        "    for i in range(1, len(values) + 1):  # Off-by-one risk\n"
        "        window = values[:i]\n"
        "        averages.append(sum(window) / i)\n"
        "    return averages\n"
    ),
    # Logic (off-by-one safe loop) fixed only
    _key("edge_logic_01"): (
        "def running_average(values):\n"
        '    """Return list of running averages."""\n'
        "    averages = []\n"
        "    values.sort()  # Still mutates!\n"
        "    for i in range(len(values)):\n"
        "        window = values[: i + 1]\n"
        "        averages.append(sum(window) / (i + 1))\n"
        "    return averages\n"
    ),
    # Mutation fixed only (copy made before sorting)
    _key("edge_mutate_01"): (
        "def running_average(values):\n"
        '    """Return list of running averages."""\n'
        "    averages = []\n"
        "    sorted_values = sorted(values)  # Non-destructive\n"
        "    for i in range(1, len(sorted_values) + 1):\n"
        "        window = sorted_values[:i]\n"
        "        averages.append(sum(window) / i)\n"
        "    return averages\n"
    ),
    # Edge-case (empty list) fixed only
    _key("edge_empty_01"): (
        "def running_average(values):\n"
        '    """Return list of running averages."""\n'
        "    if not values:\n"
        "        return []\n"
        "    averages = []\n"
        "    values.sort()\n"
        "    for i in range(1, len(values) + 1):\n"
        "        window = values[:i]\n"
        "        averages.append(sum(window) / i)\n"
        "    return averages\n"
    ),
    # Logic + mutation fixed
    _key("edge_logic_01", "edge_mutate_01"): (
        "def running_average(values):\n"
        '    """Return list of running averages."""\n'
        "    sorted_values = sorted(values)\n"
        "    averages = []\n"
        "    for i in range(len(sorted_values)):\n"
        "        window = sorted_values[: i + 1]\n"
        "        averages.append(sum(window) / (i + 1))\n"
        "    return averages\n"
    ),
    # Logic + edge fixed
    _key("edge_logic_01", "edge_empty_01"): (
        "def running_average(values):\n"
        '    """Return list of running averages."""\n'
        "    if not values:\n"
        "        return []\n"
        "    values.sort()\n"
        "    averages = []\n"
        "    for i in range(len(values)):\n"
        "        window = values[: i + 1]\n"
        "        averages.append(sum(window) / (i + 1))\n"
        "    return averages\n"
    ),
    # Mutation + edge fixed
    _key("edge_mutate_01", "edge_empty_01"): (
        "def running_average(values):\n"
        '    """Return list of running averages."""\n'
        "    if not values:\n"
        "        return []\n"
        "    sorted_values = sorted(values)\n"
        "    averages = []\n"
        "    for i in range(1, len(sorted_values) + 1):\n"
        "        window = sorted_values[:i]\n"
        "        averages.append(sum(window) / i)\n"
        "    return averages\n"
    ),
    # ALL resolved (ideal)
    _key("edge_logic_01", "edge_mutate_01", "edge_empty_01"): (
        "def running_average(values):\n"
        '    """Return list of running averages."""\n'
        "    if not values:\n"
        "        return []\n"
        "    sorted_values = sorted(values)\n"
        "    averages = []\n"
        "    for i in range(len(sorted_values)):\n"
        "        window = sorted_values[: i + 1]\n"
        "        averages.append(sum(window) / (i + 1))\n"
        "    return averages\n"
    ),
}

EDGE_TASK: Dict[str, Any] = {
    "task_id": "hard_edge_case",
    "difficulty": "hard",
    "context": (
        "This utility computes a running average of a series of values, "
        "sorting them before processing. During a recent code review sprint, "
        "a tester reported that calling this function sometimes corrupts "
        "the caller's original data. A separate report mentions incorrect "
        "averages being returned for certain inputs. Handle the most "
        "critical issue first."
    ),
    "issues": [
        {
            "id": "edge_mutate_01",
            "type": "logic_error",
            "hint": (
                "After calling this function, the original list passed by "
                "the caller is in a different order than expected. "
                "Investigate whether the function has side effects."
            ),
            "description": (
                "values.sort() mutates the caller's list in-place; "
                "use sorted() to avoid modifying the input."
            ),
            "expected_action": "fix_bug",
            "severity": 0.9,
            "priority_gate": True,
            "impact": (
                "Silent data corruption — caller's list is permanently "
                "reordered, causing subtle downstream calculation errors"
            ),
        },
        {
            "id": "edge_logic_01",
            "type": "logic_error",
            "hint": (
                "The averaging calculation produces values that are slightly "
                "inconsistent with manual calculations on the same input. "
                "Examine how the loop index maps to window size."
            ),
            "description": (
                "Loop uses range(1, len+1) with window = values[:i], "
                "causing the final window to double-count the last element "
                "in edge inputs. Correct form: range(len) with values[:i+1]."
            ),
            "expected_action": "fix_bug",
            "severity": 0.7,
            "impact": (
                "Incorrect analytics — running averages are wrong, "
                "leading to bad decisions in any system relying on this data"
            ),
        },
        {
            "id": "edge_empty_01",
            "type": "edge_case",
            "hint": (
                "What happens when this function is called with an empty "
                "collection? Check boundary conditions in the calculation."
            ),
            "description": (
                "Calling with an empty list causes ZeroDivisionError "
                "because the loop executes range(1, 1) and sum([]) / 0 "
                "is attempted on the final pass."
            ),
            "expected_action": "fix_bug",
            "severity": 0.6,
            "hidden": True,  # Revealed after mutation bug is fixed
            "impact": (
                "Application crash on empty input — unhandled ZeroDivisionError "
                "propagates to the caller with no graceful fallback"
            ),
        },
    ],
    # Mutation must be fixed first — it affects the logic test results
    "expected_sequence": ["fix_bug", "fix_bug", "fix_bug"],
    "code_versions": EDGE_CODE_VERSIONS,
    "max_steps": 8,
    "order_constraints": {
        "edge_mutate_01": {
            "must_before": ["edge_logic_01"],
            "violation_penalty": -0.25,
            "reason": (
                "Testing the logic fix on a mutated list produces "
                "misleading pass/fail signals — mutation must be "
                "eliminated first to get clean test results."
            ),
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 — MEDIUM/HARD: Inefficient algorithm (O(n²) matrix ops + memory hog)
# Baseline should score ≈ 0.65–0.75 because:
#   - Primary issue is O(n²) — recognized by keyword matching
#   - Secondary memory issue is less obvious (hint is vague)
#   - Hidden readability issue revealed after performance fix
# ══════════════════════════════════════════════════════════════════════════════

PERF_CODE_VERSIONS: Dict[frozenset, str] = {
    # Initial: O(n²) string concat + builds full intermediate list in memory
    _key(): (
        "def build_report(records):\n"
        '    """Build a formatted report string from a list of records."""\n'
        "    report = ''\n"
        "    filtered = [r for r in records if r.get('active')]\n"
        "    sorted_records = []\n"
        "    for i in range(len(filtered)):\n"
        "        min_idx = i\n"
        "        for j in range(i + 1, len(filtered)):\n"
        "            if filtered[j]['score'] < filtered[min_idx]['score']:\n"
        "                min_idx = j\n"
        "        filtered[i], filtered[min_idx] = filtered[min_idx], filtered[i]\n"
        "        sorted_records.append(filtered[i])\n"
        "    for record in sorted_records:\n"
        "        report = report + f\"{record['name']}: {record['score']}\\n\"\n"
        "    return report\n"
    ),
    # Sort optimized only
    _key("perf_sort_01"): (
        "def build_report(records):\n"
        '    """Build a formatted report string from a list of records."""\n'
        "    report = ''\n"
        "    filtered = [r for r in records if r.get('active')]\n"
        "    sorted_records = sorted(filtered, key=lambda r: r['score'])\n"
        "    for record in sorted_records:\n"
        "        report = report + f\"{record['name']}: {record['score']}\\n\"\n"
        "    return report\n"
    ),
    # String concat optimized only
    _key("perf_concat_01"): (
        "def build_report(records):\n"
        '    """Build a formatted report string from a list of records."""\n'
        "    lines = []\n"
        "    filtered = [r for r in records if r.get('active')]\n"
        "    sorted_records = []\n"
        "    for i in range(len(filtered)):\n"
        "        min_idx = i\n"
        "        for j in range(i + 1, len(filtered)):\n"
        "            if filtered[j]['score'] < filtered[min_idx]['score']:\n"
        "                min_idx = j\n"
        "        filtered[i], filtered[min_idx] = filtered[min_idx], filtered[i]\n"
        "        sorted_records.append(filtered[i])\n"
        "    for record in sorted_records:\n"
        "        lines.append(f\"{record['name']}: {record['score']}\")\n"
        "    return '\\n'.join(lines) + '\\n'\n"
    ),
    # Style/readability fixed only
    _key("perf_style_01"): (
        "def build_report(records):\n"
        '    """Build a formatted report string from a list of records."""\n'
        "    report = ''\n"
        "    active = (r for r in records if r.get('active'))\n"
        "    sorted_records = []\n"
        "    filtered = list(active)\n"
        "    for i in range(len(filtered)):\n"
        "        min_idx = i\n"
        "        for j in range(i + 1, len(filtered)):\n"
        "            if filtered[j]['score'] < filtered[min_idx]['score']:\n"
        "                min_idx = j\n"
        "        filtered[i], filtered[min_idx] = filtered[min_idx], filtered[i]\n"
        "        sorted_records.append(filtered[i])\n"
        "    for record in sorted_records:\n"
        "        report = report + f\"{record['name']}: {record['score']}\\n\"\n"
        "    return report\n"
    ),
    # Sort + concat optimized
    _key("perf_sort_01", "perf_concat_01"): (
        "def build_report(records):\n"
        '    """Build a formatted report string from a list of records."""\n'
        "    filtered = [r for r in records if r.get('active')]\n"
        "    sorted_records = sorted(filtered, key=lambda r: r['score'])\n"
        "    lines = [f\"{r['name']}: {r['score']}\" for r in sorted_records]\n"
        "    return '\\n'.join(lines) + '\\n'\n"
    ),
    # Sort + style fixed
    _key("perf_sort_01", "perf_style_01"): (
        "def build_report(records):\n"
        '    """Build a formatted report string from a list of records."""\n'
        "    report = ''\n"
        "    active = [r for r in records if r.get('active')]\n"
        "    sorted_records = sorted(active, key=lambda r: r['score'])\n"
        "    for record in sorted_records:\n"
        "        report = report + f\"{record['name']}: {record['score']}\\n\"\n"
        "    return report\n"
    ),
    # Concat + style fixed
    _key("perf_concat_01", "perf_style_01"): (
        "def build_report(records):\n"
        '    """Build a formatted report string from a list of records."""\n'
        "    active = [r for r in records if r.get('active')]\n"
        "    sorted_records = []\n"
        "    for i in range(len(active)):\n"
        "        min_idx = i\n"
        "        for j in range(i + 1, len(active)):\n"
        "            if active[j]['score'] < active[min_idx]['score']:\n"
        "                min_idx = j\n"
        "        active[i], active[min_idx] = active[min_idx], active[i]\n"
        "        sorted_records.append(active[i])\n"
        "    lines = [f\"{r['name']}: {r['score']}\" for r in sorted_records]\n"
        "    return '\\n'.join(lines) + '\\n'\n"
    ),
    # ALL resolved (ideal)
    _key("perf_sort_01", "perf_concat_01", "perf_style_01"): (
        "def build_report(records):\n"
        '    """Build a formatted report string from a list of records."""\n'
        "    active = [r for r in records if r.get('active')]\n"
        "    sorted_records = sorted(active, key=lambda r: r['score'])\n"
        "    lines = [f\"{r['name']}: {r['score']}\" for r in sorted_records]\n"
        "    return '\\n'.join(lines) + '\\n'\n"
    ),
}

PERF_TASK: Dict[str, Any] = {
    "task_id": "performance_heavy",
    "difficulty": "medium",
    "context": (
        "This function builds a formatted report from a list of record dicts, "
        "filtering active entries and sorting by score. It is called frequently "
        "in a batch processing pipeline and has been flagged as a performance "
        "bottleneck. Profiling suggests at least two independent inefficiencies. "
        "Prioritise the most significant one first."
    ),
    "issues": [
        {
            "id": "perf_sort_01",
            "type": "performance",
            "hint": (
                "The sorting step uses a pattern that scales poorly with "
                "input size. A more expressive alternative exists in the "
                "standard library."
            ),
            "description": (
                "Manual selection-sort O(n²) is used instead of Python's "
                "built-in sorted() which is O(n log n)."
            ),
            "expected_action": "optimize_code",
            "severity": 0.9,
            "impact": (
                "Pipeline bottleneck — report generation slows to minutes "
                "for large datasets, blocking downstream batch jobs"
            ),
        },
        {
            "id": "perf_concat_01",
            "type": "performance",
            "hint": (
                "The string assembly pattern inside the loop creates "
                "excessive temporary objects. Consider whether the "
                "standard library offers a more efficient idiom."
            ),
            "description": (
                "Repeated string concatenation (report = report + ...) "
                "is O(n²) in total allocations; should use '\\n'.join() "
                "with a list comprehension."
            ),
            "expected_action": "optimize_code",
            "severity": 0.7,
            "impact": (
                "High memory pressure — quadratic allocations cause GC "
                "pauses and OOM errors in large report runs"
            ),
        },
        {
            "id": "perf_style_01",
            "type": "style",
            "hint": (
                "The filtering step materialises an intermediate list "
                "that is iterated only once. A lazy evaluation pattern "
                "would reduce peak memory usage."
            ),
            "description": (
                "The list comprehension for filtering could be replaced "
                "with a generator expression consumed directly by sorted()."
            ),
            "expected_action": "optimize_code",
            "severity": 0.3,
            "hidden": True,  # Revealed after perf_sort_01 is resolved
            "impact": (
                "Unnecessary memory allocation — full intermediate list "
                "held in memory before being consumed once"
            ),
        },
    ],
    # Fix the bigger O(n²) sort first, then string concat, then codestyle
    "expected_sequence": ["optimize_code", "optimize_code", "optimize_code"],
    "code_versions": PERF_CODE_VERSIONS,
    "max_steps": 6,
}


# ══════════════════════════════════════════════════════════════════════════════
# Extended Registry (all 6 tasks)
# ══════════════════════════════════════════════════════════════════════════════

TASK_REGISTRY.update({
    "medium_security_variant": SEC_TASK,
    "hard_edge_case": EDGE_TASK,
    "performance_heavy": PERF_TASK,
})


# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 — MEDIUM: Data validation pipeline
# Scenario: API handler that processes user form data without proper guards.
# Baseline should score ≈ 0.65–0.75 because:
#   - Primary issue looks like a missing feature, not a bug (ambiguous hint)
#   - Hidden crash on None input revealed after first fix
#   - Secondary flag_issue (unsafe assumption) requires different action type
# ══════════════════════════════════════════════════════════════════════════════

VAL_CODE_VERSIONS: Dict[frozenset, str] = {
    # Initial: no type/None checks, silent wrong output on bad input
    _key(): (
        "def process_registration(payload):\n"
        '    """Process a new user registration payload."""\n'
        "    username = payload['username'].strip().lower()\n"
        "    age = int(payload['age'])\n"
        "    email = payload['email']\n"
        "    if len(username) < 3:\n"
        "        return {'error': 'Username too short'}\n"
        "    if age < 13:\n"
        "        return {'error': 'User too young'}\n"
        "    return {\n"
        "        'username': username,\n"
        "        'age': age,\n"
        "        'email': email,\n"
        "        'status': 'registered',\n"
        "    }\n"
    ),
    # Missing-field guard added only
    _key("val_guard_01"): (
        "def process_registration(payload):\n"
        '    """Process a new user registration payload."""\n'
        "    if not payload or not isinstance(payload, dict):\n"
        "        return {'error': 'Invalid payload'}\n"
        "    username = payload.get('username', '').strip().lower()\n"
        "    age_raw = payload.get('age')\n"
        "    email = payload.get('email', '')\n"
        "    if not username or len(username) < 3:\n"
        "        return {'error': 'Username too short'}\n"
        "    if age_raw is None:\n"
        "        return {'error': 'Age is required'}\n"
        "    age = int(age_raw)\n"
        "    if age < 13:\n"
        "        return {'error': 'User too young'}\n"
        "    return {\n"
        "        'username': username,\n"
        "        'age': age,\n"
        "        'email': email,\n"
        "        'status': 'registered',\n"
        "    }\n"
    ),
    # Email format flagged only
    _key("val_email_01"): (
        "def process_registration(payload):\n"
        '    """Process a new user registration payload."""\n'
        "    username = payload['username'].strip().lower()\n"
        "    age = int(payload['age'])\n"
        "    email = payload['email']\n"
        "    if '@' not in email or '.' not in email.split('@')[-1]:\n"
        "        return {'error': 'Invalid email format'}\n"
        "    if len(username) < 3:\n"
        "        return {'error': 'Username too short'}\n"
        "    if age < 13:\n"
        "        return {'error': 'User too young'}\n"
        "    return {\n"
        "        'username': username,\n"
        "        'age': age,\n"
        "        'email': email,\n"
        "        'status': 'registered',\n"
        "    }\n"
    ),
    # Age cast crash fixed only (try/except around int())
    _key("val_cast_01"): (
        "def process_registration(payload):\n"
        '    """Process a new user registration payload."""\n'
        "    username = payload['username'].strip().lower()\n"
        "    try:\n"
        "        age = int(payload['age'])\n"
        "    except (ValueError, TypeError):\n"
        "        return {'error': 'Age must be a number'}\n"
        "    email = payload['email']\n"
        "    if len(username) < 3:\n"
        "        return {'error': 'Username too short'}\n"
        "    if age < 13:\n"
        "        return {'error': 'User too young'}\n"
        "    return {\n"
        "        'username': username,\n"
        "        'age': age,\n"
        "        'email': email,\n"
        "        'status': 'registered',\n"
        "    }\n"
    ),
    # Guard + email fixed
    _key("val_guard_01", "val_email_01"): (
        "def process_registration(payload):\n"
        '    """Process a new user registration payload."""\n'
        "    if not payload or not isinstance(payload, dict):\n"
        "        return {'error': 'Invalid payload'}\n"
        "    username = payload.get('username', '').strip().lower()\n"
        "    age_raw = payload.get('age')\n"
        "    email = payload.get('email', '')\n"
        "    if '@' not in email or '.' not in email.split('@')[-1]:\n"
        "        return {'error': 'Invalid email format'}\n"
        "    if not username or len(username) < 3:\n"
        "        return {'error': 'Username too short'}\n"
        "    if age_raw is None:\n"
        "        return {'error': 'Age is required'}\n"
        "    age = int(age_raw)\n"
        "    if age < 13:\n"
        "        return {'error': 'User too young'}\n"
        "    return {\n"
        "        'username': username,\n"
        "        'age': age,\n"
        "        'email': email,\n"
        "        'status': 'registered',\n"
        "    }\n"
    ),
    # Guard + cast fixed
    _key("val_guard_01", "val_cast_01"): (
        "def process_registration(payload):\n"
        '    """Process a new user registration payload."""\n'
        "    if not payload or not isinstance(payload, dict):\n"
        "        return {'error': 'Invalid payload'}\n"
        "    username = payload.get('username', '').strip().lower()\n"
        "    age_raw = payload.get('age')\n"
        "    email = payload.get('email', '')\n"
        "    if not username or len(username) < 3:\n"
        "        return {'error': 'Username too short'}\n"
        "    if age_raw is None:\n"
        "        return {'error': 'Age is required'}\n"
        "    try:\n"
        "        age = int(age_raw)\n"
        "    except (ValueError, TypeError):\n"
        "        return {'error': 'Age must be a number'}\n"
        "    if age < 13:\n"
        "        return {'error': 'User too young'}\n"
        "    return {\n"
        "        'username': username,\n"
        "        'age': age,\n"
        "        'email': email,\n"
        "        'status': 'registered',\n"
        "    }\n"
    ),
    # Email + cast fixed
    _key("val_email_01", "val_cast_01"): (
        "def process_registration(payload):\n"
        '    """Process a new user registration payload."""\n'
        "    username = payload['username'].strip().lower()\n"
        "    try:\n"
        "        age = int(payload['age'])\n"
        "    except (ValueError, TypeError):\n"
        "        return {'error': 'Age must be a number'}\n"
        "    email = payload['email']\n"
        "    if '@' not in email or '.' not in email.split('@')[-1]:\n"
        "        return {'error': 'Invalid email format'}\n"
        "    if len(username) < 3:\n"
        "        return {'error': 'Username too short'}\n"
        "    if age < 13:\n"
        "        return {'error': 'User too young'}\n"
        "    return {\n"
        "        'username': username,\n"
        "        'age': age,\n"
        "        'email': email,\n"
        "        'status': 'registered',\n"
        "    }\n"
    ),
    # ALL resolved (ideal)
    _key("val_guard_01", "val_email_01", "val_cast_01"): (
        "def process_registration(payload):\n"
        '    """Process a new user registration payload."""\n'
        "    if not payload or not isinstance(payload, dict):\n"
        "        return {'error': 'Invalid payload'}\n"
        "    username = payload.get('username', '').strip().lower()\n"
        "    age_raw = payload.get('age')\n"
        "    email = payload.get('email', '')\n"
        "    if not username or len(username) < 3:\n"
        "        return {'error': 'Username too short'}\n"
        "    if '@' not in email or '.' not in email.split('@')[-1]:\n"
        "        return {'error': 'Invalid email format'}\n"
        "    if age_raw is None:\n"
        "        return {'error': 'Age is required'}\n"
        "    try:\n"
        "        age = int(age_raw)\n"
        "    except (ValueError, TypeError):\n"
        "        return {'error': 'Age must be a number'}\n"
        "    if age < 13:\n"
        "        return {'error': 'User too young'}\n"
        "    return {\n"
        "        'username': username,\n"
        "        'age': age,\n"
        "        'email': email,\n"
        "        'status': 'registered',\n"
        "    }\n"
    ),
}

VAL_TASK: Dict[str, Any] = {
    "task_id": "data_validation_pipeline",
    "difficulty": "medium",
    "context": (
        "This API handler processes user registration payloads received from a "
        "web form. Under normal conditions the function works correctly, but QA "
        "reported crashes when optional fields are omitted and unexpected output "
        "when the age field contains a non-numeric string. A security reviewer "
        "also noted that email addresses are never validated."
    ),
    "issues": [
        {
            "id": "val_guard_01",
            "type": "logic_error",
            "hint": (
                "The function directly accesses dict keys without checking "
                "whether the payload is present or well-formed. What happens "
                "if a required field is missing?"
            ),
            "description": (
                "payload['username'] and payload['age'] raise KeyError when "
                "fields are absent; no None/missing-key guard exists."
            ),
            "expected_action": "fix_bug",
            "severity": 0.9,
            "priority_gate": True,
            "impact": (
                "Service crash on incomplete form submissions — unhandled "
                "KeyError propagates as HTTP 500, leaking stack traces"
            ),
        },
        {
            "id": "val_cast_01",
            "type": "edge_case",
            "hint": (
                "The age field is cast to an integer directly. "
                "Trace what happens if the value supplied is a string "
                "like 'twenty' or an empty string."
            ),
            "description": (
                "int(payload['age']) raises ValueError/TypeError when the "
                "field contains a non-numeric value; no try/except present."
            ),
            "expected_action": "fix_bug",
            "severity": 0.7,
            "hidden": True,  # Revealed after val_guard_01 is fixed
            "impact": (
                "Crash on malformed age input — attackers can reliably trigger "
                "500 errors by submitting non-numeric age values"
            ),
        },
        {
            "id": "val_email_01",
            "type": "security_vulnerability",
            "hint": (
                "User-supplied contact information is stored without structural "
                "verification. Flag any field that could carry unexpected data "
                "into downstream systems."
            ),
            "description": (
                "The email field is accepted and stored verbatim with no "
                "format check — arbitrary strings pass through to the database."
            ),
            "expected_action": "flag_issue",
            "severity": 0.5,
            "impact": (
                "Invalid data in database — storing malformed emails breaks "
                "notification pipelines and may enable injection in mail clients"
            ),
        },
    ],
    "expected_sequence": ["fix_bug", "fix_bug", "flag_issue"],
    "code_versions": VAL_CODE_VERSIONS,
    "max_steps": 6,
    "order_constraints": {
        "val_guard_01": {
            "must_before": ["val_cast_01"],
            "violation_penalty": -0.25,
            "reason": (
                "Fixing the cast before the missing-field guard leaves the "
                "function still crashing on absent keys — the outer guard "
                "must be established first."
            ),
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK 8 — HARD: Shared-state / mutable-default concurrency bug
# Scenario: batch job accumulates results using a shared mutable default arg.
# Baseline should score ≈ 0.50–0.65 because:
#   - Mutable default argument is a well-known pitfall but the hint is indirect
#   - Counter drift is described as a performance anomaly, not a bug
#   - Hidden stale-accumulator issue only visible after the default-arg fix
#   - Correct ordering requires fixing mutability before the counter
# ══════════════════════════════════════════════════════════════════════════════

CONC_CODE_VERSIONS: Dict[frozenset, str] = {
    # Initial: mutable default arg + shared counter drift + no reset between calls
    _key(): (
        "def process_batch(items, results=[]):\n"
        '    """Process a batch of items and accumulate results."""\n'
        "    processed = 0\n"
        "    for item in items:\n"
        "        if item.get('valid'):\n"
        "            results.append(item['value'] * 2)\n"
        "            processed += 1\n"
        "    return {'results': results, 'processed': processed}\n"
        "\n"
        "def summarise(batches):\n"
        '    """Run process_batch on multiple batches and return totals."""\n'
        "    total = 0\n"
        "    for batch in batches:\n"
        "        out = process_batch(batch)\n"
        "        total += out['processed']\n"
        "    return {'total_processed': total, 'all_results': out['results']}\n"
    ),
    # Mutable default fixed only
    _key("conc_mutable_01"): (
        "def process_batch(items, results=None):\n"
        '    """Process a batch of items and accumulate results."""\n'
        "    if results is None:\n"
        "        results = []\n"
        "    processed = 0\n"
        "    for item in items:\n"
        "        if item.get('valid'):\n"
        "            results.append(item['value'] * 2)\n"
        "            processed += 1\n"
        "    return {'results': results, 'processed': processed}\n"
        "\n"
        "def summarise(batches):\n"
        '    """Run process_batch on multiple batches and return totals."""\n'
        "    total = 0\n"
        "    for batch in batches:\n"
        "        out = process_batch(batch)\n"
        "        total += out['processed']\n"
        "    return {'total_processed': total, 'all_results': out['results']}\n"
    ),
    # Counter drift fixed only (summarise collects all results)
    _key("conc_counter_01"): (
        "def process_batch(items, results=[]):\n"
        '    """Process a batch of items and accumulate results."""\n'
        "    processed = 0\n"
        "    for item in items:\n"
        "        if item.get('valid'):\n"
        "            results.append(item['value'] * 2)\n"
        "            processed += 1\n"
        "    return {'results': results, 'processed': processed}\n"
        "\n"
        "def summarise(batches):\n"
        '    """Run process_batch on multiple batches and return totals."""\n'
        "    total = 0\n"
        "    all_results = []\n"
        "    for batch in batches:\n"
        "        out = process_batch(batch)\n"
        "        total += out['processed']\n"
        "        all_results.extend(out['results'])\n"
        "    return {'total_processed': total, 'all_results': all_results}\n"
    ),
    # Stale-ref fixed only (out scoped inside loop)
    _key("conc_stale_01"): (
        "def process_batch(items, results=[]):\n"
        '    """Process a batch of items and accumulate results."""\n'
        "    processed = 0\n"
        "    for item in items:\n"
        "        if item.get('valid'):\n"
        "            results.append(item['value'] * 2)\n"
        "            processed += 1\n"
        "    return {'results': results, 'processed': processed}\n"
        "\n"
        "def summarise(batches):\n"
        '    """Run process_batch on multiple batches and return totals."""\n'
        "    total = 0\n"
        "    all_results = []\n"
        "    for batch in batches:\n"
        "        out = process_batch(batch)\n"
        "        total += out['processed']\n"
        "        all_results.extend(out['results'])\n"
        "    return {'total_processed': total, 'all_results': all_results}\n"
    ),
    # Mutable + counter fixed
    _key("conc_mutable_01", "conc_counter_01"): (
        "def process_batch(items, results=None):\n"
        '    """Process a batch of items and accumulate results."""\n'
        "    if results is None:\n"
        "        results = []\n"
        "    processed = 0\n"
        "    for item in items:\n"
        "        if item.get('valid'):\n"
        "            results.append(item['value'] * 2)\n"
        "            processed += 1\n"
        "    return {'results': results, 'processed': processed}\n"
        "\n"
        "def summarise(batches):\n"
        '    """Run process_batch on multiple batches and return totals."""\n'
        "    total = 0\n"
        "    all_results = []\n"
        "    for batch in batches:\n"
        "        out = process_batch(batch)\n"
        "        total += out['processed']\n"
        "        all_results.extend(out['results'])\n"
        "    return {'total_processed': total, 'all_results': all_results}\n"
    ),
    # Mutable + stale fixed
    _key("conc_mutable_01", "conc_stale_01"): (
        "def process_batch(items, results=None):\n"
        '    """Process a batch of items and accumulate results."""\n'
        "    if results is None:\n"
        "        results = []\n"
        "    processed = 0\n"
        "    for item in items:\n"
        "        if item.get('valid'):\n"
        "            results.append(item['value'] * 2)\n"
        "            processed += 1\n"
        "    return {'results': results, 'processed': processed}\n"
        "\n"
        "def summarise(batches):\n"
        '    """Run process_batch on multiple batches and return totals."""\n'
        "    total = 0\n"
        "    all_results = []\n"
        "    for batch in batches:\n"
        "        out = process_batch(batch)\n"
        "        total += out['processed']\n"
        "        all_results.extend(out['results'])\n"
        "    return {'total_processed': total, 'all_results': all_results}\n"
    ),
    # Counter + stale fixed (mutable default still present)
    _key("conc_counter_01", "conc_stale_01"): (
        "def process_batch(items, results=[]):\n"
        '    """Process a batch of items and accumulate results."""\n'
        "    processed = 0\n"
        "    for item in items:\n"
        "        if item.get('valid'):\n"
        "            results.append(item['value'] * 2)\n"
        "            processed += 1\n"
        "    return {'results': results, 'processed': processed}\n"
        "\n"
        "def summarise(batches):\n"
        '    """Run process_batch on multiple batches and return totals."""\n'
        "    total = 0\n"
        "    all_results = []\n"
        "    for batch in batches:\n"
        "        out = process_batch(batch)\n"
        "        total += out['processed']\n"
        "        all_results.extend(out['results'])\n"
        "    return {'total_processed': total, 'all_results': all_results}\n"
    ),
    # ALL resolved (ideal)
    _key("conc_mutable_01", "conc_counter_01", "conc_stale_01"): (
        "def process_batch(items, results=None):\n"
        '    """Process a batch of items and accumulate results."""\n'
        "    if results is None:\n"
        "        results = []\n"
        "    processed = 0\n"
        "    for item in items:\n"
        "        if item.get('valid'):\n"
        "            results.append(item['value'] * 2)\n"
        "            processed += 1\n"
        "    return {'results': results, 'processed': processed}\n"
        "\n"
        "def summarise(batches):\n"
        '    """Run process_batch on multiple batches and return totals."""\n'
        "    total = 0\n"
        "    all_results = []\n"
        "    for batch in batches:\n"
        "        out = process_batch(batch)\n"
        "        total += out['processed']\n"
        "        all_results.extend(out['results'])\n"
        "    return {'total_processed': total, 'all_results': all_results}\n"
    ),
}

CONC_TASK: Dict[str, Any] = {
    "task_id": "concurrency_bug",
    "difficulty": "hard",
    "context": (
        "This batch processing module runs against multiple data batches in "
        "sequence. During a production incident, the results list was found to "
        "contain data from previous unrelated runs — as if internal state was "
        "leaking between calls. A separate report noted that the processed "
        "count in summarise sometimes underreports the total. Investigate and "
        "fix the root cause before addressing secondary symptoms."
    ),
    "issues": [
        {
            "id": "conc_mutable_01",
            "type": "logic_error",
            "hint": (
                "The function signature includes a default parameter that "
                "appears safe but is shared across all calls. Investigate "
                "what happens to that parameter when the function is called "
                "multiple times without passing it explicitly."
            ),
            "description": (
                "results=[] is a mutable default argument — Python creates "
                "it once at function definition time, so all calls share the "
                "same list object, causing cross-call contamination."
            ),
            "expected_action": "fix_bug",
            "severity": 1.0,
            "priority_gate": True,
            "impact": (
                "Silent data contamination — results from previous batch runs "
                "accumulate into subsequent calls, corrupting all output"
            ),
        },
        {
            "id": "conc_counter_01",
            "type": "logic_error",
            "hint": (
                "The summarise function reports a total count that does not "
                "match the sum of individually processed batches. Trace how "
                "results from each batch are collected into the final return "
                "value."
            ),
            "description": (
                "summarise references `out['results']` after the loop, "
                "returning only the last batch's results. It should collect "
                "all_results across iterations using extend()."
            ),
            "expected_action": "fix_bug",
            "severity": 0.7,
            "impact": (
                "Incomplete reporting — only the last batch's results appear "
                "in the summary, silently discarding all earlier batches"
            ),
        },
        {
            "id": "conc_stale_01",
            "type": "edge_case",
            "hint": (
                "If the input list of batches is empty, the final return "
                "statement references a variable that was never assigned. "
                "Check whether all code paths define every variable used."
            ),
            "description": (
                "When batches=[], the loop body never executes so `out` is "
                "undefined; `out['results']` on the return line raises "
                "UnboundLocalError."
            ),
            "expected_action": "fix_bug",
            "severity": 0.6,
            "hidden": True,  # Revealed after conc_mutable_01 is fixed
            "impact": (
                "Crash on empty input — UnboundLocalError surfaces when "
                "a job runs with zero batches, halting the pipeline"
            ),
        },
    ],
    "expected_sequence": ["fix_bug", "fix_bug", "fix_bug"],
    "code_versions": CONC_CODE_VERSIONS,
    "max_steps": 8,
    "order_constraints": {
        "conc_mutable_01": {
            "must_before": ["conc_counter_01"],
            "violation_penalty": -0.25,
            "reason": (
                "Fixing the counter logic while the mutable default is "
                "still present gives misleading results — shared state "
                "must be eliminated first for a clean baseline."
            ),
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Extended Registry (all 8 tasks)
# ══════════════════════════════════════════════════════════════════════════════

TASK_REGISTRY.update({
    "data_validation_pipeline": VAL_TASK,
    "concurrency_bug": CONC_TASK,
})

