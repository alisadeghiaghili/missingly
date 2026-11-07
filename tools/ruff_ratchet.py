"""Ruff lint and format ratchet for the missingly quality gate.

The gate counts ruff violations per (file, rule) and the set of files that
``ruff format`` would rewrite, then compares those numbers with the committed
baseline in ``tools/ruff_baseline.json``. The gate fails when debt grows: a
count increases, a new (file, rule) pair appears, or a formatted file becomes
unformatted. Reduced debt passes and prints a hint to refresh the baseline.

Usage
-----
Check the working tree against the baseline::

    python tools/ruff_ratchet.py

Rewrite the baseline after debt has been paid down::

    python tools/ruff_ratchet.py --update

Exit codes are 0 for pass, 1 for a regression, and 2 when ruff cannot run or
the baseline cannot be read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "tools" / "ruff_baseline.json"
TARGETS = ("missingly", "tests")
UNFORMATTED_CODE = "unformatted"
SYNTAX_ERROR_CODE = "SYNTAX"
# ruff exits 1 when it reports findings; that is data, not a tool failure.
RUFF_FINDINGS_EXIT_CODES = (0, 1)


class RatchetError(RuntimeError):
    """Raised when ruff cannot run or the baseline cannot be read."""


@dataclass(frozen=True)
class Snapshot:
    """Lint and format debt of one tree.

    Attributes
    ----------
    violations : dict of str to dict of str to int
        Violation count keyed by repository-relative file, then rule code.
    unformatted : frozenset of str
        Repository-relative files that ``ruff format`` would rewrite.
    """

    violations: dict[str, dict[str, int]]
    unformatted: frozenset[str]


@dataclass(frozen=True)
class Comparison:
    """Outcome of comparing a snapshot with the baseline.

    Attributes
    ----------
    regressions : list of str
        Human-readable descriptions of debt that grew. Any entry fails the gate.
    improvements : list of str
        Human-readable descriptions of debt that shrank.
    """

    regressions: list[str]
    improvements: list[str]


def _relative_key(filename: str, root: Path) -> str:
    """Return ``filename`` relative to ``root`` in POSIX form.

    Parameters
    ----------
    filename : str
        Path reported by ruff.
    root : Path
        Repository root that keys are relative to.

    Returns
    -------
    str
        Relative POSIX path, or the POSIX form of ``filename`` when it lies
        outside ``root``.
    """
    path = Path(filename)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_snapshot(
    lint_items: Sequence[dict[str, Any]],
    format_items: Sequence[dict[str, Any]],
    root: Path,
) -> Snapshot:
    """Aggregate ruff JSON diagnostics into a snapshot.

    Parameters
    ----------
    lint_items : sequence of dict
        Diagnostics from ``ruff check --output-format json``.
    format_items : sequence of dict
        Diagnostics from ``ruff format --check --output-format json``.
    root : Path
        Repository root used to build relative keys.

    Returns
    -------
    Snapshot
        Violation counts per file and rule, and the unformatted file set.

    Examples
    --------
    >>> from pathlib import Path
    >>> items = [{"filename": "/r/missingly/a.py", "code": "E501"}]
    >>> build_snapshot(items, [], Path("/r")).violations
    {'missingly/a.py': {'E501': 1}}
    """
    violations: dict[str, dict[str, int]] = {}
    for item in lint_items:
        rules = violations.setdefault(_relative_key(str(item["filename"]), root), {})
        code = str(item.get("code") or SYNTAX_ERROR_CODE)
        rules[code] = rules.get(code, 0) + 1
    unformatted = frozenset(
        _relative_key(str(item["filename"]), root)
        for item in format_items
        if item.get("code") == UNFORMATTED_CODE
    )
    return Snapshot(violations=violations, unformatted=unformatted)


def _compare_counts(
    baseline: dict[str, dict[str, int]],
    current: dict[str, dict[str, int]],
) -> tuple[list[str], list[str]]:
    """Compare violation counts for every (file, rule) pair.

    Parameters
    ----------
    baseline : dict of str to dict of str to int
        Counts recorded in the baseline.
    current : dict of str to dict of str to int
        Counts measured on the working tree.

    Returns
    -------
    tuple of (list of str, list of str)
        Regression descriptions and improvement descriptions.
    """
    regressions: list[str] = []
    improvements: list[str] = []
    for file_key in sorted(set(baseline) | set(current)):
        old_rules = baseline.get(file_key, {})
        new_rules = current.get(file_key, {})
        for code in sorted(set(old_rules) | set(new_rules)):
            old = old_rules.get(code, 0)
            new = new_rules.get(code, 0)
            if new > old:
                regressions.append(f"{file_key}: {code} {old} -> {new}")
            elif new < old:
                improvements.append(f"{file_key}: {code} {old} -> {new}")
    return regressions, improvements


def compare(baseline: Snapshot, current: Snapshot) -> Comparison:
    """Compare a current snapshot with the baseline.

    Parameters
    ----------
    baseline : Snapshot
        Debt recorded in the committed baseline.
    current : Snapshot
        Debt measured on the working tree.

    Returns
    -------
    Comparison
        Regressions (fail the gate) and improvements (pass with a hint).

    Examples
    --------
    >>> none = frozenset()
    >>> base = Snapshot(violations={"missingly/a.py": {"E501": 2}}, unformatted=none)
    >>> cur = Snapshot(violations={"missingly/a.py": {"E501": 3}}, unformatted=none)
    >>> compare(base, cur).regressions
    ['missingly/a.py: E501 2 -> 3']
    """
    regressions, improvements = _compare_counts(baseline.violations, current.violations)
    for file_key in sorted(current.unformatted - baseline.unformatted):
        regressions.append(f"{file_key}: no longer formatted with ruff format")
    for file_key in sorted(baseline.unformatted - current.unformatted):
        improvements.append(f"{file_key}: now formatted with ruff format")
    return Comparison(regressions=regressions, improvements=improvements)


def dump_baseline(snapshot: Snapshot) -> str:
    """Serialise a snapshot as stable, sorted JSON.

    Parameters
    ----------
    snapshot : Snapshot
        Snapshot to serialise.

    Returns
    -------
    str
        JSON text ending in a newline.

    Examples
    --------
    >>> snap = Snapshot(violations={}, unformatted=frozenset({"tests/b.py"}))
    >>> print(dump_baseline(snap), end="")
    {
      "unformatted": [
        "tests/b.py"
      ],
      "violations": {}
    }
    """
    payload = {
        "violations": snapshot.violations,
        "unformatted": sorted(snapshot.unformatted),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def load_baseline(path: Path) -> Snapshot:
    """Read a baseline file written by :func:`dump_baseline`.

    Parameters
    ----------
    path : Path
        Baseline JSON file.

    Returns
    -------
    Snapshot
        The recorded debt.

    Raises
    ------
    RatchetError
        If the file is missing or is not a valid baseline.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        message = f"cannot read baseline {path}: {exc}; create it with --update"
        raise RatchetError(message) from exc
    violations = {
        str(file_key): {str(code): int(count) for code, count in rules.items()}
        for file_key, rules in data.get("violations", {}).items()
    }
    unformatted = frozenset(str(item) for item in data.get("unformatted", []))
    return Snapshot(violations=violations, unformatted=unformatted)


def run_ruff(args: Sequence[str], root: Path) -> list[dict[str, Any]]:
    """Run ``python -m ruff`` with JSON output and return its diagnostics.

    Parameters
    ----------
    args : sequence of str
        Arguments after ``ruff``, for example ``["check", "missingly"]``.
    root : Path
        Working directory for ruff.

    Returns
    -------
    list of dict
        Parsed JSON diagnostics.

    Raises
    ------
    RatchetError
        If ruff exits with an error or its output is not a JSON list.
    """
    command = [sys.executable, "-m", "ruff", *args]
    completed = subprocess.run(
        command,
        cwd=str(root),
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode not in RUFF_FINDINGS_EXIT_CODES:
        message = f"ruff exited with {completed.returncode}: {completed.stderr.strip()}"
        raise RatchetError(message)
    try:
        data = json.loads(completed.stdout or "[]")
    except ValueError as exc:
        message = f"ruff output is not JSON: {exc}"
        raise RatchetError(message) from exc
    if not isinstance(data, list):
        message = "ruff JSON output is not a list"
        raise RatchetError(message)
    return data


def collect_snapshot(root: Path) -> Snapshot:
    """Measure lint and format debt of the working tree.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    Snapshot
        Current debt for :data:`TARGETS`.
    """
    lint_items = run_ruff(["check", *TARGETS, "--output-format", "json"], root)
    format_items = run_ruff(
        ["format", "--check", *TARGETS, "--output-format", "json"], root
    )
    return build_snapshot(lint_items, format_items, root)


def format_report(result: Comparison) -> str:
    """Render a comparison as the text printed by the gate.

    Parameters
    ----------
    result : Comparison
        Outcome to describe.

    Returns
    -------
    str
        Multi-line report.
    """
    lines = [f"REGRESSION: {item}" for item in result.regressions]
    lines.extend(f"improved: {item}" for item in result.improvements)
    if result.regressions:
        lines.append(
            f"ruff ratchet failed with {len(result.regressions)} regression(s). "
            "Fix the new violations; do not raise the baseline."
        )
    elif result.improvements:
        lines.append(
            "ruff ratchet passed; debt went down. Refresh the baseline with "
            "`python tools/ruff_ratchet.py --update` and commit it."
        )
    else:
        lines.append("ruff ratchet passed: no change against the baseline.")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite tools/ruff_baseline.json from the current tree",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ratchet.

    Parameters
    ----------
    argv : sequence of str or None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 when the gate passes or the baseline was updated, 1 on regression,
        2 when ruff or the baseline cannot be read.
    """
    args = _parse_args(argv)
    try:
        current = collect_snapshot(REPO_ROOT)
        if args.update:
            BASELINE_PATH.write_text(dump_baseline(current), encoding="utf-8")
            print(f"ruff ratchet: baseline written to {BASELINE_PATH}")
            return 0
        result = compare(load_baseline(BASELINE_PATH), current)
    except RatchetError as exc:
        print(f"ruff ratchet: {exc}", file=sys.stderr)
        return 2
    print(format_report(result))
    return 1 if result.regressions else 0


if __name__ == "__main__":
    sys.exit(main())
