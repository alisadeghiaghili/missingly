"""Tests for the ruff lint ratchet in ``tools/ruff_ratchet.py``.

The comparison logic is exercised with small synthetic snapshots. One
integration test proves that the committed baseline matches the current tree.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RATCHET_SCRIPT = REPOSITORY_ROOT / "tools" / "ruff_ratchet.py"


def _load_ratchet() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ruff_ratchet", RATCHET_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ratchet = _load_ratchet()


def _snapshot(
    violations: Dict[str, Dict[str, int]] | None = None,
    unformatted: List[str] | None = None,
) -> Any:
    return ratchet.Snapshot(
        violations=violations or {},
        unformatted=frozenset(unformatted or []),
    )


def test_identical_snapshots_pass() -> None:
    snap = _snapshot({"missingly/a.py": {"E501": 2}}, ["missingly/b.py"])
    result = ratchet.compare(snap, snap)
    assert result.regressions == []
    assert result.improvements == []


def test_new_rule_in_existing_file_is_a_regression() -> None:
    baseline = _snapshot({"missingly/a.py": {"E501": 2}})
    current = _snapshot({"missingly/a.py": {"E501": 2, "B006": 1}})
    result = ratchet.compare(baseline, current)
    assert len(result.regressions) == 1
    assert "B006" in result.regressions[0]
    assert "missingly/a.py" in result.regressions[0]


def test_violations_in_new_file_are_a_regression() -> None:
    baseline = _snapshot({"missingly/a.py": {"E501": 2}})
    current = _snapshot(
        {"missingly/a.py": {"E501": 2}, "missingly/new.py": {"S101": 1}}
    )
    result = ratchet.compare(baseline, current)
    assert any("missingly/new.py" in line for line in result.regressions)


def test_increased_count_is_a_regression() -> None:
    baseline = _snapshot({"missingly/a.py": {"E501": 2}})
    current = _snapshot({"missingly/a.py": {"E501": 3}})
    result = ratchet.compare(baseline, current)
    assert len(result.regressions) == 1
    assert "2" in result.regressions[0] and "3" in result.regressions[0]


def test_decreased_count_passes_and_is_reported_as_improvement() -> None:
    baseline = _snapshot({"missingly/a.py": {"E501": 3}})
    current = _snapshot({"missingly/a.py": {"E501": 1}})
    result = ratchet.compare(baseline, current)
    assert result.regressions == []
    assert len(result.improvements) == 1


def test_removed_rule_is_an_improvement() -> None:
    baseline = _snapshot({"missingly/a.py": {"E501": 3}})
    current = _snapshot({})
    result = ratchet.compare(baseline, current)
    assert result.regressions == []
    assert len(result.improvements) == 1


def test_newly_unformatted_file_is_a_regression() -> None:
    baseline = _snapshot(unformatted=[])
    current = _snapshot(unformatted=["missingly/a.py"])
    result = ratchet.compare(baseline, current)
    assert len(result.regressions) == 1
    assert "missingly/a.py" in result.regressions[0]


def test_file_that_is_formatted_again_is_an_improvement() -> None:
    baseline = _snapshot(unformatted=["missingly/a.py"])
    current = _snapshot(unformatted=[])
    result = ratchet.compare(baseline, current)
    assert result.regressions == []
    assert len(result.improvements) == 1


def test_build_snapshot_counts_per_file_and_rule(tmp_path: Path) -> None:
    lint_items = [
        {"filename": str(tmp_path / "missingly" / "a.py"), "code": "E501"},
        {"filename": str(tmp_path / "missingly" / "a.py"), "code": "E501"},
        {"filename": str(tmp_path / "missingly" / "a.py"), "code": "B006"},
    ]
    format_items = [
        {"filename": str(tmp_path / "missingly" / "b.py"), "code": "unformatted"},
    ]
    snap = ratchet.build_snapshot(lint_items, format_items, tmp_path)
    assert snap.violations == {"missingly/a.py": {"B006": 1, "E501": 2}}
    assert snap.unformatted == frozenset({"missingly/b.py"})


def test_dump_and_load_round_trip_is_sorted_and_stable(tmp_path: Path) -> None:
    snap = _snapshot(
        {"tests/z.py": {"S101": 1}, "missingly/a.py": {"E501": 2, "B006": 1}},
        ["tests/z.py", "missingly/a.py"],
    )
    text = ratchet.dump_baseline(snap)
    assert text.endswith("\n")
    data = json.loads(text)
    assert list(data["violations"]) == ["missingly/a.py", "tests/z.py"]
    assert list(data["violations"]["missingly/a.py"]) == ["B006", "E501"]
    assert data["unformatted"] == ["missingly/a.py", "tests/z.py"]
    path = tmp_path / "baseline.json"
    path.write_text(text, encoding="utf-8")
    assert ratchet.load_baseline(path) == snap
    assert ratchet.dump_baseline(snap) == text


def test_run_ruff_rejects_tool_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args, 2, stdout="", stderr="boom")

    monkeypatch.setattr(ratchet.subprocess, "run", fake_run)
    with pytest.raises(ratchet.RatchetError):
        ratchet.run_ruff(["check", "missingly"], REPOSITORY_ROOT)


def test_run_ruff_rejects_unparseable_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args, 0, stdout="not json", stderr="")

    monkeypatch.setattr(ratchet.subprocess, "run", fake_run)
    with pytest.raises(ratchet.RatchetError):
        ratchet.run_ruff(["check", "missingly"], REPOSITORY_ROOT)


def test_main_update_writes_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "ruff_baseline.json"
    monkeypatch.setattr(ratchet, "BASELINE_PATH", target)
    monkeypatch.setattr(
        ratchet, "collect_snapshot", lambda root: _snapshot({"missingly/a.py": {"E501": 1}})
    )
    assert ratchet.main(["--update"]) == 0
    assert ratchet.load_baseline(target) == _snapshot({"missingly/a.py": {"E501": 1}})


def test_main_fails_on_regression_and_hints_on_improvement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    target = tmp_path / "ruff_baseline.json"
    target.write_text(ratchet.dump_baseline(_snapshot({"missingly/a.py": {"E501": 2}})))
    monkeypatch.setattr(ratchet, "BASELINE_PATH", target)

    monkeypatch.setattr(
        ratchet, "collect_snapshot", lambda root: _snapshot({"missingly/a.py": {"E501": 3}})
    )
    assert ratchet.main([]) == 1
    assert "REGRESSION" in capsys.readouterr().out

    monkeypatch.setattr(
        ratchet, "collect_snapshot", lambda root: _snapshot({"missingly/a.py": {"E501": 1}})
    )
    assert ratchet.main([]) == 0
    output = capsys.readouterr().out
    assert "--update" in output


def test_committed_baseline_matches_current_tree() -> None:
    completed = subprocess.run(
        [sys.executable, str(RATCHET_SCRIPT)],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
