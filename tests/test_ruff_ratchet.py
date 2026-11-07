"""Tests for the ruff lint ratchet in ``tools/ruff_ratchet.py``.

The comparison logic is exercised with small synthetic snapshots. One
integration test proves that the committed baseline matches the current tree.
"""

from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RATCHET_SCRIPT = REPOSITORY_ROOT / "tools" / "ruff_ratchet.py"
GATE_TARGET = "missingly/a.py"
_MISSING_RUFF = "ruff is missing"


def _load_ratchet() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ruff_ratchet", RATCHET_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ratchet = _load_ratchet()


def _snapshot(
    violations: dict[str, dict[str, int]] | None = None,
    unformatted: list[str] | None = None,
) -> ratchet.Snapshot:
    return ratchet.Snapshot(
        violations=violations or {},
        unformatted=frozenset(unformatted or []),
    )


def _fake_collect(snapshot: ratchet.Snapshot) -> object:
    def collect(_root: Path) -> ratchet.Snapshot:
        return snapshot

    return collect


def test_identical_snapshots_pass() -> None:
    snap = _snapshot({GATE_TARGET: {"E501": 2}}, ["missingly/b.py"])
    result = ratchet.compare(snap, snap)
    assert result.regressions == []
    assert result.improvements == []


def test_new_rule_in_existing_file_is_a_regression() -> None:
    baseline = _snapshot({GATE_TARGET: {"E501": 2}})
    current = _snapshot({GATE_TARGET: {"E501": 2, "B006": 1}})
    result = ratchet.compare(baseline, current)
    assert len(result.regressions) == 1
    assert "B006" in result.regressions[0]
    assert GATE_TARGET in result.regressions[0]


def test_violations_in_new_file_are_a_regression() -> None:
    baseline = _snapshot({GATE_TARGET: {"E501": 2}})
    current = _snapshot({GATE_TARGET: {"E501": 2}, "missingly/new.py": {"S101": 1}})
    result = ratchet.compare(baseline, current)
    assert any("missingly/new.py" in line for line in result.regressions)


def test_increased_count_is_a_regression() -> None:
    baseline = _snapshot({GATE_TARGET: {"E501": 2}})
    current = _snapshot({GATE_TARGET: {"E501": 3}})
    result = ratchet.compare(baseline, current)
    assert len(result.regressions) == 1
    assert "2 -> 3" in result.regressions[0]


def test_decreased_count_passes_and_is_reported_as_improvement() -> None:
    baseline = _snapshot({GATE_TARGET: {"E501": 3}})
    current = _snapshot({GATE_TARGET: {"E501": 1}})
    result = ratchet.compare(baseline, current)
    assert result.regressions == []
    assert len(result.improvements) == 1


def test_removed_rule_is_an_improvement() -> None:
    baseline = _snapshot({GATE_TARGET: {"E501": 3}})
    current = _snapshot({})
    result = ratchet.compare(baseline, current)
    assert result.regressions == []
    assert len(result.improvements) == 1


def test_newly_unformatted_file_is_a_regression() -> None:
    baseline = _snapshot(unformatted=[])
    current = _snapshot(unformatted=[GATE_TARGET])
    result = ratchet.compare(baseline, current)
    assert len(result.regressions) == 1
    assert GATE_TARGET in result.regressions[0]


def test_file_that_is_formatted_again_is_an_improvement() -> None:
    baseline = _snapshot(unformatted=[GATE_TARGET])
    current = _snapshot(unformatted=[])
    result = ratchet.compare(baseline, current)
    assert result.regressions == []
    assert len(result.improvements) == 1


def test_build_snapshot_counts_per_file_and_rule(tmp_path: Path) -> None:
    source = str(tmp_path / "missingly" / "a.py")
    lint_items = [
        {"filename": source, "code": "E501"},
        {"filename": source, "code": "E501"},
        {"filename": source, "code": "B006"},
        {"filename": source, "code": None},
    ]
    format_items = [
        {"filename": str(tmp_path / "missingly" / "b.py"), "code": "unformatted"},
    ]
    snap = ratchet.build_snapshot(lint_items, format_items, tmp_path)
    assert snap.violations == {
        GATE_TARGET: {"B006": 1, "E501": 2, "SYNTAX": 1},
    }
    assert snap.unformatted == frozenset({"missingly/b.py"})


def test_dump_and_load_round_trip_is_sorted_and_stable(tmp_path: Path) -> None:
    snap = _snapshot(
        {"tests/z.py": {"S101": 1}, GATE_TARGET: {"E501": 2, "B006": 1}},
        ["tests/z.py", GATE_TARGET],
    )
    text = ratchet.dump_baseline(snap)
    assert text.endswith("\n")
    data = json.loads(text)
    assert list(data["violations"]) == [GATE_TARGET, "tests/z.py"]
    assert list(data["violations"][GATE_TARGET]) == ["B006", "E501"]
    assert data["unformatted"] == [GATE_TARGET, "tests/z.py"]
    path = tmp_path / "baseline.json"
    path.write_text(text, encoding="utf-8")
    assert ratchet.load_baseline(path) == snap
    assert ratchet.dump_baseline(snap) == text


def test_load_baseline_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ratchet.RatchetError, match="--update"):
        ratchet.load_baseline(tmp_path / "missing.json")


def test_load_baseline_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ratchet.RatchetError, match="--update"):
        ratchet.load_baseline(path)


def _completed(code: int, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=stdout)


def test_run_ruff_rejects_tool_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=2, stderr="boom")

    monkeypatch.setattr(ratchet.subprocess, "run", fake_run)
    with pytest.raises(ratchet.RatchetError, match="exited with 2"):
        ratchet.run_ruff(["check", "missingly"], REPOSITORY_ROOT)


def test_run_ruff_rejects_unparseable_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        return _completed(0, "not json")

    monkeypatch.setattr(ratchet.subprocess, "run", fake_run)
    with pytest.raises(ratchet.RatchetError, match="not JSON"):
        ratchet.run_ruff(["check", "missingly"], REPOSITORY_ROOT)


def test_run_ruff_rejects_non_list_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        return _completed(0, "{}")

    monkeypatch.setattr(ratchet.subprocess, "run", fake_run)
    with pytest.raises(ratchet.RatchetError, match="not a list"):
        ratchet.run_ruff(["check", "missingly"], REPOSITORY_ROOT)


def test_run_ruff_accepts_findings_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = '[{"filename": "x.py", "code": "E501"}]'

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        return _completed(1, payload)

    monkeypatch.setattr(ratchet.subprocess, "run", fake_run)
    assert ratchet.run_ruff(["check"], REPOSITORY_ROOT) == [
        {"filename": "x.py", "code": "E501"},
    ]


def test_main_update_writes_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "ruff_baseline.json"
    expected = _snapshot({GATE_TARGET: {"E501": 1}})
    monkeypatch.setattr(ratchet, "BASELINE_PATH", target)
    monkeypatch.setattr(ratchet, "collect_snapshot", _fake_collect(expected))
    assert ratchet.main(["--update"]) == 0
    assert ratchet.load_baseline(target) == expected


def test_main_fails_on_regression_and_hints_on_improvement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "ruff_baseline.json"
    target.write_text(ratchet.dump_baseline(_snapshot({GATE_TARGET: {"E501": 2}})))
    monkeypatch.setattr(ratchet, "BASELINE_PATH", target)

    worse = _snapshot({GATE_TARGET: {"E501": 3}})
    monkeypatch.setattr(ratchet, "collect_snapshot", _fake_collect(worse))
    assert ratchet.main([]) == 1
    assert "REGRESSION" in capsys.readouterr().out

    better = _snapshot({GATE_TARGET: {"E501": 1}})
    monkeypatch.setattr(ratchet, "collect_snapshot", _fake_collect(better))
    assert ratchet.main([]) == 0
    assert "--update" in capsys.readouterr().out


def test_main_returns_two_when_ruff_cannot_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_root: Path) -> ratchet.Snapshot:
        raise ratchet.RatchetError(_MISSING_RUFF)

    monkeypatch.setattr(ratchet, "collect_snapshot", broken)
    assert ratchet.main([]) == 2
    assert _MISSING_RUFF in capsys.readouterr().err


def test_committed_baseline_matches_current_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(RATCHET_SCRIPT)])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(RATCHET_SCRIPT), run_name="__main__")
    assert excinfo.value.code == 0


def test_quality_extra_pins_the_ruff_version_the_baseline_was_built_with() -> None:
    """A floating ruff version can add rules and fail the ratchet spuriously."""
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    quality = pyproject["project"]["optional-dependencies"]["quality"]
    ruff_specs = [spec for spec in quality if spec.replace(" ", "").startswith("ruff")]
    assert len(ruff_specs) == 1
    assert ruff_specs[0].replace(" ", "").startswith("ruff==")
