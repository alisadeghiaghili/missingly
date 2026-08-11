"""Contract tests for the scoped Missingly-versus-R ``mice`` comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parent
    / "fixtures"
    / "mice_numeric_fcs"
    / "validate_cross_tool_comparison.py"
)
_SPEC = importlib.util.spec_from_file_location("r_mice_cross_tool_validator", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _reference() -> dict[str, object]:
    """Return a minimal scalar R reference artifact."""
    return {
        "reference_tool": "R mice",
        "reference_tool_version": "3.19.0",
        "seed": 7,
        "dataset_sha256": "a" * 64,
        "metrics": {
            "chain_1_mean_x": 1.0,
            "chain_2_mean_x": 3.0,
        },
    }


def _subject() -> dict[str, object]:
    """Return a minimal scalar Missingly comparison artifact."""
    return {
        "subject_tool": "Missingly",
        "seed": 7,
        "dataset_sha256": "a" * 64,
        "metrics": {"pooled_mean_x": 2.2},
    }


def _contract() -> dict[str, object]:
    """Return a reviewed single-estimand comparison contract."""
    return {
        "schema_version": 1,
        "reference_tool": "R mice",
        "reference_tool_version": "3.19.0",
        "seed": 7,
        "dataset_sha256": "a" * 64,
        "estimand": "pooled completed-data means",
        "tolerances": {"pooled_mean_x": 0.25},
    }


def test_validator_accepts_scoped_pooled_mean_comparison() -> None:
    """The comparison accepts a subject result inside the reviewed bound."""
    _MODULE.validate_cross_tool_comparison(_reference(), _subject(), _contract())


def test_validator_rejects_subject_metric_outside_bound() -> None:
    """A cross-tool drift outside the reviewed bound fails loudly."""
    subject = _subject()
    subject["metrics"] = {"pooled_mean_x": 2.3}

    with pytest.raises(ValueError, match="exceeds reviewed tolerance"):
        _MODULE.validate_cross_tool_comparison(_reference(), subject, _contract())
