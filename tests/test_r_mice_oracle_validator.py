"""Contract tests for the R ``mice`` evidence-only artifact validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parent
    / "fixtures"
    / "mice_numeric_fcs"
    / "validate_mice_reference.py"
)
_SPEC = importlib.util.spec_from_file_location("r_mice_oracle_validator", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _valid_artifact() -> dict[str, object]:
    """Return a minimal valid evidence-only R ``mice`` artifact."""
    return {
        "schema_version": 1,
        "artifact_kind": "evidence-only",
        "reference_tool": "R mice",
        "reference_tool_version": "3.18.0",
        "r_version": "4.4.2",
        "platform": "x86_64-pc-linux-gnu",
        "method": "mice::mice(method='norm', m=3, maxit=5)",
        "seed": 20260811,
        "dataset_sha256": "a" * 64,
        "source_script_sha256": "b" * 64,
        "metrics": {"chain_1_mean_x1": 3.5},
    }


def test_validator_accepts_complete_evidence_only_artifact() -> None:
    """The public schema accepts finite aggregate provenance evidence."""
    _MODULE.validate_reference(_valid_artifact())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_kind", "conformance", "evidence-only"),
        ("reference_tool", "Python", "unexpected tool"),
        ("dataset_sha256", "not-a-digest", "SHA-256"),
        ("metrics", {"chain_1_mean_x1": float("inf")}, "finite"),
    ],
)
def test_validator_rejects_invalid_artifacts(
    field: str,
    value: object,
    message: str,
) -> None:
    """Invalid provenance or aggregate metrics fail loudly."""
    artifact = _valid_artifact()
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        _MODULE.validate_reference(artifact)
