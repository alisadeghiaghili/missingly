"""Contract tests for validation of the locked R Little MCAR evidence artifact."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parent / "fixtures" / "little_mcar" / "validate_naniar_reference.py"
_MANIFEST = Path(__file__).parent / "fixtures" / "little_mcar" / "benchmark_manifest.json"


@pytest.fixture(scope="module")
def validator_module():
    """Load the standalone CI validator without changing package imports."""
    spec = importlib.util.spec_from_file_location("r_oracle_validator", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_accepts_matching_scalar_evidence(tmp_path, validator_module):
    """A complete locked-R artifact matching manifest metrics is accepted."""
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    reference = {
        "dataset_sha256": manifest["dataset_sha256"],
        "reference_tool": "R naniar",
        "source_script_sha256": "a" * 64,
        "metrics": {
            "chi_square": manifest["expected_metrics"]["chi_square"],
            "degrees_of_freedom": manifest["expected_metrics"]["degrees_of_freedom"],
            "p_value": manifest["expected_metrics"]["p_value"],
            "missing_patterns": manifest["expected_metrics"]["missing_patterns"],
        },
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    validator_module.main(reference_path, _MANIFEST)


def test_validator_rejects_metric_or_identity_drift(tmp_path, validator_module):
    """Evidence drift cannot silently pass the cross-tool regression gate."""
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    reference = {
        "dataset_sha256": "b" * 64,
        "reference_tool": "R naniar",
        "source_script_sha256": "a" * 64,
        "metrics": {
            "chi_square": manifest["expected_metrics"]["chi_square"],
            "degrees_of_freedom": manifest["expected_metrics"]["degrees_of_freedom"],
            "p_value": manifest["expected_metrics"]["p_value"],
            "missing_patterns": manifest["expected_metrics"]["missing_patterns"],
        },
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        validator_module.main(reference_path, _MANIFEST)
