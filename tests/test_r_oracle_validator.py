"""Contract tests for validation of the locked R Little MCAR evidence artifact."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "little_mcar"
_SCRIPT = _FIXTURE_DIR / "validate_naniar_reference.py"
_MANIFEST = _FIXTURE_DIR / "benchmark_manifest.json"
_GENERATOR = _FIXTURE_DIR / "generate_naniar_reference.R"
_METRICS = ("chi_square", "degrees_of_freedom", "p_value", "missing_patterns")


@pytest.fixture(scope="module")
def validator_module():
    """Load the standalone CI validator without changing package imports."""
    spec = importlib.util.spec_from_file_location("r_oracle_validator", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict:
    """Return the committed Little's MCAR benchmark manifest."""
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def _reference(manifest: dict) -> dict:
    """Return a locked-R artifact that matches the committed manifest."""
    expected = manifest["expected_metrics"]
    return {
        "dataset_sha256": manifest["dataset_sha256"],
        "reference_tool": "R naniar",
        "reference_tool_version": manifest["reference_tool_version"],
        "r_version": "4.4.2",
        "platform": "x86_64-pc-linux-gnu",
        "source_script_sha256": manifest["reference_script_sha256"],
        "metrics": {name: expected[name] for name in _METRICS},
    }


def _write(directory: Path, name: str, payload: dict) -> Path:
    """Write a JSON artifact and return its path."""
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validator_accepts_matching_scalar_evidence(tmp_path, validator_module):
    """A complete locked-R artifact matching manifest identity and metrics is accepted."""
    reference_path = _write(tmp_path, "reference.json", _reference(_manifest()))

    validator_module.main(reference_path, _MANIFEST)


def test_validator_rejects_metric_or_identity_drift(tmp_path, validator_module):
    """Evidence drift cannot silently pass the cross-tool regression gate."""
    reference = _reference(_manifest())
    reference["dataset_sha256"] = "b" * 64
    reference_path = _write(tmp_path, "reference.json", reference)

    with pytest.raises(ValueError, match="SHA-256"):
        validator_module.main(reference_path, _MANIFEST)


def test_validator_rejects_tampered_generator_script(tmp_path, validator_module):
    """A one-byte edit to the generator script fails against the recorded SHA-256."""
    fixture = tmp_path / "little_mcar"
    fixture.mkdir()
    manifest_path = fixture / "benchmark_manifest.json"
    shutil.copy(_MANIFEST, manifest_path)
    tampered = bytearray(_GENERATOR.read_bytes())
    tampered[0] ^= 0x01
    (fixture / _GENERATOR.name).write_bytes(bytes(tampered))
    reference = _reference(_manifest())
    reference["source_script_sha256"] = sha256(bytes(tampered)).hexdigest()
    reference_path = _write(tmp_path, "reference.json", reference)

    with pytest.raises(ValueError, match="generator script SHA-256"):
        validator_module.main(reference_path, manifest_path)


def test_validator_rejects_script_hash_recorded_by_a_different_script(tmp_path, validator_module):
    """An artifact whose recorded script hash differs from the committed script fails."""
    reference = _reference(_manifest())
    reference["source_script_sha256"] = "c" * 64
    reference_path = _write(tmp_path, "reference.json", reference)

    with pytest.raises(ValueError, match="source-script SHA-256"):
        validator_module.main(reference_path, _MANIFEST)


def test_validator_rejects_wrong_recorded_tool_version(tmp_path, validator_module):
    """A naniar version other than the locked 1.1.0 fails even when scalars match."""
    reference = _reference(_manifest())
    reference["reference_tool_version"] = "1.0.0"
    reference_path = _write(tmp_path, "reference.json", reference)

    with pytest.raises(ValueError, match="tool version 1.0.0"):
        validator_module.main(reference_path, _MANIFEST)


@pytest.mark.parametrize(
    ("r_version", "platform"),
    [("4.3.3", "x86_64-pc-linux-gnu"), ("4.4.2", "aarch64-apple-darwin20")],
)
def test_validator_rejects_wrong_recorded_r_version_or_platform(
    tmp_path, validator_module, r_version, platform
):
    """The R version and platform must match the locked reference platform."""
    reference = _reference(_manifest())
    reference["r_version"] = r_version
    reference["platform"] = platform
    reference_path = _write(tmp_path, "reference.json", reference)

    with pytest.raises(ValueError, match="R version and platform"):
        validator_module.main(reference_path, _MANIFEST)
