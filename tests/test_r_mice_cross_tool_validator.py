"""Contract tests for the scoped Missingly-versus-R ``mice`` comparison."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

import missingly


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mice_numeric_fcs"
_SCRIPT = _FIXTURE_DIR / "validate_cross_tool_comparison.py"
_SPEC = importlib.util.spec_from_file_location("r_mice_cross_tool_validator", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_DIGEST = "a" * 64


def _reference() -> dict[str, object]:
    """Return a minimal scalar R reference artifact."""
    return {
        "reference_tool": "R mice",
        "reference_tool_version": "3.19.0",
        "seed": 7,
        "dataset_sha256": _DIGEST,
        "metrics": {
            "chain_1_mean_x": 1.0,
            "chain_2_mean_x": 3.0,
        },
    }


def _subject() -> dict[str, object]:
    """Return a minimal scalar Missingly comparison artifact."""
    return {
        "subject_tool": "Missingly",
        "subject_tool_version": missingly.__version__,
        "seed": 7,
        "dataset_sha256": _DIGEST,
        "metrics": {"pooled_mean_x": 2.2},
    }


def _contract() -> dict[str, object]:
    """Return a reviewed single-estimand comparison contract."""
    return {
        "schema_version": 1,
        "reference_tool": "R mice",
        "reference_tool_version": "3.19.0",
        "seed": 7,
        "dataset_sha256": _DIGEST,
        "estimand": "pooled completed-data means",
        "tolerances": {"pooled_mean_x": 0.25},
    }


def test_committed_fixture_bytes_match_contract_digest() -> None:
    """The committed CSV hashes to the digest that the contract records."""
    csv_bytes = (_FIXTURE_DIR / "numeric_fcs.csv").read_bytes()
    contract = json.loads((_FIXTURE_DIR / "cross_tool_contract.json").read_text(encoding="utf-8"))

    assert sha256(csv_bytes).hexdigest() == contract["dataset_sha256"]


def test_validator_accepts_scoped_pooled_mean_comparison() -> None:
    """The comparison accepts a subject result inside the reviewed bound."""
    _MODULE.validate_cross_tool_comparison(_reference(), _subject(), _contract(), _DIGEST)


def test_validator_rejects_subject_metric_outside_bound() -> None:
    """A cross-tool drift outside the reviewed bound fails loudly."""
    subject = _subject()
    subject["metrics"] = {"pooled_mean_x": 2.3}

    with pytest.raises(ValueError, match="exceeds reviewed tolerance"):
        _MODULE.validate_cross_tool_comparison(_reference(), subject, _contract(), _DIGEST)


def test_validator_rejects_empty_tolerance_set() -> None:
    """An empty tolerance map must not make any subject result pass."""
    contract = _contract()
    contract["tolerances"] = {}

    with pytest.raises(ValueError, match="must not be empty"):
        _MODULE.validate_cross_tool_comparison(_reference(), _subject(), contract, _DIGEST)


def test_validator_rejects_subject_dataset_hash_not_matching_the_data() -> None:
    """The subject digest must match the hash computed from the CSV actually read."""
    with pytest.raises(ValueError, match="dataset_sha256"):
        _MODULE.validate_cross_tool_comparison(
            _reference(), _subject(), _contract(), "b" * 64
        )


@pytest.mark.parametrize("version", ["1.0.0", None])
def test_validator_rejects_subject_version_not_running_package(version) -> None:
    """A subject artifact from another, or an unrecorded, Missingly version fails."""
    subject = _subject()
    if version is None:
        del subject["subject_tool_version"]
    else:
        subject["subject_tool_version"] = version

    with pytest.raises(ValueError, match="version"):
        _MODULE.validate_cross_tool_comparison(_reference(), subject, _contract(), _DIGEST)


def test_main_hashes_the_fixture_csv_it_reads(tmp_path) -> None:
    """A CSV edited after the subject artifact was written is rejected by main."""
    shutil.copy(_FIXTURE_DIR / "cross_tool_contract.json", tmp_path / "cross_tool_contract.json")
    contract = json.loads((tmp_path / "cross_tool_contract.json").read_text(encoding="utf-8"))
    csv_bytes = bytearray((_FIXTURE_DIR / "numeric_fcs.csv").read_bytes())
    csv_bytes[-1] ^= 0x01
    (tmp_path / "numeric_fcs.csv").write_bytes(bytes(csv_bytes))
    identity = {
        "seed": contract["seed"],
        "dataset_sha256": contract["dataset_sha256"],
    }
    reference = {"reference_tool": "R mice", "reference_tool_version": "3.19.0", "metrics": {}, **identity}
    subject = {
        "subject_tool": "Missingly",
        "subject_tool_version": missingly.__version__,
        "metrics": {},
        **identity,
    }
    reference_path = _write_json(tmp_path / "reference.json", reference)
    subject_path = _write_json(tmp_path / "subject.json", subject)

    with pytest.raises(ValueError, match="dataset_sha256"):
        _MODULE.main(reference_path, subject_path, tmp_path / "cross_tool_contract.json")


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    """Write a JSON artifact and return its path."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
