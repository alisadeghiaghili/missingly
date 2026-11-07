"""Validate a narrowly scoped Missingly-versus-R ``mice`` comparison."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import missingly


_DATASET_FILENAME = "numeric_fcs.csv"
_SCOPED_PREFIX = "pooled_mean_"


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return sha256(path.read_bytes()).hexdigest()


def _check_contract_identity(
    reference: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    """Check the contract schema and that the R reference matches the contract."""
    if contract.get("schema_version") != 1:
        raise ValueError("cross-tool contract schema_version must be 1")
    for key in ("reference_tool", "reference_tool_version", "seed", "dataset_sha256"):
        if reference.get(key) != contract.get(key):
            raise ValueError(f"R reference {key} does not match comparison contract")


def _check_subject_identity(
    subject: Mapping[str, Any],
    contract: Mapping[str, Any],
    dataset_sha256: str,
) -> None:
    """Check the Missingly subject tool, running version, seed and data digest."""
    if subject.get("subject_tool") != "Missingly":
        raise ValueError("comparison subject must be Missingly")
    if subject.get("subject_tool_version") != missingly.__version__:
        raise ValueError(
            f"Missingly subject version {subject.get('subject_tool_version')!r} "
            f"does not match running version {missingly.__version__!r}"
        )
    if subject.get("seed") != contract.get("seed"):
        raise ValueError("Missingly subject seed does not match comparison contract")
    if subject.get("dataset_sha256") != contract.get("dataset_sha256"):
        raise ValueError(
            "Missingly subject dataset_sha256 does not match comparison contract"
        )
    if dataset_sha256 != contract.get("dataset_sha256"):
        raise ValueError(
            "fixture CSV dataset_sha256 does not match comparison contract"
        )


def _metric_mappings(
    reference: Mapping[str, Any],
    subject: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    """Return the reference metrics, subject metrics and tolerances as mappings."""
    mappings = (
        reference.get("metrics"),
        subject.get("metrics"),
        contract.get("tolerances"),
    )
    if not all(isinstance(item, Mapping) for item in mappings):
        raise ValueError("comparison artifacts must provide metric mappings")
    return mappings


def _check_scoped_mean(
    subject_name: Any,
    tolerance: Any,
    reference_metrics: Mapping[str, Any],
    subject_metrics: Mapping[str, Any],
) -> None:
    """Compare one pooled column mean with the average of the R chain means."""
    if not isinstance(subject_name, str) or not subject_name.startswith(_SCOPED_PREFIX):
        raise ValueError("comparison tolerances must target pooled mean metrics")
    column = subject_name.removeprefix(_SCOPED_PREFIX)
    chain_values = [
        float(value)
        for name, value in reference_metrics.items()
        if name.endswith(f"_mean_{column}")
    ]
    if not chain_values or subject_name not in subject_metrics:
        raise ValueError(f"missing scoped metric for column {column!r}")
    target = sum(chain_values) / len(chain_values)
    actual = float(subject_metrics[subject_name])
    if not all(
        math.isfinite(number) for number in (*chain_values, actual, float(tolerance))
    ):
        raise ValueError("cross-tool metrics and tolerances must be finite")
    if float(tolerance) < 0 or abs(actual - target) > float(tolerance):
        raise ValueError(f"{subject_name} exceeds reviewed tolerance")


def validate_cross_tool_comparison(
    reference: Mapping[str, Any],
    subject: Mapping[str, Any],
    contract: Mapping[str, Any],
    dataset_sha256: str,
) -> None:
    """Compare pooled completed-data means under a reviewed contract.

    Parameters
    ----------
    reference : mapping of str to Any
        Scalar artifact emitted by the pinned R ``mice`` generator.
    subject : mapping of str to Any
        Scalar artifact emitted by Missingly on the identical fixture. Its
        ``subject_tool_version`` must equal the running ``missingly.__version__``.
    contract : mapping of str to Any
        Reviewed identity fields and per-estimand absolute tolerances.
    dataset_sha256 : str
        SHA-256 hex digest computed from the fixture bytes actually read.

    Returns
    -------
    None
        Returns normally only when each scoped pooled mean is within tolerance.

    Raises
    ------
    ValueError
        If identity fields differ, the subject version or data digest does not
        match, the tolerance set is empty, metrics are malformed, or a reviewed
        bound is exceeded.

    Examples
    --------
    >>> import missingly
    >>> digest = "a" * 64
    >>> reference = {"reference_tool": "R mice", "reference_tool_version": "1", "seed": 1, "dataset_sha256": digest, "metrics": {"chain_1_mean_x": 2.0}}
    >>> subject = {"subject_tool": "Missingly", "subject_tool_version": missingly.__version__, "seed": 1, "dataset_sha256": digest, "metrics": {"pooled_mean_x": 2.0}}
    >>> contract = {"schema_version": 1, "reference_tool": "R mice", "reference_tool_version": "1", "seed": 1, "dataset_sha256": digest, "estimand": "mean", "tolerances": {"pooled_mean_x": 0.0}}
    >>> validate_cross_tool_comparison(reference, subject, contract, digest)
    """
    _check_contract_identity(reference, contract)
    _check_subject_identity(subject, contract, dataset_sha256)
    reference_metrics, subject_metrics, tolerances = _metric_mappings(
        reference, subject, contract
    )
    if not tolerances:
        raise ValueError("comparison tolerance set must not be empty")
    for subject_name, tolerance in tolerances.items():
        _check_scoped_mean(subject_name, tolerance, reference_metrics, subject_metrics)


def main(reference_path: Path, subject_path: Path, contract_path: Path) -> None:
    """Validate three JSON artifacts supplied by the CI workflow.

    The dataset digest is computed from the fixture CSV that sits next to the
    contract, not taken from either artifact.

    Parameters
    ----------
    reference_path : pathlib.Path
        R ``mice`` scalar artifact path.
    subject_path : pathlib.Path
        Missingly scalar artifact path.
    contract_path : pathlib.Path
        Reviewed cross-tool comparison contract path.

    Returns
    -------
    None
        Returns normally only after the scoped comparison passes.
    """
    reference, subject, contract = (
        json.loads(path.read_text(encoding="utf-8"))
        for path in (reference_path, subject_path, contract_path)
    )
    dataset_sha256 = _sha256_file(contract_path.parent / _DATASET_FILENAME)
    validate_cross_tool_comparison(reference, subject, contract, dataset_sha256)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: validate_cross_tool_comparison.py <r.json> <missingly.json> <contract.json>"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
