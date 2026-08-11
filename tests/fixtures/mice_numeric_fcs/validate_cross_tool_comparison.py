"""Validate a narrowly scoped Missingly-versus-R ``mice`` comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


def validate_cross_tool_comparison(
    reference: Mapping[str, Any],
    subject: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Compare pooled completed-data means under a reviewed contract.

    Parameters
    ----------
    reference : mapping of str to Any
        Scalar artifact emitted by the pinned R ``mice`` generator.
    subject : mapping of str to Any
        Scalar artifact emitted by Missingly on the identical fixture.
    contract : mapping of str to Any
        Reviewed identity fields and per-estimand absolute tolerances.

    Returns
    -------
    None
        Returns normally only when each scoped pooled mean is within tolerance.

    Raises
    ------
    ValueError
        If identity fields differ, metrics are malformed, or a reviewed bound
        is exceeded.

    Examples
    --------
    >>> reference = {"reference_tool": "R mice", "reference_tool_version": "1", "seed": 1, "dataset_sha256": "a", "metrics": {"chain_1_mean_x": 2.0}}
    >>> subject = {"subject_tool": "Missingly", "seed": 1, "dataset_sha256": "a", "metrics": {"pooled_mean_x": 2.0}}
    >>> contract = {"schema_version": 1, "reference_tool": "R mice", "reference_tool_version": "1", "seed": 1, "dataset_sha256": "a", "estimand": "mean", "tolerances": {"pooled_mean_x": 0.0}}
    >>> validate_cross_tool_comparison(reference, subject, contract)
    """
    if contract.get("schema_version") != 1:
        raise ValueError("cross-tool contract schema_version must be 1")
    for key in ("reference_tool", "reference_tool_version", "seed", "dataset_sha256"):
        if reference.get(key) != contract.get(key):
            raise ValueError(f"R reference {key} does not match comparison contract")
    if subject.get("subject_tool") != "Missingly":
        raise ValueError("comparison subject must be Missingly")
    for key in ("seed", "dataset_sha256"):
        if subject.get(key) != contract.get(key):
            raise ValueError(f"Missingly subject {key} does not match comparison contract")

    reference_metrics = reference.get("metrics")
    subject_metrics = subject.get("metrics")
    tolerances = contract.get("tolerances")
    if not all(isinstance(item, Mapping) for item in (reference_metrics, subject_metrics, tolerances)):
        raise ValueError("comparison artifacts must provide metric mappings")
    for subject_name, tolerance in tolerances.items():
        if not isinstance(subject_name, str) or not subject_name.startswith("pooled_mean_"):
            raise ValueError("comparison tolerances must target pooled mean metrics")
        column = subject_name.removeprefix("pooled_mean_")
        chain_values = [
            float(value) for name, value in reference_metrics.items()
            if name.endswith(f"_mean_{column}")
        ]
        if not chain_values or subject_name not in subject_metrics:
            raise ValueError(f"missing scoped metric for column {column!r}")
        target = sum(chain_values) / len(chain_values)
        actual = float(subject_metrics[subject_name])
        if not all(math.isfinite(number) for number in (*chain_values, actual, float(tolerance))):
            raise ValueError("cross-tool metrics and tolerances must be finite")
        if float(tolerance) < 0 or abs(actual - target) > float(tolerance):
            raise ValueError(f"{subject_name} exceeds reviewed tolerance")


def main(reference_path: Path, subject_path: Path, contract_path: Path) -> None:
    """Validate three JSON artifacts supplied by the CI workflow.

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
    paths = (reference_path, subject_path, contract_path)
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    validate_cross_tool_comparison(*values)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: validate_cross_tool_comparison.py <r.json> <missingly.json> <contract.json>")
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
