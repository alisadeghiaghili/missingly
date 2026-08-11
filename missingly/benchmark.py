"""Versioned, privacy-preserving contracts for cross-tool benchmarks.

Benchmark manifests capture the evidence needed to reproduce a numerical
comparison without embedding the benchmark's raw records.  They are immutable,
strictly validated, and have a canonical JSON representation suitable for
reviewed fixtures and artifact indexes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from missingly._version import __version__

__all__ = ["BenchmarkManifest"]

_SCHEMA_VERSION = 2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = {
    "schema_version",
    "dataset_name",
    "dataset_license",
    "dataset_sha256",
    "dataset_source_url",
    "reference_tool",
    "reference_tool_version",
    "reference_platform",
    "reference_script_sha256",
    "method",
    "missing_value_policy",
    "estimand",
    "seed",
    "expected_metrics",
    "tolerances",
    "known_differences",
    "missingly_version",
}


def _validate_nonempty_string(value: str, name: str) -> None:
    """Reject empty or non-string manifest identity fields."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_numeric_mapping(
    value: Mapping[str, float],
    name: str,
    *,
    allow_negative: bool,
) -> None:
    """Validate named finite numerical results or tolerances."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping of metric names to numbers")
    if not value:
        raise ValueError(f"{name} must contain at least one metric")
    for metric, number in value.items():
        _validate_nonempty_string(metric, f"{name} metric")
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise TypeError(f"{name}[{metric!r}] must be a finite number")
        if not math.isfinite(float(number)):
            raise ValueError(f"{name}[{metric!r}] must be finite")
        if not allow_negative and float(number) < 0:
            raise ValueError(f"{name}[{metric!r}] must be non-negative")


@dataclass(frozen=True)
class BenchmarkManifest:
    """Describe one reproducible cross-tool benchmark without raw records.

    Parameters
    ----------
    dataset_name : str
        Stable dataset identifier used in the repository or public catalogue.
    dataset_license : str
        Licence or terms that permit the benchmark fixture to be used.
    dataset_sha256 : str
        Lowercase SHA-256 of the exact source dataset bytes.
    dataset_source_url : str
        Canonical source URL for the dataset, not a path to a local copy.
    reference_tool : str
        Tool that generated the frozen reference results, such as ``"R mice"``.
    reference_tool_version : str
        Exact version of the reference tool or package.
    reference_platform : str
        Operating system and runtime/platform used for the reference output.
    reference_script_sha256 : str
        Lowercase SHA-256 of the script that produced the reference output.
    method : str
        Fully qualified reference method and material configuration summary.
    missing_value_policy : str
        How each tool interprets sentinels, nulls, and dropped rows.
    estimand : str
        Quantity being compared, for example ``"Little MCAR chi-square"``.
    expected_metrics : mapping of str to float
        Frozen scalar reference results.  These are outputs, not raw records.
    tolerances : mapping of str to float
        Non-negative absolute or relative tolerances keyed by metric name.
    known_differences : tuple of str, optional
        Intentional algorithmic differences that reviewers must consider.
    seed : int, optional
        Random seed used by the reference tool. ``None`` means deterministic.
    schema_version : int, default=1
        Manifest schema version. Only the current version is accepted.
    missingly_version : str, optional
        Missingly version used to reproduce the comparison. Defaults to the
        installed package version.

    Raises
    ------
    TypeError
        If a field has an incompatible type.
    ValueError
        If an identity field is empty, a digest is invalid, a metric is not
        finite, or a tolerance is negative.

    Examples
    --------
    >>> manifest = BenchmarkManifest(
    ...     dataset_name="airquality",
    ...     dataset_license="CC0-1.0",
    ...     dataset_sha256="a" * 64,
    ...     dataset_source_url="https://example.test/airquality.csv",
    ...     reference_tool="R naniar",
    ...     reference_tool_version="1.1.0",
    ...     reference_platform="R 4.4 on Linux",
    ...     reference_script_sha256="b" * 64,
    ...     method="naniar::mcar_test",
    ...     missing_value_policy="NA values retained",
    ...     estimand="Little MCAR chi-square",
    ...     expected_metrics={"chi_square": 25.1},
    ...     tolerances={"chi_square": 1e-6},
    ... )
    >>> len(manifest.sha256())
    64
    >>> "raw records" in manifest.to_json()
    False
    """

    dataset_name: str
    dataset_license: str
    dataset_sha256: str
    dataset_source_url: str
    reference_tool: str
    reference_tool_version: str
    reference_platform: str
    reference_script_sha256: str
    method: str
    missing_value_policy: str
    estimand: str
    expected_metrics: Mapping[str, float]
    tolerances: Mapping[str, float]
    known_differences: Tuple[str, ...] = ()
    seed: Optional[int] = None
    schema_version: int = _SCHEMA_VERSION
    missingly_version: str = __version__

    def __post_init__(self) -> None:
        """Enforce the manifest's stable and privacy-preserving schema."""
        for name in (
            "dataset_name", "dataset_license", "dataset_source_url",
            "reference_tool", "reference_tool_version", "reference_platform",
            "method", "missing_value_policy", "estimand", "missingly_version",
        ):
            _validate_nonempty_string(getattr(self, name), name)
        if not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if not isinstance(self.dataset_sha256, str):
            raise TypeError("dataset_sha256 must be a lowercase hexadecimal string")
        if not _SHA256_PATTERN.fullmatch(self.dataset_sha256):
            raise ValueError("dataset_sha256 must be a lowercase 64-character SHA-256")
        if not isinstance(self.reference_script_sha256, str):
            raise TypeError("reference_script_sha256 must be a lowercase hexadecimal string")
        if not _SHA256_PATTERN.fullmatch(self.reference_script_sha256):
            raise ValueError(
                "reference_script_sha256 must be a lowercase 64-character SHA-256"
            )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        _validate_numeric_mapping(self.expected_metrics, "expected_metrics", allow_negative=True)
        _validate_numeric_mapping(self.tolerances, "tolerances", allow_negative=False)
        if not isinstance(self.known_differences, tuple):
            raise TypeError("known_differences must be a tuple of strings")
        for difference in self.known_differences:
            _validate_nonempty_string(difference, "known_differences item")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible manifest dictionary without raw records.

        Returns
        -------
        dict
            Strict schema fields only. Mappings are copied so callers cannot
            mutate the manifest through the returned representation.

        Examples
        --------
        >>> manifest = BenchmarkManifest(
        ...     "toy", "CC0", "b" * 64, "https://example.test/toy",
        ...     "reference", "1.0", "Linux", "b" * 64, "method", "NA", "mean",
        ...     {"estimate": 1.0}, {"estimate": 0.01},
        ... )
        >>> sorted(manifest.to_dict())[0]
        'dataset_license'
        """
        payload = asdict(self)
        payload["expected_metrics"] = dict(self.expected_metrics)
        payload["tolerances"] = dict(self.tolerances)
        return payload

    def to_json(self) -> str:
        """Serialize the manifest canonically for committed evidence artifacts.

        Returns
        -------
        str
            UTF-8-safe JSON with sorted keys and compact, deterministic spacing.

        Examples
        --------
        >>> manifest = BenchmarkManifest(
        ...     "toy", "CC0", "c" * 64, "https://example.test/toy",
        ...     "reference", "1.0", "Linux", "c" * 64, "method", "NA", "mean",
        ...     {"estimate": 1.0}, {"estimate": 0.01},
        ... )
        >>> manifest.to_json() == manifest.to_json()
        True
        """
        return json.dumps(
            self.to_dict(), allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        )

    def sha256(self) -> str:
        """Return the SHA-256 identity of the canonical manifest JSON.

        Returns
        -------
        str
            Lowercase, 64-character hexadecimal digest of :meth:`to_json`.

        Examples
        --------
        >>> manifest = BenchmarkManifest(
        ...     "toy", "CC0", "d" * 64, "https://example.test/toy",
        ...     "reference", "1.0", "Linux", "d" * 64, "method", "NA", "mean",
        ...     {"estimate": 1.0}, {"estimate": 0.01},
        ... )
        >>> len(manifest.sha256())
        64
        """
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkManifest":
        """Create a manifest from an exact-schema JSON-compatible mapping.

        Parameters
        ----------
        payload : mapping of str to Any
            Decoded manifest data. Unknown fields, including fields that could
            hold raw rows or records, are rejected rather than silently kept.

        Returns
        -------
        BenchmarkManifest
            Validated immutable benchmark manifest.

        Raises
        ------
        TypeError
            If ``payload`` is not a mapping.
        ValueError
            If the mapping keys differ from the published schema.

        Examples
        --------
        >>> manifest = BenchmarkManifest(
        ...     "toy", "CC0", "e" * 64, "https://example.test/toy",
        ...     "reference", "1.0", "Linux", "e" * 64, "method", "NA", "mean",
        ...     {"estimate": 1.0}, {"estimate": 0.01},
        ... )
        >>> BenchmarkManifest.from_dict(manifest.to_dict()) == manifest
        True
        """
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        keys = set(payload)
        if keys != _MANIFEST_KEYS:
            missing = sorted(_MANIFEST_KEYS - keys)
            unexpected = sorted(keys - _MANIFEST_KEYS)
            raise ValueError(
                "payload keys must match the benchmark manifest schema; "
                f"missing={missing}, unexpected={unexpected}"
            )
        values = dict(payload)
        values["known_differences"] = tuple(values["known_differences"])
        return cls(**values)
