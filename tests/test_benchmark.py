"""Contract tests for versioned cross-tool benchmark evidence."""

from __future__ import annotations

from dataclasses import replace
import json
import re

import pytest

import missingly
from missingly.benchmark import BenchmarkManifest


@pytest.fixture()
def manifest() -> BenchmarkManifest:
    """Return a valid benchmark manifest with no raw-record field."""
    return BenchmarkManifest(
        dataset_name="airquality",
        dataset_license="CC0-1.0",
        dataset_sha256="7d6e9e34d108111271ef4b3223060811095510ce16c6ea1fd3eacb2617a77fb3",
        dataset_source_url="https://vincentarelbundock.github.io/Rdatasets/csv/datasets/airquality.csv",
        reference_tool="R naniar",
        reference_tool_version="1.1.0",
        reference_platform="R 4.4.2 on Ubuntu 24.04",
        method="naniar::mcar_test",
        missing_value_policy="R NA values retained; complete rows are not pre-dropped",
        estimand="Little MCAR chi-square and degrees of freedom",
        expected_metrics={"chi_square": 25.101, "degrees_of_freedom": 15.0},
        tolerances={"chi_square": 1e-6, "degrees_of_freedom": 0.0},
        known_differences=("EM convergence tolerance is explicitly recorded by each tool.",),
        seed=1729,
    )


def test_manifest_is_public_deterministic_and_round_trips(manifest):
    """Canonical serialisation has stable identity and reconstructs exactly."""
    assert missingly.BenchmarkManifest is BenchmarkManifest
    assert manifest.to_json() == manifest.to_json()
    assert re.fullmatch(r"[0-9a-f]{64}", manifest.sha256())
    assert BenchmarkManifest.from_dict(json.loads(manifest.to_json())) == manifest


def test_manifest_digest_is_mapping_order_independent_but_content_sensitive(manifest):
    """Mapping insertion order is irrelevant while benchmark content is material."""
    reordered = replace(
        manifest,
        expected_metrics={"degrees_of_freedom": 15.0, "chi_square": 25.101},
    )
    changed = replace(manifest, reference_tool_version="1.1.1")

    assert reordered.sha256() == manifest.sha256()
    assert changed.sha256() != manifest.sha256()


def test_manifest_rejects_invalid_or_unsafe_contracts(manifest):
    """Invalid identity, tolerance, and raw-record-shaped fields fail loudly."""
    with pytest.raises(ValueError, match="SHA-256"):
        replace(manifest, dataset_sha256="not-a-digest")
    with pytest.raises(ValueError, match="non-negative"):
        replace(manifest, tolerances={"chi_square": -0.1})
    with pytest.raises(TypeError, match="tuple"):
        replace(manifest, known_differences=["list is intentionally not accepted"])

    payload = manifest.to_dict()
    payload["rows"] = [["private", "record"]]
    with pytest.raises(ValueError, match="unexpected=.*rows"):
        BenchmarkManifest.from_dict(payload)


def test_manifest_dict_only_contains_evidence_metadata(manifest):
    """The public artifact has fixed metadata fields and no raw data container."""
    payload = manifest.to_dict()

    assert set(payload) == {
        "schema_version", "dataset_name", "dataset_license", "dataset_sha256",
        "dataset_source_url", "reference_tool", "reference_tool_version",
        "reference_platform", "method", "missing_value_policy", "estimand", "seed",
        "expected_metrics", "tolerances", "known_differences", "missingly_version",
    }
    assert not {"data", "rows", "records", "raw_data"} & set(payload)
