"""Contract tests for deterministic analysis provenance."""

from __future__ import annotations

import re
from datetime import date, datetime, time
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import missingly
from missingly.provenance import _canonical_json, analysis_provenance, dataframe_fingerprint


@pytest.fixture()
def mixed_frame() -> pd.DataFrame:
    """Return a representative frame with semantic pandas dtypes."""
    return pd.DataFrame(
        {
            "score": [1.5, np.nan, 3.25],
            "group": pd.Categorical(
                ["control", "treated", None],
                categories=["control", "treated"],
                ordered=True,
            ),
            "observed_at": pd.to_datetime(
                ["2026-01-01T00:00:00Z", None, "2026-01-03T00:00:00Z"]
            ),
        },
        index=pd.Index([101, 102, 103], name="case_id"),
    )


def test_fingerprint_is_stable_and_does_not_mutate_input(mixed_frame):
    """Equivalent calls must be stable and preserve the caller's frame."""
    before = mixed_frame.copy(deep=True)

    first = dataframe_fingerprint(mixed_frame)
    second = dataframe_fingerprint(mixed_frame.copy(deep=True))

    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    pd.testing.assert_frame_equal(mixed_frame, before)


@pytest.mark.parametrize("change", ["value", "index", "dtype", "category_order"])
def test_fingerprint_detects_semantically_relevant_frame_changes(mixed_frame, change):
    """Values, index, dtypes, and category metadata belong to the contract."""
    changed = mixed_frame.copy(deep=True)
    if change == "value":
        changed.loc[101, "score"] = 99.0
    elif change == "index":
        changed.index = pd.Index([201, 202, 203], name="case_id")
    elif change == "dtype":
        changed["score"] = changed["score"].astype("Float32")
    else:
        changed["group"] = changed["group"].cat.reorder_categories(
            ["treated", "control"], ordered=True
        )

    assert dataframe_fingerprint(changed) != dataframe_fingerprint(mixed_frame)


def test_fingerprint_context_is_order_independent_but_setting_sensitive(mixed_frame):
    """Mapping order is irrelevant while analytical settings are material."""
    first = dataframe_fingerprint(
        mixed_frame,
        operation="impute_mice",
        parameters={"seed": 42, "m": 5},
    )
    reordered = dataframe_fingerprint(
        mixed_frame,
        operation="impute_mice",
        parameters={"m": 5, "seed": 42},
    )
    changed = dataframe_fingerprint(
        mixed_frame,
        operation="impute_mice",
        parameters={"m": 10, "seed": 42},
    )

    assert first == reordered
    assert first != changed


def test_analysis_provenance_separates_input_and_analysis_identity(mixed_frame):
    """The manifest must expose package, input, and configured-run identity."""
    manifest = analysis_provenance(
        mixed_frame,
        operation="impute_mice",
        parameters={"m": 5, "seed": 42},
    )

    assert manifest == {
        "schema_version": 1,
        "package": "missingly",
        "package_version": missingly.__version__,
        "operation": "impute_mice",
        "parameters": {"m": 5, "seed": 42},
        "input_sha256": dataframe_fingerprint(mixed_frame),
        "analysis_sha256": dataframe_fingerprint(
            mixed_frame,
            operation="impute_mice",
            parameters={"m": 5, "seed": 42},
        ),
    }


def test_provenance_rejects_ambiguous_inputs(mixed_frame):
    """Unsupported input contracts must fail loudly rather than hash reprs."""
    with pytest.raises(TypeError, match="pandas DataFrame"):
        dataframe_fingerprint([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="operation must be a string"):
        dataframe_fingerprint(mixed_frame, operation=123)
    with pytest.raises(TypeError, match="parameters must be a mapping"):
        dataframe_fingerprint(mixed_frame, parameters=[("m", 5)])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (pd.NA, {"__missing__": True}),
        (np.float64("nan"), {"__float__": "nan"}),
        (float("inf"), {"__float__": "+inf"}),
        (float("-inf"), {"__float__": "-inf"}),
        (Decimal("1.20"), {"__decimal__": "1.20"}),
        (pd.Timestamp("2026-01-01T00:00:00Z"), {"__timestamp__": "2026-01-01T00:00:00+00:00"}),
        (pd.Timedelta("2 days"), {"__timedelta__": "P2DT0H0M0S"}),
        (datetime(2026, 1, 1, 12, 0), {"__datetime__": "2026-01-01T12:00:00"}),
        (date(2026, 1, 1), {"__date__": "2026-01-01"}),
        (time(12, 0), {"__time__": "12:00:00"}),
        (b"ab", {"__bytes_hex__": "6162"}),
    ],
)
def test_canonical_json_encodes_supported_scalar_types(value, expected):
    """Every documented scalar provenance type has a stable representation."""
    assert _canonical_json(value) == expected


def test_canonical_json_encodes_intervals_and_order_independent_containers():
    """Composite provenance values preserve type while ignoring set/map order."""
    interval = pd.Interval(1, 2, closed="both")

    assert _canonical_json(interval)["__interval__"]["closed"] == "both"
    assert _canonical_json({"b": 2, "a": 1}) == {"a": 1, "b": 2}
    assert _canonical_json({2, 1}) == _canonical_json({1, 2})
    assert _canonical_json(("x", 1)) == {"__tuple__": ["x", 1]}


def test_canonical_json_rejects_unstable_custom_values():
    """Opaque custom objects cannot accidentally enter a provenance digest."""
    with pytest.raises(TypeError, match="Unsupported provenance value"):
        _canonical_json(object())


def test_analysis_provenance_rejects_blank_operation(mixed_frame):
    """Replay manifests require a meaningful operation identity."""
    with pytest.raises(ValueError, match="must not be empty"):
        analysis_provenance(mixed_frame, "   ")
