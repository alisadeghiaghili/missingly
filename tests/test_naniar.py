"""Behavior and rendering coverage for tidy naniar-compatible APIs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from missingly.naniar import (
    gg_miss_case,
    gg_miss_fct,
    gg_miss_upset,
    gg_miss_var,
    n_miss_case,
    n_miss_var,
    pct_miss_var,
    shadow_long,
    shadow_matrix,
    shadow_wide,
    special_missing,
)


@pytest.fixture
def tidy_frame() -> pd.DataFrame:
    """Return a mixed frame with native and sentinel missing values."""
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "score": [1.0, np.nan, -99.0, 4.0],
            "label": ["ok", "N/A", "ok", None],
        }
    )


@pytest.fixture(autouse=True)
def close_figures():
    """Close matplotlib figures created by each rendering test."""
    yield
    plt.close("all")


def test_shadow_representations_preserve_labels_and_sentinels(tidy_frame):
    """All shadow representations agree on native and configured missingness."""
    shadow = shadow_matrix(tidy_frame, missing_values=[-99.0, "N/A"])
    wide = shadow_wide(tidy_frame, missing_values=[-99.0, "N/A"])
    long = shadow_long(tidy_frame, missing_values=[-99.0, "N/A"])

    assert shadow.loc[2, "score"] == "NA"
    assert shadow.loc[0, "label"] == "!"
    assert wide.loc[1, "label"]
    assert long.shape == (tidy_frame.size, 4)
    assert long.loc[(long["row"] == 3) & (long["variable"] == "label"), "missing"].item()


def test_missing_summaries_handle_native_and_configured_missing_values(tidy_frame):
    """Counts and percentages use the same configured sentinel policy."""
    counts = n_miss_var(tidy_frame, missing_values=[-99.0, "N/A"])
    cases = n_miss_case(tidy_frame, missing_values=[-99.0, "N/A"])
    percentages = pct_miss_var(tidy_frame, missing_values=[-99.0, "N/A"])

    assert counts.to_dict() == {"group": 0, "score": 2, "label": 2}
    assert cases.to_dict() == {0: 0, 1: 2, 2: 1, 3: 1}
    assert percentages["score"] == 50.0


def test_special_missing_replaces_scalar_and_list_values_without_mutation(tidy_frame):
    """Configured values are replaced while the original frame remains intact."""
    result = special_missing(tidy_frame, {"score": -99.0, "label": ["N/A"]})

    assert result[["score", "label"]].isna().sum().to_dict() == {"score": 2, "label": 2}
    assert tidy_frame.loc[2, "score"] == -99.0


@pytest.mark.parametrize("show_pct", [True, False])
def test_gg_miss_var_draws_on_supplied_axes(tidy_frame, show_pct):
    """Variable bar charts support both percentage-label rendering branches."""
    _, axis = plt.subplots()

    result = gg_miss_var(tidy_frame, missing_values=[-99.0], show_pct=show_pct, ax=axis)

    assert result is axis
    assert len(axis.patches) == tidy_frame.shape[1]


@pytest.mark.parametrize("show_pct", [True, False])
def test_gg_miss_case_draws_on_supplied_axes(tidy_frame, show_pct):
    """Case bar charts support both percentage-label rendering branches."""
    _, axis = plt.subplots()

    result = gg_miss_case(tidy_frame, missing_values=[-99.0], show_pct=show_pct, ax=axis)

    assert result is axis
    assert len(axis.patches) == len(tidy_frame)


def test_gg_miss_fct_draws_heatmap_and_rejects_unknown_factor(tidy_frame):
    """Factor plots render grouped missingness and validate the grouping key."""
    _, axis = plt.subplots()

    assert gg_miss_fct(tidy_frame, "group", ax=axis) is axis
    with pytest.raises(ValueError, match="Factor column"):
        gg_miss_fct(tidy_frame, "unknown")


def test_gg_miss_upset_draws_complete_and_missing_combinations(tidy_frame):
    """The UpSet-style plot labels both missing and complete combinations."""
    _, axis = plt.subplots()

    assert gg_miss_upset(tidy_frame, missing_values=[-99.0], n_sets=2, ax=axis) is axis
    assert len(axis.patches) == 2
