"""Pattern identification in Little's MCAR test must not depend on column count.

The test statistic is computed per missingness pattern.  Identifying patterns
by packing each row into a 64-bit integer (one bit per column) silently merges
distinct patterns once a frame has 64 or more columns, which corrupts the
reported pattern count, the degrees of freedom and the p-value.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from missingly.diagnostics import _missingness_patterns, mcar_test


def _frame_with_known_patterns(
    n_cols: int,
    hole_columns: Tuple[int, ...],
    n_rows: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """Normal data with independent 20% holes in *hole_columns* only."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_rows, n_cols))
    for col in hole_columns:
        values[rng.random(n_rows) < 0.2, col] = np.nan
    return pd.DataFrame(values)


def _independent_pattern_count(frame: pd.DataFrame) -> int:
    """Count distinct missingness rows without any integer encoding."""
    return len({tuple(row) for row in frame.isna().to_numpy()})


def _independent_df(frame: pd.DataFrame) -> int:
    """Little's df: sum over patterns of observed-column count, minus d."""
    mask = frame.isna().to_numpy()
    patterns = {tuple(row) for row in mask}
    return sum(int((~np.array(p)).sum()) for p in patterns) - frame.shape[1]


def test_mcar_test_counts_patterns_exactly_with_70_columns() -> None:
    """Holes in bits 0, 64 and 65 give 2**3 = 8 patterns, not 2."""
    frame = _frame_with_known_patterns(70, hole_columns=(0, 64, 65))
    assert _independent_pattern_count(frame) == 8

    result = mcar_test(frame)

    assert result["missing_patterns"] == 8
    assert result["df"] == _independent_df(frame)


def test_mcar_test_70_columns_matches_same_holes_in_30_columns() -> None:
    """Moving the holes to low-order columns must not change the pattern count."""
    wide = _frame_with_known_patterns(70, hole_columns=(0, 64, 65))
    narrow = wide.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 64, 65]].copy()
    narrow.columns = range(narrow.shape[1])

    assert mcar_test(wide)["missing_patterns"] == mcar_test(narrow)["missing_patterns"]


@pytest.mark.parametrize("n_cols", range(1, 131))
def test_missingness_patterns_agree_with_independent_count(n_cols: int) -> None:
    """The pattern helper agrees with a tuple-set count for 1 to 130 columns."""
    rng = np.random.default_rng(n_cols)
    n_rows = 150
    mask = rng.random((n_rows, n_cols)) < 0.05
    # Force some very high-order columns to carry their own holes.
    mask[:, -1] |= rng.random(n_rows) < 0.3
    mask[:, n_cols // 2] |= rng.random(n_rows) < 0.3

    patterns, inverse = _missingness_patterns(mask)

    assert len(patterns) == len({tuple(row) for row in mask})
    assert inverse.shape == (n_rows,)
    np.testing.assert_array_equal(patterns[inverse], mask)


def test_missingness_patterns_order_matches_integer_codes_below_64_columns() -> None:
    """Ordering is unchanged for narrow frames (bit i is column i)."""
    rng = np.random.default_rng(7)
    mask = rng.random((200, 12)) < 0.3
    codes = mask.astype(np.int64).dot(2 ** np.arange(12, dtype=np.int64))

    patterns, inverse = _missingness_patterns(mask)

    expected_codes = np.sort(np.unique(codes))
    pattern_codes = patterns.astype(np.int64).dot(2 ** np.arange(12, dtype=np.int64))
    np.testing.assert_array_equal(pattern_codes, expected_codes)
    np.testing.assert_array_equal(pattern_codes[inverse], codes)
