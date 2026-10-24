"""Rubin pooling must say when it has no between-imputation variance.

When all ``m`` estimates are identical (identical imputations, or an estimand
that does not depend on the imputed cells) the between-imputation variance is
zero, ``df`` is infinite and ``fmi`` is zero.  That is a valid output but it
is also exactly what degenerate multiple imputation looks like, so it must not
pass silently.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from missingly.mi import pool_scalar_estimates


def test_identical_estimates_warn_and_keep_the_documented_values() -> None:
    with pytest.warns(UserWarning, match="between-imputation variance is 0") as record:
        pooled = pool_scalar_estimates([2.0, 2.0, 2.0, 2.0], [0.1, 0.1, 0.1, 0.1])

    assert pooled["b"] == 0.0
    assert pooled["df"] == np.inf
    assert pooled["fmi"] == 0.0
    assert pooled["t"] == pooled["u_bar"]
    assert len(record) == 1
    assert "m=4" in str(record[0].message)
    assert "identical imputations" in str(record[0].message)


def test_varying_estimates_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pooled = pool_scalar_estimates([1.0, 3.0, 5.0], [0.25, 0.25, 0.25])
    assert pooled["b"] > 0


def test_warning_points_at_the_caller() -> None:
    with pytest.warns(UserWarning) as record:
        pool_scalar_estimates([1.0, 1.0], [0.5, 0.5])
    assert record[0].filename == __file__
