"""Independent closed-form and guardrail tests for statistical contracts."""

from __future__ import annotations

from fractions import Fraction

import pytest

from missingly.mi import pool_scalar_estimates


def test_rubin_scalar_pooling_matches_hand_calculation():
    """Rubin quantities must agree with an exact rational-number oracle."""
    result = pool_scalar_estimates(
        estimates=[1.0, 2.0, 3.0],
        variances=[0.25, 0.50, 0.75],
    )

    expected_fmi = (
        Fraction(8, 3) + Fraction(64, 217)
    ) / Fraction(11, 3)
    assert result["q_bar"] == pytest.approx(2.0, abs=1e-12)
    assert result["u_bar"] == pytest.approx(0.5, abs=1e-12)
    assert result["b"] == pytest.approx(1.0, abs=1e-12)
    assert result["t"] == pytest.approx(float(Fraction(11, 6)), abs=1e-12)
    assert result["r"] == pytest.approx(float(Fraction(8, 3)), abs=1e-12)
    assert result["df"] == pytest.approx(float(Fraction(121, 32)), abs=1e-12)
    assert result["lambda"] == pytest.approx(float(Fraction(8, 11)), abs=1e-12)
    assert result["fmi"] == pytest.approx(float(expected_fmi), abs=1e-12)


@pytest.mark.parametrize(
    ("estimates", "variances", "message"),
    [
        ([1.0, 2.0], [0.1, -0.2], "non-negative"),
        ([1.0, float("nan")], [0.1, 0.2], "finite"),
        ([1.0, 2.0], [0.1, float("inf")], "finite"),
    ],
)
def test_rubin_scalar_pooling_rejects_invalid_numeric_inputs(
    estimates,
    variances,
    message,
):
    """Invalid estimates must fail before producing plausible-looking output."""
    with pytest.raises(ValueError, match=message):
        pool_scalar_estimates(estimates, variances)
