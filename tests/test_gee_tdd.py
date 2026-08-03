"""TDD contracts for population-averaged GEE models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from analytics import AnalyticsError, generalized_estimating_equations


def _longitudinal_records(groups: int = 20) -> list[dict[str, Any]]:
    """Create deterministic clustered outcomes with known positive effects.

    Args:
        groups: Number of independent subject clusters.

    Returns:
        Long-format records for Gaussian, binary, and Poisson GEE tests.

    Examples:
        >>> len(_longitudinal_records(8))
        40
    """
    generator = np.random.default_rng(20260803)
    random_intercepts = generator.normal(0, 0.55, groups)
    records: list[dict[str, Any]] = []
    for group in range(groups):
        for visit in range(5):
            centered_time = float(visit - 2)
            records.append(
                {
                    "subject": f"subject-{group}",
                    "visit": float(visit),
                    "time": centered_time,
                    "arm": "active" if group % 2 else "control",
                    "continuous": float(
                        2.5
                        + 0.9 * centered_time
                        + 0.6 * (group % 2)
                        + random_intercepts[group]
                        + generator.normal(0, 0.3)
                    ),
                    "binary": "event"
                    if generator.random() < 1 / (1 + np.exp(-(-0.3 + 0.5 * centered_time)))
                    else "no-event",
                    "count": int(generator.poisson(np.exp(0.4 + 0.22 * centered_time))),
                }
            )
    return records


def _assert_json_finite(value: Any) -> None:
    """Recursively assert that an API result contains no non-finite numbers.

    Args:
        value: Nested JSON-like value to validate.

    Returns:
        None.

    Raises:
        AssertionError: If any floating-point value is NaN or infinite.

    Examples:
        >>> _assert_json_finite({"estimate": 1.25})
    """
    if isinstance(value, dict):
        for nested in value.values():
            _assert_json_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_json_finite(nested)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_gee_default_robust_covariance_does_not_shadow_configuration():
    result = generalized_estimating_equations(
        _longitudinal_records(),
        "continuous",
        ["time", "arm"],
        "subject",
        categorical_predictors=["arm"],
    )
    assert result["variance_estimator"] == "robust sandwich"
    assert result["groups"] == 20
    _assert_json_finite(result)


def test_gee_bias_reduced_covariance_is_explicit_and_reproducible():
    records = _longitudinal_records()
    first = generalized_estimating_equations(
        records, "continuous", ["time"], "subject", standard_errors="bias_reduced"
    )
    second = generalized_estimating_equations(
        records, "continuous", ["time"], "subject", standard_errors="bias_reduced"
    )
    assert first["variance_estimator"] == "bias-reduced sandwich"
    assert first["reproducibility"] == second["reproducibility"]
    assert first["coefficients"] == second["coefficients"]


def test_gee_bias_reduced_covariance_rejects_too_few_clusters():
    with pytest.raises(AnalyticsError, match="at least eight groups"):
        generalized_estimating_equations(
            _longitudinal_records(7),
            "continuous",
            ["time"],
            "subject",
            standard_errors="bias_reduced",
        )


def test_autoregressive_gee_requires_unique_numeric_time_within_subject():
    records = _longitudinal_records()
    records[1]["visit"] = records[0]["visit"]
    with pytest.raises(AnalyticsError, match="unique complete time"):
        generalized_estimating_equations(
            records,
            "continuous",
            ["time"],
            "subject",
            working_correlation="autoregressive",
            time_column="visit",
        )


@pytest.mark.parametrize(
    ("family", "outcome", "effect_key"),
    [
        ("binomial", "binary", "odds_ratio"),
        ("poisson", "count", "rate_ratio"),
    ],
)
def test_non_gaussian_gee_reports_finite_population_averaged_effects(
    family: str, outcome: str, effect_key: str
):
    result = generalized_estimating_equations(
        _longitudinal_records(),
        outcome,
        ["time"],
        "subject",
        family=family,
    )
    slope = next(coefficient for coefficient in result["coefficients"] if coefficient["term"] == "time")
    if family == "binomial":
        expected_direction = 1 if result["event"]["event"] == "event" else -1
        assert (slope["estimate"] > 0) == (expected_direction > 0)
    else:
        assert slope[effect_key] > 1
    assert result["estimand"] == "population-averaged"
    _assert_json_finite(result)
