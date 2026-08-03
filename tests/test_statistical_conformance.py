"""Closed-form numerical conformance tests for core statistical primitives."""

from __future__ import annotations

import pytest

from analytics import kaplan_meier_analysis, run_statistical_test, weighted_ols


def test_ols_matches_exact_closed_form_line():
    records = [{"x": x, "y": 1 + 2 * x} for x in range(1, 8)]
    result = run_statistical_test(records, "ols_regression", "y", predictors=["x"])
    coefficients = {item["term"]: item for item in result["coefficients"]}
    assert coefficients["intercept"]["estimate"] == pytest.approx(1.0, abs=1e-10)
    assert coefficients["x"]["estimate"] == pytest.approx(2.0, abs=1e-10)
    assert result["r_squared"] == pytest.approx(1.0)


def test_weighted_ols_matches_exact_line_under_unequal_weights():
    records = [{"x": x, "y": 3 - 0.75 * x, "weight": x + 1} for x in range(1, 9)]
    result = weighted_ols(records, "y", ["x"], "weight")
    coefficients = {item["term"]: item for item in result["coefficients"]}
    assert coefficients["intercept"]["estimate"] == pytest.approx(3.0, abs=1e-10)
    assert coefficients["x"]["estimate"] == pytest.approx(-0.75, abs=1e-10)
    assert result["r_squared"] == pytest.approx(1.0)


def test_kaplan_meier_matches_hand_calculated_survival_steps():
    records = [
        {"time": 1, "status": "event"},
        {"time": 2, "status": "event"},
        {"time": 2, "status": "censor"},
        {"time": 3, "status": "censor"},
    ]
    result = kaplan_meier_analysis(records, "time", "status")
    points = result["curves"][0]["points"]
    assert points[1]["survival"] == pytest.approx(0.75)
    assert points[2]["survival"] == pytest.approx(0.5)
