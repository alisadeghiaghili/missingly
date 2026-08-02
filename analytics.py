"""Deterministic statistical analysis primitives for the Missingly API.

The module deliberately keeps data in memory for a single request. It does not
persist raw datasets, and every response includes a stable input fingerprint.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


MAX_ROWS = 10_000
MAX_COLUMNS = 100
MAX_CELLS = 250_000
MAX_CELL_TEXT = 10_000
ENGINE_VERSION = "missingly-analytics/0.1"


class AnalyticsError(ValueError):
    """Raised when a requested analysis is not statistically valid."""


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _round(value: Any, digits: int = 8) -> float | None:
    finite = _finite_number(value)
    return round(finite, digits) if finite is not None else None


def _fingerprint(records: list[dict[str, Any]], settings: dict[str, Any]) -> str:
    payload = json.dumps({"records": records, "settings": settings}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise AnalyticsError("At least one record is required")
    if len(records) > MAX_ROWS:
        raise AnalyticsError(f"At most {MAX_ROWS} records are allowed per analysis")
    if not all(isinstance(record, dict) for record in records):
        raise AnalyticsError("Each record must be a JSON object")
    columns: list[str] = []
    cell_count = 0
    for record in records:
        cell_count += len(record)
        for key, value in record.items():
            if not isinstance(key, str) or not key.strip() or len(key) > 128:
                raise AnalyticsError("Column names must be non-empty strings up to 128 characters")
            if key not in columns:
                columns.append(key)
            if isinstance(value, (dict, list, tuple, set)):
                raise AnalyticsError(f"Nested values are not supported in column '{key}'")
            if isinstance(value, str) and len(value) > MAX_CELL_TEXT:
                raise AnalyticsError(f"Text values in '{key}' exceed {MAX_CELL_TEXT} characters")
    if not columns:
        raise AnalyticsError("Records must contain at least one column")
    if len(columns) > MAX_COLUMNS:
        raise AnalyticsError(f"At most {MAX_COLUMNS} columns are allowed per analysis")
    if cell_count > MAX_CELLS:
        raise AnalyticsError(f"At most {MAX_CELLS} supplied cells are allowed per analysis")
    frame = pd.DataFrame(records, columns=columns)
    frame = frame.replace(r"^\s*$", np.nan, regex=True).replace([np.inf, -np.inf], np.nan)
    return frame


def _numeric_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    numeric: dict[str, pd.Series] = {}
    for column in frame.columns:
        source = frame[column]
        if pd.api.types.is_bool_dtype(source):
            continue
        converted = pd.to_numeric(source, errors="coerce")
        observed = int(source.notna().sum())
        if observed and int(converted.notna().sum()) == observed:
            numeric[str(column)] = converted.astype(float)
    return numeric


def _column_profile(name: str, series: pd.Series, numeric: pd.Series | None) -> dict[str, Any]:
    total = len(series)
    missing = int(series.isna().sum())
    base = {
        "name": name,
        "missing": missing,
        "missing_rate": _round(missing / total if total else 0),
        "non_missing": total - missing,
        "distinct": int(series.nunique(dropna=True)),
    }
    if numeric is not None:
        values = numeric.dropna()
        base.update(
            {
                "kind": "numeric",
                "mean": _round(values.mean()) if len(values) else None,
                "std_dev": _round(values.std(ddof=1)) if len(values) > 1 else None,
                "min": _round(values.min()) if len(values) else None,
                "q1": _round(values.quantile(0.25)) if len(values) else None,
                "median": _round(values.median()) if len(values) else None,
                "q3": _round(values.quantile(0.75)) if len(values) else None,
                "max": _round(values.max()) if len(values) else None,
            }
        )
        return base
    frequencies = series.dropna().astype(str).value_counts().head(8)
    base.update(
        {
            "kind": "categorical",
            "top_values": [{"value": value, "count": int(count)} for value, count in frequencies.items()],
        }
    )
    return base


def _correlations(numeric: dict[str, pd.Series]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for left, right in itertools.combinations(numeric, 2):
        paired = pd.concat([numeric[left], numeric[right]], axis=1).dropna()
        if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
            continue
        statistic, p_value = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
        result.append(
            {
                "left": left,
                "right": right,
                "n": int(len(paired)),
                "pearson_r": _round(statistic),
                "p_value": _round(p_value),
            }
        )
    return sorted(result, key=lambda item: abs(item["pearson_r"] or 0), reverse=True)


def profile_dataset(records: list[dict[str, Any]], alpha: float = 0.05) -> dict[str, Any]:
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    frame = _frame(records)
    numeric = _numeric_columns(frame)
    missing_cells = int(frame.isna().sum().sum())
    return {
        "dataset": {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "complete_rows": int(len(frame.dropna())),
            "duplicate_rows": int(frame.duplicated().sum()),
            "missing_cells": missing_cells,
            "missing_cell_rate": _round(missing_cells / (len(frame) * len(frame.columns))),
        },
        "columns": [_column_profile(str(column), frame[column], numeric.get(str(column))) for column in frame.columns],
        "correlations": _correlations(numeric),
        "reproducibility": {
            "engine": ENGINE_VERSION,
            "input_sha256": _fingerprint(records, {"analysis": "profile", "alpha": alpha}),
        },
    }


def missingness_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = _frame(records)
    numeric = _numeric_columns(frame)
    patterns: dict[tuple[str, ...], int] = {}
    for _, row in frame.isna().iterrows():
        pattern = tuple(str(column) for column, is_missing in row.items() if is_missing)
        patterns[pattern] = patterns.get(pattern, 0) + 1
    associations: list[dict[str, Any]] = []
    for target in frame.columns:
        indicator = frame[target].isna().astype(int)
        if not indicator.any() or indicator.all():
            continue
        for source_name, source in numeric.items():
            if source_name == target:
                continue
            valid = source.notna()
            groups = indicator[valid]
            if len(groups) < 3 or groups.nunique() != 2 or source[valid].nunique() < 2:
                continue
            statistic, p_value = stats.pointbiserialr(groups, source[valid])
            associations.append(
                {
                    "missing_column": str(target),
                    "observed_numeric_column": source_name,
                    "n": int(valid.sum()),
                    "point_biserial_r": _round(statistic),
                    "p_value": _round(p_value),
                }
            )
    column_rates = frame.isna().mean()
    warnings: list[str] = []
    if float(column_rates.max()) >= 0.40:
        warnings.append("At least one column has 40% or more missing values; do not apply automatic imputation without domain review.")
    complete_case_loss = 1 - (len(frame.dropna()) / len(frame))
    if complete_case_loss >= 0.20:
        warnings.append("Complete-case analysis would discard at least 20% of rows; assess the missing-data mechanism first.")
    if not warnings:
        warnings.append("No high-loss threshold was triggered; this is not evidence that data are MCAR.")
    ordered_patterns = sorted(patterns.items(), key=lambda item: (-item[1], item[0]))[:20]
    return {
        "rows": int(len(frame)),
        "complete_rows": int(len(frame.dropna())),
        "column_missingness": [
            {"column": str(column), "missing": int(frame[column].isna().sum()), "missing_rate": _round(column_rates[column])}
            for column in frame.columns
        ],
        "patterns": [
            {"missing_columns": list(pattern), "rows": count, "row_rate": _round(count / len(frame))}
            for pattern, count in ordered_patterns
        ],
        "associations": sorted(associations, key=lambda item: abs(item["point_biserial_r"] or 0), reverse=True),
        "warnings": warnings,
        "reproducibility": {
            "engine": ENGINE_VERSION,
            "input_sha256": _fingerprint(records, {"analysis": "missingness"}),
        },
    }


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise AnalyticsError(f"Column '{name}' was not found")
    numeric = _numeric_columns(frame)
    if name not in numeric:
        raise AnalyticsError(f"Column '{name}' must contain only numeric observed values")
    return numeric[name]


def run_statistical_test(
    records: list[dict[str, Any]],
    test: str,
    outcome: str,
    group: str | None = None,
    predictors: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    frame = _frame(records)
    test = test.strip().lower()
    settings = {"test": test, "outcome": outcome, "group": group, "predictors": predictors or [], "alpha": alpha}
    if test == "pearson_correlation":
        if not group:
            raise AnalyticsError("group must name the second numeric column for pearson_correlation")
        left, right = _numeric_column(frame, outcome), _numeric_column(frame, group)
        paired = pd.concat([left, right], axis=1).dropna()
        if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
            raise AnalyticsError("Pearson correlation requires at least three non-constant paired observations")
        statistic, p_value = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
        result = {"method": "Pearson product-moment correlation", "n": int(len(paired)), "statistic": _round(statistic), "p_value": _round(p_value)}
    elif test == "welch_t_test":
        if not group or group not in frame.columns:
            raise AnalyticsError("group must name a categorical column for welch_t_test")
        values = _numeric_column(frame, outcome)
        subset = pd.DataFrame({"value": values, "group": frame[group]}).dropna()
        labels = list(subset["group"].astype(str).drop_duplicates())
        if len(labels) != 2:
            raise AnalyticsError("Welch t-test requires exactly two observed groups")
        samples = [subset.loc[subset["group"].astype(str) == label, "value"] for label in labels]
        if min(len(sample) for sample in samples) < 2:
            raise AnalyticsError("Each group needs at least two numeric observations")
        statistic, p_value = stats.ttest_ind(samples[0], samples[1], equal_var=False)
        result = {
            "method": "Welch two-sample t-test (two-sided)",
            "groups": [{"label": label, "n": int(len(sample)), "mean": _round(sample.mean())} for label, sample in zip(labels, samples)],
            "statistic": _round(statistic),
            "p_value": _round(p_value),
        }
    elif test == "chi_square":
        if not group or outcome not in frame.columns or group not in frame.columns:
            raise AnalyticsError("outcome and group must name categorical columns for chi_square")
        table = pd.crosstab(frame[outcome], frame[group], dropna=True)
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise AnalyticsError("Chi-square requires at least two observed categories in each variable")
        statistic, p_value, degrees_freedom, expected = stats.chi2_contingency(table)
        result = {
            "method": "Pearson chi-square test of independence",
            "n": int(table.to_numpy().sum()),
            "statistic": _round(statistic),
            "degrees_freedom": int(degrees_freedom),
            "p_value": _round(p_value),
            "minimum_expected_count": _round(expected.min()),
            "warning": "Expected cell count below 5; asymptotic p-value may be unreliable." if expected.min() < 5 else None,
        }
    elif test == "ols_regression":
        predictor_names = predictors or []
        if not predictor_names:
            raise AnalyticsError("predictors must include one or more numeric columns for ols_regression")
        target = _numeric_column(frame, outcome)
        predictor_series = [_numeric_column(frame, name) for name in predictor_names]
        data = pd.concat([target, *predictor_series], axis=1).dropna()
        parameter_count = len(predictor_names) + 1
        if len(data) <= parameter_count:
            raise AnalyticsError("OLS needs more complete rows than fitted parameters")
        y = data.iloc[:, 0].to_numpy(dtype=float)
        matrix = np.column_stack([np.ones(len(data)), *[data.iloc[:, index].to_numpy(dtype=float) for index in range(1, len(data.columns))]])
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
        if rank != parameter_count:
            raise AnalyticsError("OLS predictors are perfectly collinear")
        residuals = y - matrix @ coefficients
        degrees_freedom = len(y) - parameter_count
        mse = float(np.sum(residuals**2) / degrees_freedom)
        standard_errors = np.sqrt(np.diag(mse * np.linalg.inv(matrix.T @ matrix)))
        t_values = coefficients / standard_errors
        p_values = 2 * stats.t.sf(np.abs(t_values), degrees_freedom)
        critical = stats.t.ppf(1 - alpha / 2, degrees_freedom)
        names = ["intercept", *predictor_names]
        result = {
            "method": "Ordinary least squares regression",
            "n": int(len(y)),
            "degrees_freedom": int(degrees_freedom),
            "r_squared": _round(1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)) if np.sum((y - y.mean()) ** 2) else None,
            "coefficients": [
                {
                    "term": name,
                    "estimate": _round(coefficient),
                    "std_error": _round(error),
                    "t": _round(t_value),
                    "p_value": _round(p_value),
                    "ci_lower": _round(coefficient - critical * error),
                    "ci_upper": _round(coefficient + critical * error),
                }
                for name, coefficient, error, t_value, p_value in zip(names, coefficients, standard_errors, t_values, p_values)
            ],
        }
    else:
        raise AnalyticsError("Unsupported test. Use pearson_correlation, welch_t_test, chi_square, or ols_regression")
    result.update(
        {
            "alpha": alpha,
            "reject_null": bool((result.get("p_value") or 1) < alpha) if "p_value" in result else None,
            "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
        }
    )
    return result
