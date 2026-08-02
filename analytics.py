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
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError


MAX_ROWS = 10_000
MAX_COLUMNS = 100
MAX_CELLS = 250_000
MAX_CELL_TEXT = 10_000
ENGINE_VERSION = "missingly-analytics/0.1"

__all__ = [
    "AnalyticsError",
    "advanced_numeric_imputation",
    "count_regression",
    "count_regression_with_categorical_predictors",
    "cox_proportional_hazards",
    "generalized_estimating_equations",
    "hurdle_poisson_regression",
    "kaplan_meier_analysis",
    "linear_mixed_effects",
    "missingness_report",
    "multiple_imputation_ols",
    "multinomial_logistic_regression",
    "ordinal_logistic_regression",
    "profile_dataset",
    "records_to_csv",
    "regression_with_categorical_predictors",
    "run_statistical_test",
    "simple_imputation",
    "weighted_ols",
    "zero_inflated_count_regression",
]


class AnalyticsError(ValueError):
    """Raised when validated input cannot support the requested statistical analysis.

    Examples:
        >>> raise AnalyticsError("Outcome must be numeric")
        Traceback (most recent call last):
        ...
        analytics.AnalyticsError: Outcome must be numeric
    """


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _round(value: Any, digits: int = 8) -> float | None:
    finite = _finite_number(value)
    return round(finite, digits) if finite is not None else None


def _exp_round(value: Any) -> float | None:
    finite = _finite_number(value)
    if finite is None or finite > 700:
        return None
    return _round(math.exp(finite))


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
    """Profile an in-memory rectangular dataset without persisting raw records.

    Args:
        records: JSON-like row objects with scalar values and string column names.
        alpha: Reserved confidence level included in the reproducibility fingerprint.

    Returns:
        A schema, numeric and categorical summaries, missingness totals, correlations,
        and a deterministic input fingerprint.

    Raises:
        AnalyticsError: If records violate limits or do not form a valid tabular dataset.

    Examples:
        >>> profile_dataset([{"score": 10}, {"score": 20}])["dataset"]["rows"]
        2
    """
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
    """Describe observed missingness without asserting a missing-data mechanism.

    Args:
        records: JSON-like row objects to inspect in memory.

    Returns:
        Column-level missingness, row patterns, observed numeric associations, warnings,
        and a reproducibility fingerprint.

    Raises:
        AnalyticsError: If records are invalid, oversized, or contain unsupported values.

    Examples:
        >>> missingness_report([{"x": 1}, {"x": None}])["column_missingness"][0]["missing"]
        1
    """
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


def simple_imputation(
    records: list[dict[str, Any]],
    method: str,
    columns: list[str] | None = None,
    constant: Any | None = None,
) -> dict[str, Any]:
    """Fill selected missing cells using a transparent univariate rule.

    Args:
        records: JSON-like row objects to transform in memory.
        method: One of mean, median, mode, or constant.
        columns: Optional target columns; omitted selects all eligible columns.
        constant: Replacement value required when method is constant.

    Returns:
        Transformed records, an audit of filled cells, and a reproducibility fingerprint.

    Raises:
        AnalyticsError: If a method, column, or fill value is incompatible with the data.

    Examples:
        >>> simple_imputation([{"x": 2}, {"x": None}], "mean")["records"][1]["x"]
        2.0
    """
    frame = _frame(records)
    method = method.strip().lower()
    if method not in {"mean", "median", "mode", "constant"}:
        raise AnalyticsError("Unsupported imputation method. Use mean, median, mode, or constant")
    selected = columns or [str(column) for column in frame.columns]
    if len(set(selected)) != len(selected):
        raise AnalyticsError("Imputation columns must not contain duplicates")
    numeric = _numeric_columns(frame)
    changes: list[dict[str, Any]] = []
    for column in selected:
        if column not in frame.columns:
            raise AnalyticsError(f"Column '{column}' was not found")
        missing = int(frame[column].isna().sum())
        if not missing:
            continue
        if method in {"mean", "median"}:
            if column not in numeric:
                raise AnalyticsError(f"{method} imputation requires a numeric column: '{column}'")
            observed = numeric[column].dropna()
            if observed.empty:
                raise AnalyticsError(f"Cannot calculate {method} for an all-missing column: '{column}'")
            fill_value: Any = observed.mean() if method == "mean" else observed.median()
        elif method == "mode":
            observed = frame[column].dropna()
            if observed.empty:
                raise AnalyticsError(f"Cannot calculate mode for an all-missing column: '{column}'")
            modes = observed.mode(dropna=True)
            fill_value = modes.iloc[0]
        else:
            if constant is None:
                raise AnalyticsError("constant imputation requires a non-null constant value")
            fill_value = constant
        frame[column] = frame[column].fillna(fill_value)
        changes.append({"column": column, "method": method, "imputed_cells": missing, "fill_value": _json_value(fill_value)})
    return {
        "records": _frame_to_records(frame),
        "imputations": changes,
        "reproducibility": {
            "engine": ENGINE_VERSION,
            "input_sha256": _fingerprint(records, {"analysis": "simple_imputation", "method": method, "columns": selected, "constant": constant}),
        },
    }


def advanced_numeric_imputation(
    records: list[dict[str, Any]],
    method: str,
    columns: list[str] | None = None,
    n_neighbors: int = 5,
    max_iter: int = 10,
    random_state: int = 2026,
) -> dict[str, Any]:
    """Apply KNN or deterministic iterative single imputation to numeric columns.

    Args:
        records: JSON-like row objects to transform in memory.
        method: The KNN or iterative imputation strategy.
        columns: Numeric columns to include in the imputation model.
        n_neighbors: Neighbor count for KNN imputation.
        max_iter: Maximum iterations for iterative regression imputation.
        random_state: Deterministic seed for iterative imputation.

    Returns:
        Imputed records, imputation metadata, and a settings-aware fingerprint.

    Raises:
        AnalyticsError: If selected columns are not eligible numeric imputation inputs.

    Examples:
        >>> advanced_numeric_imputation([{"x": 1, "y": 2}, {"x": 2, "y": None}], "knn", ["x", "y"])["records"][1]["y"]
        2.0
    """
    frame = _frame(records)
    method = method.strip().lower()
    if method not in {"knn", "iterative"}:
        raise AnalyticsError("Unsupported advanced method. Use knn or iterative")
    numeric = _numeric_columns(frame)
    selected = columns or list(numeric)
    if len(set(selected)) != len(selected):
        raise AnalyticsError("Imputation columns must not contain duplicates")
    if len(selected) < 2:
        raise AnalyticsError("KNN and iterative imputation require at least two numeric columns")
    for column in selected:
        if column not in numeric:
            raise AnalyticsError(f"Advanced imputation requires a numeric column: '{column}'")
        if numeric[column].dropna().empty:
            raise AnalyticsError(f"Cannot impute an all-missing column: '{column}'")
    data = pd.DataFrame({column: numeric[column] for column in selected})
    missing_counts = {column: int(data[column].isna().sum()) for column in selected}
    if not any(missing_counts.values()):
        return {
            "records": _frame_to_records(frame),
            "imputations": [],
            "reproducibility": {
                "engine": ENGINE_VERSION,
                "input_sha256": _fingerprint(records, {"analysis": "advanced_imputation", "method": method, "columns": selected}),
            },
        }
    if method == "knn":
        if not 1 <= n_neighbors <= 100:
            raise AnalyticsError("n_neighbors must be between 1 and 100")
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        if not 1 <= max_iter <= 100:
            raise AnalyticsError("max_iter must be between 1 and 100")
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state, sample_posterior=False, initial_strategy="median")
    try:
        imputed = imputer.fit_transform(data)
    except Exception as exc:
        raise AnalyticsError(f"{method} imputation could not fit this numeric dataset") from exc
    if imputed.shape[1] != len(selected) or not np.isfinite(imputed).all():
        raise AnalyticsError(f"{method} imputation did not produce a complete finite numeric result")
    imputed_frame = pd.DataFrame(imputed, columns=selected, index=frame.index)
    for column in selected:
        missing_mask = frame[column].isna()
        frame.loc[missing_mask, column] = imputed_frame.loc[missing_mask, column]
    changes = [
        {
            "column": column,
            "method": method,
            "imputed_cells": missing_counts[column],
            "imputed_min": _round(imputed_frame.loc[data[column].isna(), column].min()) if missing_counts[column] else None,
            "imputed_max": _round(imputed_frame.loc[data[column].isna(), column].max()) if missing_counts[column] else None,
        }
        for column in selected
        if missing_counts[column]
    ]
    settings = {
        "analysis": "advanced_imputation",
        "method": method,
        "columns": selected,
        "n_neighbors": n_neighbors if method == "knn" else None,
        "max_iter": max_iter if method == "iterative" else None,
        "random_state": random_state if method == "iterative" else None,
    }
    return {"records": _frame_to_records(frame), "imputations": changes, "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)}}


def records_to_csv(records: list[dict[str, Any]]) -> str:
    """Export validated records as reproducible UTF-8 compatible CSV text.

    Args:
        records: JSON-like row objects to validate and export.

    Returns:
        CSV text with a header and LF line endings.

    Raises:
        AnalyticsError: If records violate the tabular input contract.

    Examples:
        >>> records_to_csv([{"x": 1}])
        'x\n1\n'
    """
    return _frame(records).to_csv(index=False, lineterminator="\n")


def _fit_complete_case_ols(frame: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[np.ndarray, np.ndarray, int, float | None]:
    data = frame[[outcome, *predictors]].dropna()
    parameter_count = len(predictors) + 1
    if len(data) <= parameter_count:
        raise AnalyticsError("Each imputed OLS fit needs more complete rows than fitted parameters")
    y = data[outcome].to_numpy(dtype=float)
    matrix = np.column_stack([np.ones(len(data)), *[data[name].to_numpy(dtype=float) for name in predictors]])
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if rank != parameter_count:
        raise AnalyticsError("OLS predictors are perfectly collinear after imputation")
    residuals = y - matrix @ coefficients
    degrees_freedom = len(y) - parameter_count
    mse = float(np.sum(residuals**2) / degrees_freedom)
    covariance = mse * np.linalg.inv(matrix.T @ matrix)
    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - float(np.sum(residuals**2)) / total_sum_squares if total_sum_squares else None
    return coefficients, covariance, int(len(y)), _round(r_squared)


def multiple_imputation_ols(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    impute_columns: list[str] | None = None,
    m: int = 5,
    max_iter: int = 10,
    random_state: int = 2026,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Pool numeric OLS estimates across posterior iterative imputations using Rubin's rules.

    Args:
        records: JSON-like row objects containing numeric analysis and imputation columns.
        outcome: Numeric OLS outcome column.
        predictors: Unique numeric predictor columns.
        impute_columns: Numeric columns included in each imputation model.
        m: Number of completed datasets to generate.
        max_iter: Maximum iterative-imputer iterations per completed dataset.
        random_state: Base seed; each draw receives a deterministic derived seed.
        alpha: Two-sided confidence level complement.

    Returns:
        Rubin-pooled coefficients, within/between/total variance, diagnostics, and a fingerprint.

    Raises:
        AnalyticsError: If model columns, imputation settings, or complete-data fits are invalid.

    Examples:
        >>> multiple_imputation_ols([{"x": 1, "y": 3}, {"x": 2, "y": None}, {"x": 3, "y": 7}], "y", ["x"], ["x", "y"], m=2)["m"]
        2
    """
    if not 2 <= m <= 50:
        raise AnalyticsError("m must be between 2 and 50 for multiple imputation")
    if not 1 <= max_iter <= 100:
        raise AnalyticsError("max_iter must be between 1 and 100")
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    frame = _frame(records)
    numeric = _numeric_columns(frame)
    required = [outcome, *predictors]
    for column in required:
        if column not in numeric:
            raise AnalyticsError(f"Multiple-imputation OLS requires numeric column: '{column}'")
    selected = impute_columns or list(numeric)
    if len(set(selected)) != len(selected) or len(selected) < 2:
        raise AnalyticsError("impute_columns must contain at least two unique numeric columns")
    for column in selected:
        if column not in numeric:
            raise AnalyticsError(f"Multiple imputation requires numeric column: '{column}'")
        if numeric[column].dropna().empty:
            raise AnalyticsError(f"Cannot impute an all-missing column: '{column}'")
    for column in required:
        if frame[column].isna().any() and column not in selected:
            raise AnalyticsError(f"Missing model column '{column}' must be included in impute_columns")
    source = pd.DataFrame({column: numeric[column] for column in selected})
    missing_counts = {column: int(source[column].isna().sum()) for column in selected}
    if not any(missing_counts.values()):
        raise AnalyticsError("Multiple imputation requires at least one missing value in impute_columns")
    estimates: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    r_squared_values: list[float] = []
    sample_sizes: list[int] = []
    for draw in range(m):
        try:
            imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=random_state + draw,
                sample_posterior=True,
                initial_strategy="median",
            )
            values = imputer.fit_transform(source)
        except Exception as exc:
            raise AnalyticsError("Posterior iterative imputation could not fit this numeric dataset") from exc
        if values.shape[1] != len(selected) or not np.isfinite(values).all():
            raise AnalyticsError("Posterior iterative imputation did not produce a complete finite result")
        completed = frame.copy()
        completed_values = pd.DataFrame(values, columns=selected, index=frame.index)
        for column in selected:
            mask = completed[column].isna()
            completed.loc[mask, column] = completed_values.loc[mask, column]
        coefficients, covariance, sample_size, r_squared = _fit_complete_case_ols(completed, outcome, predictors)
        estimates.append(coefficients)
        covariances.append(covariance)
        sample_sizes.append(sample_size)
        if r_squared is not None:
            r_squared_values.append(r_squared)
    estimate_matrix = np.vstack(estimates)
    covariance_stack = np.stack(covariances)
    pooled_estimates = estimate_matrix.mean(axis=0)
    within = covariance_stack.mean(axis=0)
    between = np.cov(estimate_matrix, rowvar=False, ddof=1)
    if np.ndim(between) == 0:
        between = np.array([[float(between)]])
    total = within + (1 + 1 / m) * between
    names = ["intercept", *predictors]
    coefficients: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        within_variance = max(float(within[index, index]), 0.0)
        between_variance = max(float(between[index, index]), 0.0)
        total_variance = max(float(total[index, index]), 0.0)
        standard_error = math.sqrt(total_variance)
        relative_increase = ((1 + 1 / m) * between_variance / within_variance) if within_variance > 0 else math.inf
        degrees_freedom = ((m - 1) * (1 + 1 / relative_increase) ** 2) if math.isfinite(relative_increase) and relative_increase > 0 else None
        statistic = float(pooled_estimates[index] / standard_error) if standard_error > 0 else None
        if statistic is None:
            p_value = None
            critical = None
        elif degrees_freedom is None:
            p_value = float(2 * stats.norm.sf(abs(statistic)))
            critical = float(stats.norm.ppf(1 - alpha / 2))
        else:
            p_value = float(2 * stats.t.sf(abs(statistic), degrees_freedom))
            critical = float(stats.t.ppf(1 - alpha / 2, degrees_freedom))
        fraction_missing_information = ((1 + 1 / m) * between_variance / total_variance) if total_variance > 0 else None
        coefficients.append(
            {
                "term": name,
                "estimate": _round(pooled_estimates[index]),
                "std_error": _round(standard_error),
                "statistic": _round(statistic),
                "p_value": _round(p_value),
                "degrees_freedom": _round(degrees_freedom) if degrees_freedom is not None else None,
                "within_variance": _round(within_variance),
                "between_variance": _round(between_variance),
                "total_variance": _round(total_variance),
                "fraction_missing_information": _round(fraction_missing_information),
                "ci_lower": _round(pooled_estimates[index] - critical * standard_error) if critical is not None else None,
                "ci_upper": _round(pooled_estimates[index] + critical * standard_error) if critical is not None else None,
            }
        )
    settings = {
        "analysis": "multiple_imputation_ols",
        "outcome": outcome,
        "predictors": predictors,
        "impute_columns": selected,
        "m": m,
        "max_iter": max_iter,
        "random_state": random_state,
        "alpha": alpha,
    }
    diagnostics = []
    if m < 5:
        diagnostics.append("Fewer than five imputations were requested; increase m when missing information is substantial.")
    if any((item["fraction_missing_information"] or 0) > 0.50 for item in coefficients):
        diagnostics.append("At least one coefficient has high fraction of missing information; inspect the imputation model and sensitivity analyses.")
    return {
        "method": "Posterior iterative multiple imputation with Rubin-pooled OLS",
        "m": m,
        "sample_size_per_imputation": sample_sizes,
        "pooled_r_squared_mean": _round(np.mean(r_squared_values)) if r_squared_values else None,
        "imputed_cells_per_dataset": missing_counts,
        "coefficients": coefficients,
        "diagnostics": diagnostics,
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def kaplan_meier_analysis(
    records: list[dict[str, Any]],
    time_column: str,
    event_column: str,
    group_column: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate Kaplan-Meier curves and a two-group log-rank comparison when applicable.

    Args:
        records: JSON-like row objects containing follow-up time and event status.
        time_column: Non-negative numeric follow-up-time column.
        event_column: Observed two-level event/censor column.
        group_column: Optional grouping column for separate curves.
        alpha: Confidence interval complement.

    Returns:
        Event coding, curve points with Greenwood log-log intervals, optional log-rank
        inference, and a reproducibility fingerprint.

    Raises:
        AnalyticsError: If times, event coding, group count, or alpha are invalid.

    Examples:
        >>> kaplan_meier_analysis([{"t": 1, "e": "event"}, {"t": 2, "e": "censor"}], "t", "e")["n"]
        2
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    frame = _frame(records)
    time = _numeric_column(frame, time_column)
    if event_column not in frame.columns:
        raise AnalyticsError(f"Column '{event_column}' was not found")
    group = frame[group_column] if group_column else pd.Series("All", index=frame.index)
    data = pd.DataFrame({"time": time, "event": frame[event_column], "group": group}).dropna()
    if data.empty or (data["time"] < 0).any():
        raise AnalyticsError("Survival time must contain non-negative complete numeric values")
    event_labels = sorted(data["event"].astype(str).unique().tolist())
    if len(event_labels) != 2:
        raise AnalyticsError("event_column must contain exactly two observed categories")
    data["event_binary"] = (data["event"].astype(str) == event_labels[1]).astype(int)
    group_labels = sorted(data["group"].astype(str).unique().tolist())
    if len(group_labels) > 20:
        raise AnalyticsError("Kaplan-Meier analysis supports at most 20 observed groups")
    critical = float(stats.norm.ppf(1 - alpha / 2))
    curves = []
    for label in group_labels:
        subset = data.loc[data["group"].astype(str) == label]
        event_times = sorted(subset.loc[subset["event_binary"] == 1, "time"].unique().tolist())
        survival = 1.0
        greenwood_sum = 0.0
        points = [{"time": 0.0, "at_risk": int(len(subset)), "events": 0, "censored": 0, "survival": 1.0, "ci_lower": 1.0, "ci_upper": 1.0}]
        for event_time in event_times:
            at_risk = int((subset["time"] >= event_time).sum())
            events = int(((subset["time"] == event_time) & (subset["event_binary"] == 1)).sum())
            censored = int(((subset["time"] == event_time) & (subset["event_binary"] == 0)).sum())
            survival *= 1 - events / at_risk
            if at_risk > events:
                greenwood_sum += events / (at_risk * (at_risk - events))
                variance = survival**2 * greenwood_sum
            else:
                variance = 0.0
            if survival <= 0 or survival >= 1 or variance <= 0:
                lower, upper = survival, survival
            else:
                theta = math.log(-math.log(survival))
                se_theta = math.sqrt(variance / (survival**2 * math.log(survival) ** 2))
                lower = math.exp(-math.exp(theta + critical * se_theta))
                upper = math.exp(-math.exp(theta - critical * se_theta))
            points.append(
                {
                    "time": _round(event_time),
                    "at_risk": at_risk,
                    "events": events,
                    "censored": censored,
                    "survival": _round(survival),
                    "ci_lower": _round(lower),
                    "ci_upper": _round(upper),
                }
            )
        curves.append({"group": label, "n": int(len(subset)), "events": int(subset["event_binary"].sum()), "points": points})
    log_rank = None
    if len(group_labels) == 2:
        left = data.loc[data["group"].astype(str) == group_labels[0]]
        right = data.loc[data["group"].astype(str) == group_labels[1]]
        observed_minus_expected = 0.0
        variance = 0.0
        event_times = sorted(data.loc[data["event_binary"] == 1, "time"].unique().tolist())
        for event_time in event_times:
            n_left, n_right = int((left["time"] >= event_time).sum()), int((right["time"] >= event_time).sum())
            d_left, d_right = int(((left["time"] == event_time) & (left["event_binary"] == 1)).sum()), int(((right["time"] == event_time) & (right["event_binary"] == 1)).sum())
            total_risk, total_events = n_left + n_right, d_left + d_right
            if total_risk <= 1 or total_events == 0:
                continue
            expected_left = total_events * n_left / total_risk
            observed_minus_expected += d_left - expected_left
            variance += n_left * n_right * total_events * (total_risk - total_events) / (total_risk**2 * (total_risk - 1))
        if variance > 0:
            statistic = observed_minus_expected**2 / variance
            log_rank = {"method": "Log-rank test", "groups": group_labels, "chi_square": _round(statistic), "degrees_freedom": 1, "p_value": _round(stats.chi2.sf(statistic, 1))}
    settings = {"analysis": "kaplan_meier", "time": time_column, "event": event_column, "group": group_column, "alpha": alpha}
    return {
        "method": "Kaplan-Meier survival estimate",
        "n": int(len(data)),
        "event": {"censor_or_reference": event_labels[0], "event": event_labels[1]},
        "curves": curves,
        "log_rank": log_rank,
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def _cox_schoenfeld_style_diagnostic(
    data: pd.DataFrame,
    exog: pd.DataFrame,
    fitted_params: np.ndarray,
    strata_column: str | None,
) -> dict[str, Any]:
    """Calculate event-level Schoenfeld-style residual/time correlations for Cox diagnostics."""
    residuals: dict[str, list[float]] = {term: [] for term in exog.columns}
    event_times: list[float] = []
    strata = data.groupby("strata", observed=True) if strata_column else [(None, data)]
    for _, subset in strata:
        for event_time in sorted(subset.loc[subset["event_binary"] == 1, "time"].unique().tolist()):
            event_rows = subset.loc[(subset["time"] == event_time) & (subset["event_binary"] == 1)]
            risk_rows = subset.loc[subset["time"] >= event_time]
            risk_exog = exog.loc[risk_rows.index].to_numpy()
            linear_predictor = risk_exog @ fitted_params
            weights = np.exp(linear_predictor - np.max(linear_predictor))
            weighted_mean = np.average(risk_exog, axis=0, weights=weights)
            event_exog = exog.loc[event_rows.index].to_numpy()
            for row in event_exog:
                event_times.append(float(event_time))
                for index, term in enumerate(exog.columns):
                    residuals[term].append(float(row[index] - weighted_mean[index]))
    transformed_time = np.log(np.asarray(event_times, dtype=float)) if event_times else np.array([])
    diagnostics = []
    for term in exog.columns:
        values = np.asarray(residuals[term], dtype=float)
        if len(values) < 5 or np.unique(transformed_time).size < 2 or np.unique(values).size < 2:
            diagnostics.append(
                {
                    "term": term,
                    "events_with_residuals": int(len(values)),
                    "log_time_pearson_r": None,
                    "p_value": None,
                    "bonferroni_p_value": None,
                    "status": "insufficient variation for an exploratory residual-time correlation",
                }
            )
            continue
        correlation, p_value = stats.pearsonr(transformed_time, values)
        diagnostics.append(
            {
                "term": term,
                "events_with_residuals": int(len(values)),
                "log_time_pearson_r": _round(correlation),
                "p_value": _round(p_value),
                "bonferroni_p_value": _round(min(1.0, p_value * len(exog.columns))),
                "status": "exploratory residual-time correlation",
            }
        )
    return {
        "method": "Schoenfeld-style residual correlation with log(time)",
        "event_residuals": int(len(event_times)),
        "diagnostics": diagnostics,
        "warning": "Exploratory only: this is not a global proportional-hazards test and should be interpreted with plots and study-specific diagnostics.",
    }


def linear_mixed_effects(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    group_column: str,
    alpha: float = 0.05,
    categorical_predictors: list[str] | None = None,
    category_references: dict[str, str] | None = None,
    interactions: list[tuple[str, str]] | None = None,
    random_slope_column: str | None = None,
) -> dict[str, Any]:
    """Fit a Gaussian mixed model with a random intercept and optional numeric random slope.

    Args:
        records: JSON-like clustered row objects.
        outcome: Numeric continuous outcome column.
        predictors: Fixed-effect predictor columns.
        group_column: Cluster identifier for random effects.
        alpha: Confidence interval complement.
        categorical_predictors: Predictors to treatment-code with explicit references.
        category_references: Optional categorical predictor to reference-level mapping.
        interactions: Selected pairs of fixed predictors to interact.
        random_slope_column: Optional numeric fixed predictor with a group-specific slope.

    Returns:
        Fixed-effect inference, encoded-factor metadata, random-effect variance/correlation,
        ICC metadata, and a reproducibility fingerprint.

    Raises:
        AnalyticsError: If the design is singular, groups are too small, or fitting fails.

    Examples:
        >>> linear_mixed_effects([{"id": "a", "t": 0, "y": 1}, {"id": "a", "t": 1, "y": 2}, {"id": "b", "t": 0, "y": 3}, {"id": "b", "t": 1, "y": 4}], "y", ["t"], "id")["groups"]
        2
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    if group_column in {outcome, *predictors}:
        raise AnalyticsError("group_column must differ from outcome and predictors")
    if random_slope_column and random_slope_column not in predictors:
        raise AnalyticsError("random_slope_column must be one of the fixed-effect predictors")
    if random_slope_column and random_slope_column in (categorical_predictors or []):
        raise AnalyticsError("random_slope_column must be a numeric, not categorical, predictor")
    frame = _frame(records)
    if group_column not in frame.columns:
        raise AnalyticsError(f"Column '{group_column}' was not found")
    numeric = _numeric_columns(frame)
    if outcome not in numeric:
        raise AnalyticsError(f"Linear mixed-effects requires numeric column: '{outcome}'")
    if random_slope_column and random_slope_column not in numeric:
        raise AnalyticsError(f"Random slope requires numeric column: '{random_slope_column}'")
    design, encoding, normalized_interactions = _treatment_coded_design(
        frame, predictors, categorical_predictors, category_references, interactions
    )
    data = pd.concat(
        [pd.DataFrame({"outcome": numeric[outcome], "group": frame[group_column].astype("string")}), design], axis=1
    )
    data = data.dropna()
    group_sizes = data["group"].value_counts()
    if len(data) <= design.shape[1] + 2 or len(group_sizes) < 2:
        raise AnalyticsError("Linear mixed-effects requires more complete rows and at least two groups")
    minimum_group_size = 3 if random_slope_column else 2
    if int(group_sizes.min()) < minimum_group_size:
        raise AnalyticsError(f"Each observed group must contain at least {minimum_group_size} complete rows")
    exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for term in design.columns:
        exog[term] = data[term].astype(float)
    if np.linalg.matrix_rank(exog.to_numpy()) != exog.shape[1]:
        raise AnalyticsError("Linear mixed-effects predictors are perfectly collinear")
    exog_re = pd.DataFrame({"intercept": 1.0}, index=data.index)
    if random_slope_column:
        exog_re[random_slope_column] = data[random_slope_column].astype(float)
        if exog_re[random_slope_column].nunique() < 2:
            raise AnalyticsError("random_slope_column must vary across complete observations")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = sm.MixedLM(data["outcome"].astype(float), exog, groups=data["group"], exog_re=exog_re).fit(
                reml=True, method="lbfgs", disp=False
            )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Linear mixed-effects model could not fit this dataset") from exc
    random_variance = float(fitted.cov_re.iloc[0, 0])
    residual_variance = float(fitted.scale)
    random_slope_variance = float(fitted.cov_re.iloc[1, 1]) if random_slope_column else None
    random_effect_correlation = (
        float(fitted.cov_re.iloc[0, 1] / math.sqrt(random_variance * random_slope_variance))
        if random_slope_variance is not None and random_variance > 0 and random_slope_variance > 0
        else None
    )
    if (
        not fitted.converged
        or not math.isfinite(random_variance)
        or random_variance <= 0
        or not math.isfinite(residual_variance)
        or residual_variance <= 0
        or (random_slope_variance is not None and (not math.isfinite(random_slope_variance) or random_slope_variance <= 0))
        or (random_effect_correlation is not None and (not math.isfinite(random_effect_correlation) or abs(random_effect_correlation) > 1))
        or not np.isfinite(fitted.fe_params.to_numpy()).all()
        or not np.isfinite(fitted.bse_fe.to_numpy()).all()
    ):
        raise AnalyticsError("Linear mixed-effects fit was singular or did not converge")
    confidence_intervals = fitted.conf_int(alpha=alpha)
    coefficients = []
    for term in exog.columns:
        coefficients.append(
            {
                "term": term,
                "estimate": _round(fitted.fe_params[term]),
                "std_error": _round(fitted.bse_fe[term]),
                "z_value": _round(fitted.tvalues[term]),
                "p_value": _round(fitted.pvalues[term]),
                "ci_lower": _round(confidence_intervals.loc[term, 0]),
                "ci_upper": _round(confidence_intervals.loc[term, 1]),
            }
        )
    settings = {
        "analysis": "linear_mixed_effects_random_intercept",
        "outcome": outcome,
        "predictors": predictors,
        "group_column": group_column,
        "random_slope_column": random_slope_column,
        "categorical_predictors": categorical_predictors or [],
        "category_references": category_references or {},
        "interactions": normalized_interactions,
        "alpha": alpha,
    }
    return {
        "method": "Gaussian linear mixed-effects model (random intercept + slope)" if random_slope_column else "Gaussian linear mixed-effects model (random intercept)",
        "n": int(len(data)),
        "groups": int(len(group_sizes)),
        "minimum_group_size": int(group_sizes.min()),
        "estimation": "REML",
        "categorical_encoding": encoding,
        "interactions": [{"left": left, "right": right} for left, right in normalized_interactions],
        "coefficients": coefficients,
        "random_intercept_variance": _round(random_variance),
        "random_slope_column": random_slope_column,
        "random_slope_variance": _round(random_slope_variance),
        "random_intercept_slope_correlation": _round(random_effect_correlation),
        "residual_variance": _round(residual_variance),
        "intraclass_correlation": _round(random_variance / (random_variance + residual_variance)),
        "intraclass_correlation_at_zero": _round(random_variance / (random_variance + residual_variance)),
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def generalized_estimating_equations(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    group_column: str,
    family: str = "gaussian",
    working_correlation: str = "exchangeable",
    alpha: float = 0.05,
    categorical_predictors: list[str] | None = None,
    category_references: dict[str, str] | None = None,
    interactions: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Fit a population-averaged GEE with robust sandwich standard errors.

    Args:
        records: JSON-like rows containing correlated observations.
        outcome: Numeric Gaussian or Poisson outcome, or a two-level Binomial outcome.
        predictors: Unique numeric or explicitly categorical fixed predictors.
        group_column: Identifier that groups correlated observations.
        family: One of gaussian, binomial, or poisson.
        working_correlation: Either independence or exchangeable.
        alpha: Confidence interval complement.
        categorical_predictors: Predictor subset to treatment-code.
        category_references: Optional factor-to-reference-level mapping.
        interactions: Selected pairs of fixed predictor names to interact.

    Returns:
        Population-averaged coefficients, robust confidence intervals, working-correlation
        metadata, categorical encoding, and a reproducibility fingerprint.

    Raises:
        AnalyticsError: If the response, cluster structure, design, or GEE fit is invalid.

    Examples:
        >>> generalized_estimating_equations([{"id": "a", "x": 0, "y": 1}, {"id": "a", "x": 1, "y": 2}, {"id": "b", "x": 0, "y": 3}, {"id": "b", "x": 1, "y": 4}, {"id": "c", "x": 0, "y": 5}, {"id": "c", "x": 1, "y": 6}], "y", ["x"], "id")["groups"]
        3
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if family not in {"gaussian", "binomial", "poisson"}:
        raise AnalyticsError("family must be gaussian, binomial, or poisson")
    if working_correlation not in {"independence", "exchangeable"}:
        raise AnalyticsError("working_correlation must be independence or exchangeable")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    if group_column in {outcome, *predictors}:
        raise AnalyticsError("group_column must differ from outcome and predictors")
    frame = _frame(records)
    if group_column not in frame.columns:
        raise AnalyticsError(f"Column '{group_column}' was not found")
    design, encoding, normalized_interactions = _treatment_coded_design(
        frame, predictors, categorical_predictors, category_references, interactions
    )
    numeric = _numeric_columns(frame)
    event: dict[str, str] | None = None
    if family == "binomial":
        if outcome not in frame.columns:
            raise AnalyticsError(f"Column '{outcome}' was not found")
        labels = sorted(frame[outcome].dropna().astype(str).unique().tolist())
        if len(labels) != 2:
            raise AnalyticsError("Binomial GEE outcome must contain exactly two observed categories")
        outcome_values = (frame[outcome].astype("string") == labels[1]).astype(float)
        event = {"reference": labels[0], "event": labels[1]}
        gee_family: Any = sm.families.Binomial()
    else:
        if outcome not in numeric:
            raise AnalyticsError(f"{family.title()} GEE requires numeric column: '{outcome}'")
        outcome_values = numeric[outcome]
        if family == "poisson":
            observed = outcome_values.dropna()
            if (observed < 0).any() or not np.isclose(observed, np.round(observed)).all():
                raise AnalyticsError("Poisson GEE outcome must contain non-negative integer counts")
            gee_family = sm.families.Poisson()
        else:
            gee_family = sm.families.Gaussian()
    data = pd.concat(
        [
            pd.DataFrame({"outcome": outcome_values, "group": frame[group_column].astype("string")}),
            design,
        ],
        axis=1,
    ).dropna()
    group_sizes = data["group"].value_counts()
    if len(group_sizes) < 3 or len(data) <= design.shape[1] + 2:
        raise AnalyticsError("GEE requires at least three groups and more complete rows than fitted parameters")
    if working_correlation == "exchangeable" and int(group_sizes.min()) < 2:
        raise AnalyticsError("Exchangeable GEE requires at least two complete observations in every group")
    if family == "binomial" and data["outcome"].nunique() != 2:
        raise AnalyticsError("Binomial GEE requires both outcome categories after complete-case filtering")
    if family == "poisson" and data["outcome"].sum() <= 0:
        raise AnalyticsError("Poisson GEE requires at least one observed count")
    exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for term in design.columns:
        exog[term] = data[term].astype(float)
    if np.linalg.matrix_rank(exog.to_numpy()) != exog.shape[1]:
        raise AnalyticsError("GEE predictors are perfectly collinear")
    covariance_structure = (
        sm.cov_struct.Independence()
        if working_correlation == "independence"
        else sm.cov_struct.Exchangeable()
    )
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = sm.GEE(
                data["outcome"].astype(float),
                exog,
                groups=data["group"],
                family=gee_family,
                cov_struct=covariance_structure,
            ).fit()
        if not bool(getattr(fitted, "converged", True)):
            raise AnalyticsError("GEE did not converge")
        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise AnalyticsError("GEE did not converge")
    except AnalyticsError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("GEE could not fit this dataset") from exc
    estimates = np.asarray(fitted.params, dtype=float)
    standard_errors = np.asarray(fitted.bse, dtype=float)
    p_values = np.asarray(fitted.pvalues, dtype=float)
    if not np.isfinite(estimates).all() or not np.isfinite(standard_errors).all() or (standard_errors <= 0).any():
        raise AnalyticsError("GEE produced non-finite or zero-precision estimates")
    critical = float(stats.norm.ppf(1 - alpha / 2))
    coefficients: list[dict[str, Any]] = []
    for index, term in enumerate(exog.columns):
        estimate = float(estimates[index])
        standard_error = float(standard_errors[index])
        lower = estimate - critical * standard_error
        upper = estimate + critical * standard_error
        coefficient = {
            "term": term,
            "estimate": _round(estimate),
            "std_error": _round(standard_error),
            "z_value": _round(estimate / standard_error),
            "p_value": _round(p_values[index]),
            "ci_lower": _round(lower),
            "ci_upper": _round(upper),
        }
        if family == "binomial":
            coefficient.update(
                {
                    "odds_ratio": _exp_round(estimate),
                    "odds_ratio_ci_lower": _exp_round(lower),
                    "odds_ratio_ci_upper": _exp_round(upper),
                }
            )
        elif family == "poisson":
            coefficient.update(
                {
                    "rate_ratio": _exp_round(estimate),
                    "rate_ratio_ci_lower": _exp_round(lower),
                    "rate_ratio_ci_upper": _exp_round(upper),
                }
            )
        coefficients.append(coefficient)
    dependence = getattr(fitted.cov_struct, "dep_params", None)
    settings = {
        "analysis": "generalized_estimating_equations",
        "outcome": outcome,
        "predictors": predictors,
        "group_column": group_column,
        "family": family,
        "working_correlation": working_correlation,
        "categorical_predictors": categorical_predictors or [],
        "category_references": category_references or {},
        "interactions": normalized_interactions,
        "alpha": alpha,
    }
    return {
        "method": "Generalized estimating equations",
        "estimand": "population-averaged",
        "n": int(len(data)),
        "groups": int(len(group_sizes)),
        "minimum_group_size": int(group_sizes.min()),
        "maximum_group_size": int(group_sizes.max()),
        "family": family,
        "event": event,
        "working_correlation": working_correlation,
        "working_correlation_estimate": _round(dependence),
        "variance_estimator": "robust sandwich",
        "scale": _round(fitted.scale),
        "categorical_encoding": encoding,
        "interactions": [{"left": left, "right": right} for left, right in normalized_interactions],
        "coefficients": coefficients,
        "warning": "GEE estimates population-averaged effects. The working correlation is not a proof of the true within-group covariance, and few clusters can make sandwich standard errors unreliable.",
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def cox_proportional_hazards(
    records: list[dict[str, Any]],
    time_column: str,
    event_column: str,
    predictors: list[str],
    strata_column: str | None = None,
    cluster_column: str | None = None,
    ties: str = "efron",
    alpha: float = 0.05,
    categorical_predictors: list[str] | None = None,
    category_references: dict[str, str] | None = None,
    interactions: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Fit Cox proportional-hazards regression with validated encoding and diagnostics.

    Args:
        records: JSON-like follow-up rows.
        time_column: Strictly positive numeric follow-up time.
        event_column: Observed two-level event/censor column.
        predictors: Fixed predictor columns.
        strata_column: Optional baseline-hazard stratum column.
        cluster_column: Optional cluster column for robust standard errors.
        ties: Efron or Breslow tie method.
        alpha: Confidence interval complement.
        categorical_predictors: Predictors to treatment-code.
        category_references: Explicit reference level for each categorical predictor.
        interactions: Selected fixed-effect interaction pairs.

    Returns:
        Hazard-ratio inference, fit statistics, exploratory Schoenfeld-style diagnostics,
        and a reproducibility fingerprint.

    Raises:
        AnalyticsError: If data are invalid, the model is non-identifiable, or fitting fails.

    Examples:
        >>> cox_proportional_hazards([{"t": 1, "e": "event", "x": 0}, {"t": 2, "e": "censor", "x": 1}, {"t": 3, "e": "event", "x": 2}], "t", "e", ["x"])["events"]
        2
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if ties not in {"breslow", "efron"}:
        raise AnalyticsError("ties must be either 'breslow' or 'efron'")
    if not predictors or len(set(predictors)) != len(predictors) or time_column in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain time_column")
    reserved = {time_column, event_column, *predictors}
    if strata_column and strata_column in reserved:
        raise AnalyticsError("strata_column must differ from time, event, and predictor columns")
    if cluster_column and (cluster_column in reserved or cluster_column == strata_column):
        raise AnalyticsError("cluster_column must differ from time, event, strata, and predictor columns")
    if strata_column and cluster_column:
        raise AnalyticsError("Cox regression currently supports either strata or clustered standard errors, not both together")
    frame = _frame(records)
    if event_column not in frame.columns:
        raise AnalyticsError(f"Column '{event_column}' was not found")
    numeric = _numeric_columns(frame)
    if time_column not in numeric:
        raise AnalyticsError(f"Cox regression requires numeric column: '{time_column}'")
    if strata_column and strata_column not in frame.columns:
        raise AnalyticsError(f"Column '{strata_column}' was not found")
    if cluster_column and cluster_column not in frame.columns:
        raise AnalyticsError(f"Column '{cluster_column}' was not found")
    design, encoding, normalized_interactions = _treatment_coded_design(
        frame, predictors, categorical_predictors, category_references, interactions
    )
    data = pd.concat(
        [pd.DataFrame({"time": numeric[time_column], "event": frame[event_column].astype("string")}), design], axis=1
    )
    if strata_column:
        data["strata"] = frame[strata_column].astype("string")
    if cluster_column:
        data["cluster"] = frame[cluster_column].astype("string")
    data = data.dropna()
    if data.empty or (data["time"] <= 0).any():
        raise AnalyticsError("Cox regression requires strictly positive complete follow-up times")
    event_labels = sorted(data["event"].unique().tolist())
    if len(event_labels) != 2:
        raise AnalyticsError("event_column must contain exactly two observed categories")
    data["event_binary"] = (data["event"] == event_labels[1]).astype(int)
    event_count = int(data["event_binary"].sum())
    if event_count <= design.shape[1]:
        raise AnalyticsError("Cox regression requires more observed events than fitted predictors")
    if strata_column:
        strata_events = data.groupby("strata", observed=True)["event_binary"].sum()
        if (strata_events == 0).any():
            raise AnalyticsError("Each observed stratum must contain at least one event")
    if cluster_column and data["cluster"].nunique() < 2:
        raise AnalyticsError("Robust clustered standard errors require at least two clusters")
    exog = data.loc[:, design.columns].astype(float)
    if np.linalg.matrix_rank(exog.to_numpy()) != exog.shape[1]:
        raise AnalyticsError("Cox regression predictors are perfectly collinear")
    model_kwargs: dict[str, Any] = {"status": data["event_binary"].to_numpy(), "ties": ties}
    if strata_column:
        model_kwargs["strata"] = data["strata"].to_numpy()
    groups = data["cluster"].to_numpy() if cluster_column else None
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = sm.duration.PHReg(data["time"].to_numpy(), exog.to_numpy(), **model_kwargs).fit(groups=groups)
        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise AnalyticsError("Cox regression did not converge")
    except AnalyticsError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Cox regression could not fit this dataset") from exc
    if not np.isfinite(fitted.params).all() or not np.isfinite(fitted.bse).all():
        raise AnalyticsError("Cox regression produced non-finite estimates")
    try:
        null_model = sm.duration.PHReg(data["time"].to_numpy(), np.empty((len(data), 0)), **model_kwargs).fit()
        likelihood_ratio = max(0.0, 2 * (float(fitted.llf) - float(null_model.llf)))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Cox regression could not calculate the null-model comparison") from exc
    confidence_intervals = fitted.conf_int(alpha=alpha)
    coefficients = []
    for index, column in enumerate(exog.columns):
        lower, upper = confidence_intervals[index]
        coefficients.append(
            {
                "term": column,
                "log_hazard_ratio": _round(fitted.params[index]),
                "std_error": _round(fitted.bse[index]),
                "z_value": _round(fitted.params[index] / fitted.bse[index]),
                "p_value": _round(fitted.pvalues[index]),
                "hazard_ratio": _exp_round(fitted.params[index]),
                "hazard_ratio_ci_lower": _exp_round(lower),
                "hazard_ratio_ci_upper": _exp_round(upper),
            }
        )
    proportional_hazards_diagnostic = _cox_schoenfeld_style_diagnostic(
        data, exog, np.asarray(fitted.params, dtype=float), strata_column
    )
    settings = {
        "analysis": "cox_proportional_hazards",
        "time_column": time_column,
        "event_column": event_column,
        "predictors": predictors,
        "categorical_predictors": categorical_predictors or [],
        "category_references": category_references or {},
        "interactions": normalized_interactions,
        "strata_column": strata_column,
        "cluster_column": cluster_column,
        "ties": ties,
        "alpha": alpha,
    }
    return {
        "method": "Cox proportional hazards regression",
        "n": int(len(data)),
        "events": event_count,
        "outcome": {"censor_or_reference": event_labels[0], "event": event_labels[1]},
        "ties": ties,
        "strata": int(data["strata"].nunique()) if strata_column else None,
        "variance_estimator": "cluster-robust" if cluster_column else "model-based",
        "partial_log_likelihood": _round(fitted.llf),
        "likelihood_ratio_chi_square": _round(likelihood_ratio),
        "likelihood_ratio_degrees_freedom": exog.shape[1],
        "likelihood_ratio_p_value": _round(stats.chi2.sf(likelihood_ratio, exog.shape[1])),
        "categorical_encoding": encoding,
        "interactions": [{"left": left, "right": right} for left, right in normalized_interactions],
        "coefficients": coefficients,
        "proportional_hazards_diagnostic": proportional_hazards_diagnostic,
        "warning": "The residual-time diagnostic is exploratory, not a definitive global proportional-hazards test; assess it with plots and study-specific diagnostics before interpretation.",
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def count_regression(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    distribution: str = "poisson",
    exposure_column: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit Poisson or NB2 count regression with an optional log-exposure offset.

    Args:
        records: JSON-like row objects with a non-negative integer outcome.
        outcome: Count outcome column.
        predictors: Unique numeric rate-predictor columns.
        distribution: Either poisson or negative_binomial for an NB2 variance model.
        exposure_column: Optional strictly positive exposure; its logarithm is an offset.
        alpha: Confidence interval complement.

    Returns:
        Rate-ratio inference, dispersion and fit diagnostics, and reproducibility metadata.

    Raises:
        AnalyticsError: If the data, design, distribution, or fitted estimates are invalid.

    Examples:
        >>> count_regression([{"y": 1, "x": 0}, {"y": 2, "x": 1}, {"y": 4, "x": 2}], "y", ["x"])["distribution"]
        'poisson'
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if distribution not in {"poisson", "negative_binomial"}:
        raise AnalyticsError("distribution must be either 'poisson' or 'negative_binomial'")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    if exposure_column and exposure_column in {outcome, *predictors}:
        raise AnalyticsError("exposure_column must differ from outcome and predictors")
    frame = _frame(records)
    numeric = _numeric_columns(frame)
    for column in [outcome, *predictors, *([exposure_column] if exposure_column else [])]:
        if column not in numeric:
            raise AnalyticsError(f"Count regression requires numeric column: '{column}'")
    observed_outcome = numeric[outcome].dropna()
    if (observed_outcome < 0).any() or not np.isclose(observed_outcome, np.round(observed_outcome)).all():
        raise AnalyticsError("Count regression outcome must contain non-negative integer counts")
    data = pd.DataFrame({"outcome": numeric[outcome]})
    for index, column in enumerate(predictors):
        data[f"predictor_{index}"] = numeric[column]
    if exposure_column:
        data["exposure"] = numeric[exposure_column]
    data = data.dropna()
    parameter_count = len(predictors) + 1
    if len(data) <= parameter_count or data["outcome"].sum() <= 0:
        raise AnalyticsError("Count regression requires enough complete rows and at least one observed count")
    if exposure_column and (data["exposure"] <= 0).any():
        raise AnalyticsError("exposure_column must contain strictly positive values")
    exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for index, column in enumerate(predictors):
        values = data[f"predictor_{index}"].astype(float)
        if values.nunique() < 2:
            raise AnalyticsError(f"Count predictor '{column}' must not be constant")
        exog[column] = values
    if np.linalg.matrix_rank(exog.to_numpy()) != parameter_count:
        raise AnalyticsError("Count regression predictors are perfectly collinear")
    offset = np.log(data["exposure"].to_numpy()) if exposure_column else None
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if distribution == "poisson":
                fitted = sm.GLM(
                    data["outcome"].to_numpy(), exog, family=sm.families.Poisson(), offset=offset
                ).fit()
                converged = bool(fitted.converged)
            else:
                fitted = sm.NegativeBinomial(
                    data["outcome"].to_numpy(), exog, loglike_method="nb2", offset=offset
                ).fit(disp=False)
                converged = bool(fitted.mle_retvals.get("converged", False))
        if not converged or any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise AnalyticsError("Count regression did not converge")
    except AnalyticsError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Count regression could not fit this dataset") from exc
    parameter_values = np.asarray(fitted.params, dtype=float)
    standard_errors = np.asarray(fitted.bse, dtype=float)
    p_values = np.asarray(fitted.pvalues, dtype=float)
    confidence_intervals = np.asarray(fitted.conf_int(alpha=alpha), dtype=float)
    dispersion_alpha = None
    if distribution == "negative_binomial":
        dispersion_alpha = float(parameter_values[-1])
        parameter_values, standard_errors, p_values = parameter_values[:-1], standard_errors[:-1], p_values[:-1]
        confidence_intervals = confidence_intervals[:-1]
        if not math.isfinite(dispersion_alpha) or dispersion_alpha <= 0:
            raise AnalyticsError("Negative-binomial dispersion estimate was invalid")
    if not np.isfinite(parameter_values).all() or not np.isfinite(standard_errors).all():
        raise AnalyticsError("Count regression produced non-finite estimates")
    coefficients = []
    for index, term in enumerate(exog.columns):
        lower, upper = confidence_intervals[index]
        coefficients.append(
            {
                "term": term,
                "log_rate_ratio": _round(parameter_values[index]),
                "std_error": _round(standard_errors[index]),
                "z_value": _round(parameter_values[index] / standard_errors[index]),
                "p_value": _round(p_values[index]),
                "rate_ratio": _exp_round(parameter_values[index]),
                "rate_ratio_ci_lower": _exp_round(lower),
                "rate_ratio_ci_upper": _exp_round(upper),
            }
        )
    predicted = np.asarray(fitted.predict(), dtype=float)
    variance = predicted if distribution == "poisson" else predicted + float(dispersion_alpha) * predicted**2
    pearson_chi_square = float(np.sum((data["outcome"].to_numpy() - predicted) ** 2 / variance))
    overdispersion_ratio = pearson_chi_square / float(fitted.df_resid)
    settings = {
        "analysis": "count_regression",
        "outcome": outcome,
        "predictors": predictors,
        "distribution": distribution,
        "exposure_column": exposure_column,
        "alpha": alpha,
    }
    warning = (
        "Poisson dispersion appears elevated; consider negative binomial and inspect zero inflation."
        if distribution == "poisson" and overdispersion_ratio > 1.5
        else "This model does not diagnose zero inflation, hurdle processes, dependence, or causal effects."
    )
    return {
        "method": "Poisson count regression" if distribution == "poisson" else "Negative-binomial (NB2) count regression",
        "n": int(len(data)),
        "total_count": int(data["outcome"].sum()),
        "exposure_column": exposure_column,
        "coefficients": coefficients,
        "aic": _round(fitted.aic),
        "residual_degrees_freedom": _round(fitted.df_resid),
        "pearson_chi_square": _round(pearson_chi_square),
        "overdispersion_ratio": _round(overdispersion_ratio),
        "dispersion_alpha": _round(dispersion_alpha),
        "warning": warning,
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def zero_inflated_count_regression(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    distribution: str = "poisson",
    exposure_column: str | None = None,
    inflation_predictors: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit ZIP or ZINB2 regression with a separately modeled structural-zero process.

    Args:
        records: JSON-like count-outcome rows.
        outcome: Non-negative integer count column.
        predictors: Numeric predictors for the count component.
        distribution: Either poisson or negative_binomial for the count component.
        exposure_column: Optional positive count-process exposure offset.
        inflation_predictors: Optional numeric predictors for structural-zero probability.
        alpha: Confidence interval complement.

    Returns:
        Count and inflation-component estimates, fit metrics, and reproducibility metadata.

    Raises:
        AnalyticsError: If data are unsuitable or either model component cannot converge.

    Examples:
        >>> zero_inflated_count_regression([{"y": 0, "x": 0}, {"y": 0, "x": 1}, {"y": 3, "x": 2}, {"y": 5, "x": 3}], "y", ["x"])["distribution"]
        'poisson'
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if distribution not in {"poisson", "negative_binomial"}:
        raise AnalyticsError("distribution must be either 'poisson' or 'negative_binomial'")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    inflation = inflation_predictors or []
    if len(set(inflation)) != len(inflation) or outcome in inflation:
        raise AnalyticsError("inflation_predictors must be unique and must not contain outcome")
    if exposure_column and exposure_column in {outcome, *predictors, *inflation}:
        raise AnalyticsError("exposure_column must differ from outcome and predictor columns")
    frame = _frame(records)
    numeric = _numeric_columns(frame)
    required = [outcome, *predictors, *inflation, *([exposure_column] if exposure_column else [])]
    for column in required:
        if column not in numeric:
            raise AnalyticsError(f"Zero-inflated count regression requires numeric column: '{column}'")
    observed_outcome = numeric[outcome].dropna()
    if (observed_outcome < 0).any() or not np.isclose(observed_outcome, np.round(observed_outcome)).all():
        raise AnalyticsError("Zero-inflated count regression outcome must contain non-negative integer counts")
    data = pd.DataFrame({"outcome": numeric[outcome]})
    for column in set([*predictors, *inflation]):
        data[column] = numeric[column]
    if exposure_column:
        data["exposure"] = numeric[exposure_column]
    data = data.dropna()
    if len(data) <= len(predictors) + len(inflation) + 2 or data["outcome"].sum() <= 0:
        raise AnalyticsError("Zero-inflated count regression requires enough complete rows and at least one observed count")
    if exposure_column and (data["exposure"] <= 0).any():
        raise AnalyticsError("exposure_column must contain strictly positive values")
    count_exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for column in predictors:
        values = data[column].astype(float)
        if values.nunique() < 2:
            raise AnalyticsError(f"Count predictor '{column}' must not be constant")
        count_exog[column] = values
    inflation_exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for column in inflation:
        values = data[column].astype(float)
        if values.nunique() < 2:
            raise AnalyticsError(f"Inflation predictor '{column}' must not be constant")
        inflation_exog[column] = values
    if np.linalg.matrix_rank(count_exog.to_numpy()) != count_exog.shape[1] or np.linalg.matrix_rank(inflation_exog.to_numpy()) != inflation_exog.shape[1]:
        raise AnalyticsError("Zero-inflated count regression predictors are perfectly collinear")
    offset = np.log(data["exposure"].to_numpy()) if exposure_column else None
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model_class = sm.ZeroInflatedPoisson if distribution == "poisson" else sm.ZeroInflatedNegativeBinomialP
            fitted = model_class(
                data["outcome"].to_numpy(),
                count_exog,
                exog_infl=inflation_exog,
                offset=offset,
                inflation="logit",
            ).fit(disp=False, maxiter=100)
        if not fitted.mle_retvals.get("converged", False) or any(
            issubclass(item.category, ConvergenceWarning) for item in captured
        ):
            raise AnalyticsError("Zero-inflated count regression did not converge")
    except AnalyticsError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Zero-inflated count regression could not fit this dataset") from exc
    parameters = fitted.params
    standard_errors = fitted.bse
    p_values = fitted.pvalues
    if not np.isfinite(parameters.to_numpy()).all() or not np.isfinite(standard_errors.to_numpy()).all():
        raise AnalyticsError("Zero-inflated count regression produced non-finite estimates")
    critical = float(stats.norm.ppf(1 - alpha / 2))

    def coefficient(term: str, prefix: str = "") -> dict[str, Any]:
        name = f"{prefix}{term}"
        estimate = float(parameters[name])
        standard_error = float(standard_errors[name])
        return {
            "term": term,
            "estimate": _round(estimate),
            "std_error": _round(standard_error),
            "z_value": _round(estimate / standard_error),
            "p_value": _round(p_values[name]),
            "ci_lower": _round(estimate - critical * standard_error),
            "ci_upper": _round(estimate + critical * standard_error),
        }

    count_coefficients = []
    for term in count_exog.columns:
        item = coefficient(term)
        item.update(
            {
                "rate_ratio": _exp_round(item["estimate"]),
                "rate_ratio_ci_lower": _exp_round(item["ci_lower"]),
                "rate_ratio_ci_upper": _exp_round(item["ci_upper"]),
            }
        )
        count_coefficients.append(item)
    inflation_coefficients = []
    for term in inflation_exog.columns:
        item = coefficient(term, "inflate_")
        item.update(
            {
                "structural_zero_odds_ratio": _exp_round(item["estimate"]),
                "structural_zero_odds_ratio_ci_lower": _exp_round(item["ci_lower"]),
                "structural_zero_odds_ratio_ci_upper": _exp_round(item["ci_upper"]),
            }
        )
        inflation_coefficients.append(item)
    dispersion_alpha = _round(parameters["alpha"]) if distribution == "negative_binomial" else None
    if dispersion_alpha is not None and dispersion_alpha <= 0:
        raise AnalyticsError("Zero-inflated negative-binomial dispersion estimate was invalid")
    settings = {
        "analysis": "zero_inflated_count_regression",
        "outcome": outcome,
        "predictors": predictors,
        "distribution": distribution,
        "exposure_column": exposure_column,
        "inflation_predictors": inflation,
        "alpha": alpha,
    }
    return {
        "method": "Zero-inflated Poisson regression" if distribution == "poisson" else "Zero-inflated negative-binomial (ZINB2) regression",
        "n": int(len(data)),
        "total_count": int(data["outcome"].sum()),
        "zero_count": int((data["outcome"] == 0).sum()),
        "zero_rate": _round(float((data["outcome"] == 0).mean())),
        "exposure_column": exposure_column,
        "count_coefficients": count_coefficients,
        "inflation_coefficients": inflation_coefficients,
        "dispersion_alpha": dispersion_alpha,
        "log_likelihood": _round(fitted.llf),
        "aic": _round(fitted.aic),
        "bic": _round(fitted.bic_llf if hasattr(fitted, "bic_llf") else fitted.bic),
        "warning": "The inflation component models structural-zero odds. Assess fit, sparse covariate patterns, dependence, and study design before treating observed zeros as structural.",
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def hurdle_poisson_regression(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    exposure_column: str | None = None,
    hurdle_predictors: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit a logit hurdle and a zero-truncated Poisson count model for positive counts.

    Args:
        records: JSON-like count-outcome rows.
        outcome: Non-negative integer count column.
        predictors: Numeric predictors for positive counts.
        exposure_column: Optional positive exposure offset for the positive-count component.
        hurdle_predictors: Optional numeric predictors of a positive rather than zero count.
        alpha: Confidence interval complement.

    Returns:
        Hurdle-logit and zero-truncated Poisson inference with reproducibility metadata.

    Raises:
        AnalyticsError: If data are invalid or either component does not converge.

    Examples:
        >>> hurdle_poisson_regression([{"y": 0, "x": 0}, {"y": 1, "x": 1}, {"y": 2, "x": 2}, {"y": 4, "x": 3}], "y", ["x"])["method"]
        'Poisson hurdle model'
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    hurdle = hurdle_predictors or predictors
    if len(set(hurdle)) != len(hurdle) or outcome in hurdle:
        raise AnalyticsError("hurdle_predictors must be unique and must not contain outcome")
    if exposure_column and exposure_column in {outcome, *predictors, *hurdle}:
        raise AnalyticsError("exposure_column must differ from outcome and predictor columns")
    frame = _frame(records)
    numeric = _numeric_columns(frame)
    required = [outcome, *predictors, *hurdle, *([exposure_column] if exposure_column else [])]
    for column in required:
        if column not in numeric:
            raise AnalyticsError(f"Hurdle Poisson regression requires numeric column: '{column}'")
    observed_outcome = numeric[outcome].dropna()
    if (observed_outcome < 0).any() or not np.isclose(observed_outcome, np.round(observed_outcome)).all():
        raise AnalyticsError("Hurdle Poisson regression outcome must contain non-negative integer counts")
    data = pd.DataFrame({"outcome": numeric[outcome]})
    for column in dict.fromkeys([*predictors, *hurdle]):
        data[column] = numeric[column]
    if exposure_column:
        data["exposure"] = numeric[exposure_column]
    data = data.dropna()
    positive = data["outcome"] > 0
    if not positive.any() or positive.all():
        raise AnalyticsError("Hurdle Poisson regression requires both zero and positive observed counts")
    if exposure_column and (data["exposure"] <= 0).any():
        raise AnalyticsError("exposure_column must contain strictly positive values")
    count_exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    hurdle_exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for column in predictors:
        values = data[column].astype(float)
        if values.nunique() < 2:
            raise AnalyticsError(f"Count predictor '{column}' must not be constant")
        count_exog[column] = values
    for column in hurdle:
        values = data[column].astype(float)
        if values.nunique() < 2:
            raise AnalyticsError(f"Hurdle predictor '{column}' must not be constant")
        hurdle_exog[column] = values
    if np.linalg.matrix_rank(count_exog.to_numpy()) != count_exog.shape[1] or np.linalg.matrix_rank(hurdle_exog.to_numpy()) != hurdle_exog.shape[1]:
        raise AnalyticsError("Hurdle Poisson regression predictors are perfectly collinear")
    positive_count_exog = count_exog.loc[positive]
    if len(positive_count_exog) <= count_exog.shape[1]:
        raise AnalyticsError("Hurdle Poisson count component requires more positive rows than fitted parameters")
    offset = np.log(data["exposure"].to_numpy()) if exposure_column else np.zeros(len(data))
    positive_offset = offset[positive.to_numpy()]
    positive_counts = data.loc[positive, "outcome"].to_numpy(dtype=float)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            hurdle_fit = sm.Logit(positive.astype(float), hurdle_exog).fit(disp=False, maxiter=100)
        if not hurdle_fit.mle_retvals.get("converged", False) or any(
            issubclass(item.category, ConvergenceWarning) for item in captured
        ):
            raise AnalyticsError("Hurdle-logit component did not converge")
        start = sm.GLM(
            positive_counts, positive_count_exog, family=sm.families.Poisson(), offset=positive_offset
        ).fit().params.to_numpy()
    except AnalyticsError:
        raise
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Hurdle Poisson regression could not fit the hurdle component") from exc

    count_matrix = positive_count_exog.to_numpy()

    def truncated_objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = count_matrix @ beta + positive_offset
        mu = np.exp(np.clip(eta, -30, 30))
        log_normalizer = np.log(-np.expm1(-mu))
        log_likelihood = np.sum(positive_counts * eta - mu - special.gammaln(positive_counts + 1) - log_normalizer)
        adjustment = np.zeros_like(mu)
        manageable = mu < 50
        adjustment[manageable] = mu[manageable] / np.expm1(mu[manageable])
        score = positive_counts - mu - adjustment
        return -float(log_likelihood), -(count_matrix.T @ score)

    try:
        optimized = optimize.minimize(
            lambda beta: truncated_objective(beta)[0],
            start,
            jac=lambda beta: truncated_objective(beta)[1],
            method="BFGS",
            options={"maxiter": 200, "gtol": 1e-7},
        )
        objective, gradient = truncated_objective(optimized.x)
        if not optimized.success and float(np.max(np.abs(gradient))) > 1e-5:
            raise AnalyticsError("Zero-truncated Poisson component did not converge")
        eta = count_matrix @ optimized.x + positive_offset
        mu = np.exp(np.clip(eta, -30, 30))
        derivative_adjustment = np.zeros_like(mu)
        manageable = mu < 50
        denominator = np.expm1(mu[manageable])
        derivative_adjustment[manageable] = (
            mu[manageable] * (denominator - mu[manageable] * np.exp(mu[manageable])) / denominator**2
        )
        information_weights = mu + derivative_adjustment
        information = count_matrix.T @ (count_matrix * information_weights[:, None])
        covariance = np.linalg.inv(information)
        standard_errors = np.sqrt(np.diag(covariance))
    except AnalyticsError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError) as exc:
        raise AnalyticsError("Zero-truncated Poisson component could not fit this dataset") from exc
    if not np.isfinite(optimized.x).all() or not np.isfinite(standard_errors).all() or (standard_errors <= 0).any():
        raise AnalyticsError("Zero-truncated Poisson component produced non-finite estimates")
    critical = float(stats.norm.ppf(1 - alpha / 2))
    count_coefficients = []
    for index, term in enumerate(count_exog.columns):
        estimate, standard_error = float(optimized.x[index]), float(standard_errors[index])
        z_value = estimate / standard_error
        count_coefficients.append(
            {
                "term": term,
                "log_rate_ratio": _round(estimate),
                "std_error": _round(standard_error),
                "z_value": _round(z_value),
                "p_value": _round(2 * stats.norm.sf(abs(z_value))),
                "rate_ratio": _exp_round(estimate),
                "rate_ratio_ci_lower": _exp_round(estimate - critical * standard_error),
                "rate_ratio_ci_upper": _exp_round(estimate + critical * standard_error),
            }
        )
    hurdle_coefficients = []
    for term in hurdle_exog.columns:
        estimate, standard_error = float(hurdle_fit.params[term]), float(hurdle_fit.bse[term])
        hurdle_coefficients.append(
            {
                "term": term,
                "log_positive_count_odds": _round(estimate),
                "std_error": _round(standard_error),
                "z_value": _round(estimate / standard_error),
                "p_value": _round(hurdle_fit.pvalues[term]),
                "positive_count_odds_ratio": _exp_round(estimate),
                "positive_count_odds_ratio_ci_lower": _exp_round(estimate - critical * standard_error),
                "positive_count_odds_ratio_ci_upper": _exp_round(estimate + critical * standard_error),
            }
        )
    log_likelihood = float(hurdle_fit.llf - objective)
    parameter_count = len(count_exog.columns) + len(hurdle_exog.columns)
    settings = {
        "analysis": "hurdle_poisson_regression",
        "outcome": outcome,
        "predictors": predictors,
        "exposure_column": exposure_column,
        "hurdle_predictors": hurdle,
        "alpha": alpha,
    }
    return {
        "method": "Hurdle Poisson regression (logit + zero-truncated Poisson)",
        "n": int(len(data)),
        "positive_count_n": int(positive.sum()),
        "zero_count": int((~positive).sum()),
        "exposure_column": exposure_column,
        "count_coefficients": count_coefficients,
        "hurdle_coefficients": hurdle_coefficients,
        "log_likelihood": _round(log_likelihood),
        "aic": _round(2 * parameter_count - 2 * log_likelihood),
        "bic": _round(math.log(len(data)) * parameter_count - 2 * log_likelihood),
        "warning": "The hurdle component models odds of any positive count; the count component is conditional on positive counts. Assess fit, dependence, and study design before interpretation.",
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def ordinal_logistic_regression(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    category_order: list[str],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit a proportional-odds ordinal logistic model with explicit category order.

    Args:
        records: JSON-like rows containing an ordered categorical outcome.
        outcome: Ordered outcome column.
        predictors: Unique numeric predictor columns.
        category_order: Every observed outcome label in increasing order.
        alpha: Confidence interval complement.

    Returns:
        Proportional-odds coefficients, thresholds, odds ratios, and fingerprint metadata.

    Raises:
        AnalyticsError: If ordering, data, design, or model convergence is invalid.

    Examples:
        >>> ordinal_logistic_regression([{"y": "low", "x": 0}, {"y": "mid", "x": 1}, {"y": "high", "x": 2}], "y", ["x"], ["low", "mid", "high"])["category_order"]
        ['low', 'mid', 'high']
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    if len(category_order) < 3 or len(category_order) > 20 or len(set(category_order)) != len(category_order):
        raise AnalyticsError("category_order must contain between three and twenty unique ordered labels")
    frame = _frame(records)
    if outcome not in frame.columns:
        raise AnalyticsError(f"Column '{outcome}' was not found")
    numeric = _numeric_columns(frame)
    for column in predictors:
        if column not in numeric:
            raise AnalyticsError(f"Ordinal logistic regression requires numeric predictor: '{column}'")
    data = pd.DataFrame({"outcome": frame[outcome].astype("string")})
    for index, column in enumerate(predictors):
        data[f"predictor_{index}"] = numeric[column]
    data = data.dropna()
    observed_labels = set(data["outcome"].unique().tolist())
    requested_labels = set(category_order)
    if observed_labels != requested_labels:
        raise AnalyticsError("category_order must contain each observed outcome label exactly once")
    if len(data) <= len(predictors) + len(category_order):
        raise AnalyticsError("Ordinal logistic regression requires more complete rows than model parameters")
    exog = pd.DataFrame(index=data.index)
    for index, column in enumerate(predictors):
        values = data[f"predictor_{index}"].astype(float)
        if values.nunique() < 2:
            raise AnalyticsError(f"Ordinal logistic predictor '{column}' must not be constant")
        exog[column] = values
    if np.linalg.matrix_rank(exog.to_numpy()) != exog.shape[1]:
        raise AnalyticsError("Ordinal logistic predictors are perfectly collinear")
    ordered_outcome = pd.Series(
        pd.Categorical(data["outcome"], categories=category_order, ordered=True), index=data.index
    )
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = OrderedModel(ordered_outcome, exog, distr="logit").fit(method="bfgs", disp=False)
        if not fitted.mle_retvals.get("converged", False) or any(
            issubclass(item.category, ConvergenceWarning) for item in captured
        ):
            raise AnalyticsError("Ordinal logistic regression did not converge")
    except AnalyticsError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Ordinal logistic regression could not fit this dataset") from exc
    parameters = fitted.params
    standard_errors = fitted.bse
    p_values = fitted.pvalues
    confidence_intervals = fitted.conf_int(alpha=alpha)
    if not np.isfinite(parameters.loc[predictors].to_numpy()).all() or not np.isfinite(standard_errors.loc[predictors].to_numpy()).all():
        raise AnalyticsError("Ordinal logistic regression produced non-finite estimates")
    coefficients = []
    for column in predictors:
        estimate = float(parameters[column])
        lower, upper = confidence_intervals.loc[column]
        coefficients.append(
            {
                "term": column,
                "log_cumulative_odds_ratio": _round(estimate),
                "std_error": _round(standard_errors[column]),
                "z_value": _round(estimate / standard_errors[column]),
                "p_value": _round(p_values[column]),
                "cumulative_odds_ratio": _exp_round(estimate),
                "cumulative_odds_ratio_ci_lower": _exp_round(lower),
                "cumulative_odds_ratio_ci_upper": _exp_round(upper),
            }
        )
    transformed_thresholds = fitted.model.transform_threshold_params(parameters.to_numpy())[1:-1]
    settings = {
        "analysis": "ordinal_logistic_regression",
        "outcome": outcome,
        "predictors": predictors,
        "category_order": category_order,
        "alpha": alpha,
    }
    return {
        "method": "Ordinal logistic regression (proportional odds)",
        "n": int(len(data)),
        "categories": category_order,
        "coefficients": coefficients,
        "thresholds": [
            {"lower_category": category_order[index], "upper_category": category_order[index + 1], "cutpoint": _round(value)}
            for index, value in enumerate(transformed_thresholds)
        ],
        "log_likelihood": _round(fitted.llf),
        "aic": _round(fitted.aic),
        "bic": _round(fitted.bic),
        "warning": "The proportional-odds assumption is not tested automatically; assess it before interpreting one common odds ratio across thresholds.",
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def multinomial_logistic_regression(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    reference_category: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit a baseline-category multinomial logistic model with explicit reference.

    Args:
        records: JSON-like rows containing a nominal multi-class outcome.
        outcome: Nominal outcome column.
        predictors: Unique numeric predictor columns.
        reference_category: Observed category used as the baseline.
        alpha: Confidence interval complement.

    Returns:
        Per-comparison coefficients, relative-risk ratios, and reproducibility metadata.

    Raises:
        AnalyticsError: If categories, data, design, or optimization are invalid.

    Examples:
        >>> multinomial_logistic_regression([{"y": "a", "x": 0}, {"y": "b", "x": 1}, {"y": "c", "x": 2}], "y", ["x"], "a")["reference_category"]
        'a'
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    frame = _frame(records)
    if outcome not in frame.columns:
        raise AnalyticsError(f"Column '{outcome}' was not found")
    numeric = _numeric_columns(frame)
    for column in predictors:
        if column not in numeric:
            raise AnalyticsError(f"Multinomial logistic regression requires numeric predictor: '{column}'")
    data = pd.DataFrame({"outcome": frame[outcome].astype("string")})
    for index, column in enumerate(predictors):
        data[f"predictor_{index}"] = numeric[column]
    data = data.dropna()
    observed_categories = sorted(data["outcome"].unique().tolist())
    if not 3 <= len(observed_categories) <= 20:
        raise AnalyticsError("Multinomial logistic regression requires between three and twenty observed outcome categories")
    if reference_category not in observed_categories:
        raise AnalyticsError("reference_category must be one of the observed outcome labels")
    categories = [reference_category, *[label for label in observed_categories if label != reference_category]]
    parameter_count = (len(predictors) + 1) * (len(categories) - 1)
    if len(data) <= parameter_count:
        raise AnalyticsError("Multinomial logistic regression requires more complete rows than fitted parameters")
    exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for index, column in enumerate(predictors):
        values = data[f"predictor_{index}"].astype(float)
        if values.nunique() < 2:
            raise AnalyticsError(f"Multinomial logistic predictor '{column}' must not be constant")
        exog[column] = values
    if np.linalg.matrix_rank(exog.to_numpy()) != exog.shape[1]:
        raise AnalyticsError("Multinomial logistic predictors are perfectly collinear")
    category_codes = data["outcome"].map({label: index for index, label in enumerate(categories)}).astype(int)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = sm.MNLogit(category_codes, exog).fit(method="newton", maxiter=100, disp=False)
        if not fitted.mle_retvals.get("converged", False) or any(
            issubclass(item.category, ConvergenceWarning) for item in captured
        ):
            raise AnalyticsError("Multinomial logistic regression did not converge")
    except AnalyticsError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Multinomial logistic regression could not fit this dataset") from exc
    if not np.isfinite(fitted.params.to_numpy()).all() or not np.isfinite(fitted.bse.to_numpy()).all():
        raise AnalyticsError("Multinomial logistic regression produced non-finite estimates")
    critical = float(stats.norm.ppf(1 - alpha / 2))
    coefficient_sets = []
    for category_index, category in enumerate(categories[1:]):
        coefficients = []
        for term in exog.columns:
            estimate = float(fitted.params.loc[term, category_index])
            standard_error = float(fitted.bse.loc[term, category_index])
            coefficients.append(
                {
                    "term": term,
                    "log_relative_risk_ratio": _round(estimate),
                    "std_error": _round(standard_error),
                    "z_value": _round(estimate / standard_error),
                    "p_value": _round(fitted.pvalues.loc[term, category_index]),
                    "relative_risk_ratio": _exp_round(estimate),
                    "relative_risk_ratio_ci_lower": _exp_round(estimate - critical * standard_error),
                    "relative_risk_ratio_ci_upper": _exp_round(estimate + critical * standard_error),
                }
            )
        coefficient_sets.append({"category": category, "reference_category": reference_category, "coefficients": coefficients})
    likelihood_ratio = max(0.0, 2 * (float(fitted.llf) - float(fitted.llnull)))
    settings = {
        "analysis": "multinomial_logistic_regression",
        "outcome": outcome,
        "predictors": predictors,
        "reference_category": reference_category,
        "alpha": alpha,
    }
    return {
        "method": "Multinomial logistic regression (baseline-category)",
        "n": int(len(data)),
        "categories": categories,
        "reference_category": reference_category,
        "coefficient_sets": coefficient_sets,
        "log_likelihood": _round(fitted.llf),
        "aic": _round(fitted.aic),
        "bic": _round(fitted.bic),
        "mcfadden_pseudo_r_squared": _round(fitted.prsquared),
        "likelihood_ratio_chi_square": _round(likelihood_ratio),
        "likelihood_ratio_degrees_freedom": len(predictors) * (len(categories) - 1),
        "likelihood_ratio_p_value": _round(stats.chi2.sf(likelihood_ratio, len(predictors) * (len(categories) - 1))),
        "warning": "Classes are modeled as unordered. This model does not diagnose sparse classes, separation, dependence, or causal effects.",
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def weighted_ols(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    weight_column: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit weighted least squares for strictly positive analytic weights.

    Args:
        records: JSON-like rows containing numeric analysis values.
        outcome: Numeric continuous outcome column.
        predictors: Unique numeric predictor columns.
        weight_column: Strictly positive analytic-weight column.
        alpha: Confidence interval complement.

    Returns:
        Weighted least-squares coefficient inference and reproducibility metadata.

    Raises:
        AnalyticsError: If weights, data, design, or model estimates are invalid.

    Examples:
        >>> weighted_ols([{"y": 1, "x": 0, "w": 1}, {"y": 3, "x": 1, "w": 2}, {"y": 5, "x": 2, "w": 1}], "y", ["x"], "w")["method"]
        'Weighted least squares'
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    if weight_column in {outcome, *predictors}:
        raise AnalyticsError("weight_column must differ from outcome and predictors")
    frame = _frame(records)
    numeric = _numeric_columns(frame)
    for column in [outcome, *predictors, weight_column]:
        if column not in numeric:
            raise AnalyticsError(f"Weighted OLS requires numeric column: '{column}'")
    data = pd.DataFrame({"outcome": numeric[outcome], "weight": numeric[weight_column]})
    for index, column in enumerate(predictors):
        data[f"predictor_{index}"] = numeric[column]
    data = data.dropna()
    parameter_count = len(predictors) + 1
    if len(data) <= parameter_count:
        raise AnalyticsError("Weighted OLS requires more complete rows than fitted parameters")
    if (data["weight"] <= 0).any() or not np.isfinite(data["weight"].to_numpy()).all():
        raise AnalyticsError("weight_column must contain strictly positive finite values")
    exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for index, column in enumerate(predictors):
        exog[column] = data[f"predictor_{index}"].astype(float)
    if np.linalg.matrix_rank(exog.to_numpy()) != parameter_count:
        raise AnalyticsError("Weighted OLS predictors are perfectly collinear")
    try:
        fitted = sm.WLS(data["outcome"].astype(float), exog, weights=data["weight"].astype(float)).fit()
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Weighted OLS model could not fit this dataset") from exc
    if not np.isfinite(fitted.params.to_numpy()).all() or not np.isfinite(fitted.bse.to_numpy()).all():
        raise AnalyticsError("Weighted OLS model produced non-finite estimates")
    critical = float(stats.t.ppf(1 - alpha / 2, fitted.df_resid))
    coefficients = []
    for term in exog.columns:
        estimate = float(fitted.params[term])
        standard_error = float(fitted.bse[term])
        coefficients.append(
            {
                "term": term,
                "estimate": _round(estimate),
                "std_error": _round(standard_error),
                "t_value": _round(fitted.tvalues[term]),
                "p_value": _round(fitted.pvalues[term]),
                "ci_lower": _round(estimate - critical * standard_error),
                "ci_upper": _round(estimate + critical * standard_error),
            }
        )
    settings = {
        "analysis": "weighted_ols",
        "outcome": outcome,
        "predictors": predictors,
        "weight_column": weight_column,
        "alpha": alpha,
    }
    return {
        "method": "Weighted least squares (analytic weights)",
        "n": int(len(data)),
        "sum_weights": _round(data["weight"].sum()),
        "r_squared": _round(fitted.rsquared),
        "adjusted_r_squared": _round(fitted.rsquared_adj),
        "residual_degrees_freedom": _round(fitted.df_resid),
        "coefficients": coefficients,
        "warning": "Analytic weights only: this is not a complex-survey estimator and does not model strata, clusters, or finite-population corrections.",
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_record in frame.to_dict(orient="records"):
        records.append({str(key): _json_value(value) for key, value in raw_record.items()})
    return records


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise AnalyticsError(f"Column '{name}' was not found")
    numeric = _numeric_columns(frame)
    if name not in numeric:
        raise AnalyticsError(f"Column '{name}' must contain only numeric observed values")
    return numeric[name]


def regression_with_categorical_predictors(
    records: list[dict[str, Any]],
    model: str,
    outcome: str,
    predictors: list[str],
    categorical_predictors: list[str] | None = None,
    category_references: dict[str, str] | None = None,
    interactions: list[tuple[str, str]] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit OLS or binary logistic regression with explicit categorical encoding.

    Args:
        records: JSON-like rows containing the analysis fields.
        model: Either linear for OLS or logistic for binary logistic regression.
        outcome: Numeric OLS outcome or observed two-level logistic outcome.
        predictors: Numeric or categorical fixed predictors.
        categorical_predictors: Predictor subset to treatment-code.
        category_references: Optional factor-to-reference-level mapping.
        interactions: Selected pairs of predictor names to interact.
        alpha: Confidence interval complement.

    Returns:
        Encoded-design metadata, model-specific inference, and reproducibility metadata.

    Raises:
        AnalyticsError: If factor levels, design, response, or fitting are invalid.

    Examples:
        >>> regression_with_categorical_predictors([{"y": 1, "x": 0, "g": "a"}, {"y": 3, "x": 1, "g": "b"}, {"y": 5, "x": 2, "g": "a"}], "linear", "y", ["x", "g"], ["g"])["model"]
        'linear'
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if model not in {"linear", "logistic"}:
        raise AnalyticsError("model must be either 'linear' or 'logistic'")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    categorical = categorical_predictors or []
    references = category_references or {}
    if len(set(categorical)) != len(categorical) or not set(categorical).issubset(predictors):
        raise AnalyticsError("categorical_predictors must be a unique subset of predictors")
    if not set(references).issubset(categorical):
        raise AnalyticsError("category_references may only name categorical_predictors")
    normalized_interactions: list[tuple[str, str]] = []
    for pair in interactions or []:
        if len(pair) != 2 or pair[0] == pair[1] or pair[0] not in predictors or pair[1] not in predictors:
            raise AnalyticsError("Each interaction must contain two distinct predictor names")
        normalized = tuple(sorted(pair))
        if normalized not in normalized_interactions:
            normalized_interactions.append(normalized)
    frame = _frame(records)
    if outcome not in frame.columns:
        raise AnalyticsError(f"Column '{outcome}' was not found")
    numeric = _numeric_columns(frame)
    if model == "linear" and outcome not in numeric:
        raise AnalyticsError(f"Linear regression requires numeric outcome: '{outcome}'")
    design = pd.DataFrame(index=frame.index)
    feature_groups: dict[str, list[str]] = {}
    encoding: list[dict[str, Any]] = []
    for predictor in predictors:
        if predictor not in frame.columns:
            raise AnalyticsError(f"Column '{predictor}' was not found")
        if predictor not in categorical:
            if predictor not in numeric:
                raise AnalyticsError(f"Numeric predictor '{predictor}' must contain only numeric observed values")
            design[predictor] = numeric[predictor]
            feature_groups[predictor] = [predictor]
            continue
        source = frame[predictor].astype("string")
        levels = sorted(source.dropna().unique().tolist())
        if not 2 <= len(levels) <= 30:
            raise AnalyticsError(f"Categorical predictor '{predictor}' must have between two and thirty observed levels")
        reference = references.get(predictor, levels[0])
        if reference not in levels:
            raise AnalyticsError(f"Reference '{reference}' was not observed in categorical predictor '{predictor}'")
        terms = []
        for level in levels:
            if level == reference:
                continue
            term = f"{predictor}[{level}]"
            design[term] = np.where(source.isna(), np.nan, (source == level).astype(float))
            terms.append(term)
        feature_groups[predictor] = terms
        encoding.append({"predictor": predictor, "reference": reference, "levels": levels, "terms": terms})
    for left, right in normalized_interactions:
        for left_term in feature_groups[left]:
            for right_term in feature_groups[right]:
                term = f"{left_term} × {right_term}"
                design[term] = design[left_term] * design[right_term]
    if design.shape[1] > 100:
        raise AnalyticsError("Categorical encoding and interactions may produce at most 100 model features")
    if model == "linear":
        response = numeric[outcome]
        outcome_description = None
    else:
        labels = sorted(frame[outcome].dropna().astype(str).unique().tolist())
        if len(labels) != 2:
            raise AnalyticsError("Binary logistic regression requires exactly two observed outcome categories")
        response = (frame[outcome].astype("string") == labels[1]).astype(float).where(frame[outcome].notna())
        outcome_description = {"reference": labels[0], "event": labels[1]}
    data = pd.concat([response.rename("outcome"), design], axis=1).dropna()
    exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for term in design.columns:
        exog[term] = data[term].astype(float)
    if len(data) <= exog.shape[1] or np.linalg.matrix_rank(exog.to_numpy()) != exog.shape[1]:
        raise AnalyticsError("Regression design is rank-deficient or has insufficient complete rows")
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if model == "linear":
                fitted = sm.OLS(data["outcome"].astype(float), exog).fit()
                converged = True
            else:
                fitted = sm.Logit(data["outcome"].astype(float), exog).fit(disp=False, maxiter=100)
                converged = bool(fitted.mle_retvals.get("converged", False))
        if not converged or any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise AnalyticsError("Regression model did not converge")
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Regression model could not fit this dataset") from exc
    if not np.isfinite(fitted.params.to_numpy()).all() or not np.isfinite(fitted.bse.to_numpy()).all():
        raise AnalyticsError("Regression model produced non-finite estimates")
    critical = float(stats.t.ppf(1 - alpha / 2, fitted.df_resid)) if model == "linear" else float(stats.norm.ppf(1 - alpha / 2))
    coefficients = []
    for term in exog.columns:
        estimate = float(fitted.params[term])
        standard_error = float(fitted.bse[term])
        result = {
            "term": term,
            "estimate": _round(estimate),
            "std_error": _round(standard_error),
            "statistic": _round(fitted.tvalues[term]),
            "p_value": _round(fitted.pvalues[term]),
            "ci_lower": _round(estimate - critical * standard_error),
            "ci_upper": _round(estimate + critical * standard_error),
        }
        if model == "logistic":
            result.update(
                {
                    "odds_ratio": _exp_round(estimate),
                    "odds_ratio_ci_lower": _exp_round(estimate - critical * standard_error),
                    "odds_ratio_ci_upper": _exp_round(estimate + critical * standard_error),
                }
            )
        coefficients.append(result)
    settings = {
        "analysis": "regression_with_categorical_predictors",
        "model": model,
        "outcome": outcome,
        "predictors": predictors,
        "categorical_predictors": categorical,
        "category_references": references,
        "interactions": normalized_interactions,
        "alpha": alpha,
    }
    result = {
        "method": "OLS regression with categorical predictors" if model == "linear" else "Binary logistic regression with categorical predictors",
        "n": int(len(data)),
        "categorical_encoding": encoding,
        "interactions": [{"left": left, "right": right} for left, right in normalized_interactions],
        "coefficients": coefficients,
        "aic": _round(fitted.aic),
        "bic": _round(fitted.bic),
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }
    if model == "linear":
        result.update({"r_squared": _round(fitted.rsquared), "adjusted_r_squared": _round(fitted.rsquared_adj), "residual_degrees_freedom": _round(fitted.df_resid)})
    else:
        result.update({"outcome": outcome_description, "mcfadden_pseudo_r_squared": _round(1 - fitted.llf / fitted.llnull)})
    return result


def _treatment_coded_design(
    frame: pd.DataFrame,
    predictors: list[str],
    categorical_predictors: list[str] | None,
    category_references: dict[str, str] | None,
    interactions: list[tuple[str, str]] | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[tuple[str, str]]]:
    """Build a bounded treatment-coded design matrix without converting missing values to a level."""
    categorical = categorical_predictors or []
    references = category_references or {}
    if len(set(categorical)) != len(categorical) or not set(categorical).issubset(predictors):
        raise AnalyticsError("categorical_predictors must be a unique subset of predictors")
    if not set(references).issubset(categorical):
        raise AnalyticsError("category_references may only name categorical_predictors")
    normalized_interactions: list[tuple[str, str]] = []
    for pair in interactions or []:
        if len(pair) != 2 or pair[0] == pair[1] or pair[0] not in predictors or pair[1] not in predictors:
            raise AnalyticsError("Each interaction must contain two distinct predictor names")
        normalized = tuple(sorted(pair))
        if normalized not in normalized_interactions:
            normalized_interactions.append(normalized)
    numeric = _numeric_columns(frame)
    design = pd.DataFrame(index=frame.index)
    feature_groups: dict[str, list[str]] = {}
    encoding: list[dict[str, Any]] = []
    for predictor in predictors:
        if predictor not in frame.columns:
            raise AnalyticsError(f"Column '{predictor}' was not found")
        if predictor not in categorical:
            if predictor not in numeric:
                raise AnalyticsError(f"Numeric predictor '{predictor}' must contain only numeric observed values")
            design[predictor] = numeric[predictor]
            feature_groups[predictor] = [predictor]
            continue
        source = frame[predictor].astype("string")
        levels = sorted(source.dropna().unique().tolist())
        if not 2 <= len(levels) <= 30:
            raise AnalyticsError(f"Categorical predictor '{predictor}' must have between two and thirty observed levels")
        reference = references.get(predictor, levels[0])
        if reference not in levels:
            raise AnalyticsError(f"Reference '{reference}' was not observed in categorical predictor '{predictor}'")
        terms = []
        for level in levels:
            if level != reference:
                term = f"{predictor}[{level}]"
                design[term] = np.where(source.isna(), np.nan, (source == level).astype(float))
                terms.append(term)
        feature_groups[predictor] = terms
        encoding.append({"predictor": predictor, "reference": reference, "levels": levels, "terms": terms})
    for left, right in normalized_interactions:
        for left_term in feature_groups[left]:
            for right_term in feature_groups[right]:
                term = f"{left_term} × {right_term}"
                design[term] = design[left_term] * design[right_term]
    if design.shape[1] > 100:
        raise AnalyticsError("Categorical encoding and interactions may produce at most 100 model features")
    return design, encoding, normalized_interactions


def count_regression_with_categorical_predictors(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    distribution: str = "poisson",
    exposure_column: str | None = None,
    categorical_predictors: list[str] | None = None,
    category_references: dict[str, str] | None = None,
    interactions: list[tuple[str, str]] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit Poisson or NB2 count regression with treatment-coded factors and interactions.

    Args:
        records: JSON-like count-outcome rows.
        outcome: Non-negative integer count column.
        predictors: Numeric or categorical fixed predictors.
        distribution: Either poisson or negative_binomial for an NB2 variance model.
        exposure_column: Optional strictly positive exposure offset.
        categorical_predictors: Predictor subset to treatment-code.
        category_references: Optional factor-to-reference-level mapping.
        interactions: Selected predictor pairs to interact after treatment coding.
        alpha: Confidence interval complement.

    Returns:
        Rate-ratio inference, categorical encoding, fit diagnostics, and fingerprint metadata.

    Raises:
        AnalyticsError: If factor levels, outcome, design, or model fitting is invalid.

    Examples:
        >>> count_regression_with_categorical_predictors([{"y": 1, "x": 0, "g": "a"}, {"y": 2, "x": 1, "g": "b"}, {"y": 4, "x": 2, "g": "a"}, {"y": 5, "x": 3, "g": "b"}], "y", ["x", "g"], categorical_predictors=["g"])["distribution"]
        'poisson'
    """
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if distribution not in {"poisson", "negative_binomial"}:
        raise AnalyticsError("distribution must be either 'poisson' or 'negative_binomial'")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    if exposure_column and exposure_column in {outcome, *predictors}:
        raise AnalyticsError("exposure_column must differ from outcome and predictors")
    frame = _frame(records)
    numeric = _numeric_columns(frame)
    if outcome not in numeric:
        raise AnalyticsError(f"Count regression requires numeric column: '{outcome}'")
    if exposure_column and exposure_column not in numeric:
        raise AnalyticsError(f"Count regression requires numeric column: '{exposure_column}'")
    observed_outcome = numeric[outcome].dropna()
    if (observed_outcome < 0).any() or not np.isclose(observed_outcome, np.round(observed_outcome)).all():
        raise AnalyticsError("Count regression outcome must contain non-negative integer counts")
    design, encoding, normalized_interactions = _treatment_coded_design(
        frame, predictors, categorical_predictors, category_references, interactions
    )
    source = pd.DataFrame({"outcome": numeric[outcome]})
    if exposure_column:
        source["exposure"] = numeric[exposure_column]
    data = pd.concat([source, design], axis=1).dropna()
    if len(data) <= design.shape[1] + 1 or data["outcome"].sum() <= 0:
        raise AnalyticsError("Count regression requires enough complete rows and at least one observed count")
    if exposure_column and (data["exposure"] <= 0).any():
        raise AnalyticsError("exposure_column must contain strictly positive values")
    exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for term in design.columns:
        exog[term] = data[term].astype(float)
    if np.linalg.matrix_rank(exog.to_numpy()) != exog.shape[1]:
        raise AnalyticsError("Count regression design is perfectly collinear")
    offset = np.log(data["exposure"].to_numpy()) if exposure_column else None
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if distribution == "poisson":
                fitted = sm.GLM(data["outcome"].to_numpy(), exog, family=sm.families.Poisson(), offset=offset).fit()
                converged = bool(fitted.converged)
            else:
                fitted = sm.NegativeBinomial(data["outcome"].to_numpy(), exog, loglike_method="nb2", offset=offset).fit(disp=False)
                converged = bool(fitted.mle_retvals.get("converged", False))
        if not converged or any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise AnalyticsError("Count regression did not converge")
    except AnalyticsError:
        raise
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Count regression could not fit this dataset") from exc
    params, standard_errors, p_values = np.asarray(fitted.params, dtype=float), np.asarray(fitted.bse, dtype=float), np.asarray(fitted.pvalues, dtype=float)
    dispersion_alpha = None
    if distribution == "negative_binomial":
        dispersion_alpha = float(params[-1])
        params, standard_errors, p_values = params[:-1], standard_errors[:-1], p_values[:-1]
        if not math.isfinite(dispersion_alpha) or dispersion_alpha <= 0:
            raise AnalyticsError("Negative-binomial dispersion estimate was invalid")
    if not np.isfinite(params).all() or not np.isfinite(standard_errors).all():
        raise AnalyticsError("Count regression produced non-finite estimates")
    critical = float(stats.norm.ppf(1 - alpha / 2))
    coefficients = []
    for index, term in enumerate(exog.columns):
        estimate, standard_error = float(params[index]), float(standard_errors[index])
        coefficients.append(
            {
                "term": term,
                "log_rate_ratio": _round(estimate),
                "std_error": _round(standard_error),
                "z_value": _round(estimate / standard_error),
                "p_value": _round(p_values[index]),
                "rate_ratio": _exp_round(estimate),
                "rate_ratio_ci_lower": _exp_round(estimate - critical * standard_error),
                "rate_ratio_ci_upper": _exp_round(estimate + critical * standard_error),
            }
        )
    predicted = np.asarray(fitted.predict(), dtype=float)
    variance = predicted if distribution == "poisson" else predicted + float(dispersion_alpha) * predicted**2
    pearson_chi_square = float(np.sum((data["outcome"].to_numpy() - predicted) ** 2 / variance))
    settings = {
        "analysis": "count_regression_with_categorical_predictors",
        "outcome": outcome,
        "predictors": predictors,
        "distribution": distribution,
        "exposure_column": exposure_column,
        "categorical_predictors": categorical_predictors or [],
        "category_references": category_references or {},
        "interactions": normalized_interactions,
        "alpha": alpha,
    }
    return {
        "method": "Poisson count regression with categorical predictors" if distribution == "poisson" else "Negative-binomial (NB2) count regression with categorical predictors",
        "n": int(len(data)),
        "total_count": int(data["outcome"].sum()),
        "exposure_column": exposure_column,
        "categorical_encoding": encoding,
        "interactions": [{"left": left, "right": right} for left, right in normalized_interactions],
        "coefficients": coefficients,
        "aic": _round(fitted.aic),
        "bic": _round(fitted.bic_llf if hasattr(fitted, "bic_llf") else fitted.bic),
        "pearson_chi_square": _round(pearson_chi_square),
        "overdispersion_ratio": _round(pearson_chi_square / float(fitted.df_resid)),
        "dispersion_alpha": _round(dispersion_alpha),
        "warning": "This model does not diagnose zero inflation, dependence, or a complex-survey design.",
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def run_statistical_test(
    records: list[dict[str, Any]],
    test: str,
    outcome: str,
    group: str | None = None,
    predictors: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run a validated classical inference procedure on in-memory tabular data.

    Args:
        records: JSON-like rows containing fields required by the selected test.
        test: Supported test identifier such as pearson_correlation, t_test, anova, or ols.
        outcome: Outcome or first analysis column, depending on the procedure.
        group: Optional grouping column or second analysis column.
        predictors: Optional unique numeric OLS predictor columns.
        alpha: Confidence interval complement where a model supplies intervals.

    Returns:
        Test-specific statistic, p-value, sample-size details, and reproducibility metadata.

    Raises:
        AnalyticsError: If the test, columns, assumptions, or data shape are unsupported.

    Examples:
        >>> run_statistical_test([{"x": 1, "y": 2}, {"x": 2, "y": 4}, {"x": 3, "y": 6}], "pearson_correlation", "x", "y")["test"]
        'pearson_correlation'
    """
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
    elif test == "logistic_regression":
        predictor_names = predictors or []
        if not predictor_names:
            raise AnalyticsError("predictors must include one or more numeric columns for logistic_regression")
        if outcome not in frame.columns:
            raise AnalyticsError(f"Column '{outcome}' was not found")
        if len(set(predictor_names)) != len(predictor_names) or outcome in predictor_names:
            raise AnalyticsError("logistic regression predictors must be unique and cannot include outcome")
        predictor_series = [_numeric_column(frame, name) for name in predictor_names]
        data = pd.concat([frame[outcome], *predictor_series], axis=1).dropna()
        labels = sorted(data.iloc[:, 0].astype(str).unique().tolist())
        if len(labels) != 2:
            raise AnalyticsError("Logistic regression requires exactly two observed outcome categories")
        parameter_count = len(predictor_names) + 1
        if len(data) <= parameter_count:
            raise AnalyticsError("Logistic regression needs more complete rows than fitted parameters")
        y = (data.iloc[:, 0].astype(str) == labels[1]).astype(int).to_numpy()
        matrix = sm.add_constant(data.iloc[:, 1:].astype(float).to_numpy(), has_constant="add")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                fitted = sm.Logit(y, matrix).fit(disp=False, maxiter=100)
        except (PerfectSeparationError, np.linalg.LinAlgError, Warning) as exc:
            raise AnalyticsError("Logistic regression could not converge; check predictors for complete or quasi-complete separation") from exc
        converged = bool(fitted.mle_retvals.get("converged", False))
        if not converged or np.any(np.abs(fitted.params) > 25) or not np.all(np.isfinite(fitted.bse)):
            raise AnalyticsError("Logistic regression did not converge reliably; check for complete or quasi-complete separation")
        confidence = fitted.conf_int(alpha=alpha)
        names = ["intercept", *predictor_names]
        coefficients = []
        for index, name in enumerate(names):
            estimate = float(fitted.params[index])
            lower, upper = float(confidence[index, 0]), float(confidence[index, 1])
            coefficients.append(
                {
                    "term": name,
                    "estimate_log_odds": _round(estimate),
                    "std_error": _round(fitted.bse[index]),
                    "z": _round(fitted.tvalues[index]),
                    "p_value": _round(fitted.pvalues[index]),
                    "ci_lower_log_odds": _round(lower),
                    "ci_upper_log_odds": _round(upper),
                    "odds_ratio": _exp_round(estimate),
                    "odds_ratio_ci_lower": _exp_round(lower),
                    "odds_ratio_ci_upper": _exp_round(upper),
                }
            )
        result = {
            "method": "Binary logistic regression (maximum likelihood)",
            "n": int(len(data)),
            "outcome": {"reference": labels[0], "event": labels[1]},
            "log_likelihood": _round(fitted.llf),
            "aic": _round(fitted.aic),
            "bic": _round(fitted.bic),
            "mcfadden_pseudo_r_squared": _round(fitted.prsquared),
            "coefficients": coefficients,
        }
    else:
        raise AnalyticsError("Unsupported test. Use pearson_correlation, welch_t_test, chi_square, ols_regression, or logistic_regression")
    result.update(
        {
            "alpha": alpha,
            "reject_null": bool((result.get("p_value") or 1) < alpha) if "p_value" in result else None,
            "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
        }
    )
    return result
