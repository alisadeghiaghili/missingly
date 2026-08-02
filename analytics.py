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
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


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


def simple_imputation(
    records: list[dict[str, Any]],
    method: str,
    columns: list[str] | None = None,
    constant: Any | None = None,
) -> dict[str, Any]:
    """Apply a transparent single-value imputation method and report each fill."""
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
    """Impute numeric columns with KNN or deterministic chained regression.

    This is single imputation. It intentionally does not represent Rubin-pooled
    multiple imputation or attach inferential certainty to imputed values.
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
    """Export validated records with predictable LF line endings for reproducible sharing."""
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
    """Run posterior-draw iterative imputation and pool an OLS model with Rubin's rules."""
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
    """Estimate Kaplan-Meier curves and a two-group log-rank test when applicable."""
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


def linear_mixed_effects(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    group_column: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit a Gaussian random-intercept model for clustered continuous outcomes."""
    if not 0 < alpha < 1:
        raise AnalyticsError("alpha must be between 0 and 1")
    if not predictors or len(set(predictors)) != len(predictors) or outcome in predictors:
        raise AnalyticsError("predictors must be a non-empty unique list that does not contain outcome")
    if group_column in {outcome, *predictors}:
        raise AnalyticsError("group_column must differ from outcome and predictors")
    frame = _frame(records)
    if group_column not in frame.columns:
        raise AnalyticsError(f"Column '{group_column}' was not found")
    numeric = _numeric_columns(frame)
    for column in [outcome, *predictors]:
        if column not in numeric:
            raise AnalyticsError(f"Linear mixed-effects requires numeric column: '{column}'")
    data = pd.DataFrame({"outcome": numeric[outcome], "group": frame[group_column].astype("string")})
    for index, column in enumerate(predictors):
        data[f"predictor_{index}"] = numeric[column]
    data = data.dropna()
    group_sizes = data["group"].value_counts()
    if len(data) <= len(predictors) + 2 or len(group_sizes) < 2:
        raise AnalyticsError("Linear mixed-effects requires more complete rows and at least two groups")
    if int(group_sizes.min()) < 2:
        raise AnalyticsError("Each observed group must contain at least two complete rows")
    exog = pd.DataFrame({"intercept": 1.0}, index=data.index)
    for index, column in enumerate(predictors):
        exog[column] = data[f"predictor_{index}"].astype(float)
    if np.linalg.matrix_rank(exog.to_numpy()) != exog.shape[1]:
        raise AnalyticsError("Linear mixed-effects predictors are perfectly collinear")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = sm.MixedLM(data["outcome"].astype(float), exog, groups=data["group"]).fit(
                reml=True, method="lbfgs", disp=False
            )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        raise AnalyticsError("Linear mixed-effects model could not fit this dataset") from exc
    random_variance = float(fitted.cov_re.iloc[0, 0])
    residual_variance = float(fitted.scale)
    if (
        not fitted.converged
        or not math.isfinite(random_variance)
        or random_variance <= 0
        or not math.isfinite(residual_variance)
        or residual_variance <= 0
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
        "alpha": alpha,
    }
    return {
        "method": "Gaussian linear mixed-effects model (random intercept)",
        "n": int(len(data)),
        "groups": int(len(group_sizes)),
        "minimum_group_size": int(group_sizes.min()),
        "estimation": "REML",
        "coefficients": coefficients,
        "random_intercept_variance": _round(random_variance),
        "residual_variance": _round(residual_variance),
        "intraclass_correlation": _round(random_variance / (random_variance + residual_variance)),
        "reproducibility": {"engine": ENGINE_VERSION, "input_sha256": _fingerprint(records, settings)},
    }


def weighted_ols(
    records: list[dict[str, Any]],
    outcome: str,
    predictors: list[str],
    weight_column: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit weighted least squares for strictly positive analytic weights."""
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
