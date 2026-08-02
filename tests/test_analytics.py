import importlib
import io
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from analytics import advanced_numeric_imputation, count_regression, count_regression_with_categorical_predictors, cox_proportional_hazards, hurdle_poisson_regression, kaplan_meier_analysis, linear_mixed_effects, missingness_report, multiple_imputation_ols, multinomial_logistic_regression, ordinal_logistic_regression, profile_dataset, records_to_csv, regression_with_categorical_predictors, run_statistical_test, simple_imputation, weighted_ols, zero_inflated_count_regression
from ingestion import import_tabular_bytes
from reporting import build_descriptive_report


RECORDS = [
    {"group": "control", "score": 10, "age": 20, "outcome": "no"},
    {"group": "control", "score": 12, "age": 21, "outcome": "no"},
    {"group": "control", "score": 11, "age": 22, "outcome": "yes"},
    {"group": "treatment", "score": 20, "age": 30, "outcome": "yes"},
    {"group": "treatment", "score": 22, "age": 31, "outcome": "yes"},
    {"group": "treatment", "score": None, "age": 32, "outcome": "yes"},
]


def _cox_records():
    generator = np.random.default_rng(19)
    predictor = generator.normal(size=80)
    event_time = generator.exponential(scale=np.exp(-0.65 * predictor))
    censor_time = generator.exponential(scale=1.4, size=80)
    return [
        {
            "time": float(min(event, censor)),
            "event": "event" if event <= censor else "censor",
            "x": float(value),
            "site": f"site-{index % 4}",
            "cluster": f"cluster-{index % 10}",
        }
        for index, (value, event, censor) in enumerate(zip(predictor, event_time, censor_time))
    ]


def _count_records():
    generator = np.random.default_rng(23)
    predictor = generator.normal(size=120)
    exposure = generator.uniform(0.5, 2.5, size=120)
    mean = exposure * np.exp(0.3 + 0.45 * predictor)
    counts = generator.negative_binomial(4, 4 / (4 + mean))
    return [
        {"count": int(count), "x": float(value), "person_time": float(duration)}
        for count, value, duration in zip(counts, predictor, exposure)
    ]


def _ordinal_records():
    generator = np.random.default_rng(29)
    predictor = generator.normal(size=120)
    latent = 0.9 * predictor + generator.logistic(size=120)
    outcomes = np.where(latent < -0.7, "low", np.where(latent < 0.8, "medium", "high"))
    return [{"severity": str(outcome), "x": float(value)} for outcome, value in zip(outcomes, predictor)]


def _multinomial_records():
    generator = np.random.default_rng(41)
    predictor = generator.normal(size=200)
    logits = np.column_stack([np.zeros(200), -0.2 + 0.6 * predictor, 0.1 - 0.5 * predictor])
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    category_codes = [generator.choice(3, p=probability) for probability in probabilities]
    labels = np.array(["baseline", "positive", "negative"])
    return [{"class": str(labels[code]), "x": float(value)} for code, value in zip(category_codes, predictor)]


def _categorical_regression_records():
    generator = np.random.default_rng(61)
    records = []
    for index in range(180):
        treatment = index % 2
        age = float(25 + (index % 40))
        linear_score = 5 + 0.5 * age + 2 * treatment + 0.12 * age * treatment + generator.normal(0, 0.4)
        probability = 1 / (1 + np.exp(-(-5 + 0.15 * age + 0.7 * treatment)))
        records.append(
            {
                "group": "treatment" if treatment else "control",
                "age": age,
                "score": float(linear_score),
                "event": "yes" if generator.random() < probability else "no",
            }
        )
    return records


def _zero_inflated_records():
    generator = np.random.default_rng(72)
    predictor = generator.normal(size=500)
    zero_driver = generator.normal(size=500)
    exposure = generator.uniform(0.7, 2.0, size=500)
    structural_zero = generator.random(500) < (1 / (1 + np.exp(-(-1.5 + 0.4 * zero_driver))))
    mean = exposure * np.exp(0.3 + 0.35 * predictor)
    counts = generator.negative_binomial(1.5, 1.5 / (1.5 + mean))
    counts[structural_zero] = 0
    return [
        {"count": int(count), "x": float(value), "zero_driver": float(driver), "exposure": float(duration)}
        for count, value, driver, duration in zip(counts, predictor, zero_driver, exposure)
    ]


def _hurdle_records():
    generator = np.random.default_rng(83)
    predictor = generator.normal(size=400)
    hurdle_driver = generator.normal(size=400)
    exposure = generator.uniform(0.7, 2.0, size=400)
    has_positive_count = generator.random(400) < (1 / (1 + np.exp(-(-0.6 + 0.6 * hurdle_driver))))
    mean = exposure * np.exp(0.3 + 0.4 * predictor)
    positive_counts = generator.poisson(mean)
    while (positive_counts == 0).any():
        mask = positive_counts == 0
        positive_counts[mask] = generator.poisson(mean[mask])
    counts = np.where(has_positive_count, positive_counts, 0)
    return [
        {"count": int(count), "x": float(value), "hurdle_driver": float(driver), "exposure": float(duration)}
        for count, value, driver, duration in zip(counts, predictor, hurdle_driver, exposure)
    ]


def _categorical_count_records():
    generator = np.random.default_rng(95)
    predictor = generator.normal(size=300)
    exposure = generator.uniform(0.7, 2.0, size=300)
    treatment = np.arange(300) % 2
    mean = exposure * np.exp(0.2 + 0.25 * predictor + 0.4 * treatment + 0.08 * predictor * treatment)
    counts = generator.negative_binomial(4, 4 / (4 + mean))
    return [
        {"count": int(count), "x": float(value), "group": "treatment" if group else "control", "exposure": float(duration)}
        for count, value, group, duration in zip(counts, predictor, treatment, exposure)
    ]


def _categorical_cox_records():
    generator = np.random.default_rng(101)
    predictor = generator.normal(size=300)
    treatment = np.arange(300) % 2
    event_time = generator.exponential(scale=np.exp(-(0.4 * predictor + 0.5 * treatment)))
    censor_time = generator.exponential(scale=1.6, size=300)
    return [
        {
            "time": float(min(event, censor)),
            "event": "event" if event <= censor else "censor",
            "x": float(value),
            "group": "treatment" if group else "control",
        }
        for value, group, event, censor in zip(predictor, treatment, event_time, censor_time)
    ]


def test_profile_and_missingness_are_deterministic_and_do_not_claim_mcar():
    profile = profile_dataset(RECORDS)
    assert profile["dataset"]["rows"] == 6
    assert profile["dataset"]["missing_cells"] == 1
    score = next(column for column in profile["columns"] if column["name"] == "score")
    assert score["kind"] == "numeric"
    assert score["mean"] == 15.0
    assert profile["reproducibility"]["input_sha256"] == profile_dataset(RECORDS)["reproducibility"]["input_sha256"]

    missingness = missingness_report(RECORDS)
    score_missingness = next(item for item in missingness["column_missingness"] if item["column"] == "score")
    assert score_missingness == {"column": "score", "missing": 1, "missing_rate": pytest.approx(1 / 6)}
    assert all("MCAR" not in warning or "not evidence" in warning for warning in missingness["warnings"])


def test_statistical_test_contracts_match_known_direction():
    correlation = run_statistical_test(RECORDS, "pearson_correlation", "score", "age")
    assert correlation["method"].startswith("Pearson")
    assert correlation["statistic"] == pytest.approx(0.98964006)

    welch = run_statistical_test(RECORDS, "welch_t_test", "score", "group")
    assert welch["groups"][1]["mean"] > welch["groups"][0]["mean"]
    assert welch["statistic"] < 0

    chi_square = run_statistical_test(RECORDS, "chi_square", "outcome", "group")
    assert chi_square["degrees_freedom"] == 1
    assert chi_square["n"] == 6

    regression_rows = [{"y": 2 + 3 * x + (0.2 if x % 2 else -0.2), "x": x} for x in range(1, 10)]
    regression = run_statistical_test(regression_rows, "ols_regression", "y", predictors=["x"])
    slope = next(item for item in regression["coefficients"] if item["term"] == "x")
    assert slope["estimate"] == pytest.approx(3, abs=0.1)
    assert regression["r_squared"] > 0.99

    logistic_rows = []
    for x in range(6):
        logistic_rows.extend({"x": float(x), "event": "yes"} for _ in range(x + 1))
        logistic_rows.extend({"x": float(x), "event": "no"} for _ in range(6 - (x + 1)))
    logistic = run_statistical_test(logistic_rows, "logistic_regression", "event", predictors=["x"])
    slope = next(item for item in logistic["coefficients"] if item["term"] == "x")
    assert logistic["outcome"] == {"reference": "no", "event": "yes"}
    assert slope["odds_ratio"] > 1
    assert logistic["n"] == len(logistic_rows)


def test_tabular_import_and_simple_imputation_contracts():
    csv_result = import_tabular_bytes("study.csv", b"group,score\ncontrol,10\ntreatment,\n")
    assert csv_result["format"] == "csv"
    assert csv_result["rows"] == 2
    assert csv_result["records"][1]["score"] is None

    spreadsheet = io.BytesIO()
    pd.DataFrame([{"group": "control", "score": 10}, {"group": "treatment", "score": 20}]).to_excel(spreadsheet, index=False)
    xlsx_result = import_tabular_bytes("study.xlsx", spreadsheet.getvalue())
    assert xlsx_result["columns"] == ["group", "score"]

    imputed = simple_imputation(csv_result["records"], "mean", ["score"])
    assert imputed["imputations"] == [{"column": "score", "method": "mean", "imputed_cells": 1, "fill_value": 10.0}]
    assert imputed["records"][1]["score"] == 10.0
    categorical = simple_imputation([{"label": "yes"}, {"label": None}, {"label": "yes"}], "mode")
    assert categorical["records"][1]["label"] == "yes"


def test_advanced_numeric_imputation_is_reproducible_and_preserves_observed_values():
    records = [
        {"x": 1.0, "y": 2.0},
        {"x": 2.0, "y": 4.1},
        {"x": 3.0, "y": None},
        {"x": 4.0, "y": 8.1},
        {"x": None, "y": 10.0},
    ]
    knn = advanced_numeric_imputation(records, "knn", ["x", "y"], n_neighbors=2)
    assert knn["records"][0] == records[0]
    assert all(item["x"] is not None and item["y"] is not None for item in knn["records"])
    first_iterative = advanced_numeric_imputation(records, "iterative", ["x", "y"], max_iter=20, random_state=19)
    second_iterative = advanced_numeric_imputation(records, "iterative", ["x", "y"], max_iter=20, random_state=19)
    assert first_iterative["records"] == second_iterative["records"]
    assert records_to_csv(first_iterative["records"]).startswith("x,y\n")


def test_multiple_imputation_ols_uses_rubin_pooling_and_is_reproducible():
    records = [
        {"x": float(x) if x not in {3, 9} else None, "y": (2 + 3 * x + (0.25 if x % 2 else -0.25)) if x != 6 else None}
        for x in range(1, 15)
    ]
    first = multiple_imputation_ols(records, "y", ["x"], ["x", "y"], m=4, max_iter=15, random_state=11)
    second = multiple_imputation_ols(records, "y", ["x"], ["x", "y"], m=4, max_iter=15, random_state=11)
    assert first["method"].endswith("Rubin-pooled OLS")
    assert first["sample_size_per_imputation"] == [14, 14, 14, 14]
    assert first["imputed_cells_per_dataset"] == {"x": 2, "y": 1}
    slope = next(item for item in first["coefficients"] if item["term"] == "x")
    assert slope["estimate"] == pytest.approx(3, abs=0.5)
    assert slope["total_variance"] >= slope["within_variance"]
    assert first["reproducibility"] == second["reproducibility"]


def test_kaplan_meier_log_rank_and_escaped_html_report():
    records = [
        {"time": 1, "event": "event", "group": "control", "score": 10},
        {"time": 2, "event": "event", "group": "control", "score": 11},
        {"time": 3, "event": "censor", "group": "control", "score": 12},
        {"time": 1, "event": "censor", "group": "treatment", "score": 18},
        {"time": 2, "event": "event", "group": "treatment", "score": 19},
        {"time": 4, "event": "event", "group": "treatment", "score": 20},
    ]
    survival = kaplan_meier_analysis(records, "time", "event", "group")
    assert survival["event"] == {"censor_or_reference": "censor", "event": "event"}
    assert len(survival["curves"]) == 2
    assert survival["log_rank"]["degrees_freedom"] == 1
    report = build_descriptive_report(records, "<img src=x onerror=alert(1)>")
    assert "<img src=x" not in report
    assert "&lt;img src=x" in report
    assert "Input SHA-256" in report
    assert "Missing-data patterns" in report


def test_random_intercept_mixed_model_and_analytic_weighted_ols_contracts():
    clustered_records = []
    for group, effect in enumerate([-2.0, -1.0, 0.5, 1.5, 2.5, 3.0]):
        for observation in range(6):
            clustered_records.append(
                {
                    "patient": f"p-{group}",
                    "x": float(observation),
                    "y": 10 + 1.75 * observation + effect + (0.12 if observation % 2 else -0.12),
                }
            )
    mixed = linear_mixed_effects(clustered_records, "y", ["x"], "patient")
    slope = next(item for item in mixed["coefficients"] if item["term"] == "x")
    assert mixed["groups"] == 6
    assert mixed["estimation"] == "REML"
    assert slope["estimate"] == pytest.approx(1.75, abs=0.05)
    assert mixed["intraclass_correlation"] > 0

    weighted_records = [
        {"x": float(x), "y": 3 + 2.5 * x + (0.4 if x % 3 else -0.4), "weight": float((x % 4) + 1)}
        for x in range(1, 16)
    ]
    weighted = weighted_ols(weighted_records, "y", ["x"], "weight")
    weighted_slope = next(item for item in weighted["coefficients"] if item["term"] == "x")
    assert weighted["method"].startswith("Weighted least squares")
    assert weighted["sum_weights"] == sum(row["weight"] for row in weighted_records)
    assert weighted_slope["estimate"] == pytest.approx(2.5, abs=0.15)
    assert "not a complex-survey estimator" in weighted["warning"]


def test_cox_proportional_hazards_supports_strata_and_clustered_standard_errors():
    records = _cox_records()
    stratified = cox_proportional_hazards(records, "time", "event", ["x"], strata_column="site")
    coefficient = stratified["coefficients"][0]
    assert stratified["method"] == "Cox proportional hazards regression"
    assert stratified["events"] == 48
    assert stratified["strata"] == 4
    assert stratified["variance_estimator"] == "model-based"
    assert coefficient["hazard_ratio"] > 1
    assert stratified["likelihood_ratio_p_value"] < 0.05

    robust = cox_proportional_hazards(records, "time", "event", ["x"], cluster_column="cluster", ties="breslow")
    assert robust["ties"] == "breslow"
    assert robust["variance_estimator"] == "cluster-robust"
    assert robust["coefficients"][0]["std_error"] > 0


def test_poisson_and_negative_binomial_count_regression_with_exposure():
    records = _count_records()
    poisson = count_regression(records, "count", ["x"], "poisson", "person_time")
    poisson_slope = next(item for item in poisson["coefficients"] if item["term"] == "x")
    assert poisson["method"] == "Poisson count regression"
    assert poisson["exposure_column"] == "person_time"
    assert poisson_slope["rate_ratio"] > 1
    assert poisson["overdispersion_ratio"] > 0

    negative_binomial = count_regression(records, "count", ["x"], "negative_binomial", "person_time")
    nb_slope = next(item for item in negative_binomial["coefficients"] if item["term"] == "x")
    assert negative_binomial["method"].startswith("Negative-binomial")
    assert negative_binomial["dispersion_alpha"] > 0
    assert nb_slope["rate_ratio"] == pytest.approx(np.exp(0.45), rel=0.35)


def test_ordinal_logistic_requires_explicit_category_order_and_reports_common_odds():
    ordinal = ordinal_logistic_regression(_ordinal_records(), "severity", ["x"], ["low", "medium", "high"])
    coefficient = ordinal["coefficients"][0]
    assert ordinal["method"].startswith("Ordinal logistic")
    assert ordinal["categories"] == ["low", "medium", "high"]
    assert coefficient["cumulative_odds_ratio"] > 1
    assert len(ordinal["thresholds"]) == 2
    assert "proportional-odds assumption" in ordinal["warning"]


def test_multinomial_logistic_uses_explicit_reference_category():
    multinomial = multinomial_logistic_regression(_multinomial_records(), "class", ["x"], "baseline")
    coefficients = {item["category"]: item["coefficients"] for item in multinomial["coefficient_sets"]}
    positive_slope = next(item for item in coefficients["positive"] if item["term"] == "x")
    negative_slope = next(item for item in coefficients["negative"] if item["term"] == "x")
    assert multinomial["method"].startswith("Multinomial logistic")
    assert multinomial["reference_category"] == "baseline"
    assert positive_slope["relative_risk_ratio"] > 1
    assert negative_slope["relative_risk_ratio"] < 1
    assert multinomial["likelihood_ratio_p_value"] < 0.001


def test_categorical_regression_uses_explicit_reference_and_pairwise_interaction():
    records = _categorical_regression_records()
    linear = regression_with_categorical_predictors(
        records, "linear", "score", ["age", "group"], ["group"], {"group": "control"}, [("age", "group")]
    )
    terms = {item["term"] for item in linear["coefficients"]}
    assert linear["categorical_encoding"] == [
        {"predictor": "group", "reference": "control", "levels": ["control", "treatment"], "terms": ["group[treatment]"]}
    ]
    assert "group[treatment]" in terms
    assert "age × group[treatment]" in terms
    assert linear["r_squared"] > 0.99

    logistic = regression_with_categorical_predictors(
        records, "logistic", "event", ["age", "group"], ["group"], {"group": "control"}
    )
    group_effect = next(item for item in logistic["coefficients"] if item["term"] == "group[treatment]")
    assert logistic["outcome"] == {"reference": "no", "event": "yes"}
    assert group_effect["odds_ratio"] > 1


def test_zero_inflated_poisson_and_zinb_separate_count_and_structural_zero_processes():
    records = _zero_inflated_records()
    zip_result = zero_inflated_count_regression(records, "count", ["x"], "poisson", "exposure", ["zero_driver"])
    zip_count_slope = next(item for item in zip_result["count_coefficients"] if item["term"] == "x")
    zip_inflation_slope = next(item for item in zip_result["inflation_coefficients"] if item["term"] == "zero_driver")
    assert zip_result["method"] == "Zero-inflated Poisson regression"
    assert zip_result["zero_rate"] > 0.3
    assert zip_count_slope["rate_ratio"] > 1
    assert zip_inflation_slope["structural_zero_odds_ratio"] > 1

    zinb_result = zero_inflated_count_regression(records, "count", ["x"], "negative_binomial", "exposure", ["zero_driver"])
    assert zinb_result["method"].startswith("Zero-inflated negative-binomial")
    assert zinb_result["dispersion_alpha"] > 0


def test_hurdle_poisson_models_positive_count_and_positive_size_separately():
    hurdle = hurdle_poisson_regression(_hurdle_records(), "count", ["x"], "exposure", ["hurdle_driver"])
    count_slope = next(item for item in hurdle["count_coefficients"] if item["term"] == "x")
    hurdle_slope = next(item for item in hurdle["hurdle_coefficients"] if item["term"] == "hurdle_driver")
    assert hurdle["method"].startswith("Hurdle Poisson")
    assert hurdle["zero_count"] > 0
    assert hurdle["positive_count_n"] > 0
    assert count_slope["rate_ratio"] > 1
    assert hurdle_slope["positive_count_odds_ratio"] > 1


def test_count_regression_supports_categorical_predictors_and_interactions():
    records = _categorical_count_records()
    poisson = count_regression_with_categorical_predictors(
        records, "count", ["x", "group"], "poisson", "exposure", ["group"], {"group": "control"}, [("x", "group")]
    )
    poisson_terms = {item["term"] for item in poisson["coefficients"]}
    assert poisson["categorical_encoding"][0]["reference"] == "control"
    assert "group[treatment]" in poisson_terms
    assert "group[treatment] × x" in poisson_terms

    negative_binomial = count_regression_with_categorical_predictors(
        records, "count", ["x", "group"], "negative_binomial", "exposure", ["group"], {"group": "control"}
    )
    group_effect = next(item for item in negative_binomial["coefficients"] if item["term"] == "group[treatment]")
    assert negative_binomial["dispersion_alpha"] > 0
    assert group_effect["rate_ratio"] > 1


def test_cox_regression_supports_categorical_predictors_and_interactions():
    cox = cox_proportional_hazards(
        _categorical_cox_records(),
        "time",
        "event",
        ["x", "group"],
        categorical_predictors=["group"],
        category_references={"group": "control"},
        interactions=[("x", "group")],
    )
    terms = {item["term"]: item for item in cox["coefficients"]}
    assert cox["categorical_encoding"][0]["reference"] == "control"
    assert terms["group[treatment]"]["hazard_ratio"] > 1
    assert "group[treatment] × x" in terms


@pytest.fixture
def analytics_client(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-that-is-long-enough-for-security-checks")
    monkeypatch.setenv("DATABASE_NAME", str(tmp_path / "analytics.db"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    module.create_tables()
    module.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        ("Analyst", "analyst@example.com", module.hash_password("safe-password")),
    )
    user = module.fetch_one("SELECT * FROM users WHERE email = ?", ("analyst@example.com",))
    return TestClient(module.app), {"Authorization": f"Bearer {module.create_auth_session(user['id'])}"}


def test_analysis_endpoints_require_auth_and_return_results(analytics_client):
    client, headers = analytics_client
    assert client.post("/analysis/profile", json={"records": RECORDS}).status_code == 401
    profile = client.post("/analysis/profile", headers=headers, json={"records": RECORDS})
    assert profile.status_code == 200
    assert profile.json()["dataset"]["rows"] == 6
    invalid = client.post(
        "/analysis/tests",
        headers=headers,
        json={"records": RECORDS, "test": "ols_regression", "outcome": "score", "predictors": []},
    )
    assert invalid.status_code == 422


def test_analysis_file_import_and_imputation_endpoints(analytics_client):
    client, headers = analytics_client
    imported = client.post(
        "/analysis/import",
        headers=headers,
        files={"file": ("scores.csv", b"group,score\ncontrol,10\ntreatment,\n", "text/csv")},
    )
    assert imported.status_code == 200
    assert imported.json()["format"] == "csv"
    imputed = client.post(
        "/analysis/impute",
        headers=headers,
        json={"records": imported.json()["records"], "method": "mean", "columns": ["score"]},
    )
    assert imputed.status_code == 200
    assert imputed.json()["records"][1]["score"] == 10.0

    exported = client.post("/analysis/export.csv", headers=headers, json={"records": imputed.json()["records"]})
    assert exported.status_code == 200
    assert exported.headers["content-disposition"].startswith("attachment;")
    assert "group,score" in exported.text

    mi_records = [
        {"x": float(x) if x != 3 else None, "y": float(2 + 3 * x + (0.2 if x % 2 else -0.2)) if x != 5 else None}
        for x in range(1, 11)
    ]
    pooled = client.post(
        "/analysis/multiple-imputation/ols",
        headers=headers,
        json={"records": mi_records, "outcome": "y", "predictors": ["x"], "impute_columns": ["x", "y"], "m": 3},
    )
    assert pooled.status_code == 200
    assert pooled.json()["m"] == 3

    logistic_rows = [
        {"x": float(x), "event": "yes" if (index + x) % 3 else "no"}
        for x in range(1, 8)
        for index in range(5)
    ]
    logistic = client.post(
        "/analysis/tests",
        headers=headers,
        json={"records": logistic_rows, "test": "logistic_regression", "outcome": "event", "predictors": ["x"]},
    )
    assert logistic.status_code == 200
    assert logistic.json()["method"].startswith("Binary logistic")

    survival_records = [
        {"time": 1, "event": "event", "group": "a"},
        {"time": 2, "event": "censor", "group": "a"},
        {"time": 1, "event": "censor", "group": "b"},
        {"time": 3, "event": "event", "group": "b"},
    ]
    survival = client.post(
        "/analysis/survival",
        headers=headers,
        json={"records": survival_records, "time_column": "time", "event_column": "event", "group_column": "group"},
    )
    assert survival.status_code == 200
    clustered_records = [
        {"patient": f"p-{group}", "x": float(observation), "y": 5 + 1.5 * observation + group}
        for group in range(4)
        for observation in range(5)
    ]
    mixed = client.post(
        "/analysis/mixed-linear",
        headers=headers,
        json={"records": clustered_records, "outcome": "y", "predictors": ["x"], "group_column": "patient"},
    )
    assert mixed.status_code == 200
    weighted = client.post(
        "/analysis/weighted-ols",
        headers=headers,
        json={
            "records": [{"x": float(x), "y": 4 + 2 * x + (0.1 if x % 2 else -0.1), "weight": float((x % 3) + 1)} for x in range(1, 12)],
            "outcome": "y",
            "predictors": ["x"],
            "weight_column": "weight",
        },
    )
    assert weighted.status_code == 200
    cox = client.post(
        "/analysis/cox",
        headers=headers,
        json={"records": _cox_records(), "time_column": "time", "event_column": "event", "predictors": ["x"], "strata_column": "site"},
    )
    assert cox.status_code == 200
    assert cox.json()["method"] == "Cox proportional hazards regression"
    count = client.post(
        "/analysis/count-regression",
        headers=headers,
        json={"records": _count_records(), "outcome": "count", "predictors": ["x"], "distribution": "negative_binomial", "exposure_column": "person_time"},
    )
    assert count.status_code == 200
    assert count.json()["method"].startswith("Negative-binomial")
    ordinal = client.post(
        "/analysis/ordinal-logistic",
        headers=headers,
        json={"records": _ordinal_records(), "outcome": "severity", "predictors": ["x"], "category_order": ["low", "medium", "high"]},
    )
    assert ordinal.status_code == 200
    assert ordinal.json()["categories"] == ["low", "medium", "high"]
    multinomial = client.post(
        "/analysis/multinomial-logistic",
        headers=headers,
        json={"records": _multinomial_records(), "outcome": "class", "predictors": ["x"], "reference_category": "baseline"},
    )
    assert multinomial.status_code == 200
    assert multinomial.json()["reference_category"] == "baseline"
    categorical_regression = client.post(
        "/analysis/regression",
        headers=headers,
        json={
            "records": _categorical_regression_records(),
            "model": "linear",
            "outcome": "score",
            "predictors": ["age", "group"],
            "categorical_predictors": ["group"],
            "category_references": {"group": "control"},
            "interactions": [["age", "group"]],
        },
    )
    assert categorical_regression.status_code == 200
    assert categorical_regression.json()["categorical_encoding"][0]["reference"] == "control"
    zero_inflated = client.post(
        "/analysis/zero-inflated-count",
        headers=headers,
        json={
            "records": _zero_inflated_records(),
            "outcome": "count",
            "predictors": ["x"],
            "distribution": "negative_binomial",
            "exposure_column": "exposure",
            "inflation_predictors": ["zero_driver"],
        },
    )
    assert zero_inflated.status_code == 200
    assert zero_inflated.json()["dispersion_alpha"] > 0
    hurdle = client.post(
        "/analysis/hurdle-poisson",
        headers=headers,
        json={
            "records": _hurdle_records(),
            "outcome": "count",
            "predictors": ["x"],
            "exposure_column": "exposure",
            "hurdle_predictors": ["hurdle_driver"],
        },
    )
    assert hurdle.status_code == 200
    assert hurdle.json()["method"].startswith("Hurdle Poisson")
    categorical_count = client.post(
        "/analysis/count-regression-categorical",
        headers=headers,
        json={
            "records": _categorical_count_records(),
            "outcome": "count",
            "predictors": ["x", "group"],
            "distribution": "negative_binomial",
            "exposure_column": "exposure",
            "categorical_predictors": ["group"],
            "category_references": {"group": "control"},
            "interactions": [["x", "group"]],
        },
    )
    assert categorical_count.status_code == 200
    assert categorical_count.json()["categorical_encoding"][0]["reference"] == "control"
    categorical_cox = client.post(
        "/analysis/cox",
        headers=headers,
        json={
            "records": _categorical_cox_records(),
            "time_column": "time",
            "event_column": "event",
            "predictors": ["x", "group"],
            "categorical_predictors": ["group"],
            "category_references": {"group": "control"},
            "interactions": [["x", "group"]],
        },
    )
    assert categorical_cox.status_code == 200
    assert categorical_cox.json()["categorical_encoding"][0]["reference"] == "control"
    report = client.post("/analysis/report.html", headers=headers, json={"records": survival_records, "title": "Report"})
    assert report.status_code == 200
    assert report.headers["content-disposition"].startswith("attachment;")
