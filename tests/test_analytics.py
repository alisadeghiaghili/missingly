import importlib
import io
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from analytics import advanced_numeric_imputation, missingness_report, profile_dataset, records_to_csv, run_statistical_test, simple_imputation
from ingestion import import_tabular_bytes


RECORDS = [
    {"group": "control", "score": 10, "age": 20, "outcome": "no"},
    {"group": "control", "score": 12, "age": 21, "outcome": "no"},
    {"group": "control", "score": 11, "age": 22, "outcome": "yes"},
    {"group": "treatment", "score": 20, "age": 30, "outcome": "yes"},
    {"group": "treatment", "score": 22, "age": 31, "outcome": "yes"},
    {"group": "treatment", "score": None, "age": 32, "outcome": "yes"},
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
