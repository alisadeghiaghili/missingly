import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from analytics import missingness_report, profile_dataset, run_statistical_test


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
