"""Correctness tests for the hot-deck imputers (WP-18).

Statistical claims are checked against closed-form expectations: the random
and weighted donor draws are compared to exact probabilities with a fixed-seed
chi-square statistic, and the vectorised Gower distance is compared to an
independent pure-Python reference implementation.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

import missingly.hotdeck as hotdeck
from missingly.hotdeck import (
    impute_hotdeck_random,
    impute_hotdeck_sequential,
    impute_hotdeck_weighted,
)

# Chi-square critical values at significance 0.001 for the degrees of freedom
# used below; with a fixed seed the tests are deterministic, the level only
# documents how far from the expectation a failure is.
CHI2_CRIT_001 = {1: 10.828, 2: 13.816, 3: 16.266, 4: 18.467, 9: 27.877}


def _chi_square(observed: np.ndarray, probabilities: np.ndarray) -> float:
    expected = probabilities * observed.sum()
    return float(((observed - expected) ** 2 / expected).sum())


def _counts(values: pd.Series, levels: list[float]) -> np.ndarray:
    return np.array([(values == level).sum() for level in levels], dtype=float)


# ---------------------------------------------------------------------------
# Characterisation: behaviour that was correct and must not change
# ---------------------------------------------------------------------------


class TestCharacterisation:
    def test_random_fixed_seed_output_is_unchanged(self):
        df = pd.DataFrame(
            {
                "age": [25.0, np.nan, 35.0, 40.0, np.nan, 52.0, 61.0, np.nan],
                "city": pd.Categorical(
                    ["Paris", "Lyon", np.nan, "Paris", "Lyon", "Paris", "Nice", "Nice"]
                ),
                "n": pd.array([1, 2, pd.NA, 4, 5, 6, pd.NA, 8], dtype="Int64"),
            }
        )
        result = impute_hotdeck_random(df, random_state=7)
        assert result["age"].tolist() == [25.0, 61.0, 35.0, 40.0, 52.0, 52.0, 61.0, 52.0]
        assert result["city"].astype(str).tolist() == [
            "Paris", "Lyon", "Nice", "Paris", "Lyon", "Paris", "Nice", "Nice",
        ]
        assert result["n"].tolist() == [1, 2, 5, 4, 5, 6, 6, 8]

    def test_weighted_nearest_neighbour_output_is_unchanged(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.7, np.nan, 9.0, 10.5, np.nan, 20.0],
                "g": ["a", "a", "b", "a", "c", "c", "b", "c"],
                "y": [10.0, np.nan, 31.0, 40.0, 50.0, np.nan, 33.0, 61.0],
            }
        )
        result = impute_hotdeck_weighted(df, n_donors=1, random_state=0)
        assert result["x"].tolist() == [1.0, 2.0, 3.7, 2.0, 9.0, 10.5, 3.7, 20.0]
        assert result["y"].tolist() == [10.0, 40.0, 31.0, 40.0, 50.0, 50.0, 33.0, 61.0]

    def test_sequential_leading_and_trailing_gaps_are_unchanged(self):
        df = pd.DataFrame({"a": [np.nan, 2.0, 3.0, np.nan]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nearest = impute_hotdeck_sequential(df)
        assert nearest["a"].tolist() == [2.0, 2.0, 3.0, 3.0]
        assert impute_hotdeck_sequential(df, "forward")["a"].isna().sum() == 1
        assert impute_hotdeck_sequential(df, "backward")["a"].isna().sum() == 1

    def test_random_marginal_is_uniform_over_observed_values(self):
        observed = [1.0, 2.0, 3.0, 4.0]
        df = pd.DataFrame({"a": observed + [np.nan] * 800})
        result = impute_hotdeck_random(df, random_state=11)
        counts = _counts(result["a"].iloc[4:], observed)
        stat = _chi_square(counts, np.full(4, 0.25))
        assert stat < CHI2_CRIT_001[3]


# ---------------------------------------------------------------------------
# F-013: ties are broken at random
# ---------------------------------------------------------------------------


class TestWeightedTies:
    LEVELS = [float(v) for v in range(10)]

    def _tied_frame(self, n_recipients: int = 1000) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "m": ["x"] * (10 + n_recipients),
                "v": self.LEVELS + [np.nan] * n_recipients,
            }
        )

    def test_every_tied_donor_is_used_uniformly(self):
        result = impute_hotdeck_weighted(self._tied_frame(), n_donors=3, random_state=5)
        counts = _counts(result["v"].iloc[10:], self.LEVELS)
        assert (counts > 0).all()
        assert _chi_square(counts, np.full(10, 0.1)) < CHI2_CRIT_001[9]

    def test_single_donor_pool_picks_uniformly_among_ties(self):
        result = impute_hotdeck_weighted(self._tied_frame(), n_donors=1, random_state=5)
        counts = _counts(result["v"].iloc[10:], self.LEVELS)
        assert _chi_square(counts, np.full(10, 0.1)) < CHI2_CRIT_001[9]

    def test_no_matching_columns_means_all_donors_tie(self):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0] + [np.nan] * 800})
        result = impute_hotdeck_weighted(df, n_donors=1, random_state=3)
        counts = _counts(result["v"].iloc[4:], [1.0, 2.0, 3.0, 4.0])
        assert _chi_square(counts, np.full(4, 0.25)) < CHI2_CRIT_001[3]


# ---------------------------------------------------------------------------
# F-087: documented, bounded weight function
# ---------------------------------------------------------------------------


class TestWeightedProbabilities:
    DONOR_X = [0.0, 0.1, 0.5, 1.0]
    LEVELS = [0.0, 1.0, 2.0, 3.0]

    def _frame(self, n_recipients: int = 4000) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": self.DONOR_X + [0.0] * n_recipients,
                "v": self.LEVELS + [np.nan] * n_recipients,
            }
        )

    @pytest.mark.parametrize("n_donors", [4, 2])
    def test_donor_probabilities_follow_documented_weights(self, n_donors):
        # Gower distance to the recipient (x = 0, column range 1) is x itself;
        # the documented weight is 1 / (1 + distance), normalised over the pool.
        weights = 1.0 / (1.0 + np.array(self.DONOR_X[:n_donors]))
        probabilities = weights / weights.sum()
        result = impute_hotdeck_weighted(self._frame(), n_donors=n_donors, random_state=1)
        counts = _counts(result["v"].iloc[4:], self.LEVELS[:n_donors])
        assert counts.sum() == 4000
        assert _chi_square(counts, probabilities) < CHI2_CRIT_001[n_donors - 1]

    def test_exact_match_does_not_take_all_the_weight(self):
        result = impute_hotdeck_weighted(self._frame(1000), n_donors=4, random_state=2)
        share_of_exact = float((result["v"].iloc[4:] == 0.0).mean())
        assert share_of_exact < 0.5


# ---------------------------------------------------------------------------
# F-342: datetime and timedelta matchers are numeric
# ---------------------------------------------------------------------------


class TestDatetimeMatchers:
    @pytest.mark.parametrize("seed", range(5))
    def test_nearest_date_wins(self, seed):
        dates = pd.to_datetime(["2015-01-01", "2020-01-01", "2020-01-02", "2010-06-01"])
        df = pd.DataFrame({"when": dates, "v": [1.0, 2.0, np.nan, 4.0]})
        result = impute_hotdeck_weighted(df, n_donors=1, random_state=seed)
        assert result.loc[2, "v"] == 2.0

    def test_timezone_aware_dates_are_numeric(self):
        dates = pd.to_datetime(
            ["2015-01-01", "2020-01-01", "2020-01-02", "2010-06-01"], utc=True
        ).tz_convert("Asia/Tehran")
        df = pd.DataFrame({"when": dates, "v": [1.0, 2.0, np.nan, 4.0]})
        result = impute_hotdeck_weighted(df, n_donors=1, random_state=0)
        assert result.loc[2, "v"] == 2.0

    def test_timedelta_matchers_are_numeric(self):
        deltas = pd.to_timedelta([100, 10, 11, 500], unit="D")
        df = pd.DataFrame({"gap": deltas, "v": [1.0, 2.0, np.nan, 4.0]})
        result = impute_hotdeck_weighted(df, n_donors=1, random_state=0)
        assert result.loc[2, "v"] == 2.0

    def test_missing_dates_are_skipped_pairwise(self):
        dates = pd.to_datetime(["2015-01-01", None, "2020-01-02", "2020-01-01"])
        df = pd.DataFrame({"when": dates, "v": [1.0, 2.0, np.nan, 4.0]})
        result = impute_hotdeck_weighted(df, n_donors=1, random_state=0)
        assert result.loc[2, "v"] == 4.0


# ---------------------------------------------------------------------------
# F-085: vectorised distances equal an independent reference
# ---------------------------------------------------------------------------


def _reference_gower(a: pd.Series, b: pd.Series, numeric: list, ranges: list) -> float:
    total, count = 0.0, 0
    for j, (u, v) in enumerate(zip(a, b)):
        if pd.isna(u) or pd.isna(v):
            continue
        count += 1
        if numeric[j]:
            total += 0.0 if ranges[j] == 0 else abs(float(u) - float(v)) / ranges[j]
        else:
            total += 0.0 if u == v else 1.0
    return 1.0 if count == 0 else total / count


def _reference_nearest(df: pd.DataFrame, target: str, as_numeric: dict) -> dict:
    features = [c for c in df.columns if c != target]
    numeric = [c in as_numeric for c in features]
    ranges = [
        float(as_numeric[c].max() - as_numeric[c].min()) if c in as_numeric else 0.0
        for c in features
    ]
    num_frame = df[features].copy()
    for c, values in as_numeric.items():
        num_frame[c] = values
    donors = df.index[df[target].notna()]
    expected = {}
    for i in df.index[df[target].isna()]:
        dists = {
            d: _reference_gower(num_frame.loc[i], num_frame.loc[d], numeric, ranges)
            for d in donors
        }
        best = min(dists.values())
        winners = [d for d, dist in dists.items() if np.isclose(dist, best)]
        expected[i] = {df.loc[d, target] for d in winners}
    return expected


def _random_frame(seed: int, with_dates: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 40
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.integers(0, 50, size=n).astype(float),
            "c": rng.choice(["a", "b", "c"], size=n),
            "flag": rng.random(n) < 0.5,
            "y": rng.normal(size=n),
        }
    )
    if with_dates:
        df["when"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(
            rng.integers(0, 3000, size=n), unit="D"
        )
    for col in ("x", "z", "y"):
        df.loc[rng.random(n) < 0.15, col] = np.nan
    df["c"] = df["c"].where(rng.random(n) >= 0.1, np.nan)
    df.loc[[0, 1], "y"] = [0.5, 0.25]
    return df


class TestVectorisedDistance:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("with_dates", [False, True])
    def test_nearest_donor_matches_pure_python_reference(self, seed, with_dates):
        df = _random_frame(seed, with_dates)
        as_numeric = {"x": df["x"], "z": df["z"]}
        if with_dates:
            as_numeric["when"] = df["when"].astype("int64").astype(float)
        expected = _reference_nearest(df, "y", as_numeric)
        result = impute_hotdeck_weighted(df, n_donors=1, random_state=seed)
        for row, allowed in expected.items():
            assert result.loc[row, "y"] in allowed

    def test_large_frame_runs_in_reasonable_time(self):
        rng = np.random.default_rng(0)
        n = 4000
        df = pd.DataFrame(
            {
                "a": rng.normal(size=n),
                "b": rng.normal(size=n),
                "c": rng.choice(["u", "v", "w"], size=n),
            }
        )
        for col in df.columns:
            df.loc[rng.random(n) < 0.15, col] = np.nan
        start = time.perf_counter()
        impute_hotdeck_weighted(df, n_donors=5, random_state=0)
        assert time.perf_counter() - start < 6.0

    def test_warns_when_pair_count_is_large(self, monkeypatch):
        monkeypatch.setattr(hotdeck, "_PAIR_WARN_THRESHOLD", 20)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan, np.nan, 6.0, 7.0, 8.0]})
        with pytest.warns(UserWarning, match="donor pairs"):
            impute_hotdeck_weighted(df, random_state=0)

    def test_no_pair_warning_for_small_problem(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            impute_hotdeck_weighted(df, random_state=0)


# ---------------------------------------------------------------------------
# Invariants: donors are observed values from rows that have the variable
# ---------------------------------------------------------------------------


class TestDonorInvariants:
    @pytest.mark.parametrize("seed", range(5))
    def test_weighted_imputed_values_come_from_observed_rows(self, seed):
        df = _random_frame(seed, with_dates=True)
        result = impute_hotdeck_weighted(df, n_donors=4, random_state=seed)
        assert result["y"].notna().all()
        assert set(result.loc[df["y"].isna(), "y"]).issubset(set(df["y"].dropna()))
        assert set(result.loc[df["c"].isna(), "c"]).issubset(set(df["c"].dropna()))

    @pytest.mark.parametrize("seed", range(5))
    def test_random_imputed_values_come_from_observed_rows(self, seed):
        df = _random_frame(seed, with_dates=False)
        result = impute_hotdeck_random(df, random_state=seed)
        for col in ("x", "y", "z"):
            assert set(result.loc[df[col].isna(), col]).issubset(set(df[col].dropna()))


# ---------------------------------------------------------------------------
# F-086 / F-340: imputation classes
# ---------------------------------------------------------------------------


class TestImputationClasses:
    def test_random_donors_stay_inside_their_cell(self):
        df = pd.DataFrame(
            {
                "g": ["a"] * 903 + ["b"] * 4,
                "v": [1.0, 2.0, 3.0] + [np.nan] * 900 + [100.0, 200.0, np.nan, np.nan],
            }
        )
        result = impute_hotdeck_random(df, random_state=4, by="g")
        counts = _counts(result["v"].iloc[3:903], [1.0, 2.0, 3.0])
        assert counts.sum() == 900
        assert _chi_square(counts, np.full(3, 1 / 3)) < CHI2_CRIT_001[2]
        assert set(result.loc[[905, 906], "v"]).issubset({100.0, 200.0})

    def test_random_by_list_of_columns(self):
        df = pd.DataFrame(
            {
                "g": ["a", "a", "a", "b", "b", "b"],
                "h": ["x", "x", "y", "x", "x", "y"],
                "v": [1.0, np.nan, 7.0, 5.0, 6.0, np.nan],
            }
        )
        result = impute_hotdeck_random(df, random_state=0, by=["g", "h"])
        assert result.loc[1, "v"] == 1.0
        assert result.loc[5, "v"] == 7.0

    def test_random_by_missing_key_forms_its_own_cell(self):
        df = pd.DataFrame(
            {
                "g": ["a", "a", None, None, None],
                "v": [1.0, np.nan, 50.0, 60.0, np.nan],
            }
        )
        for seed in range(10):
            result = impute_hotdeck_random(df, random_state=seed, by="g")
            assert result.loc[1, "v"] == 1.0
            assert result.loc[4, "v"] in {50.0, 60.0}

    def test_by_columns_are_not_imputed(self):
        df = pd.DataFrame({"g": ["a", None, "a", "b"], "v": [1.0, 2.0, np.nan, 4.0]})
        result = impute_hotdeck_random(df, random_state=0, by="g")
        assert pd.isna(result.loc[1, "g"])

    def test_random_cell_without_donor_raises(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, np.nan]})
        with pytest.raises(ValueError, match="no observed values"):
            impute_hotdeck_random(df, by="g")

    def test_unknown_by_column_raises(self):
        df = pd.DataFrame({"v": [1.0, np.nan]})
        with pytest.raises(ValueError, match="by"):
            impute_hotdeck_random(df, by="missing")

    def test_invalid_by_type_raises(self):
        df = pd.DataFrame({"v": [1.0, np.nan]})
        with pytest.raises(TypeError, match="by"):
            impute_hotdeck_random(df, by=3)

    def test_sequential_forward_does_not_cross_subjects(self):
        df = pd.DataFrame(
            {
                "subject": [1, 1, 1, 2, 2, 2],
                "v": [1.0, np.nan, 3.0, np.nan, 5.0, np.nan],
            }
        )
        forward = impute_hotdeck_sequential(df, "forward", by="subject")
        assert forward["v"].iloc[:3].tolist() == [1.0, 1.0, 3.0]
        assert pd.isna(forward.loc[3, "v"])
        assert forward.loc[5, "v"] == 5.0
        backward = impute_hotdeck_sequential(df, "backward", by="subject")
        assert backward["v"].iloc[:5].tolist() == [1.0, 3.0, 3.0, 5.0, 5.0]
        assert pd.isna(backward.loc[5, "v"])

    def test_sequential_closest_stays_inside_group(self):
        df = pd.DataFrame({"s": [1, 1, 2, 2], "v": [np.nan, 10.0, 20.0, np.nan]})
        result = impute_hotdeck_sequential(df, "closest", by="s")
        assert result["v"].tolist() == [10.0, 10.0, 20.0, 20.0]

    def test_sequential_keeps_original_row_order(self):
        df = pd.DataFrame(
            {"s": [2, 1, 2, 1], "v": [5.0, 1.0, np.nan, np.nan]},
            index=[10, 11, 12, 13],
        )
        result = impute_hotdeck_sequential(df, "forward", by="s")
        assert list(result.index) == [10, 11, 12, 13]
        assert result["v"].tolist() == [5.0, 1.0, 5.0, 1.0]

    def test_weighted_donors_stay_inside_their_cell(self):
        df = pd.DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "x": [0.0, 10.0, 0.1, 10.0],
                "v": [np.nan, 1.0, 2.0, 3.0],
            }
        )
        # The nearest row by x is in cell b, but cell a only offers value 1.
        result = impute_hotdeck_weighted(df, n_donors=1, random_state=0, by="g")
        assert result.loc[0, "v"] == 1.0

    def test_weighted_cell_without_donor_raises(self):
        df = pd.DataFrame(
            {"g": ["a", "b", "b"], "x": [1.0, 2.0, 3.0], "v": [np.nan, 2.0, 3.0]}
        )
        with pytest.raises(ValueError, match="no observed values"):
            impute_hotdeck_weighted(df, by="g")


# ---------------------------------------------------------------------------
# F-341: closest direction for the sequential hot-deck
# ---------------------------------------------------------------------------


class TestSequentialClosest:
    def test_closest_picks_nearest_position_with_ties_to_previous(self):
        df = pd.DataFrame({"v": [1.0, np.nan, np.nan, np.nan, 100.0]})
        result = impute_hotdeck_sequential(df, "closest")
        assert result["v"].tolist() == [1.0, 1.0, 1.0, 100.0, 100.0]

    def test_closest_fills_leading_and_trailing_gaps(self):
        df = pd.DataFrame({"v": [np.nan, 2.0, np.nan, np.nan, 8.0, np.nan]})
        result = impute_hotdeck_sequential(df, "closest")
        assert result["v"].tolist() == [2.0, 2.0, 2.0, 8.0, 8.0, 8.0]

    def test_closest_matches_brute_force_on_random_gaps(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            values = rng.normal(size=15)
            values[rng.random(15) < 0.5] = np.nan
            if np.isnan(values).all():
                continue
            observed = np.flatnonzero(~np.isnan(values))
            expected = values.copy()
            for i in np.flatnonzero(np.isnan(values)):
                gaps = np.abs(observed - i)
                expected[i] = values[observed[np.flatnonzero(gaps == gaps.min())[0]]]
            result = impute_hotdeck_sequential(pd.DataFrame({"v": values}), "closest")
            np.testing.assert_array_equal(result["v"].to_numpy(), expected)

    def test_forward_then_backward_is_the_legacy_behaviour(self):
        df = pd.DataFrame({"v": [np.nan, 1.0, np.nan, np.nan, np.nan, 100.0]})
        result = impute_hotdeck_sequential(df, "forward_then_backward")
        assert result["v"].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 100.0]

    def test_nearest_keeps_legacy_values_and_warns_when_they_differ(self):
        df = pd.DataFrame({"v": [1.0, np.nan, np.nan, np.nan, 100.0]})
        with pytest.warns(FutureWarning, match="closest"):
            result = impute_hotdeck_sequential(df, "nearest")
        assert result["v"].tolist() == [1.0, 1.0, 1.0, 1.0, 100.0]

    def test_nearest_is_silent_when_it_agrees_with_closest(self):
        df = pd.DataFrame({"v": [np.nan, 1.0, np.nan, 3.0, np.nan]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = impute_hotdeck_sequential(df, "nearest")
        assert result["v"].tolist() == [1.0, 1.0, 1.0, 3.0, 3.0]

    def test_invalid_direction_message_lists_new_values(self):
        with pytest.raises(ValueError, match="closest"):
            impute_hotdeck_sequential(pd.DataFrame({"v": [1.0]}), "sideways")
