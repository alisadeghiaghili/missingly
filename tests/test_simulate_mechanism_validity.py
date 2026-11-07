"""Mechanism validity of ``simulate_mcar`` / ``simulate_mar`` / ``simulate_mnar``.

Every mechanism is checked against its *definition*, independently of the
implementation:

* MCAR: the missingness indicator is uncorrelated with every value.
* MAR: the probability of being missing is a function of the observed driver
  only; once the driver is fixed, the value of the blanked column adds nothing.
* MNAR: the probability of being missing depends on the column's own value.
* The realised missing proportion matches the requested ``frac``.

Statistical tolerance
---------------------
A mask is a vector of independent Bernoulli draws with probabilities ``p_i``.
For ``R`` replicate seeds the total count in a row group ``G`` has mean
``R * sum(p_i)`` and variance ``R * sum(p_i * (1 - p_i))``; every comparison
below allows ``Z = 4`` standard errors of that count. For a requested overall
rate ``frac`` the variance is at most the binomial ``R * n * frac * (1 - frac)``.
The seeds are fixed, so the tests are deterministic.

The reference probabilities come from ``_reference_probabilities``, a direct
loop-based transcription of the documented model (rank split with the tied
boundary shared fractionally, odds weights, scale solved with ``brentq``).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from missingly.simulate import (
    simulate_mar,
    simulate_mcar,
    simulate_mixed,
    simulate_mnar,
)

Z = 4.0
N = 2000
REPLICATES = 20
MAR_WEIGHTS = (2.0, 0.5)
MNAR_WEIGHTS = (2.0, 0.25)
FRACTIONS = [0.05, 0.3, 0.5, 0.7, 0.9]
TAILS = ["upper", "lower"]


def _reference_probabilities(values, tail, frac, weights):
    """Documented per-row probabilities, computed with plain loops."""
    x = np.asarray(values, dtype=float)
    x = x if tail == "upper" else -x
    n = len(x)
    w_hi, w_lo = weights
    w = np.zeros(n)
    for i in range(n):
        greater = int((x > x[i]).sum())
        ties = int((x == x[i]).sum())
        share = min(max((n / 2 - greater) / ties, 0.0), 1.0)
        w[i] = w_lo + share * (w_hi - w_lo)
    target = frac * n

    def gap(scale):
        return np.minimum(1.0, scale * w).sum() - target

    scale = brentq(gap, 0.0, 1.0 / w.min())
    return np.minimum(1.0, scale * w)


def _drivers():
    rng = np.random.default_rng(7)
    return {
        "continuous": rng.normal(size=N),
        "binary_majority": rng.permutation(np.r_[np.ones(1600), np.zeros(400)]),
        "binary_minority": rng.permutation(np.r_[np.ones(600), np.zeros(1400)]),
        "discrete3": rng.permutation(np.resize([0.0, 1.0, 2.0], N)),
    }


DRIVERS = _drivers()
DISCRETE = ["binary_majority", "binary_minority", "discrete3"]


def _masks(fn, df, target, **kwargs):
    """Boolean masks (REPLICATES x n) of the blanked target column."""
    rows = [
        fn(df, target_col=target, random_state=seed, **kwargs)[target].isna().to_numpy()
        for seed in range(REPLICATES)
    ]
    return np.vstack(rows)


def _groups(kind, driver):
    if kind == "continuous":
        median = np.median(driver)
        return {
            "below median": np.flatnonzero(driver <= median),
            "above median": np.flatnonzero(driver > median),
        }
    return {f"level {v}": np.flatnonzero(driver == v) for v in np.unique(driver)}


def _assert_group_counts(masks, probs, groups):
    replicates = masks.shape[0]
    for name, index in groups.items():
        expected = replicates * probs[index].sum()
        sd = np.sqrt(replicates * (probs[index] * (1 - probs[index])).sum())
        observed = masks[:, index].sum()
        assert abs(observed - expected) <= Z * sd + 1e-9, (
            f"group {name}: observed {observed}, expected {expected:.1f} +- {Z * sd:.1f}"
        )


def _assert_overall_rate(masks, frac):
    replicates = masks.shape[0]
    expected = frac * N * replicates
    sd = np.sqrt(replicates * N * frac * (1 - frac))
    assert abs(masks.sum() - expected) <= Z * sd, masks.sum() / (N * replicates)


def _frame(driver):
    rng = np.random.default_rng(3)
    return pd.DataFrame({"driver": driver, "y": driver + rng.normal(size=len(driver))})


# ---------------------------------------------------------------------------
# F-122: the achieved proportion equals frac (MAR and MNAR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(DRIVERS))
@pytest.mark.parametrize("tail", TAILS)
@pytest.mark.parametrize("frac", FRACTIONS)
def test_mar_achieves_requested_fraction(kind, tail, frac):
    df = _frame(DRIVERS[kind])
    masks = _masks(simulate_mar, df, "y", frac=frac, predictor_col="driver", tail=tail)
    _assert_overall_rate(masks, frac)


@pytest.mark.parametrize("kind", list(DRIVERS))
@pytest.mark.parametrize("tail", TAILS)
@pytest.mark.parametrize("frac", FRACTIONS)
def test_mnar_achieves_requested_fraction(kind, tail, frac):
    df = _frame(DRIVERS[kind])
    masks = _masks(simulate_mnar, df, "driver", frac=frac, tail=tail)
    _assert_overall_rate(masks, frac)


# ---------------------------------------------------------------------------
# F-038: binary / discrete drivers must not degenerate to MCAR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(DRIVERS))
@pytest.mark.parametrize("tail", TAILS)
def test_mar_matches_reference_rates_per_driver_group(kind, tail):
    driver = DRIVERS[kind]
    frac = 0.3
    probs = _reference_probabilities(driver, tail, frac, MAR_WEIGHTS)
    masks = _masks(
        simulate_mar, _frame(driver), "y", frac=frac, predictor_col="driver", tail=tail
    )
    _assert_group_counts(masks, probs, _groups(kind, driver))


@pytest.mark.parametrize("kind", list(DRIVERS))
@pytest.mark.parametrize("tail", TAILS)
def test_mnar_matches_reference_rates_per_value_group(kind, tail):
    driver = DRIVERS[kind]
    frac = 0.3
    probs = _reference_probabilities(driver, tail, frac, MNAR_WEIGHTS)
    masks = _masks(simulate_mnar, _frame(driver), "driver", frac=frac, tail=tail)
    _assert_group_counts(masks, probs, _groups(kind, driver))


@pytest.mark.parametrize("kind", DISCRETE)
@pytest.mark.parametrize("tail", TAILS)
def test_mar_discrete_driver_levels_have_different_rates(kind, tail):
    driver = DRIVERS[kind]
    masks = _masks(
        simulate_mar, _frame(driver), "y", frac=0.3, predictor_col="driver", tail=tail
    )
    rates = [masks[:, driver == v].mean() for v in np.unique(driver)]
    assert max(rates) > 1.5 * min(rates), rates


@pytest.mark.parametrize("kind", DISCRETE)
@pytest.mark.parametrize("tail", TAILS)
def test_mnar_discrete_levels_have_different_rates(kind, tail):
    driver = DRIVERS[kind]
    masks = _masks(simulate_mnar, _frame(driver), "driver", frac=0.3, tail=tail)
    rates = [masks[:, driver == v].mean() for v in np.unique(driver)]
    assert max(rates) > 1.5 * min(rates), rates


# ---------------------------------------------------------------------------
# MAR depends only on the driver; MNAR on the value; MCAR on neither
# ---------------------------------------------------------------------------


def test_mar_value_of_target_adds_nothing_given_driver():
    driver = DRIVERS["discrete3"]
    df = _frame(driver)
    masks = _masks(simulate_mar, df, "y", frac=0.3, predictor_col="driver")
    y = df["y"].to_numpy()
    for level in np.unique(driver):
        rows = driver == level
        high = rows & (y > np.median(y[rows]))
        low = rows & ~high
        hits_high, hits_low = masks[:, high].sum(), masks[:, low].sum()
        n_high, n_low = REPLICATES * high.sum(), REPLICATES * low.sum()
        pooled = (hits_high + hits_low) / (n_high + n_low)
        se = np.sqrt(pooled * (1 - pooled) * (1 / n_high + 1 / n_low))
        assert abs(hits_high / n_high - hits_low / n_low) <= Z * se, level


def test_mnar_depends_on_own_value():
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"y": rng.normal(size=N)})
    masks = _masks(simulate_mnar, df, "y", frac=0.3)
    y = df["y"].to_numpy()
    upper = masks[:, y > np.median(y)].mean()
    lower = masks[:, y <= np.median(y)].mean()
    assert upper > 3 * lower


def test_mcar_indicator_is_uncorrelated_with_observed_and_masked_values():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.normal(size=N), "y": rng.normal(size=N)})
    masks = np.vstack(
        [
            simulate_mcar(df, 0.3, columns=["y"], random_state=s)["y"].isna().to_numpy()
            for s in range(200)
        ]
    )
    for column in ("x", "y"):
        values = df[column].to_numpy()
        corr = np.array([np.corrcoef(m.astype(float), values)[0, 1] for m in masks])
        # corr * sqrt(n - 1) ~ N(0, 1) per replicate, so the mean has this SE.
        se = 1.0 / np.sqrt((N - 1) * len(masks))
        assert abs(corr.mean()) <= Z * se, column


# ---------------------------------------------------------------------------
# F-121: the driver cannot be the target; degenerate thresholds are flagged
# ---------------------------------------------------------------------------


def test_mar_rejects_driver_equal_to_target():
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    with pytest.raises(ValueError, match="predictor_col"):
        simulate_mar(df, 0.3, target_col="a", predictor_col="a", random_state=0)


def test_degenerate_explicit_threshold_warns():
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    with pytest.warns(UserWarning, match="threshold"):
        simulate_mar(
            df, 0.3, target_col="a", predictor_col="b", threshold=100.0, random_state=0
        )
    with pytest.warns(UserWarning, match="threshold"):
        simulate_mnar(df, 0.3, target_col="a", threshold=100.0, random_state=0)


# ---------------------------------------------------------------------------
# F-121 / F-183: simulate_mixed keeps MAR drivers observed
# ---------------------------------------------------------------------------

MAR_STEP = {
    "mechanism": "mar",
    "target_col": "y",
    "predictor_col": "driver",
    "frac": 0.2,
}


def _mixed_frame():
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "driver": DRIVERS["binary_majority"],
            "y": rng.normal(size=N),
            "z": rng.normal(size=N),
        }
    )


@pytest.mark.parametrize("mar_first", [True, False])
def test_mixed_default_mcar_does_not_mask_mar_driver(mar_first):
    mcar = {"mechanism": "mcar", "frac": 0.2}
    spec = [MAR_STEP, mcar] if mar_first else [mcar, MAR_STEP]
    result = simulate_mixed(_mixed_frame(), spec, random_state=0)
    assert not result["driver"].isna().any()
    assert result["z"].isna().any()


@pytest.mark.parametrize(
    "masking_step",
    [
        {"mechanism": "mcar", "columns": ["driver"], "frac": 0.2},
        {"mechanism": "mnar", "target_col": "driver", "frac": 0.2},
        {"mechanism": "mar", "target_col": "driver", "predictor_col": "z", "frac": 0.2},
    ],
)
@pytest.mark.parametrize("mar_first", [True, False])
def test_mixed_rejects_masking_a_mar_driver(masking_step, mar_first):
    spec = [MAR_STEP, masking_step] if mar_first else [masking_step, MAR_STEP]
    with pytest.raises(ValueError, match="driver"):
        simulate_mixed(_mixed_frame(), spec, random_state=0)


def test_mixed_rejects_mar_driver_equal_to_target():
    spec = [{"mechanism": "mar", "target_col": "y", "predictor_col": "y", "frac": 0.2}]
    with pytest.raises(ValueError, match="predictor_col"):
        simulate_mixed(_mixed_frame(), spec, random_state=0)


def test_mixed_mar_step_equals_declared_mechanism():
    df = _mixed_frame()
    driver = df["driver"].to_numpy()
    probs = _reference_probabilities(driver, "upper", 0.3, MAR_WEIGHTS)
    spec = [{**MAR_STEP, "frac": 0.3}]
    masks = np.vstack(
        [
            simulate_mixed(df, spec, random_state=s)["y"].isna().to_numpy()
            for s in range(REPLICATES)
        ]
    )
    _assert_group_counts(masks, probs, _groups("binary_majority", driver))


def test_mixed_mnar_step_after_mcar_on_other_column_equals_declared_mechanism():
    df = _mixed_frame()
    z = df["z"].to_numpy()
    probs = _reference_probabilities(z, "upper", 0.3, MNAR_WEIGHTS)
    spec = [
        {"mechanism": "mcar", "columns": ["y"], "frac": 0.2},
        {"mechanism": "mnar", "target_col": "z", "frac": 0.3},
    ]
    masks = np.vstack(
        [
            simulate_mixed(df, spec, random_state=s)["z"].isna().to_numpy()
            for s in range(REPLICATES)
        ]
    )
    _assert_group_counts(masks, probs, _groups("continuous", z))


def test_mixed_step_on_partly_masked_target_blanks_frac_of_observed_rows():
    df = _mixed_frame()
    spec = [
        {"mechanism": "mcar", "columns": ["z"], "frac": 0.5},
        {"mechanism": "mar", "target_col": "z", "predictor_col": "driver", "frac": 0.4},
    ]
    observed_rows = N // 2
    newly_missing = sum(
        simulate_mixed(df, spec, random_state=s)["z"].isna().sum() - observed_rows
        for s in range(REPLICATES)
    )
    expected = REPLICATES * 0.4 * observed_rows
    sd = np.sqrt(REPLICATES * observed_rows * 0.4 * 0.6)
    assert abs(newly_missing - expected) <= Z * sd


# ---------------------------------------------------------------------------
# Characterization: unchanged behaviour (passes before and after the fix)
# ---------------------------------------------------------------------------


def test_characterization_mcar_mask_for_fixed_seed():
    df = pd.DataFrame({"x": np.arange(20.0)})
    result = simulate_mcar(df, 0.25, random_state=3)
    expected = np.random.default_rng(3).choice(20, size=5, replace=False)
    assert sorted(np.flatnonzero(result["x"].isna())) == sorted(expected)


def test_characterization_mixed_mcar_only_matches_standalone_mcar():
    df = pd.DataFrame({"x": np.arange(30.0), "y": np.arange(30.0) * 2})
    mixed = simulate_mixed(df, [{"mechanism": "mcar", "frac": 0.2}], random_state=4)
    standalone = simulate_mcar(df, 0.2, random_state=4)
    pd.testing.assert_frame_equal(mixed, standalone)


# ---------------------------------------------------------------------------
# Edge cases and error paths
# ---------------------------------------------------------------------------


def test_mixed_step_on_fully_masked_target_is_a_no_op():
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    spec = [
        {"mechanism": "mcar", "columns": ["a"], "frac": 0.99},
        {"mechanism": "mnar", "target_col": "a", "frac": 0.5},
    ]
    result = simulate_mixed(df, spec, random_state=0)
    assert result["a"].isna().all()
    assert result["b"].notna().all()


def test_mixed_mar_step_rejects_invalid_frac():
    spec = [{**MAR_STEP, "frac": 1.5}]
    with pytest.raises(ValueError, match="frac"):
        simulate_mixed(_mixed_frame(), spec, random_state=0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"target_col": "nope", "predictor_col": "x"}, "target_col"),
        ({"target_col": "y", "predictor_col": "nope"}, "predictor_col"),
    ],
)
def test_mar_unknown_columns_raise_key_error(kwargs, match):
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    with pytest.raises(KeyError, match=match):
        simulate_mar(df, 0.5, **kwargs)


def test_mnar_unknown_column_raises_key_error():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(KeyError, match="target_col"):
        simulate_mnar(df, 0.5, target_col="nope")
