"""Seed-type and argument-validation contracts for stochastic imputers (WP-05).

Every stochastic public function accepts the same ``random_state`` types:
``None``, non-negative Python or numpy integers of any size, a
:class:`numpy.random.Generator` and a :class:`numpy.random.RandomState`.
Anything else, including negative integers, floats, booleans and strings,
raises :class:`~missingly.exceptions.ConfigurationError` before any draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly._validation import validate_positive_int
from missingly.exceptions import ConfigurationError
from missingly.hotdeck import impute_hotdeck_random, impute_hotdeck_weighted
from missingly.impute import (
    impute_gb,
    impute_logreg,
    impute_mean,
    impute_mice,
    impute_mode,
    impute_pmm,
    impute_polr,
    impute_polyreg,
    impute_rf,
)

DETERMINISTIC_SEED_FACTORIES = {
    "int": lambda: 0,
    "np.int64": lambda: np.int64(3),
    "2**40": lambda: 2**40,
    "Generator": lambda: np.random.default_rng(1),
    "RandomState": lambda: np.random.RandomState(1),
}
INVALID_SEEDS = [-1, 1.5, True, "a"]


def _seeded_frame() -> pd.DataFrame:
    """Return a small mixed-type frame with missing cells in every modelled column."""
    rng = np.random.default_rng(7)
    n_rows = 30
    x = rng.normal(size=n_rows)
    y = 2.0 * x + rng.normal(scale=0.1, size=n_rows)
    binary = np.where(x > 0, "yes", "no").astype(object)
    ordinal = np.select([x < -0.5, x < 0.5], ["low", "mid"], default="high").astype(object)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "b": binary,
            "c": pd.Categorical(ordinal, categories=["low", "mid", "high"]).astype(object),
        }
    )
    for column in ("x", "y", "b", "c"):
        frame.loc[rng.choice(n_rows, size=5, replace=False), column] = np.nan
    return frame


STOCHASTIC_CALLS = {
    "impute_hotdeck_random": lambda df, seed: impute_hotdeck_random(df, random_state=seed),
    "impute_hotdeck_weighted": lambda df, seed: impute_hotdeck_weighted(df, random_state=seed),
    "impute_pmm": lambda df, seed: impute_pmm(df, max_iter=2, random_state=seed),
    "impute_logreg": lambda df, seed: impute_logreg(df, max_iter=2, random_state=seed),
    "impute_polyreg": lambda df, seed: impute_polyreg(df, max_iter=2, random_state=seed),
    "impute_polr": lambda df, seed: impute_polr(df, max_iter=2, random_state=seed),
    "impute_rf": lambda df, seed: impute_rf(df, max_iter=1, random_state=seed),
    "impute_gb": lambda df, seed: impute_gb(df, max_iter=1, random_state=seed),
    "impute_mice": lambda df, seed: impute_mice(df, max_iter=2, random_state=seed),
    "impute_mice_multi": lambda df, seed: impute_mice(
        df, max_iter=2, random_state=seed, n_imputations=2
    ),
}


@pytest.mark.parametrize("name", sorted(STOCHASTIC_CALLS))
@pytest.mark.parametrize("seed_kind", sorted(DETERMINISTIC_SEED_FACTORIES))
def test_every_stochastic_function_accepts_each_seed_type(name, seed_kind):
    """Each stochastic function runs for every supported seed type."""
    df = _seeded_frame()
    seed_factory = DETERMINISTIC_SEED_FACTORIES[seed_kind]
    result = STOCHASTIC_CALLS[name](df, seed_factory())
    frames = result if isinstance(result, list) else [result]
    for frame in frames:
        assert frame.shape == df.shape
        assert not frame.isna().any().any()


@pytest.mark.parametrize("name", sorted(STOCHASTIC_CALLS))
@pytest.mark.parametrize("seed_kind", sorted(DETERMINISTIC_SEED_FACTORIES))
def test_fresh_equal_seeds_reproduce_every_stochastic_output(name, seed_kind):
    """Two fresh objects built from the same deterministic seed give identical output."""
    df = _seeded_frame()
    seed_factory = DETERMINISTIC_SEED_FACTORIES[seed_kind]
    first = STOCHASTIC_CALLS[name](df, seed_factory())
    second = STOCHASTIC_CALLS[name](df, seed_factory())
    if isinstance(first, list):
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)
    else:
        pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("name", sorted(STOCHASTIC_CALLS))
@pytest.mark.parametrize("bad_seed", INVALID_SEEDS, ids=repr)
def test_invalid_seed_raises_configuration_error(name, bad_seed):
    """Negative ints, floats, booleans and strings are rejected by every stochastic function."""
    with pytest.raises(ConfigurationError, match="random_state"):
        STOCHASTIC_CALLS[name](_seeded_frame(), bad_seed)


def test_none_seed_is_accepted_by_every_stochastic_function():
    """``random_state=None`` keeps working and fills every missing cell."""
    df = _seeded_frame()
    for name, call in STOCHASTIC_CALLS.items():
        assert not call(df, None).isna().any().any(), name


def test_as_generator_returns_generator_for_every_accepted_seed():
    from missingly._validation import as_generator, as_sklearn_seed

    """The shared helper returns a Generator for each supported seed type."""
    generator = np.random.default_rng(5)
    assert as_generator(generator) is generator
    assert isinstance(as_generator(None), np.random.Generator)
    assert isinstance(as_generator(np.random.RandomState(5)), np.random.Generator)
    assert isinstance(as_generator(np.int64(4)), np.random.Generator)
    assert isinstance(as_generator(2**40), np.random.Generator)


def test_as_generator_int_seed_matches_default_rng_exactly():
    from missingly._validation import as_generator, as_sklearn_seed

    """An int seed yields the same stream as ``np.random.default_rng(seed)``."""
    for seed in (0, 3, 2**32 + 5, 2**40):
        expected = np.random.default_rng(seed).integers(0, 10**9, size=8)
        actual = as_generator(seed).integers(0, 10**9, size=8)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bad_seed", INVALID_SEEDS, ids=repr)
def test_as_generator_rejects_invalid_seed(bad_seed):
    """The helper raises the package error type for every invalid seed."""
    from missingly._validation import as_generator

    with pytest.raises(ConfigurationError):
        as_generator(bad_seed)


def test_as_sklearn_seed_keeps_small_int_and_derives_large_seed():
    from missingly._validation import as_generator, as_sklearn_seed

    """Ints below 2**32 pass through unchanged; larger seeds map to a valid sklearn int."""
    assert as_sklearn_seed(7) == 7
    assert as_sklearn_seed(np.int64(7)) == 7
    assert as_sklearn_seed(None) is None
    derived = as_sklearn_seed(2**40)
    assert isinstance(derived, int)
    assert 0 <= derived < 2**32
    assert derived == as_sklearn_seed(2**40)


def test_as_sklearn_seed_derives_from_generator():
    from missingly._validation import as_generator, as_sklearn_seed

    """A Generator seed maps to a deterministic sklearn-valid integer."""
    first = as_sklearn_seed(np.random.default_rng(9))
    second = as_sklearn_seed(np.random.default_rng(9))
    assert first == second
    assert 0 <= first < 2**32


def test_generator_seed_does_not_fall_back_to_mean_imputation():
    """F-033: a Generator seed fits a real model instead of filling column means."""
    df = _seeded_frame()
    missing = df["y"].isna()
    mean_filled = impute_mean(df)["y"][missing]
    for seed in (np.random.default_rng(1), 2**40):
        imputed = impute_rf(df, random_state=seed)["y"][missing]
        assert not np.allclose(imputed.to_numpy(), mean_filled.to_numpy())


def test_generator_seed_does_not_fall_back_to_mode_imputation():
    """F-033: a Generator seed fits a real classifier instead of filling the mode."""
    df = _seeded_frame()
    missing = df["b"].isna()
    mode_filled = impute_mode(df)["b"][missing]
    imputed = impute_logreg(df, random_state=np.random.default_rng(1))["b"][missing]
    assert not (imputed.to_numpy() == mode_filled.to_numpy()).all()


def test_validate_positive_int_rejects_booleans():
    """F-203: booleans are not counts, even though bool subclasses int."""
    with pytest.raises(TypeError, match="n_neighbors"):
        validate_positive_int(True, param="n_neighbors")
    with pytest.raises(TypeError, match="n_donors"):
        validate_positive_int(False, param="n_donors", allow_zero=True)
    with pytest.raises(TypeError, match="n_imputations"):
        impute_mice(_seeded_frame(), n_imputations=True)


def test_validate_positive_int_accepts_numpy_integers():
    """numpy integer counts pass validation."""
    validate_positive_int(np.int64(3), param="max_iter")
    validate_positive_int(np.uint8(0), param="n_donors", allow_zero=True)


def test_mice_return_result_accepts_numpy_integer_arguments():
    """F-111: numpy integers for random_state, max_iter and n_imputations work with return_result."""
    df = _seeded_frame()
    typed = impute_mice(
        df,
        max_iter=np.int64(2),
        random_state=np.int64(5),
        n_imputations=np.int64(2),
        return_result=True,
    )
    reference = impute_mice(df, max_iter=2, random_state=5, n_imputations=2, return_result=True)
    assert typed.data.n_imputations == 2
    assert typed.data.plan.seed_sequence == (5, 6)
    for left, right in zip(typed.data.imputations, reference.data.imputations):
        pd.testing.assert_frame_equal(left, right)


def test_mice_return_result_accepts_generator_and_random_state_seeds():
    """Generator and RandomState seeds build a typed result with integer chain seeds."""
    df = _seeded_frame()
    for seed in (np.random.default_rng(2), np.random.RandomState(2)):
        typed = impute_mice(df, max_iter=2, random_state=seed, n_imputations=2, return_result=True)
        assert all(isinstance(value, int) for value in typed.data.plan.seed_sequence)
        assert len(set(typed.data.plan.seed_sequence)) == 2


def test_visit_sequence_accepts_non_string_labels():
    """F-244: visit_sequence accepts any column label present in the frame."""
    df = pd.DataFrame({0: [1.0, np.nan, 3.0, 4.0], 1: [2.0, 2.5, np.nan, 5.0]})
    result = impute_mice(df, max_iter=2, visit_sequence=[1, 0])
    assert not result.isna().any().any()


def test_visit_sequence_rejects_unknown_non_string_labels():
    """Unknown labels are still rejected when they are not strings."""
    df = pd.DataFrame({0: [1.0, np.nan], 1: [2.0, 3.0]})
    with pytest.raises(ValueError, match="unknown columns"):
        impute_mice(df, max_iter=2, visit_sequence=[7, 0])


def test_bounds_accept_numpy_integer_limits():
    """F-244: numpy integer bounds are valid numeric limits."""
    df = _seeded_frame()
    result = impute_mice(
        df,
        max_iter=2,
        bounds={"y": (np.int64(-50), np.int64(50))},
    )
    assert result["y"].between(-50, 50).all()
