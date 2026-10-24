"""Chain seeds of ``impute_mice`` are independent and reproducible.

Seeds used to be ``random_state + i``, so ``random_state=0`` and
``random_state=1`` with three chains shared two chains exactly. Seeds are now
spawned from ``numpy.random.SeedSequence(random_state)``.
"""

from __future__ import annotations

import itertools
from typing import List

import numpy as np
import pandas as pd
import pytest

from missingly.impute import impute_mice


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = 60
    x = rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "y": x + rng.normal(size=n), "z": rng.normal(size=n)})
    frame.loc[0::6, "y"] = np.nan
    frame.loc[2::8, "z"] = np.nan
    return frame


def _chains(random_state: int, n_imputations: int) -> List[pd.DataFrame]:
    out = impute_mice(_frame(), random_state=random_state, n_imputations=n_imputations, max_iter=4)
    return list(out) if n_imputations > 1 else [out]


def test_chains_of_adjacent_seeds_share_no_data_set() -> None:
    first, second = _chains(0, 3), _chains(1, 3)
    for left, right in itertools.product(first, second):
        assert not left.equals(right)


def test_all_chains_of_many_seeds_are_pairwise_distinct() -> None:
    chains = [chain for seed in range(5) for chain in _chains(seed, 4)]
    assert len(chains) == 20
    for left, right in itertools.combinations(chains, 2):
        assert not left.equals(right)


def test_same_seed_reproduces_every_chain() -> None:
    for first, again in zip(_chains(3, 4), _chains(3, 4)):
        pd.testing.assert_frame_equal(first, again, check_exact=True)


def test_a_chain_does_not_depend_on_how_many_chains_were_requested() -> None:
    one, two, four = _chains(5, 1), _chains(5, 2), _chains(5, 4)
    pd.testing.assert_frame_equal(one[0], two[0], check_exact=True)
    pd.testing.assert_frame_equal(two[1], four[1], check_exact=True)
    pd.testing.assert_frame_equal(two[0], four[0], check_exact=True)


def test_recorded_chain_seeds_are_distinct_across_adjacent_random_states() -> None:
    def seeds(random_state: int, m: int) -> tuple:
        result = impute_mice(
            _frame(), random_state=random_state, n_imputations=m, max_iter=2, return_result=True
        )
        return result.data.plan.seed_sequence

    zero, one = seeds(0, 3), seeds(1, 3)
    assert len(set(zero)) == 3
    assert not set(zero) & set(one)
    assert seeds(0, 5)[:3] == zero  # the plan is reproducible and extendable


def test_recorded_chain_seeds_are_integers_the_chains_can_be_rebuilt_from() -> None:
    result = impute_mice(
        _frame(), random_state=8, n_imputations=3, max_iter=4, return_result=True
    )
    again = impute_mice(
        _frame(), random_state=8, n_imputations=3, max_iter=4, return_result=True
    )
    assert result.data.plan.seed_sequence == again.data.plan.seed_sequence
    for left, right in zip(result.data.imputations, again.data.imputations):
        pd.testing.assert_frame_equal(left, right, check_exact=True)


def test_none_random_state_draws_fresh_entropy_and_records_it() -> None:
    first = impute_mice(_frame(), random_state=None, max_iter=2, return_result=True)
    second = impute_mice(_frame(), random_state=None, max_iter=2, return_result=True)
    assert first.data.plan.seed_sequence != second.data.plan.seed_sequence


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True])
def test_invalid_random_state_is_rejected_before_any_work(bad: object) -> None:
    with pytest.raises(ValueError, match="random_state"):
        impute_mice(_frame(), random_state=bad)  # type: ignore[arg-type]
