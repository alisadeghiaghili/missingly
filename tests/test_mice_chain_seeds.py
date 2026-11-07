"""Chain-seed independence for ``impute_mice`` (WP-05, F-246).

Chains inside one call, and chains of calls with consecutive base seeds, must
draw from independent streams.  Before the fix chain ``k`` of seed ``s`` used
seed ``s + k``, so chain 1 of seed ``s`` repeated chain 0 of seed ``s + 1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.impute import impute_mice

N_CHAINS = 4


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    base = rng.normal(size=(40, 3))
    frame = pd.DataFrame(
        {"a": base[:, 0], "b": base[:, 0] + base[:, 1], "c": base[:, 2]}
    )
    for column, step in (("a", 7), ("b", 5), ("c", 9)):
        frame.loc[frame.index[step - 1 :: step], column] = np.nan
    return frame


def _chains(seed: int, n_chains: int = N_CHAINS) -> list:
    return impute_mice(_frame(), n_imputations=n_chains, max_iter=3, random_state=seed)


def _same(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return bool(np.allclose(left.to_numpy(), right.to_numpy(), rtol=0, atol=1e-12))


def test_consecutive_base_seeds_share_no_chain_stream():
    first = _chains(5)
    second = _chains(6)
    shared = [
        (i, j)
        for i, left in enumerate(first)
        for j, right in enumerate(second)
        if _same(left, right)
    ]
    assert shared == []


def test_chains_within_one_call_are_pairwise_different():
    chains = _chains(5)
    equal = [
        (i, j)
        for i in range(len(chains))
        for j in range(i + 1, len(chains))
        if _same(chains[i], chains[j])
    ]
    assert equal == []


def test_chain_seeds_are_deterministic_for_a_given_seed():
    for left, right in zip(_chains(5), _chains(5)):
        pd.testing.assert_frame_equal(left, right)


def test_single_chain_equals_first_chain_of_a_multi_chain_call():
    single = impute_mice(_frame(), max_iter=3, random_state=5)
    pd.testing.assert_frame_equal(single, _chains(5)[0])


@pytest.mark.parametrize("seed", [0, 5, 2**40])
def test_result_seed_sequence_is_unique_and_not_offset_counted(seed):
    plan = impute_mice(
        _frame(),
        n_imputations=N_CHAINS,
        max_iter=2,
        random_state=seed,
        return_result=True,
    ).data.plan
    seeds = plan.seed_sequence
    assert len(set(seeds)) == N_CHAINS
    assert seeds != tuple(seed + i for i in range(N_CHAINS))
