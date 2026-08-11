"""Tests for MICE convergence diagnostics.

Covers _gelman_rubin_rhat, mice_convergence, and plot_mice_convergence
with histories from impute_mice(return_history=True).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.diagnostics import (
    _gelman_rubin_rhat,
    mice_convergence,
    plot_mice_convergence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_histories():
    """Two chains, 5 iterations each, variable 'a'."""
    return [
        {"a": [2.0, 2.1, 2.05, 2.02, 2.01]},
        {"a": [1.95, 2.02, 2.04, 2.01, 2.02]},
    ]


@pytest.fixture
def three_chain_histories():
    """Three chains, 10 iterations, two variables."""
    rng = np.random.default_rng(42)
    return [
        {
            "a": (rng.normal(2.0, 0.05, 10)).tolist(),
            "b": (rng.normal(5.0, 0.1, 10)).tolist(),
        }
        for _ in range(3)
    ]


# ---------------------------------------------------------------------------
# _gelman_rubin_rhat
# ---------------------------------------------------------------------------

class TestGelmanRubinRhat:
    def test_converged_chains_near_one(self):
        traces = np.array([
            [2.0, 2.01, 2.02, 2.01, 2.0],
            [2.01, 2.02, 2.0, 2.01, 2.01],
        ])
        rhat = _gelman_rubin_rhat(traces)
        assert np.isfinite(rhat)
        assert rhat < 1.1

    def test_diverged_chains_high_rhat(self):
        traces = np.array([
            [1.0, 1.1, 1.2, 1.3, 1.4],
            [10.0, 10.1, 10.2, 10.3, 10.4],
        ])
        rhat = _gelman_rubin_rhat(traces)
        assert rhat > 2.0

    def test_single_chain_returns_nan(self):
        traces = np.array([[1.0, 2.0, 3.0]])
        assert np.isnan(_gelman_rubin_rhat(traces))

    def test_single_iteration_returns_nan(self):
        traces = np.array([[1.0], [2.0]])
        assert np.isnan(_gelman_rubin_rhat(traces))

    def test_identical_chain_traces_are_not_convergence_evidence(self):
        traces = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        assert np.isnan(_gelman_rubin_rhat(traces))

    def test_identical_nonconstant_traces_are_not_convergence_evidence(self):
        traces = np.array([[1.0, 1.5, 1.2], [1.0, 1.5, 1.2]])
        assert np.isnan(_gelman_rubin_rhat(traces))


# ---------------------------------------------------------------------------
# mice_convergence
# ---------------------------------------------------------------------------

class TestMiceConvergence:
    def test_basic_output_keys(self, simple_histories):
        result = mice_convergence(simple_histories)
        assert set(result.keys()) == {
            "rhat", "trace_data", "converged_by_variable",
            "converged", "n_chains", "n_iter_used",
        }

    def test_n_chains_correct(self, three_chain_histories):
        result = mice_convergence(three_chain_histories)
        assert result["n_chains"] == 3

    def test_rhat_finite_for_well_behaved_chains(self, three_chain_histories):
        result = mice_convergence(three_chain_histories)
        for var, rhat in result["rhat"].items():
            assert np.isfinite(rhat), f"R-hat for {var!r} is not finite"

    def test_converged_flag_true_for_close_chains(self, simple_histories):
        result = mice_convergence(simple_histories)
        assert result["converged"] is True

    def test_not_converged_flag_for_diverged_chains(self):
        diverged = [
            {"a": [1.0, 1.1, 1.2, 1.3, 1.4]},
            {"a": [10.0, 10.1, 10.2, 10.3, 10.4]},
        ]
        result = mice_convergence(diverged)
        assert result["converged"] is False
        assert result["converged_by_variable"]["a"] is False

    def test_identical_histories_are_not_marked_converged(self):
        histories = [
            {"a": [1.0, 1.5, 1.2]},
            {"a": [1.0, 1.5, 1.2]},
        ]
        result = mice_convergence(histories)
        assert np.isnan(result["rhat"]["a"])
        assert result["converged_by_variable"]["a"] is False
        assert result["converged"] is False

    def test_fewer_than_two_chains_raises(self):
        with pytest.raises(ValueError, match="at least 2 chains"):
            mice_convergence([{"a": [1.0, 2.0, 3.0]}])

    def test_empty_histories_raises(self):
        with pytest.raises(ValueError, match="at least 2 chains"):
            mice_convergence([])

    def test_variables_filter(self, three_chain_histories):
        result = mice_convergence(three_chain_histories, variables=["a"])
        assert "a" in result["rhat"]
        assert "b" not in result["rhat"]

    def test_unknown_variable_ignored(self, three_chain_histories):
        result = mice_convergence(three_chain_histories, variables=["nonexistent"])
        assert result["rhat"] == {}
        assert result["converged"] is None

    def test_tuple_histories_from_public_contract_are_accepted(self):
        """Tuple iteration histories produce a real convergence diagnostic."""
        histories = [
            {"a": (1.0, 1.1, 1.2, 1.3)},
            {"a": (8.0, 8.1, 8.2, 8.3)},
        ]

        result = mice_convergence(histories)

        assert "a" in result["rhat"]
        assert result["converged"] is False

    def test_trace_data_shape(self, three_chain_histories):
        result = mice_convergence(three_chain_histories)
        for var, chains in result["trace_data"].items():
            assert len(chains) == 3
            for chain_name, values in chains.items():
                assert len(values) == result["n_iter_used"][var]

    def test_mismatched_iteration_lengths_uses_minimum(self):
        histories = [
            {"a": [1.0, 1.1, 1.2, 1.3]},       # 4 iterations
            {"a": [1.0, 1.05, 1.08]},            # 3 iterations
        ]
        result = mice_convergence(histories)
        assert result["n_iter_used"]["a"] == 3

    def test_non_dict_history_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            mice_convergence([[1.0, 2.0], [1.1, 2.1]])

    def test_rhat_threshold_respected(self):
        diverged = [
            {"a": [1.0, 1.1, 1.2, 1.3, 1.4]},
            {"a": [10.0, 10.1, 10.2, 10.3, 10.4]},
        ]
        result_strict = mice_convergence(diverged, rhat_threshold=1.01)
        result_lenient = mice_convergence(diverged, rhat_threshold=100.0)
        assert result_strict["converged"] is False
        assert result_lenient["converged"] is True


# ---------------------------------------------------------------------------
# plot_mice_convergence — smoke tests (no display)
# ---------------------------------------------------------------------------

class TestPlotMiceConvergence:
    def test_smoke_no_error(self, simple_histories, monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)
        result = mice_convergence(simple_histories)
        plot_mice_convergence(result)

    def test_empty_result_prints_message(self, capsys):
        empty_result = {"trace_data": {}, "rhat": {}}
        plot_mice_convergence(empty_result)
        captured = capsys.readouterr()
        assert "No variables" in captured.out

    def test_variable_filter_in_plot(self, three_chain_histories, monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)
        result = mice_convergence(three_chain_histories)
        plot_mice_convergence(result, variables=["a"])


# ---------------------------------------------------------------------------
# Integration: accessor path
# ---------------------------------------------------------------------------

class TestAccessorMiceConvergence:
    def test_accessor_returns_same_as_direct_call(self, simple_histories):
        import missingly  # noqa: F401 — registers accessor
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan]})
        direct = mice_convergence(simple_histories)
        via_accessor = df.miss.mice_convergence(simple_histories)
        assert direct["n_chains"] == via_accessor["n_chains"]
        assert set(direct["rhat"].keys()) == set(via_accessor["rhat"].keys())
