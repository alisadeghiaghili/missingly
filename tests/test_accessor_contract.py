"""Contract and parity tests for the ``df.miss`` accessor (WP-07).

Covers:
- regression tests for the accessor crashes (dendrogram, vis_miss_fct,
  vis_impute_dist, vis_miss_span),
- signature parity between every accessor method and the function it wraps,
- one contract call per accessor wrapper (return type + one semantic property),
- identical handling of a partly missing target in both association tests,
- deprecation of silently ignored parameters.
"""

from __future__ import annotations

import inspect
import sys
import types
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

import missingly  # noqa: E402,F401 - registers the .miss accessor
from missingly import diagnostics, impute, manipulation, visualise  # noqa: E402
from missingly.accessor import MissinglyAccessor  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def frame() -> pd.DataFrame:
    """Numeric frame with missing values, a fully observed factor and target."""
    rng = np.random.default_rng(7)
    data = pd.DataFrame(
        {
            "x1": rng.normal(size=40),
            "x2": rng.normal(size=40),
            "x3": rng.normal(size=40),
            "grp": ["a", "b"] * 20,
            "y": rng.integers(0, 2, size=40),
        }
    )
    data.loc[[1, 5, 9, 13, 17], "x1"] = np.nan
    data.loc[[2, 5, 11, 19, 23], "x2"] = np.nan
    data.loc[[3, 12, 21, 30], "x3"] = np.nan
    return data


# ---------------------------------------------------------------------------
# Regression tests for the crashes
# ---------------------------------------------------------------------------


class TestCrashRegressions:
    def test_dendrogram_does_not_leak_method_into_scipy(self, frame):
        """F-005/F-047: df.miss.dendrogram() must not raise."""
        ax = frame.miss.dendrogram()
        assert isinstance(ax, Axes)
        assert "Dendrogram" in ax.get_title()

    def test_dendrogram_accepts_linkage_method(self, frame):
        """The linkage method is a real, honoured parameter."""
        ax = frame.miss.dendrogram(method="complete")
        assert isinstance(ax, Axes)

    def test_vis_miss_fct_does_not_pass_ax_twice(self, frame):
        """F-006/F-048: df.miss.vis_miss_fct() must not raise."""
        ax = frame.miss.vis_miss_fct("grp")
        assert isinstance(ax, Axes)
        assert "grp" in ax.get_title()

    def test_vis_impute_dist_accepts_documented_call(self, frame):
        """F-072: df.miss.vis_impute_dist(imputed, column) works."""
        imputed = frame.miss.impute_mean()
        ax = frame.miss.vis_impute_dist(imputed, "x1")
        assert isinstance(ax, Axes)
        assert "before vs after" in ax.get_title()

    def test_vis_impute_dist_keeps_imputed_df_keyword(self, frame):
        """The v1 keyword name ``imputed_df`` keeps working."""
        imputed = frame.miss.impute_mean()
        ax = frame.miss.vis_impute_dist(imputed_df=imputed, column="x1")
        assert isinstance(ax, Axes)

    def test_vis_miss_span_default_span(self, frame):
        """F-209/F-210: span is optional and span_every is accepted."""
        default_ax = frame.miss.vis_miss_span("x1")
        every_ax = frame.miss.vis_miss_span("x1", span_every=7)
        alias_ax = frame.miss.vis_miss_span("x1", span=5)
        assert "window=20" in default_ax.get_title()
        assert "window=7" in every_ax.get_title()
        assert "window=5" in alias_ax.get_title()


# ---------------------------------------------------------------------------
# Signature parity
# ---------------------------------------------------------------------------

_FUNCTION_MODULES = (diagnostics, visualise, impute, manipulation)

# Accessor methods that deliberately do not mirror one public function.
NOT_MIRRORED = {
    "impute": "dispatcher over several imputation functions, no single target",
    "clean_names": "wrapped function is a deprecated (*args, **kwargs) shim",
    "remove_empty": "wrapped function is a deprecated (*args, **kwargs) shim",
    "coalesce_columns": "wrapped function is a deprecated (*args, **kwargs) shim",
}

# Deliberate parameter renames: accessor name replaces the wrapped name.
RENAMED = {
    "vis_impute_dist": {"df_after": "imputed_df"},
    "missingness_association_test": {"outcome": "target"},
    "mar_mnar_test": {"Y": "target"},
}

# Accessor methods whose wrapped function takes no leading DataFrame.
NO_FRAME_ARGUMENT = {"mice_convergence"}


def _public_methods() -> list[str]:
    return sorted(
        name
        for name, member in inspect.getmembers(MissinglyAccessor)
        if not name.startswith("_") and callable(member)
    )


def _wrapped_function(name: str):
    for module in _FUNCTION_MODULES:
        function = getattr(module, name, None)
        if callable(function):
            return function
    return None


def _normalised(parameters, rename):
    return [
        (rename.get(p.name, p.name), p.kind, p.default, p.default is p.empty)
        for p in parameters
    ]


class TestSignatureParity:
    def test_every_public_method_is_mirrored_or_allow_listed(self):
        """A method must wrap a public function or be on the allow-list."""
        for name in _public_methods():
            assert name in NOT_MIRRORED or _wrapped_function(name) is not None, name
        assert set(NOT_MIRRORED) <= set(_public_methods())

    @pytest.mark.parametrize(
        "name",
        [n for n in _public_methods() if n not in NOT_MIRRORED],
    )
    def test_accessor_signature_matches_wrapped_function(self, name):
        """Names, kinds, defaults and required-ness match the function."""
        function = _wrapped_function(name)
        wrapped = list(inspect.signature(function).parameters.values())
        if name not in NO_FRAME_ARGUMENT:
            wrapped = wrapped[1:]
        accessor = list(
            inspect.signature(getattr(MissinglyAccessor, name)).parameters.values()
        )[1:]
        assert _normalised(accessor, {}) == _normalised(
            wrapped, RENAMED.get(name, {})
        )

    def test_allow_list_entries_are_documented(self):
        """Every allow-list entry carries a reason."""
        assert all(reason for reason in NOT_MIRRORED.values())
        assert set(RENAMED) <= set(_public_methods())


# ---------------------------------------------------------------------------
# One contract call per wrapper
# ---------------------------------------------------------------------------


def _artists(ax: Axes) -> int:
    return len(ax.patches) + len(ax.lines) + len(ax.images) + len(ax.collections)


@pytest.fixture
def fake_cleaning(monkeypatch):
    """Provide a stand-in ``data_quality_toolkit.cleaning`` module."""
    calls: dict = {}

    def _record(name):
        def _fn(df, *args, **kwargs):
            calls[name] = (args, kwargs)
            return df

        return _fn

    module = types.ModuleType("data_quality_toolkit.cleaning")
    for fn_name in ("clean_names", "remove_empty", "coalesce_columns"):
        setattr(module, fn_name, _record(fn_name))
    package = types.ModuleType("data_quality_toolkit")
    package.cleaning = module
    monkeypatch.setitem(sys.modules, "data_quality_toolkit", package)
    monkeypatch.setitem(sys.modules, "data_quality_toolkit.cleaning", module)
    return calls


class TestWrapperContract:
    def test_replace_with_na(self, frame):
        out = frame.miss.replace_with_na({"x3": [frame["x3"].iloc[0]]})
        assert isinstance(out, pd.DataFrame)
        assert out["x3"].isna().sum() == frame["x3"].isna().sum() + 1

    def test_replace_with_na_all(self, frame):
        out = frame.miss.replace_with_na_all(lambda v: v == "a")
        assert isinstance(out, pd.DataFrame)
        assert out["grp"].isna().sum() == 20

    def test_shim_wrappers_forward(self, frame, fake_cleaning):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame.miss.clean_names(case="upper")
            frame.miss.remove_empty(thresh_row=0.5)
            frame.miss.coalesce_columns("x1", "x2", remove_donors=True)
        assert fake_cleaning["clean_names"][1]["case"] == "upper"
        assert fake_cleaning["remove_empty"][1]["thresh_row"] == 0.5
        assert fake_cleaning["coalesce_columns"][0] == ("x1", "x2")

    def test_miss_as_feature(self, frame):
        out = frame.miss.miss_as_feature(["x1"])
        assert isinstance(out, pd.DataFrame)
        assert int(out["x1_NA"].sum()) == int(frame["x1"].isna().sum())

    @pytest.mark.parametrize(
        "method",
        [
            "impute_mean",
            "impute_median",
            "impute_mode",
            "impute_knn",
            "impute_mice",
            "impute_rf",
            "impute_gb",
        ],
    )
    def test_imputers(self, frame, method):
        numeric = frame[["x1", "x2", "x3"]]
        out = getattr(numeric.miss, method)()
        assert isinstance(out, pd.DataFrame)
        assert int(out.isna().sum().sum()) == 0

    def test_impute_dispatch(self, frame):
        out = frame[["x1", "x2"]].miss.impute("median")
        assert int(out.isna().sum().sum()) == 0

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("n_miss", lambda f: int(f.isna().sum().sum())),
            ("n_complete", lambda f: int(f.notna().sum().sum())),
            ("pct_miss", lambda f: float(f.isna().to_numpy().mean() * 100)),
            ("pct_complete", lambda f: float(f.notna().to_numpy().mean() * 100)),
        ],
    )
    def test_scalar_summaries(self, frame, method, expected):
        assert getattr(frame.miss, method)() == pytest.approx(expected(frame))

    def test_miss_var_summary(self, frame):
        out = frame.miss.miss_var_summary()
        assert isinstance(out, pd.DataFrame)
        assert len(out) == frame.shape[1]

    def test_miss_case_summary(self, frame):
        out = frame.miss.miss_case_summary()
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(frame)

    def test_bind_shadow(self, frame):
        out = frame.miss.bind_shadow()
        assert isinstance(out, pd.DataFrame)
        assert out.shape[1] == 2 * frame.shape[1]

    def test_mcar_test(self, frame):
        out = frame[["x1", "x2", "x3"]].miss.mcar_test()
        assert isinstance(out, dict)
        assert 0.0 <= out["p_value"] <= 1.0

    def test_mcar_test_forwards_tuning_and_missing_values(self, frame):
        numeric = frame[["x1", "x2", "x3"]]
        coded = numeric.fillna(-99.0)
        expected = diagnostics.mcar_test(
            coded, max_iter=3, tol=1e-3, ridge=1e-6, missing_values=[-99.0]
        )
        got = coded.miss.mcar_test(
            max_iter=3, tol=1e-3, ridge=1e-6, missing_values=[-99.0]
        )
        assert got["p_value"] == pytest.approx(expected["p_value"])
        assert got["em_iterations"] == expected["em_iterations"]

    def test_missingness_association_test(self, frame):
        out = frame[["x1", "x2", "x3", "y"]].miss.missingness_association_test("y")
        assert isinstance(out, pd.DataFrame)
        assert set(out["feature"]) == {"x1", "x2", "x3"}

    def test_mar_mnar_test(self, frame):
        with pytest.warns(FutureWarning):
            out = frame[["x1", "x2", "x3", "y"]].miss.mar_mnar_test("y")
        assert isinstance(out, list)
        assert {row[0] for row in out} == {"x1", "x2", "x3"}

    def test_association_tests_forward_missing_values(self, frame):
        coded = frame[["x1", "x2", "x3", "y"]].fillna(-99.0)
        got = coded.miss.missingness_association_test("y", missing_values=[-99.0])
        expected = diagnostics.missingness_association_test(
            coded.drop(columns="y"), coded["y"].to_numpy(), missing_values=[-99.0]
        )
        pd.testing.assert_frame_equal(got, expected)

    def test_mice_convergence(self, frame):
        numeric = frame[["x1", "x2", "x3"]]
        _, histories = impute.impute_mice(
            numeric, n_imputations=3, max_iter=4, return_history=True
        )
        out = numeric.miss.mice_convergence(histories)
        assert isinstance(out, dict)
        assert out["n_chains"] == 3

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("matrix", ()),
            ("bar", ()),
            ("heatmap", ()),
            ("vis_miss", ()),
            ("miss_var_pct", ()),
            ("miss_cluster", ()),
            ("miss_which", ()),
            ("scatter_miss", ("x1", "x2")),
            ("miss_case", ()),
            ("vis_miss_fct", ("grp",)),
            ("vis_miss_cumsum_var", ()),
            ("vis_miss_cumsum_case", ()),
            ("vis_miss_span", ("x1",)),
            ("vis_parallel_coords", ()),
            ("dendrogram", ()),
        ],
    )
    def test_axes_visualisations(self, frame, method, args):
        ax = getattr(frame.miss, method)(*args)
        assert isinstance(ax, Axes)
        assert _artists(ax) > 0

    def test_vis_impute_dist(self, frame):
        ax = frame.miss.vis_impute_dist(frame.miss.impute_mean(), "x1")
        assert isinstance(ax, Axes)
        assert _artists(ax) > 0

    def test_upset(self, frame):
        out = frame.miss.upset()
        assert isinstance(out, dict)

    def test_vis_miss_forwards_figsize(self, frame):
        ax = frame.miss.vis_miss(figsize=(5, 3))
        assert tuple(ax.figure.get_size_inches()) == (5.0, 3.0)


# ---------------------------------------------------------------------------
# Association tests: partly missing target
# ---------------------------------------------------------------------------


class TestPartlyMissingTarget:
    def test_both_entry_points_raise_identically(self, frame):
        """F-070/F-206/F-207: both raise the same ValueError."""
        data = frame[["x1", "x2", "y"]].astype({"y": float})
        data.loc[0, "y"] = np.nan
        with pytest.raises(ValueError) as first:
            data.miss.missingness_association_test("y")
        with pytest.warns(FutureWarning):
            with pytest.raises(ValueError) as second:
                data.miss.mar_mnar_test("y")
        assert str(first.value) == str(second.value)
        assert "fully observed" in str(first.value)

    def test_unknown_target_raises_missing_column(self, frame):
        from missingly.exceptions import MissingColumnError

        with pytest.raises(MissingColumnError):
            frame.miss.missingness_association_test("nope")
        with pytest.raises(MissingColumnError):
            frame.miss.mar_mnar_test("nope")


# ---------------------------------------------------------------------------
# Silently ignored parameters are deprecated, not silent
# ---------------------------------------------------------------------------


class TestIgnoredParametersWarn:
    def test_vis_miss_show_pct_warns_on_matplotlib(self, frame):
        with pytest.warns(FutureWarning, match="show_pct"):
            ax = frame.miss.vis_miss(show_pct=False)
        assert isinstance(ax, Axes)

    def test_miss_cluster_method_warns(self, frame):
        with pytest.warns(FutureWarning, match="method"):
            ax = frame.miss.miss_cluster(method="average")
        assert isinstance(ax, Axes)

    def test_no_warning_without_deprecated_arguments(self, frame):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            frame.miss.vis_miss()
            frame.miss.miss_cluster()

    def test_vis_miss_show_pct_is_honoured_by_plotly(self, frame):
        pytest.importorskip("plotly")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = frame.miss.vis_miss(interactive=True, show_pct=False)
        assert hasattr(fig, "data")
