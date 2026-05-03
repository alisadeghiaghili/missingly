"""Pandas DataFrame accessor for the missingly library.

Registers the ``miss`` namespace on :class:`pandas.DataFrame` via
:func:`pandas.api.extensions.register_dataframe_accessor`.  This
allows the entire missingly API to be used as fluent method chains:

.. code-block:: python

    import pandas as pd
    import missingly  # noqa: F401 — import registers the accessor

    cleaned = (
        df
        .miss.replace_with_na({'score': -99, 'label': 'N/A'})
        .miss.remove_empty(thresh_col=0.9)
        .miss.clean_names()
        .miss.miss_as_feature()
    )

    # Visualisation — returns the Axes, not the DataFrame
    ax = df.miss.vis_miss()
    summary = df.miss.miss_var_summary()

Design notes
------------
* Manipulation methods (``replace_with_na``, ``remove_empty``, …)
  return the **transformed DataFrame** so they are chainable.
* Summary / stats methods (``n_miss``, ``miss_var_summary``, …)
  return their natural output type (scalar, DataFrame, dict, …).
* Visualisation methods return the matplotlib Axes object (or dict
  of Axes for multi-panel plots such as ``upset``).
* The accessor never mutates ``self._df``; every manipulation method
  works on a copy.

Compatibility
-------------
Compatible with Python 3.9+.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from . import (
    manipulation,
    summary,
    stats,
    visualise,
)
from .impute import (
    impute_mean,
    impute_median,
    impute_mode,
    impute_knn,
    impute_mice,
    impute_rf,
    impute_gb,
)


@pd.api.extensions.register_dataframe_accessor("miss")
class MissinglyAccessor:
    """Accessor registered under ``DataFrame.miss``.

    Exposes every public missingly function as a method on any
    :class:`pandas.DataFrame`.  Manipulation methods return a new
    DataFrame (enabling chaining); all other methods return their
    natural output type.

    Parameters
    ----------
    pandas_obj : pd.DataFrame
        The DataFrame instance being accessed.  Injected automatically
        by pandas when the accessor is first used.

    Raises
    ------
    TypeError
        If *pandas_obj* is not a :class:`pandas.DataFrame`.

    Example
    -------
    >>> import pandas as pd, numpy as np, missingly
    >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [None, 2, 3]})
    >>> df.miss.n_miss()
    2
    >>> df.miss.replace_with_na({'A': -99}).miss.remove_empty()
         A    B
    0  1.0  NaN
    2  3.0  3.0
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """Initialise the accessor and validate the wrapped object."""
        if not isinstance(pandas_obj, pd.DataFrame):
            raise TypeError(
                f"MissinglyAccessor requires a DataFrame; got {type(pandas_obj)!r}"
            )
        self._df = pandas_obj

    # ------------------------------------------------------------------
    # Manipulation — return DataFrame for chaining
    # ------------------------------------------------------------------

    def replace_with_na(
        self,
        replace: Dict[str, Union[List, object, Callable]],
    ) -> pd.DataFrame:
        """Replace specified values with NaN and return a new DataFrame.

        Parameters
        ----------
        replace : dict
            Passed directly to :func:`missingly.manipulation.replace_with_na`.

        Returns
        -------
        pd.DataFrame
            Transformed copy of the wrapped DataFrame.
        """
        return manipulation.replace_with_na(self._df, replace)

    def replace_with_na_all(self, condition: Callable) -> pd.DataFrame:
        """Replace all values matching *condition* with NaN.

        Parameters
        ----------
        condition : callable
            Passed directly to
            :func:`missingly.manipulation.replace_with_na_all`.

        Returns
        -------
        pd.DataFrame
            Transformed copy of the wrapped DataFrame.
        """
        return manipulation.replace_with_na_all(self._df, condition)

    def clean_names(
        self,
        *,
        case: str = "lower",
        sep: str = "_",
        strip_accents: bool = False,
    ) -> pd.DataFrame:
        """Normalise column names and return a new DataFrame.

        Parameters
        ----------
        case : str, optional
            Passed to :func:`missingly.manipulation.clean_names`.
        sep : str, optional
            Passed to :func:`missingly.manipulation.clean_names`.
        strip_accents : bool, optional
            Passed to :func:`missingly.manipulation.clean_names`.

        Returns
        -------
        pd.DataFrame
            DataFrame with cleaned column names.
        """
        return manipulation.clean_names(
            self._df, case=case, sep=sep, strip_accents=strip_accents
        )

    def remove_empty(
        self,
        *,
        axis: Union[str, int] = "both",
        missing_values: Optional[List] = None,
        thresh_row: Optional[float] = None,
        thresh_col: Optional[float] = None,
    ) -> pd.DataFrame:
        """Drop empty rows/columns and return a new DataFrame.

        Parameters
        ----------
        axis, missing_values, thresh_row, thresh_col
            Passed to :func:`missingly.manipulation.remove_empty`.

        Returns
        -------
        pd.DataFrame
            Filtered copy of the wrapped DataFrame.
        """
        return manipulation.remove_empty(
            self._df,
            axis=axis,
            missing_values=missing_values,
            thresh_row=thresh_row,
            thresh_col=thresh_col,
        )

    def coalesce_columns(
        self,
        target: str,
        *donors: str,
        remove_donors: bool = False,
    ) -> pd.DataFrame:
        """Fill missing values in *target* from donor columns.

        Parameters
        ----------
        target, *donors, remove_donors
            Passed to :func:`missingly.manipulation.coalesce_columns`.

        Returns
        -------
        pd.DataFrame
            Transformed copy of the wrapped DataFrame.
        """
        return manipulation.coalesce_columns(
            self._df, target, *donors, remove_donors=remove_donors
        )

    def miss_as_feature(
        self,
        columns: Optional[List[str]] = None,
        *,
        missing_values: Optional[List] = None,
        suffix: str = "_NA",
        keep_original: bool = True,
    ) -> pd.DataFrame:
        """Add binary missingness indicator columns and return a new DataFrame.

        Parameters
        ----------
        columns, missing_values, suffix, keep_original
            Passed to :func:`missingly.manipulation.miss_as_feature`.

        Returns
        -------
        pd.DataFrame
            DataFrame with indicator columns appended.
        """
        return manipulation.miss_as_feature(
            self._df,
            columns,
            missing_values=missing_values,
            suffix=suffix,
            keep_original=keep_original,
        )

    # ------------------------------------------------------------------
    # Imputation — return DataFrame for chaining
    # ------------------------------------------------------------------

    def impute_mean(self, **kwargs) -> pd.DataFrame:
        """Impute missing values with column means.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`missingly.impute.impute_mean`.

        Returns
        -------
        pd.DataFrame
            Imputed copy of the wrapped DataFrame.
        """
        return impute_mean(self._df, **kwargs)

    def impute_median(self, **kwargs) -> pd.DataFrame:
        """Impute missing values with column medians.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`missingly.impute.impute_median`.

        Returns
        -------
        pd.DataFrame
        """
        return impute_median(self._df, **kwargs)

    def impute_mode(self, **kwargs) -> pd.DataFrame:
        """Impute missing values with column modes.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`missingly.impute.impute_mode`.

        Returns
        -------
        pd.DataFrame
        """
        return impute_mode(self._df, **kwargs)

    def impute_knn(self, **kwargs) -> pd.DataFrame:
        """Impute missing values using k-nearest neighbours.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`missingly.impute.impute_knn`.

        Returns
        -------
        pd.DataFrame
        """
        return impute_knn(self._df, **kwargs)

    def impute_mice(self, **kwargs) -> pd.DataFrame:
        """Impute missing values using MICE (iterative imputer).

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`missingly.impute.impute_mice`.

        Returns
        -------
        pd.DataFrame
        """
        return impute_mice(self._df, **kwargs)

    def impute_rf(self, **kwargs) -> pd.DataFrame:
        """Impute missing values using a random-forest model.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`missingly.impute.impute_rf`.

        Returns
        -------
        pd.DataFrame
        """
        return impute_rf(self._df, **kwargs)

    def impute_gb(self, **kwargs) -> pd.DataFrame:
        """Impute missing values using a gradient-boosting model.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`missingly.impute.impute_gb`.

        Returns
        -------
        pd.DataFrame
        """
        return impute_gb(self._df, **kwargs)

    # ------------------------------------------------------------------
    # Summary — return natural output type (not DataFrame)
    # ------------------------------------------------------------------

    def n_miss(self) -> int:
        """Return total number of missing values.

        Returns
        -------
        int
        """
        return summary.n_miss(self._df)

    def n_complete(self) -> int:
        """Return total number of non-missing values.

        Returns
        -------
        int
        """
        return summary.n_complete(self._df)

    def pct_miss(self) -> float:
        """Return percentage of missing values across the whole DataFrame.

        Returns
        -------
        float
        """
        return summary.pct_miss(self._df)

    def pct_complete(self) -> float:
        """Return percentage of complete (non-missing) values.

        Returns
        -------
        float
        """
        return summary.pct_complete(self._df)

    def miss_var_summary(
        self, missing_values: Optional[List] = None
    ) -> pd.DataFrame:
        """Return per-variable missing value summary.

        Parameters
        ----------
        missing_values : list, optional
            Sentinel values treated as missing.

        Returns
        -------
        pd.DataFrame
        """
        return summary.miss_var_summary(self._df, missing_values)

    def miss_case_summary(
        self, missing_values: Optional[List] = None
    ) -> pd.DataFrame:
        """Return per-case (row) missing value summary.

        Parameters
        ----------
        missing_values : list, optional
            Sentinel values treated as missing.

        Returns
        -------
        pd.DataFrame
        """
        return summary.miss_case_summary(self._df, missing_values)

    def bind_shadow(self, missing_values: Optional[List] = None) -> pd.DataFrame:
        """Return the DataFrame with a shadow matrix appended.

        Parameters
        ----------
        missing_values : list, optional
            Sentinel values treated as missing.

        Returns
        -------
        pd.DataFrame
        """
        return summary.bind_shadow(self._df, missing_values)

    # ------------------------------------------------------------------
    # Stats — return natural output type
    # ------------------------------------------------------------------

    def mcar_test(self) -> dict:
        """Run Little's MCAR test.

        Returns
        -------
        dict
            Test result with keys ``statistic``, ``df``, ``p_value``.
        """
        return stats.mcar_test(self._df)

    def mar_mnar_test(
        self, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Test for MAR / MNAR patterns using logistic regression.

        Parameters
        ----------
        target_col : str, optional
            Passed to :func:`missingly.stats.mar_mnar_test`.

        Returns
        -------
        pd.DataFrame
        """
        if target_col is not None:
            return stats.mar_mnar_test(self._df, target_col)
        return stats.mar_mnar_test(self._df)

    # ------------------------------------------------------------------
    # Visualisation — return Axes (or dict of Axes)
    # ------------------------------------------------------------------

    def matrix(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a missingness matrix plot.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.matrix(self._df, ax=ax, missing_values=missing_values, **kwargs)

    def bar(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a bar chart of missing counts per column.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.bar(self._df, ax=ax, missing_values=missing_values, **kwargs)

    def upset(self, missing_values: Optional[List] = None, **kwargs):
        """Render an UpSet plot of missing value combinations.

        Returns
        -------
        dict
        """
        return visualise.upset(self._df, missing_values=missing_values, **kwargs)

    def heatmap(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a nullity-correlation heatmap.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.heatmap(self._df, ax=ax, missing_values=missing_values, **kwargs)

    def vis_miss(
        self,
        ax=None,
        missing_values: Optional[List] = None,
        show_pct: bool = True,
        cluster: bool = False,
        **kwargs,
    ):
        """Render an annotated missingness overview matrix.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.vis_miss(
            self._df,
            ax=ax,
            missing_values=missing_values,
            show_pct=show_pct,
            cluster=cluster,
            **kwargs,
        )

    def miss_var_pct(
        self,
        ax=None,
        missing_values: Optional[List] = None,
        sort: bool = True,
        **kwargs,
    ):
        """Render a horizontal bar chart of missingness % per variable.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.miss_var_pct(
            self._df, ax=ax, missing_values=missing_values, sort=sort, **kwargs
        )

    def miss_cluster(
        self,
        ax=None,
        missing_values: Optional[List] = None,
        method: str = "ward",
        **kwargs,
    ):
        """Render a clustered missingness heatmap.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.miss_cluster(
            self._df, ax=ax, missing_values=missing_values, method=method, **kwargs
        )

    def miss_which(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a binary tile plot showing which columns have missing data.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.miss_which(
            self._df, ax=ax, missing_values=missing_values, **kwargs
        )

    def scatter_miss(
        self,
        x: str,
        y: str,
        ax=None,
        missing_values: Optional[List] = None,
        **kwargs,
    ):
        """Render a scatter plot highlighting missing values.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.scatter_miss(
            self._df, x=x, y=y, ax=ax, missing_values=missing_values, **kwargs
        )

    def miss_case(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a bar chart of missing values per row.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.miss_case(
            self._df, ax=ax, missing_values=missing_values, **kwargs
        )

    def vis_miss_fct(
        self,
        fct: str,
        ax=None,
        missing_values: Optional[List] = None,
        **kwargs,
    ):
        """Render missingness by a categorical factor variable.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.vis_miss_fct(
            self._df, fct=fct, ax=ax, missing_values=missing_values, **kwargs
        )

    def vis_miss_cumsum_var(
        self, ax=None, missing_values: Optional[List] = None, **kwargs
    ):
        """Render cumulative missing count per variable.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.vis_miss_cumsum_var(
            self._df, ax=ax, missing_values=missing_values, **kwargs
        )

    def vis_miss_cumsum_case(
        self, ax=None, missing_values: Optional[List] = None, **kwargs
    ):
        """Render cumulative missing count per row.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.vis_miss_cumsum_case(
            self._df, ax=ax, missing_values=missing_values, **kwargs
        )

    def vis_miss_span(
        self,
        column: str,
        span: int,
        ax=None,
        missing_values: Optional[List] = None,
        **kwargs,
    ):
        """Render rolling missing count for a single variable.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.vis_miss_span(
            self._df,
            column=column,
            span=span,
            ax=ax,
            missing_values=missing_values,
            **kwargs,
        )

    def vis_parallel_coords(self, missing_values: Optional[List] = None, **kwargs):
        """Render a parallel coordinates missingness plot.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.vis_parallel_coords(
            self._df, missing_values=missing_values, **kwargs
        )

    def dendrogram(
        self,
        ax=None,
        missing_values: Optional[List] = None,
        method: str = "ward",
        **kwargs,
    ):
        """Render a dendrogram clustering variables by nullity correlation.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.dendrogram(
            self._df, ax=ax, missing_values=missing_values, method=method, **kwargs
        )
