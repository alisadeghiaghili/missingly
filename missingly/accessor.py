"""Pandas DataFrame accessor for the missingly library.

Registers the ``miss`` namespace on :class:`pandas.DataFrame` via
:func:`pandas.api.extensions.register_dataframe_accessor`.  This
allows the entire missingly API to be used as fluent method chains::

    import pandas as pd
    import missingly  # noqa: F401 — import registers the accessor

    cleaned = (
        df
        .miss.replace_with_na({'score': -99, 'label': 'N/A'})
        .miss.miss_as_feature()
    )

    # Unified imputation — delegates to MissinglyImputer
    imputed = df.miss.impute(method="mean")
    imputed = df.miss.impute(method="knn", columns=["age", "income"])

    # Visualisation — returns Axes, not DataFrame
    ax = df.miss.vis_miss()
    summary = df.miss.miss_var_summary()

Design notes
------------
* Manipulation / imputation methods return the **transformed DataFrame**
  so they are chainable.
* Summary / stats methods return their natural output type
  (scalar, DataFrame, dict, …).
* Visualisation methods return the matplotlib Axes object (or dict of
  Axes for multi-panel plots such as ``upset``).
* The accessor never mutates ``self._df``; every manipulation method
  works on a copy.
* ``miss.impute`` is a thin façade over
  :class:`~missingly.transformer.MissinglyImputer`.  No imputation
  algorithm is implemented inside the accessor.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd

from . import (
    diagnostics,
    manipulation,
    visualise,
)
from .exceptions import MissingColumnError
from .impute import (
    impute_mean,
    impute_median,
    impute_mode,
    impute_knn,
    impute_mice,
    impute_rf,
    impute_gb,
)

_VALID_IMPUTE_METHODS = frozenset(
    {"mean", "median", "mode", "knn", "mice", "rf", "gb", "constant"}
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
    >>> imputed = df.miss.impute(method="mean")
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """Initialise the accessor and validate the wrapped object."""
        if not isinstance(pandas_obj, pd.DataFrame):
            raise TypeError(
                f"MissinglyAccessor requires a DataFrame; got {type(pandas_obj)!r}"
            )
        self._df = pandas_obj

    # ------------------------------------------------------------------
    # Unified imputation façade
    # ------------------------------------------------------------------

    def impute(
        self,
        method: str = "mean",
        *,
        columns: Optional[List[str]] = None,
        fill_value: object = None,
        inplace: bool = False,
        random_state: int = 0,
    ) -> Optional[pd.DataFrame]:
        """Impute missing values using a named strategy.

        This method is a thin façade over
        :class:`~missingly.transformer.MissinglyImputer`.  No imputation
        algorithm is implemented here.

        Parameters
        ----------
        method : str, default ``"mean"``
            Imputation strategy.  One of:

            * ``"mean"``     — fill numeric with mean, categorical with mode.
            * ``"median"``   — fill numeric with median, categorical with mode.
            * ``"mode"``     — fill all columns with most frequent value.
            * ``"knn"``      — k-Nearest Neighbours (k=5 by default).
            * ``"mice"``     — Multiple Imputation by Chained Equations.
            * ``"rf"``       — Random Forest.
            * ``"gb"``       — Gradient Boosting.
            * ``"constant"`` — fill with *fill_value* (must be supplied).

        columns : list of str, optional
            Subset of columns to impute.  Untouched columns are copied
            verbatim.  If *None*, all columns are imputed.
        fill_value : scalar, optional
            Value used when ``method="constant"``.  Required for that
            strategy; ignored for all others.
        inplace : bool, default False
            * ``False`` — return a new DataFrame; leave caller unchanged.
            * ``True``  — mutate the caller DataFrame in place and return
              ``None``.
        random_state : int, default 0
            Random seed forwarded to stochastic strategies (knn, mice,
            rf, gb).  Has no effect on deterministic strategies.

        Returns
        -------
        pd.DataFrame or None
            New imputed DataFrame when *inplace* is ``False``.
            ``None`` when *inplace* is ``True``.

        Raises
        ------
        ValueError
            If *method* is not one of the supported strategies.
        ValueError
            If ``method="constant"`` and *fill_value* is ``None``.
        MissingColumnError
            If any name in *columns* is not present in the DataFrame.
        TypeError
            If the wrapped object is not a :class:`pandas.DataFrame`.

        Examples
        --------
        >>> import pandas as pd, numpy as np, missingly
        >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"]})
        >>> df.miss.impute(method="mean")
             a  b
        0  1.0  x
        1  2.0  x
        2  3.0  z
        >>> df.miss.impute(method="constant", fill_value=0)
             a  b
        0  1.0  x
        1  0.0  0
        2  3.0  z
        """
        if method not in _VALID_IMPUTE_METHODS:
            raise ValueError(
                f"Unknown imputation method {method!r}. "
                f"Valid methods: {sorted(_VALID_IMPUTE_METHODS)}"
            )

        if method == "constant" and fill_value is None:
            raise ValueError(
                "method='constant' requires fill_value to be provided. "
                "Example: df.miss.impute(method='constant', fill_value=0)"
            )

        # Validate requested columns
        if columns is not None:
            bad = [c for c in columns if c not in self._df.columns]
            if bad:
                raise MissingColumnError(
                    columns=bad,
                    available=list(self._df.columns),
                )

        # Determine working subset
        work_df = self._df[columns].copy() if columns is not None else self._df.copy()

        # Delegate to canonical engine
        if method == "constant":
            imputed_work = work_df.fillna(fill_value)
        else:
            from .transformer import MissinglyImputer
            imputer = MissinglyImputer(
                strategy=method,
                random_state=random_state,
            )
            imputed_work = imputer.fit_transform(work_df)

        # Reconstruct full DataFrame preserving untouched columns
        if columns is not None:
            result = self._df.copy()
            result[columns] = imputed_work
        else:
            result = imputed_work

        # Preserve index name
        result.index.name = self._df.index.name

        if inplace:
            for col in result.columns:
                self._df[col] = result[col]
            return None

        return result

    # ------------------------------------------------------------------
    # Manipulation — return DataFrame for chaining
    # ------------------------------------------------------------------

    def replace_with_na(
        self,
        replace: dict,
    ) -> pd.DataFrame:
        """Replace specified values with NaN and return a new DataFrame.

        Parameters
        ----------
        replace : dict
            Passed directly to :func:`missingly.manipulation.replace_with_na`.

        Returns
        -------
        pd.DataFrame
        """
        return manipulation.replace_with_na(self._df, replace)

    def replace_with_na_all(self, condition: object) -> pd.DataFrame:
        """Replace all values matching *condition* with NaN.

        Parameters
        ----------
        condition : callable

        Returns
        -------
        pd.DataFrame
        """
        return manipulation.replace_with_na_all(self._df, condition)  # type: ignore[arg-type]

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
        columns : list of str, optional
        missing_values : list, optional
        suffix : str, optional
        keep_original : bool, optional

        Returns
        -------
        pd.DataFrame
        """
        return manipulation.miss_as_feature(
            self._df,
            columns,
            missing_values=missing_values,
            suffix=suffix,
            keep_original=keep_original,
        )

    # ------------------------------------------------------------------
    # Imputation — typed wrappers, return DataFrame for chaining
    # ------------------------------------------------------------------

    def impute_mean(
        self,
        numeric_only: bool = False,
    ) -> pd.DataFrame:
        """Impute missing values with column means.

        Parameters
        ----------
        numeric_only : bool, default False
            If True, only numeric columns are imputed.

        Returns
        -------
        pd.DataFrame
        """
        return impute_mean(self._df, numeric_only=numeric_only)

    def impute_median(
        self,
        numeric_only: bool = False,
    ) -> pd.DataFrame:
        """Impute missing values with column medians.

        Parameters
        ----------
        numeric_only : bool, default False
            If True, only numeric columns are imputed.

        Returns
        -------
        pd.DataFrame
        """
        return impute_median(self._df, numeric_only=numeric_only)

    def impute_mode(self) -> pd.DataFrame:
        """Impute missing values with column modes (most frequent values).

        Returns
        -------
        pd.DataFrame
        """
        return impute_mode(self._df)

    def impute_knn(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
    ) -> pd.DataFrame:
        """Impute missing values using k-nearest neighbours.

        Parameters
        ----------
        n_neighbors : int, default 5
        metric : {"euclidean", "mixed"}, default "euclidean"

        Returns
        -------
        pd.DataFrame
        """
        return impute_knn(self._df, n_neighbors=n_neighbors, metric=metric)

    def impute_mice(
        self,
        max_iter: int = 10,
        random_state: int = 0,
        n_imputations: int = 1,
    ) -> pd.DataFrame:
        """Impute missing values using MICE (iterative imputer).

        Parameters
        ----------
        max_iter : int, default 10
        random_state : int, default 0
        n_imputations : int, default 1

        Returns
        -------
        pd.DataFrame
        """
        return impute_mice(  # type: ignore[return-value]
            self._df,
            max_iter=max_iter,
            random_state=random_state,
            n_imputations=n_imputations,
        )

    def impute_rf(
        self,
        max_iter: int = 1,
        random_state: int = 0,
        strict_mode: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Impute missing values using a random-forest model.

        Parameters
        ----------
        max_iter : int, default 1
        random_state : int, default 0
        strict_mode : bool or None, default None

        Returns
        -------
        pd.DataFrame
        """
        return impute_rf(
            self._df,
            max_iter=max_iter,
            random_state=random_state,
            strict_mode=strict_mode,
        )

    def impute_gb(
        self,
        max_iter: int = 1,
        random_state: int = 0,
        strict_mode: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Impute missing values using a gradient-boosting model.

        Parameters
        ----------
        max_iter : int, default 1
        random_state : int, default 0
        strict_mode : bool or None, default None

        Returns
        -------
        pd.DataFrame
        """
        return impute_gb(
            self._df,
            max_iter=max_iter,
            random_state=random_state,
            strict_mode=strict_mode,
        )

    # ------------------------------------------------------------------
    # Summary — return natural output type (not DataFrame)
    # ------------------------------------------------------------------

    def n_miss(self) -> int:
        """Return total number of missing values.

        Returns
        -------
        int
        """
        return diagnostics.n_miss(self._df)

    def n_complete(self) -> int:
        """Return total number of non-missing values.

        Returns
        -------
        int
        """
        return diagnostics.n_complete(self._df)

    def pct_miss(self) -> float:
        """Return percentage of missing values across the whole DataFrame.

        Returns
        -------
        float
        """
        return diagnostics.pct_miss(self._df)

    def pct_complete(self) -> float:
        """Return percentage of complete (non-missing) values.

        Returns
        -------
        float
        """
        return diagnostics.pct_complete(self._df)

    def miss_var_summary(
        self, missing_values: Optional[List] = None
    ) -> pd.DataFrame:
        """Return per-variable missing value summary.

        Parameters
        ----------
        missing_values : list, optional

        Returns
        -------
        pd.DataFrame
        """
        return diagnostics.miss_var_summary(self._df, missing_values)

    def miss_case_summary(
        self, missing_values: Optional[List] = None
    ) -> pd.DataFrame:
        """Return per-case (row) missing value summary.

        Parameters
        ----------
        missing_values : list, optional

        Returns
        -------
        pd.DataFrame
        """
        return diagnostics.miss_case_summary(self._df, missing_values)

    def bind_shadow(self, missing_values: Optional[List] = None) -> pd.DataFrame:
        """Return the DataFrame with a shadow matrix appended.

        Parameters
        ----------
        missing_values : list, optional

        Returns
        -------
        pd.DataFrame
        """
        return diagnostics.bind_shadow(self._df, missing_values)

    # ------------------------------------------------------------------
    # Stats — return natural output type
    # ------------------------------------------------------------------

    def mcar_test(self) -> dict:
        """Run Little's MCAR test.

        Returns
        -------
        dict
        """
        return diagnostics.mcar_test(self._df)

    def mar_mnar_test(self, target: str) -> pd.DataFrame:
        """Test for MAR / MNAR patterns with respect to a target column.

        Parameters
        ----------
        target : str

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        MissingColumnError
            If *target* is not a column in the wrapped DataFrame.
        """
        if target not in self._df.columns:
            raise MissingColumnError(
                columns=[target],
                available=list(self._df.columns),
            )
        X = self._df.drop(columns=[target])
        Y = self._df[target].to_numpy()
        return diagnostics.mar_mnar_test(X, Y)

    # ------------------------------------------------------------------
    # Visualisation — return Axes (or dict of Axes)
    # ------------------------------------------------------------------

    def matrix(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a missingness matrix plot."""
        return visualise.matrix(self._df, ax=ax, missing_values=missing_values, **kwargs)

    def bar(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a bar chart of missing counts per column."""
        return visualise.bar(self._df, ax=ax, missing_values=missing_values, **kwargs)

    def upset(self, missing_values: Optional[List] = None, **kwargs):
        """Render an UpSet plot of missing value combinations."""
        return visualise.upset(self._df, missing_values=missing_values, **kwargs)

    def heatmap(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a nullity-correlation heatmap."""
        return visualise.heatmap(self._df, ax=ax, missing_values=missing_values, **kwargs)

    def vis_miss(
        self,
        ax=None,
        missing_values: Optional[List] = None,
        show_pct: bool = True,
        cluster: bool = False,
        **kwargs,
    ):
        """Render an annotated missingness overview matrix."""
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
        """Render a horizontal bar chart of missingness % per variable."""
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
        """Render a clustered missingness heatmap."""
        return visualise.miss_cluster(
            self._df, ax=ax, missing_values=missing_values, method=method, **kwargs
        )

    def miss_which(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a binary tile plot showing which columns have missing data."""
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
        """Render a scatter plot highlighting missing values."""
        return visualise.scatter_miss(
            self._df, x=x, y=y, ax=ax, missing_values=missing_values, **kwargs
        )

    def miss_case(self, ax=None, missing_values: Optional[List] = None, **kwargs):
        """Render a bar chart of missing values per row."""
        return visualise.miss_case(
            self._df, ax=ax, missing_values=missing_values, **kwargs
        )

    def vis_impute_dist(
        self,
        imputed_df: pd.DataFrame,
        ax=None,
        **kwargs,
    ):
        """Compare distributions before and after imputation."""
        return visualise.vis_impute_dist(self._df, imputed_df, ax=ax, **kwargs)

    def vis_miss_fct(
        self,
        fct: str,
        ax=None,
        missing_values: Optional[List] = None,
        **kwargs,
    ):
        """Render missingness by a categorical factor variable."""
        return visualise.vis_miss_fct(
            self._df, fct=fct, ax=ax, missing_values=missing_values, **kwargs
        )

    def vis_miss_cumsum_var(
        self, ax=None, missing_values: Optional[List] = None, **kwargs
    ):
        """Render cumulative missing count per variable."""
        return visualise.vis_miss_cumsum_var(
            self._df, ax=ax, missing_values=missing_values, **kwargs
        )

    def vis_miss_cumsum_case(
        self, ax=None, missing_values: Optional[List] = None, **kwargs
    ):
        """Render cumulative missing count per row."""
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
        """Render rolling missing count for a single variable."""
        return visualise.vis_miss_span(
            self._df,
            column=column,
            span=span,
            ax=ax,
            missing_values=missing_values,
            **kwargs,
        )

    def vis_parallel_coords(self, missing_values: Optional[List] = None, **kwargs):
        """Render a parallel coordinates missingness plot."""
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
        """Render a dendrogram clustering variables by nullity correlation."""
        return visualise.dendrogram(
            self._df, ax=ax, missing_values=missing_values, method=method, **kwargs
        )
