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
        .miss.miss_as_feature()
    )

    # Imputation — returns new DataFrame by default
    imputed = df.miss.impute(method="mean")
    imputed_inplace = df.miss.impute(method="knn", inplace=False)

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
  works on a copy unless ``inplace=True`` is explicitly passed.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

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
    _SUPPORTED_STRATEGIES,
)
from .transformer import MissinglyImputer


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
    >>> imputed.isna().any().any()
    False
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """Initialise the accessor and validate the wrapped object."""
        if not isinstance(pandas_obj, pd.DataFrame):
            raise TypeError(
                f"MissinglyAccessor requires a DataFrame; got {type(pandas_obj)!r}"
            )
        self._df = pandas_obj

    # ------------------------------------------------------------------
    # Imputation — unified dispatcher (C3)
    # ------------------------------------------------------------------

    def impute(
        self,
        method: str = "mean",
        *,
        columns: Optional[List[str]] = None,
        fill_value: Optional[object] = None,
        inplace: bool = False,
        random_state: int = 0,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Impute missing values using any missingly strategy.

        This is a thin facade over :class:`~missingly.transformer.MissinglyImputer`.
        No imputation logic lives here — all computation is delegated to
        the canonical engine so that accessor and transformer results are
        always consistent.

        Parameters
        ----------
        method : str, default ``"mean"``
            Imputation strategy.  One of:
            ``"mean"``, ``"median"``, ``"mode"``, ``"knn"``,
            ``"mice"``, ``"rf"``, ``"gb"``.
        columns : list of str or None, default None
            Columns to impute.  If ``None``, all columns are imputed.
            Untouched columns are preserved as-is.
        fill_value : object or None, default None
            Reserved for future ``"constant"`` strategy.  Unused by
            current strategies; passing it with other methods raises
            ``ValueError``.
        inplace : bool, default False
            If ``False`` (default), return a new DataFrame and leave the
            caller unchanged.  If ``True``, mutate the caller DataFrame
            in-place and return ``None``.
        random_state : int, default 0
            Random seed forwarded to stochastic strategies
            (``"mice"``, ``"rf"``, ``"gb"``).
        **kwargs
            Additional keyword arguments forwarded to
            :class:`~missingly.transformer.MissinglyImputer`
            (e.g. ``n_neighbors=3`` for ``"knn"``).

        Returns
        -------
        pd.DataFrame or None
            New imputed DataFrame when ``inplace=False``;
            ``None`` when ``inplace=True``.

        Raises
        ------
        ValueError
            If *method* is not a recognised strategy.
        MissingColumnError
            If any name in *columns* does not exist in the DataFrame.
        ValueError
            If *fill_value* is supplied with a non-``"constant"`` method.

        Examples
        --------
        >>> import pandas as pd, numpy as np, missingly
        >>> df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': ['x', None, 'z']})
        >>> df.miss.impute(method="mean")  # numeric mean, categorical mode
             a  b
        0  1.0  x
        1  2.0  x
        2  3.0  z
        >>> df.miss.impute(method="mean", columns=["a"])  # only column 'a'
             a     b
        0  1.0     x
        1  2.0  None
        2  3.0     z
        """
        if method not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown imputation method {method!r}. "
                f"Supported methods: {sorted(_SUPPORTED_STRATEGIES)}"
            )

        if fill_value is not None:
            raise ValueError(
                "fill_value is reserved for strategy='constant' which is not "
                "yet implemented. Do not pass fill_value with other strategies."
            )

        df = self._df

        # Validate requested columns
        if columns is not None:
            unknown = [c for c in columns if c not in df.columns]
            if unknown:
                raise MissingColumnError(
                    columns=unknown,
                    available=list(df.columns),
                )

        # Select the sub-frame to impute; keep the rest untouched
        if columns is not None:
            sub = df[columns].copy()
        else:
            sub = df.copy()

        # Delegate entirely to MissinglyImputer — no logic here
        imputer = MissinglyImputer(
            strategy=method,
            random_state=random_state,
            **kwargs,
        )
        imputed_sub = imputer.fit_transform(sub)

        # Reconstruct full DataFrame preserving column order
        if columns is not None:
            result = df.copy()
            result[columns] = imputed_sub[columns].values
        else:
            result = imputed_sub

        # Preserve index
        result.index = df.index

        if inplace:
            for col in df.columns:
                df[col] = result[col]
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

    def replace_with_na_all(self, condition) -> pd.DataFrame:
        """Replace all values matching *condition* with NaN.

        Parameters
        ----------
        condition : callable

        Returns
        -------
        pd.DataFrame
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
        sep : str, optional
        strip_accents : bool, optional

        Returns
        -------
        pd.DataFrame
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
        axis : str or int, optional
        missing_values : list, optional
        thresh_row : float, optional
        thresh_col : float, optional

        Returns
        -------
        pd.DataFrame
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
        """Fill missing values in *target* from donor columns (SQL COALESCE).

        Parameters
        ----------
        target : str
        *donors : str
        remove_donors : bool, optional

        Returns
        -------
        pd.DataFrame
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
    # Imputation wrappers — typed signatures, delegate to impute module
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
        """Impute missing values using k-Nearest Neighbours.

        Parameters
        ----------
        n_neighbors : int, default 5
            Number of nearest neighbours.
        metric : {"euclidean", "mixed"}, default "euclidean"
            Distance metric.

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
            Maximum MICE iterations.
        random_state : int, default 0
            Random seed.
        n_imputations : int, default 1
            Number of imputed datasets to generate.

        Returns
        -------
        pd.DataFrame
        """
        return impute_mice(
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
            Number of imputation passes.
        random_state : int, default 0
            Random seed.
        strict_mode : bool or None, default None
            If True, raise on estimator failure instead of falling back.

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
            Number of imputation passes.
        random_state : int, default 0
            Random seed.
        strict_mode : bool or None, default None
            If True, raise on estimator failure instead of falling back.

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
            Keys: ``chi_square``, ``df``, ``p_value``, ``missing_patterns``,
            ``amount_missing``.
        """
        return diagnostics.mcar_test(self._df)

    def mar_mnar_test(self, target: str) -> pd.DataFrame:
        """Test for MAR / MNAR patterns with respect to a target column.

        Parameters
        ----------
        target : str
            Name of the outcome column.  Must exist in the DataFrame.

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
        Y = self._df[target]
        valid_mask = Y.notna()
        X = X[valid_mask]
        Y = Y[valid_mask].to_numpy()
        return diagnostics.mar_mnar_test(X, Y)

    def mice_convergence(
        self,
        histories: List[Dict[str, List[float]]],
        variables: Optional[List[str]] = None,
        rhat_threshold: float = 1.1,
    ) -> dict:
        """Assess convergence of multiple MICE chains using iteration history.

        Parameters
        ----------
        histories : list of dict
            One history dict per chain as returned by
            ``impute_mice(..., return_history=True)`` in multi-chain mode.
        variables : list of str, optional
            Specific variables to diagnose.
        rhat_threshold : float, default 1.1

        Returns
        -------
        dict
            See :func:`missingly.diagnostics.mice_convergence`.

        Examples
        --------
        >>> import pandas as pd, numpy as np, missingly
        >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan]})
        >>> _, histories = missingly.impute_mice(
        ...     df, n_imputations=3, max_iter=5, return_history=True
        ... )
        >>> result = df.miss.mice_convergence(histories)
        >>> result["n_chains"]
        3
        """
        return diagnostics.mice_convergence(
            histories,
            variables=variables,
            rhat_threshold=rhat_threshold,
        )

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

    def miss_case(
        self, ax=None, missing_values: Optional[List] = None, **kwargs
    ):
        """Render a bar chart of missing values per row.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.miss_case(
            self._df, ax=ax, missing_values=missing_values, **kwargs
        )

    def vis_impute_dist(
        self,
        imputed_df: pd.DataFrame,
        ax=None,
        **kwargs,
    ):
        """Compare distributions before and after imputation.

        Parameters
        ----------
        imputed_df : pd.DataFrame
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        matplotlib.axes.Axes
        """
        return visualise.vis_impute_dist(self._df, imputed_df, ax=ax, **kwargs)

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

    def vis_parallel_coords(
        self, missing_values: Optional[List] = None, **kwargs
    ):
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
