"""Pandas DataFrame accessor for the missingly library.

Registers the ``miss`` namespace on :class:`pandas.DataFrame` via
:func:`pandas.api.extensions.register_dataframe_accessor`.  This
allows the core missingly API to be used as fluent method chains:

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

    # Visualisation — returns Axes, not DataFrame
    ax = df.miss.vis_miss()
    summary = df.miss.miss_var_summary()

Scope
-----
The accessor covers the manipulation helpers, the simple and model-based
imputers (``mean``, ``median``, ``mode``, ``knn``, ``mice``, ``rf``, ``gb``),
the summary and diagnostic functions and the static visualisation
functions.  It does **not** expose the naniar-style API
(:mod:`missingly.naniar`), the remaining imputers (``impute_pmm``,
``impute_logreg``, ``impute_polyreg``, ``impute_polr``, the hot-deck
imputers, ``impute_ts``), the multiple-imputation workflow, the
simulation helpers or the report builder; call those as functions.

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
* Every method that wraps a public function mirrors that function's
  signature (minus the DataFrame argument) so parameters, defaults and
  keyword-only markers cannot drift; ``tests/test_accessor_contract.py``
  enforces this with an explicit allow-list of deliberate differences.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import (
    diagnostics,
    manipulation,
    visualise,
)
from . import impute as _impute
from .exceptions import InvalidStrategyError, MissingColumnError
from .impute import (
    impute_mean,
    impute_median,
    impute_mode,
    impute_knn,
    impute_mice,
    impute_rf,
    impute_gb,
)

_SELF_PARAMETER = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _mirror(
    function: Callable,
    *,
    rename: Optional[Dict[str, str]] = None,
) -> Callable[[Callable], Callable]:
    """Give an accessor method the signature of the function it wraps.

    The wrapped function's first (DataFrame) parameter is replaced by
    ``self``; every other parameter keeps its name, kind and default, except
    for the explicit renames in *rename*.

    Parameters
    ----------
    function : callable
        Public function wrapped by the accessor method.
    rename : dict of str to str, optional
        Deliberate parameter renames, ``{wrapped_name: accessor_name}``.

    Returns
    -------
    callable
        Decorator that sets ``__signature__`` on the accessor method.

    Examples
    --------
    >>> import inspect
    >>> def target(df, *, flag=True): ...
    >>> @_mirror(target)
    ... def method(self, *args, **kwargs): ...
    >>> str(inspect.signature(method))
    '(self, *, flag=True)'
    """
    names = rename or {}

    def decorator(method: Callable) -> Callable:
        signature = inspect.signature(function)
        parameters = [
            parameter.replace(name=names.get(parameter.name, parameter.name))
            for parameter in list(signature.parameters.values())[1:]
        ]
        setattr(
            method,
            "__signature__",
            signature.replace(parameters=[_SELF_PARAMETER, *parameters]),
        )
        return method

    return decorator


def _warn_ignored(kwargs: Dict[str, Any], name: str, method: str, reason: str) -> None:
    """Drop a parameter that has no effect and warn that it is deprecated.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments of the accessor call; *name* is removed in place.
    name : str
        Parameter that is accepted for backwards compatibility only.
    method : str
        Accessor method name used in the warning text.
    reason : str
        Why the parameter has no effect.

    Examples
    --------
    >>> options = {"method": "ward"}
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     _warn_ignored(options, "method", "miss_cluster", "no effect")
    >>> options, caught[0].category.__name__
    ({}, 'FutureWarning')
    """
    if name not in kwargs:
        return
    kwargs.pop(name)
    warnings.warn(
        f"`DataFrame.miss.{method}({name}=...)` is deprecated and has no effect: "
        f"{reason}. The argument will be removed in a future release.",
        FutureWarning,
        stacklevel=3,
    )


@pd.api.extensions.register_dataframe_accessor("miss")
class MissinglyAccessor:
    """Accessor registered under ``DataFrame.miss``.

    Exposes the manipulation, simple/model-based imputation, summary,
    diagnostic and static visualisation functions of missingly as methods on
    any :class:`pandas.DataFrame`.  Manipulation methods return a new
    DataFrame (enabling chaining); all other methods return their natural
    output type.  The naniar-style API and the remaining imputers are not
    exposed here; call the module functions instead.

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
    >>> cleaned = df.miss.replace_with_na({'A': -99}).miss.remove_empty()
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
        """
        return manipulation.replace_with_na(self._df, replace)

    def replace_with_na_all(self, condition: Callable) -> pd.DataFrame:
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

        The implementation lives in ``data_quality_toolkit`` (the module
        function is a deprecated shim), so this wrapper keeps its explicit
        v1 signature.

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

        The implementation lives in ``data_quality_toolkit`` (the module
        function is a deprecated shim), so this wrapper keeps its explicit
        v1 signature.

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

        The implementation lives in ``data_quality_toolkit`` (the module
        function is a deprecated shim), so this wrapper keeps its explicit
        v1 signature.

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
    # Imputation — return DataFrame for chaining
    # ------------------------------------------------------------------

    def impute(self, strategy: str = "mean", **kwargs):
        """Impute missing values with one named Missingly strategy.

        This dispatcher covers only the seven strategies listed below; the
        other imputers (``impute_pmm``, ``impute_logreg``, ``impute_polyreg``,
        ``impute_polr``, hot-deck, ``impute_ts``) are module functions.

        Parameters
        ----------
        strategy : {"mean", "median", "mode", "knn", "mice", "rf", "gb"}, default "mean"
            Imputation algorithm to apply to this accessor's DataFrame.
        **kwargs
            Keyword arguments forwarded unchanged to the selected imputation
            function. For example, ``n_neighbors`` applies to ``"knn"`` and
            ``max_iter`` applies to ``"mice"``.

        Returns
        -------
        pandas.DataFrame or list of pandas.DataFrame or ImputationResult
            The selected function's documented return value. Most strategies
            return one DataFrame; multi-imputation MICE options may return a
            list or typed result.

        Raises
        ------
        InvalidStrategyError
            If ``strategy`` is not supported by the generic accessor.

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> frame = pd.DataFrame({"score": [1.0, None, 3.0]})
        >>> float(frame.miss.impute("median").loc[1, "score"])
        2.0
        """
        strategies: Dict[str, Callable[..., Any]] = {
            "mean": impute_mean,
            "median": impute_median,
            "mode": impute_mode,
            "knn": impute_knn,
            "mice": impute_mice,
            "rf": impute_rf,
            "gb": impute_gb,
        }
        try:
            imputer = strategies[strategy]
        except (KeyError, TypeError) as error:
            raise InvalidStrategyError(
                param="strategy",
                got=strategy,
                allowed=sorted(strategies),
                example="mean",
            ) from error
        return imputer(self._df, **kwargs)

    @_mirror(_impute.impute_mean)
    def impute_mean(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Impute missing values with column means.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.impute.impute_mean` (for
            example ``numeric_only``), without the DataFrame.

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> float(pd.DataFrame({"a": [1.0, None, 3.0]}).miss.impute_mean().loc[1, "a"])
        2.0
        """
        return impute_mean(self._df, *args, **kwargs)

    @_mirror(_impute.impute_median)
    def impute_median(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Impute missing values with column medians.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.impute.impute_median` (for
            example ``numeric_only``), without the DataFrame.

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> frame = pd.DataFrame({"a": [1.0, None, 3.0, 10.0]})
        >>> float(frame.miss.impute_median().loc[1, "a"])
        3.0
        """
        return impute_median(self._df, *args, **kwargs)

    @_mirror(_impute.impute_mode)
    def impute_mode(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Impute missing values with column modes.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.impute.impute_mode`, without
            the DataFrame.

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> frame = pd.DataFrame({"a": ["x", "x", None]})
        >>> frame.miss.impute_mode().loc[2, "a"]
        'x'
        """
        return impute_mode(self._df, *args, **kwargs)

    @_mirror(_impute.impute_knn)
    def impute_knn(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Impute missing values using k-nearest neighbours.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.impute.impute_knn` (for
            example ``n_neighbors``, ``metric``), without the DataFrame.

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> frame = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        >>> int(frame.miss.impute_knn(n_neighbors=2).isna().sum().sum())
        0
        """
        return impute_knn(self._df, *args, **kwargs)

    @_mirror(_impute.impute_mice)
    def impute_mice(self, *args: Any, **kwargs: Any):
        """Impute missing values using MICE (iterative imputer).

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.impute.impute_mice` (for
            example ``max_iter``, ``n_imputations``, ``return_history``),
            without the DataFrame.

        Returns
        -------
        pd.DataFrame or list or tuple or ImputationResult
            As documented for :func:`missingly.impute.impute_mice`.

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> frame = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        >>> int(frame.miss.impute_mice(max_iter=2).isna().sum().sum())
        0
        """
        return impute_mice(self._df, *args, **kwargs)

    @_mirror(_impute.impute_rf)
    def impute_rf(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Impute missing values using a random-forest model.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.impute.impute_rf` (for
            example ``max_iter``, ``random_state``), without the DataFrame.

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> frame = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        >>> int(frame.miss.impute_rf(n_estimators=5).isna().sum().sum())
        0
        """
        return impute_rf(self._df, *args, **kwargs)

    @_mirror(_impute.impute_gb)
    def impute_gb(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Impute missing values using a gradient-boosting model.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.impute.impute_gb` (for
            example ``max_iter``, ``random_state``), without the DataFrame.

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> frame = pd.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        >>> int(frame.miss.impute_gb(n_estimators=5).isna().sum().sum())
        0
        """
        return impute_gb(self._df, *args, **kwargs)

    # ------------------------------------------------------------------
    # Summary — return natural output type (not DataFrame)
    # ------------------------------------------------------------------

    def n_miss(self, missing_values: Optional[List] = None) -> int:
        """Return total number of missing values.

        Parameters
        ----------
        missing_values : list, optional
            Extra sentinel values to count as missing.

        Returns
        -------
        int

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> pd.DataFrame({"a": [1, None, -99]}).miss.n_miss(missing_values=[-99])
        2
        """
        return diagnostics.n_miss(self._df, missing_values)

    def n_complete(self, missing_values: Optional[List] = None) -> int:
        """Return total number of non-missing values.

        Parameters
        ----------
        missing_values : list, optional
            Extra sentinel values to count as missing.

        Returns
        -------
        int

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> pd.DataFrame({"a": [1, None, -99]}).miss.n_complete(missing_values=[-99])
        1
        """
        return diagnostics.n_complete(self._df, missing_values)

    def pct_miss(self, missing_values: Optional[List] = None) -> float:
        """Return percentage of missing values across the whole DataFrame.

        Parameters
        ----------
        missing_values : list, optional
            Extra sentinel values to count as missing.

        Returns
        -------
        float

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> float(pd.DataFrame({"a": [1, None, 3, 4]}).miss.pct_miss())
        25.0
        """
        return diagnostics.pct_miss(self._df, missing_values)

    def pct_complete(self, missing_values: Optional[List] = None) -> float:
        """Return percentage of complete (non-missing) values.

        Parameters
        ----------
        missing_values : list, optional
            Extra sentinel values to count as missing.

        Returns
        -------
        float

        Examples
        --------
        >>> import pandas as pd
        >>> import missingly
        >>> float(pd.DataFrame({"a": [1, None, 3, 4]}).miss.pct_complete())
        75.0
        """
        return diagnostics.pct_complete(self._df, missing_values)

    def miss_var_summary(self, missing_values: Optional[List] = None) -> pd.DataFrame:
        """Return per-variable missing value summary.

        Parameters
        ----------
        missing_values : list, optional

        Returns
        -------
        pd.DataFrame
        """
        return diagnostics.miss_var_summary(self._df, missing_values)

    def miss_case_summary(self, missing_values: Optional[List] = None) -> pd.DataFrame:
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

    def mcar_test(
        self,
        max_iter: int = 500,
        tol: float = 1e-07,
        ridge: float = 1e-08,
        missing_values: Optional[List] = None,
    ) -> dict:
        """Run Little's MCAR test.

        Parameters
        ----------
        max_iter : int, default 500
            Maximum number of EM iterations.
        tol : float, default 1e-07
            EM convergence tolerance.
        ridge : float, default 1e-08
            Ridge added to the covariance matrix for numerical stability.
        missing_values : list, optional
            Extra sentinel values to treat as missing.

        Returns
        -------
        dict
            Keys: ``chi_square``, ``df``, ``p_value``, ``missing_patterns``,
            ``amount_missing``, and ``em_iterations``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import missingly  # noqa: F401
        >>> frame = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, 2.0, np.nan, 4.0]})
        >>> "p_value" in frame.miss.mcar_test(max_iter=100)
        True
        """
        return diagnostics.mcar_test(
            self._df,
            max_iter=max_iter,
            tol=tol,
            ridge=ridge,
            missing_values=missing_values,
        )

    def _split_target(self, target: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Split the frame into feature columns and the target values.

        Parameters
        ----------
        target : str
            Name of the auxiliary target column.

        Returns
        -------
        tuple
            ``(features, target_values)``.

        Raises
        ------
        MissingColumnError
            If *target* is not a column of the wrapped DataFrame.
        """
        if target not in self._df.columns:
            raise MissingColumnError(
                columns=[target],
                available=list(self._df.columns),
            )
        return self._df.drop(columns=[target]), self._df[target].to_numpy()

    def missingness_association_test(
        self, target: str, missing_values: Optional[List] = None
    ) -> pd.DataFrame:
        """Screen observedness associations with a fully observed target column.

        This diagnostic does not identify MAR versus MNAR. It tests whether the
        observedness of each non-target column is associated with the observed target,
        conditional on the remaining supplied columns.

        Parameters
        ----------
        target : str
            Name of the outcome column.  Must exist in the DataFrame and be
            fully observed.
        missing_values : list, optional
            Extra sentinel values to treat as missing in the non-target columns.

        Returns
        -------
        pd.DataFrame
            Named association-screen results from
            :func:`missingly.diagnostics.missingness_association_test`.

        Raises
        ------
        MissingColumnError
            If *target* is not a column in the wrapped DataFrame.
        ValueError
            If *target* has missing values (raised by the wrapped function,
            identically for :meth:`mar_mnar_test`).

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import missingly  # noqa: F401
        >>> frame = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "y": [0, 1, 0, 1]})
        >>> list(frame.miss.missingness_association_test("y").columns)
        ['feature', 'lrt_statistic', 'p_value', 'n_obs']
        """
        features, outcome = self._split_target(target)
        return diagnostics.missingness_association_test(
            features, outcome, missing_values
        )

    def mar_mnar_test(self, target: str, missing_values: Optional[List] = None) -> list:
        """Deprecated compatibility wrapper for :meth:`missingness_association_test`.

        Parameters
        ----------
        target : str
            Name of the fully observed auxiliary target column.
        missing_values : list, optional
            Extra sentinel values to treat as missing in the non-target columns.

        Returns
        -------
        list of tuple
            Legacy ``(feature, lrt_statistic, p_value)`` tuples.

        Raises
        ------
        MissingColumnError
            If *target* is not a column in the wrapped DataFrame.
        ValueError
            If *target* has missing values.  Rows are never dropped silently;
            the behaviour is identical to :meth:`missingness_association_test`.

        Warns
        -----
        FutureWarning
            Emitted by the wrapped function; the name is deprecated.

        Examples
        --------
        >>> import warnings
        >>> import numpy as np
        >>> import pandas as pd
        >>> import missingly  # noqa: F401
        >>> frame = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [0, 1, 1]})
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     isinstance(frame.miss.mar_mnar_test("y"), list)
        True
        """
        features, outcome = self._split_target(target)
        return diagnostics.mar_mnar_test(features, outcome, missing_values)

    def mice_convergence(
        self,
        histories: List[Dict[str, List[float]]],
        variables: Optional[List[str]] = None,
        rhat_threshold: float = 1.1,
    ) -> dict:
        """Assess convergence of multiple MICE chains using iteration history.

        This is a convenience wrapper around
        :func:`missingly.diagnostics.mice_convergence` that keeps the
        accessor pattern consistent.  The DataFrame context (``self._df``)
        is not used directly — pass *histories* from
        ``impute_mice(..., n_imputations=m, return_history=True)``.

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

    @_mirror(visualise.matrix)
    def matrix(self, *args: Any, **kwargs: Any):
        """Render a missingness matrix plot.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.matrix`, without
            the DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> ax = pd.DataFrame({"a": [1, np.nan, 3]}).miss.matrix()
        """
        return visualise.matrix(self._df, *args, **kwargs)

    @_mirror(visualise.bar)
    def bar(self, *args: Any, **kwargs: Any):
        """Render a bar chart of missing counts per column.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.bar`, without the
            DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> ax = pd.DataFrame({"a": [1, np.nan, 3]}).miss.bar()
        """
        return visualise.bar(self._df, *args, **kwargs)

    @_mirror(visualise.upset)
    def upset(self, *args: Any, **kwargs: Any):
        """Render an UpSet plot of missing value combinations.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.upset`, without
            the DataFrame.

        Returns
        -------
        dict
            Mapping of the UpSet panels to their Axes.

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> frame = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]})
        >>> panels = frame.miss.upset()
        """
        return visualise.upset(self._df, *args, **kwargs)

    @_mirror(visualise.heatmap)
    def heatmap(self, *args: Any, **kwargs: Any):
        """Render a nullity-correlation heatmap.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.heatmap`, without
            the DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> frame = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [np.nan, 2, 3, np.nan]})
        >>> ax = frame.miss.heatmap()
        """
        return visualise.heatmap(self._df, *args, **kwargs)

    @_mirror(visualise.vis_miss)
    def vis_miss(self, *args: Any, **kwargs: Any):
        """Render an annotated missingness overview matrix.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.vis_miss`, without
            the DataFrame.  The v1 ``show_pct`` argument is deprecated: it
            only has an effect for the interactive (Plotly) backend and is
            ignored, with a :class:`FutureWarning`, for matplotlib.

        Returns
        -------
        matplotlib.axes.Axes or plotly.graph_objects.Figure

        Warns
        -----
        FutureWarning
            If ``show_pct`` is passed while the matplotlib backend is used.

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> ax = pd.DataFrame({"a": [1, np.nan, 3]}).miss.vis_miss(sort=True)
        """
        interactive = kwargs.get("interactive") or kwargs.get("backend") == "plotly"
        if not interactive:
            _warn_ignored(
                kwargs,
                "show_pct",
                "vis_miss",
                "the matplotlib backend does not draw percentages",
            )
        return visualise.vis_miss(self._df, *args, **kwargs)

    @_mirror(visualise.miss_var_pct)
    def miss_var_pct(self, *args: Any, **kwargs: Any):
        """Render a horizontal bar chart of missingness % per variable.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.miss_var_pct`,
            without the DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> ax = pd.DataFrame({"a": [1, np.nan, 3]}).miss.miss_var_pct()
        """
        return visualise.miss_var_pct(self._df, *args, **kwargs)

    @_mirror(visualise.miss_cluster)
    def miss_cluster(self, *args: Any, **kwargs: Any):
        """Render a clustered missingness heatmap.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.miss_cluster`,
            without the DataFrame.  The v1 ``method`` argument is deprecated:
            the clustering always uses Ward linkage, so it is ignored with a
            :class:`FutureWarning`.

        Returns
        -------
        matplotlib.axes.Axes

        Warns
        -----
        FutureWarning
            If ``method`` is passed.

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> frame = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [np.nan, 2, 3, 4]})
        >>> ax = frame.miss.miss_cluster()
        """
        _warn_ignored(
            kwargs,
            "method",
            "miss_cluster",
            "the clustering always uses Ward linkage",
        )
        return visualise.miss_cluster(self._df, *args, **kwargs)

    @_mirror(visualise.miss_which)
    def miss_which(self, *args: Any, **kwargs: Any):
        """Render a binary tile plot showing which columns have missing data.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.miss_which`,
            without the DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> ax = pd.DataFrame({"a": [1, np.nan, 3]}).miss.miss_which()
        """
        return visualise.miss_which(self._df, *args, **kwargs)

    @_mirror(visualise.scatter_miss)
    def scatter_miss(self, *args: Any, **kwargs: Any):
        """Render a scatter plot highlighting missing values.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.scatter_miss`
            (``x``, ``y`` and plot options), without the DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> frame = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [1, 2, np.nan, 4]})
        >>> ax = frame.miss.scatter_miss("a", "b")
        """
        return visualise.scatter_miss(self._df, *args, **kwargs)

    @_mirror(visualise.miss_case)
    def miss_case(self, *args: Any, **kwargs: Any):
        """Render a bar chart of missing values per row.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.miss_case`,
            without the DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> ax = pd.DataFrame({"a": [1, np.nan, 3]}).miss.miss_case()
        """
        return visualise.miss_case(self._df, *args, **kwargs)

    @_mirror(visualise.vis_impute_dist, rename={"df_after": "imputed_df"})
    def vis_impute_dist(self, *args: Any, **kwargs: Any):
        """Compare distributions before and after imputation.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.vis_impute_dist`
            without the "before" DataFrame: ``imputed_df`` (the "after"
            DataFrame, ``df_after`` in the function), the required
            ``column`` and the plot options.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
        >>> ax = frame.miss.vis_impute_dist(frame.miss.impute_mean(), "a")
        """
        if "imputed_df" in kwargs:
            kwargs["df_after"] = kwargs.pop("imputed_df")
        return visualise.vis_impute_dist(self._df, *args, **kwargs)

    @_mirror(visualise.vis_miss_fct)
    def vis_miss_fct(self, *args: Any, **kwargs: Any):
        """Render missingness by a categorical factor variable.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.vis_miss_fct`
            (the required ``fct`` and plot options), without the DataFrame.
            The function creates its own facet axes, so it has no ``ax``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes of the last facet.

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> frame = pd.DataFrame({"a": [1, np.nan, 3, 4], "g": ["u", "u", "v", "v"]})
        >>> ax = frame.miss.vis_miss_fct("g")
        """
        return visualise.vis_miss_fct(self._df, *args, **kwargs)

    @_mirror(visualise.vis_miss_cumsum_var)
    def vis_miss_cumsum_var(self, *args: Any, **kwargs: Any):
        """Render cumulative missing count per variable.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as
            :func:`missingly.visualise.vis_miss_cumsum_var`, without the
            DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> ax = pd.DataFrame({"a": [1, np.nan, 3]}).miss.vis_miss_cumsum_var()
        """
        return visualise.vis_miss_cumsum_var(self._df, *args, **kwargs)

    @_mirror(visualise.vis_miss_cumsum_case)
    def vis_miss_cumsum_case(self, *args: Any, **kwargs: Any):
        """Render cumulative missing count per row.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as
            :func:`missingly.visualise.vis_miss_cumsum_case`, without the
            DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> ax = pd.DataFrame({"a": [1, np.nan, 3]}).miss.vis_miss_cumsum_case()
        """
        return visualise.vis_miss_cumsum_case(self._df, *args, **kwargs)

    @_mirror(visualise.vis_miss_span)
    def vis_miss_span(self, *args: Any, **kwargs: Any):
        """Render rolling missing count for a single variable.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.vis_miss_span`:
            the required ``column``, the window size ``span_every`` (default
            20) or its alias ``span``, and plot options.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> frame = pd.DataFrame({"a": [1, np.nan, 3, 4, np.nan, 6]})
        >>> ax = frame.miss.vis_miss_span("a", span_every=3)
        """
        return visualise.vis_miss_span(self._df, *args, **kwargs)

    @_mirror(visualise.vis_parallel_coords)
    def vis_parallel_coords(self, *args: Any, **kwargs: Any):
        """Render a parallel coordinates missingness plot.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as
            :func:`missingly.visualise.vis_parallel_coords`, without the
            DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> frame = pd.DataFrame({"a": [1, np.nan, 3], "b": [4.0, 5.0, 6.0]})
        >>> ax = frame.miss.vis_parallel_coords()
        """
        return visualise.vis_parallel_coords(self._df, *args, **kwargs)

    @_mirror(visualise.dendrogram)
    def dendrogram(self, *args: Any, **kwargs: Any):
        """Render a dendrogram clustering variables by nullity correlation.

        Parameters
        ----------
        *args, **kwargs
            Same parameters as :func:`missingly.visualise.dendrogram`
            (``method`` is the linkage method, default ``"average"``),
            without the DataFrame.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import numpy as np, pandas as pd, missingly
        >>> frame = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [np.nan, 2, 3, 4],
        ...                       "c": [1, 2, np.nan, np.nan]})
        >>> ax = frame.miss.dendrogram(method="complete")
        """
        return visualise.dendrogram(self._df, *args, **kwargs)
