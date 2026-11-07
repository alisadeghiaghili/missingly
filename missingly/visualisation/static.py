"""Static (matplotlib / seaborn) visualisations for missing-data analysis.

Return types and the ``ax`` keyword
-----------------------------------
Unless noted below, every public function accepts an optional *ax* keyword
that draws into an existing :class:`~matplotlib.axes.Axes` and returns the
:class:`~matplotlib.axes.Axes` that was drawn on.  The exceptions are:

* :func:`upset` always creates its own figure and returns a ``dict`` of axes
  keyed by ``intersections``, ``matrix`` and ``totals``.
* :func:`vis_miss_fct` creates one figure with a facet per group, has no
  *ax* keyword and returns only the last facet's axes (the others are
  reachable through ``ax.figure.axes``).
* :func:`miss_impute_compare` creates a grid of histograms, has no *ax*
  keyword and returns the :class:`~matplotlib.figure.Figure`.

Interactive output
------------------
Only :func:`vis_miss`, :func:`heatmap`, :func:`matrix`, :func:`bar`,
:func:`miss_var_pct`, :func:`miss_cooccurrence`, :func:`miss_case`,
:func:`upset` and :func:`miss_patterns` have a Plotly backend, selected with
``backend="plotly"`` or ``interactive=True``.  Styling arguments that the
Plotly backend cannot honour are reported with a warning.

Unsupported keyword arguments
-----------------------------
Functions that accept ``**kwargs`` without a documented target currently
ignore unsupported keywords.  Each such call emits a :class:`FutureWarning`
naming the argument; a future release will raise :class:`TypeError` instead.

Degenerate inputs and figure lifetime
-------------------------------------
A frame with no rows or no columns is rejected with :class:`ValueError` when
no plot can be drawn, and otherwise drawn as an annotated empty state (for
example "No missing values" or "All values are missing") without runtime
warnings. A figure created by a call is closed if the call raises, and a call
given *ax* never creates a figure.

RTL support
-----------
Column names and index values that contain Arabic / Persian characters are
automatically reshaped and bidi-reordered via :func:`_rtl_safe_labels`
before being passed to matplotlib.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, TypeVar, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from missingly.visualisation._base import (
    _apply_rtl_font,
    _is_rtl,
    _require_plotly,
    _rtl_safe,
    _rtl_safe_labels,
)

_BACKENDS = frozenset({"matplotlib", "plotly"})
_MISS_CASE_SORTED_LABEL = "Row (sorted by # missing)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _warn_unsupported_kwargs(
    func_name: str,
    kwargs: Dict[str, Any],
    stacklevel: int = 3,
) -> None:
    """Warn that unsupported keyword arguments were ignored.

    Parameters
    ----------
    func_name : str
        Public function that received the keywords.
    kwargs : dict
        Keyword arguments the function does not use.
    stacklevel : int, default 3
        Stack level of the reported warning; the default points at the caller
        of the public function.

    Warns
    -----
    FutureWarning
        If *kwargs* is not empty. A future release will raise ``TypeError``.

    Examples
    --------
    >>> _warn_unsupported_kwargs("bar", {})
    """
    if not kwargs:
        return
    names = ", ".join(sorted(kwargs))
    warnings.warn(
        f"{func_name}() ignored unsupported keyword argument(s): {names}; "
        "passing unsupported keyword arguments will raise TypeError in a "
        "future release.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def _use_plotly(func_name: str, backend: str, interactive: bool) -> bool:
    """Return whether the Plotly backend is selected, validating ``backend``.

    Parameters
    ----------
    func_name : str
        Public function that received the arguments.
    backend : str
        Requested backend name.
    interactive : bool
        Whether interactive output was requested.

    Returns
    -------
    bool
        ``True`` when ``backend == "plotly"`` or *interactive* is true.

    Warns
    -----
    FutureWarning
        If *backend* is neither ``"matplotlib"`` nor ``"plotly"``. The call
        falls back to Matplotlib; a future release will raise ``ValueError``.

    Examples
    --------
    >>> _use_plotly("bar", "matplotlib", False)
    False
    """
    if backend not in _BACKENDS:
        warnings.warn(
            f"{func_name}() received unknown backend {backend!r}; falling back "
            "to 'matplotlib'. Unknown backends will raise ValueError in a "
            "future release.",
            FutureWarning,
            stacklevel=3,
        )
    return backend == "plotly" or interactive


def _warn_ignored_on_plotly(func: Callable[..., Any], **given: Any) -> None:
    """Warn about styling arguments the Plotly backend does not honour.

    Parameters
    ----------
    func : callable
        Public function whose signature supplies the defaults.
    **given : Any
        Argument names mapped to the values the caller passed.

    Warns
    -----
    UserWarning
        If any value differs from the parameter default of *func*.

    Examples
    --------
    >>> _warn_ignored_on_plotly(bar, color="#4C72B0")
    """
    parameters = inspect.signature(func).parameters
    ignored = sorted(
        name
        for name, value in given.items()
        if value is not parameters[name].default and value != parameters[name].default
    )
    if ignored:
        warnings.warn(
            f"{func.__name__}() ignored styling argument(s) not supported by "
            f"the Plotly backend: {', '.join(ignored)}.",
            UserWarning,
            stacklevel=3,
        )


def _null_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean DataFrame: True where *df* has a missing value."""
    return df.isnull()


def _pct_missing(df: pd.DataFrame) -> pd.Series:
    """Return the percentage of missing values per column."""
    return df.isnull().mean() * 100


def _apply_sentinels(df: pd.DataFrame, missing_values=None) -> pd.DataFrame:
    """Replace custom sentinel values with NaN."""
    if missing_values:
        df = df.replace(missing_values, np.nan)
    return df


def _set_datetime_yticks(
    ax: Any,
    index: pd.DatetimeIndex,
    freq: str,
    fontsize: int,
) -> None:
    boundaries = pd.date_range(index.min(), index.max(), freq=freq)
    tick_positions = [np.searchsorted(index, dt) for dt in boundaries]
    tick_labels = [dt.strftime("%Y-%m-%d") for dt in boundaries]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=fontsize)


def _clean_ax(ax: Any, spines: Sequence[str] = ("top", "right")) -> None:
    for spine in spines:
        ax.spines[spine].set_visible(False)


def _empty_upset_axes(
    figsize: tuple[float, float] | None,
    message: str,
) -> dict[str, Any]:
    """Create the documented UpSet axes schema for an empty pattern result.

    Parameters
    ----------
    figsize : tuple of float or None
        Requested matplotlib figure size. ``None`` selects the standard size.
    message : str
        Explanation rendered on the intersections axis.

    Returns
    -------
    dict of str to matplotlib.axes.Axes
        Axes keyed by ``intersections``, ``matrix``, and ``totals``.

    Examples
    --------
    >>> sorted(_empty_upset_axes(None, "No patterns"))
    ['intersections', 'matrix', 'totals']
    """
    figure = plt.figure(figsize=figsize or (9, 5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.2, 4.0), height_ratios=(2.2, 1.4))
    blank_ax = figure.add_subplot(grid[0, 0])
    intersections_ax = figure.add_subplot(grid[0, 1])
    totals_ax = figure.add_subplot(grid[1, 0])
    matrix_ax = figure.add_subplot(grid[1, 1])
    blank_ax.axis("off")
    intersections_ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=intersections_ax.transAxes,
    )
    intersections_ax.set_axis_off()
    totals_ax.set_axis_off()
    matrix_ax.set_axis_off()
    return {
        "intersections": intersections_ax,
        "matrix": matrix_ax,
        "totals": totals_ax,
    }


# ===========================================================================
# Figure lifetime and degenerate-input guard
# ===========================================================================

_ZERO_ROWS = "zero_rows"
_ZERO_COLS = "zero_cols"
_NO_MISSING = "no_missing"
_ALL_MISSING = "all_missing"
_EMPTY_STATE_TEXT = {
    _ZERO_ROWS: "No rows to display",
    _ZERO_COLS: "No columns to display",
    _NO_MISSING: "No missing values",
    _ALL_MISSING: "All values are missing",
}
_DEFAULT_EMPTY_FIGSIZE = (6, 4)
_FuncT = TypeVar("_FuncT", bound=Callable[..., Any])
_RAISE_NO_ROWS_OR_COLS = frozenset({_ZERO_ROWS, _ZERO_COLS})
_RAISE_NO_COLS = frozenset({_ZERO_COLS})
_KEEP_NO_ROWS_OR_COLS = frozenset({_ZERO_ROWS, _ZERO_COLS})
_KEEP_NO_ROWS_COLS_OR_MISSING = frozenset({_ZERO_ROWS, _ZERO_COLS, _NO_MISSING})


def _degenerate_case(frame: Any, missing_values: Any) -> str | None:
    """Classify a frame as degenerate, or return ``None`` if it is not one."""
    if not isinstance(frame, pd.DataFrame):
        return None
    if frame.shape[1] == 0:
        return _ZERO_COLS
    if frame.shape[0] == 0:
        return _ZERO_ROWS
    missing = _apply_sentinels(frame, missing_values).isnull().to_numpy()
    if not missing.any():
        return _NO_MISSING
    if missing.all():
        return _ALL_MISSING
    return None


def _close_figures_since(before: set[int]) -> None:
    """Close every pyplot figure that was opened after *before* was taken."""
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def _empty_state(
    case: str,
    ax: Any,
    figsize: tuple[float, float] | None,
    result: str,
) -> Any:
    """Return an annotated empty plot of the shape the public function returns."""
    message = _EMPTY_STATE_TEXT[case]
    if result == "upset":
        return _empty_upset_axes(figsize, message)
    if ax is None:
        _, ax = plt.subplots(
            figsize=figsize or _DEFAULT_EMPTY_FIGSIZE,
            constrained_layout=True,
        )
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return ax.figure if result == "figure" else ax


def _needs_message(func_name: str, case: str) -> str:
    noun = "row" if case == _ZERO_ROWS else "column"
    return f"{func_name}() needs at least one {noun}"


def _guarded(
    *,
    raises: frozenset[str] = frozenset(),
    keeps: frozenset[str] = frozenset(),
    result: str = "axes",
) -> Callable[[_FuncT], _FuncT]:
    """Release figures on failure and handle degenerate inputs for a public plot.

    Parameters
    ----------
    raises : frozenset of str, optional
        Degenerate cases (zero rows or zero columns) that raise ``ValueError``.
    keeps : frozenset of str, optional
        Degenerate cases the function already annotates itself; the body runs.
    result : {"axes", "figure", "upset"}, default "axes"
        Return shape used for the generic empty state.

    Returns
    -------
    callable
        Decorator that wraps a public plotting function.

    Examples
    --------
    >>> @_guarded(raises=frozenset({_ZERO_ROWS}))
    ... def _demo(df):
    ...     return df
    >>> _demo(pd.DataFrame({"a": [1.0]})).shape
    (1, 1)
    """

    def decorate(func: _FuncT) -> _FuncT:
        signature = inspect.signature(func)
        frame_name = next(iter(signature.parameters))

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind_partial(*args, **kwargs)
            before = set(plt.get_fignums())
            try:
                arguments = bound.arguments
                case = _degenerate_case(
                    arguments.get(frame_name), arguments.get("missing_values")
                )
                plotly_requested = arguments.get("interactive") or (
                    arguments.get("backend") == "plotly"
                )
                if plotly_requested or case is None or case in keeps:
                    return func(*args, **kwargs)
                if case in raises:
                    raise ValueError(_needs_message(func.__name__, case))
                return _empty_state(
                    case,
                    arguments.get("ax"),
                    arguments.get("figsize"),
                    result,
                )
            except BaseException:
                _close_figures_since(before)
                raise

        return cast(_FuncT, wrapper)

    return decorate


# ===========================================================================
# matrix
# ===========================================================================

@_guarded(raises=_RAISE_NO_ROWS_OR_COLS)
def matrix(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    color: str | tuple[float, float, float] = (0.25, 0.25, 0.25),
    fontsize: int = 12,
    labels: bool = True,
    sparkline: bool = True,
    freq: str | None = None,
    ax: Any = None,
    backend: str = "matplotlib",
    interactive: bool = False,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Nullity matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose missing cells are drawn.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    color : str or tuple of float, default (0.25, 0.25, 0.25)
        Colour of present values and of the sparkline.  Any Matplotlib colour
        specification accepted by :func:`matplotlib.colors.to_rgb` is allowed.
    fontsize : int, default 12
        Font size for tick labels.
    labels : bool, default True
        Whether to display column labels.
    sparkline : bool, default True
        Whether to draw the per-row completeness sparkline.  It is only
        drawn when the figure is created here, not when *ax* is supplied.
    freq : str, optional
        Date offset alias for y-axis ticks when the index is a
        :class:`pandas.DatetimeIndex`.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.  ``interactive=True`` also selects Plotly.
    interactive : bool, default False
        Whether to return an interactive Plotly figure.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Forwarded to the Plotly backend.  On the Matplotlib backend they are
        ignored.

    Returns
    -------
    matplotlib.axes.Axes or plotly Figure
        The matrix axes, or an interactive Plotly figure.

    Raises
    ------
    ValueError
        If the frame has no rows or no columns.
    ValueError
        If *color* is not a valid Matplotlib colour.
    ImportError
        If Plotly output is requested but Plotly is not installed.

    Warns
    -----
    FutureWarning
        If an unsupported keyword argument or an unknown *backend* is given;
        a future release will raise instead.
    UserWarning
        If styling arguments are given together with the Plotly backend,
        which does not honour them.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]})
    >>> ax = matrix(frame, color="red", sparkline=False)
    """
    if _use_plotly("matrix", backend, interactive):
        _require_plotly()
        _warn_ignored_on_plotly(
            matrix,
            figsize=figsize,
            color=color,
            fontsize=fontsize,
            labels=labels,
            sparkline=sparkline,
            freq=freq,
            ax=ax,
        )
        from missingly.visualisation.interactive import _matrix_plotly
        return _matrix_plotly(_apply_sentinels(df, missing_values), **kwargs)

    _warn_unsupported_kwargs("matrix", kwargs)
    rgb = mcolors.to_rgb(color)
    df = _apply_sentinels(df, missing_values)
    null_mat = _null_matrix(df)
    n_rows, n_cols = null_mat.shape

    if figsize is None:
        figsize = (min(10, max(4, n_cols * 0.6)), min(8, max(3, n_rows * 0.12)))

    if ax is not None:
        fig = ax.figure
        main_ax = ax
        spark_ax = None
    elif sparkline and n_rows > 1:
        fig, (main_ax, spark_ax) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios": [20, 1]},
            constrained_layout=True,
        )
    else:
        fig, main_ax = plt.subplots(figsize=figsize, constrained_layout=True)
        spark_ax = None

    rgba = np.ones((n_rows, n_cols, 4))
    present = ~null_mat.to_numpy()
    for c in range(3):
        rgba[:, :, c] = np.where(present, rgb[c], 1.0)

    main_ax.imshow(rgba, aspect="auto", interpolation="none")
    main_ax.set_xlim(-0.5, n_cols - 0.5)
    main_ax.set_ylim(n_rows - 0.5, -0.5)

    if labels:
        col_labels = _rtl_safe_labels(list(df.columns), ax=main_ax)
        main_ax.set_xticks(range(n_cols))
        main_ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    else:
        main_ax.set_xticks([])

    if freq and isinstance(df.index, pd.DatetimeIndex):
        _set_datetime_yticks(main_ax, df.index, freq, fontsize)
    else:
        main_ax.set_yticks([])

    main_ax.tick_params(axis="x", which="both", bottom=False)
    main_ax.set_facecolor("white")
    for spine in main_ax.spines.values():
        spine.set_visible(False)

    if spark_ax is not None:
        completeness = present.sum(axis=1) / n_cols
        spark_ax.plot(completeness, range(n_rows), color=rgb, linewidth=1)
        spark_ax.set_ylim(n_rows - 0.5, -0.5)
        spark_ax.set_xlim(0, 1)
        spark_ax.set_xticks([0, 1])
        spark_ax.set_xticklabels(["0", str(n_cols)], fontsize=fontsize - 2)
        spark_ax.set_yticks([])
        for spine in spark_ax.spines.values():
            spine.set_visible(False)

    return main_ax


# ===========================================================================
# bar
# ===========================================================================

@_guarded()
def bar(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    color: str = "#4C72B0",
    log: bool = False,
    labels: bool = True,
    fontsize: int = 12,
    sort: bool = False,
    ax: Any = None,
    backend: str = "matplotlib",
    interactive: bool = False,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Bar chart of per-column missing-value counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose missing values are counted by column.
    figsize : tuple of float, optional
        Figure size for a newly-created Matplotlib axes.
    color : str, default "#4C72B0"
        Bar colour for the Matplotlib backend.
    log : bool, default False
        Whether to use a logarithmic count axis in the Matplotlib backend.
    labels : bool, default True
        Whether to display column labels in the Matplotlib backend.
    fontsize : int, default 12
        Font size for Matplotlib axis labels.
    sort : bool, default False
        Whether to sort counts descending while preserving caller order for ties.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axes to draw on.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.  ``interactive=True`` also selects Plotly.
    interactive : bool, default False
        Whether to return an interactive Plotly figure.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Forwarded to the Plotly backend.  On the Matplotlib backend they are
        ignored.

    Returns
    -------
    matplotlib.axes.Axes or plotly Figure
        A chart whose bar heights are the number of missing values per column.

    Raises
    ------
    ImportError
        If Plotly output is requested but Plotly is not installed.

    Warns
    -----
    FutureWarning
        If an unsupported keyword argument or an unknown *backend* is given;
        a future release will raise instead.
    UserWarning
        If styling arguments are given together with the Plotly backend,
        which does not honour them.

    Examples
    --------
    >>> import pandas as pd
    >>> axis = bar(pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]}))
    >>> [int(patch.get_height()) for patch in axis.patches]
    [1, 0]
    """
    if _use_plotly("bar", backend, interactive):
        _require_plotly()
        _warn_ignored_on_plotly(
            bar, figsize=figsize, color=color, log=log, labels=labels,
            fontsize=fontsize, ax=ax,
        )
        from missingly.visualisation.interactive import _bar_plotly
        return _bar_plotly(
            _apply_sentinels(df, missing_values),
            sort=sort,
            **kwargs,
        )

    _warn_unsupported_kwargs("bar", kwargs)
    df = _apply_sentinels(df, missing_values)
    counts = df.isnull().sum()
    if sort:
        counts = counts.sort_values(ascending=False, kind="stable")

    n = len(counts)
    if figsize is None:
        figsize = (min(12, max(4, n * 0.6)), 4)

    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    col_labels = _rtl_safe_labels(list(counts.index), ax=ax)
    x = np.arange(n)
    ax.bar(x, counts.values, color=color)

    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    else:
        ax.set_xticks([])

    ax.set_ylabel("Number of Missing Values", fontsize=fontsize)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, max(1, counts.max() * 1.1))
    _clean_ax(ax)
    return ax


# ===========================================================================
# miss_case  – missing values per row
# ===========================================================================

@_guarded()
def miss_case(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    color: str = "#4C72B0",
    fontsize: int = 12,
    sort: bool = True,
    ax: Any = None,
    backend: str = "matplotlib",
    interactive: bool = False,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Bar chart of missing-value counts per row (case).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose missing values are counted by row.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    color : str, default "#4C72B0"
        Bar colour for the Matplotlib backend.
    fontsize : int, default 12
        Font size for axis labels.
    sort : bool, default True
        Whether to order the bars by descending count.  When sorted, the
        x positions are ranks and the axis is labelled accordingly.  Row
        identifiers are shown as tick labels for at most 50 rows.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.  ``interactive=True`` also selects Plotly.
    interactive : bool, default False
        Whether to return an interactive Plotly figure.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Forwarded to the Plotly backend.  On the Matplotlib backend they are
        ignored.

    Returns
    -------
    matplotlib.axes.Axes or plotly Figure
        A bar chart with one bar per row.

    Raises
    ------
    ImportError
        If Plotly output is requested but Plotly is not installed.

    Warns
    -----
    FutureWarning
        If an unsupported keyword argument or an unknown *backend* is given;
        a future release will raise instead.
    UserWarning
        If styling arguments are given together with the Plotly backend,
        which does not honour them.

    Examples
    --------
    >>> import pandas as pd
    >>> ax = miss_case(pd.DataFrame({"a": [1.0, None], "b": [None, None]}))
    >>> ax.get_ylabel()
    '# Missing'
    """
    if _use_plotly("miss_case", backend, interactive):
        _require_plotly()
        _warn_ignored_on_plotly(
            miss_case, figsize=figsize, color=color, fontsize=fontsize, ax=ax,
        )
        from missingly.visualisation.interactive import _miss_case_plotly
        return _miss_case_plotly(
            _apply_sentinels(df, missing_values),
            sort=sort,
            **kwargs,
        )

    _warn_unsupported_kwargs("miss_case", kwargs)
    df = _apply_sentinels(df, missing_values)
    counts = df.isnull().sum(axis=1)
    if sort:
        counts = counts.sort_values(ascending=False, kind="stable")

    n = len(counts)
    if figsize is None:
        figsize = (min(14, max(4, n * 0.15)), 4)

    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.bar(np.arange(n), counts.values, color=color)
    x_label = _MISS_CASE_SORTED_LABEL if sort else "Row index"
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel("# Missing", fontsize=fontsize)
    ax.set_title("Missing values per case (row)")
    if n <= 50:
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels([str(i) for i in counts.index], rotation=45, ha="right", fontsize=fontsize - 2)
    else:
        ax.set_xticks([])
    _clean_ax(ax)
    return ax


# ===========================================================================
# miss_var_pct  – % missing per variable
# ===========================================================================

@_guarded()
def miss_var_pct(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    color: str = "#4C72B0",
    fontsize: int = 12,
    sort: bool = True,
    ax: Any = None,
    backend: str = "matplotlib",
    interactive: bool = False,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Horizontal bar chart of percentage missing per variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose missing share is computed per column.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    color : str, default "#4C72B0"
        Bar colour for the Matplotlib backend.
    fontsize : int, default 12
        Font size for tick and axis labels.
    sort : bool, default True
        Whether to order the bars by ascending percentage.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.  ``interactive=True`` also selects Plotly.
    interactive : bool, default False
        Whether to return an interactive Plotly figure.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Forwarded to the Plotly backend.  On the Matplotlib backend they are
        ignored.

    Returns
    -------
    matplotlib.axes.Axes or plotly Figure
        A horizontal bar chart of the percentage missing per column.

    Raises
    ------
    ImportError
        If Plotly output is requested but Plotly is not installed.

    Warns
    -----
    FutureWarning
        If an unsupported keyword argument or an unknown *backend* is given;
        a future release will raise instead.
    UserWarning
        If styling arguments are given together with the Plotly backend,
        which does not honour them.

    Examples
    --------
    >>> import pandas as pd
    >>> ax = miss_var_pct(pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]}))
    >>> ax.get_xlabel()
    '% Missing'
    """
    if _use_plotly("miss_var_pct", backend, interactive):
        _require_plotly()
        _warn_ignored_on_plotly(
            miss_var_pct, figsize=figsize, color=color, fontsize=fontsize, ax=ax,
        )
        from missingly.visualisation.interactive import _miss_var_pct_plotly
        return _miss_var_pct_plotly(
            _apply_sentinels(df, missing_values),
            sort=sort,
            **kwargs,
        )

    _warn_unsupported_kwargs("miss_var_pct", kwargs)
    df = _apply_sentinels(df, missing_values)
    pct = df.isnull().mean() * 100
    if sort:
        pct = pct.sort_values(ascending=True, kind="stable")

    n = len(pct)
    if figsize is None:
        figsize = (7, max(3, n * 0.4))

    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    col_labels = _rtl_safe_labels(list(pct.index), ax=ax)
    y = np.arange(n)
    ax.barh(y, pct.values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(col_labels, fontsize=fontsize)
    ax.set_xlabel("% Missing", fontsize=fontsize)
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    _clean_ax(ax)
    return ax


# ===========================================================================
# miss_which  – indicator grid (rows x missing cols)
# ===========================================================================

@_guarded()
def miss_which(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Heatmap showing *which* cells are missing (rows x columns).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.  Only columns with at least one missing value are shown.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    fontsize : int, default 10
        Font size for column labels; row labels are one point smaller.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Unsupported; this function has no interactive backend.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that were drawn on.

    Warns
    -----
    FutureWarning
        If any keyword argument (including ``interactive`` or ``backend``) is
        given; it is ignored and a future release will raise ``TypeError``.

    Examples
    --------
    >>> import pandas as pd
    >>> ax = miss_which(pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]}))
    >>> ax.get_title()
    'Which values are missing?'
    """
    _warn_unsupported_kwargs("miss_which", kwargs)
    df = _apply_sentinels(df, missing_values)
    null_mat = _null_matrix(df)
    cols_miss = [c for c in null_mat.columns if null_mat[c].any()]
    sub = null_mat[cols_miss] if cols_miss else null_mat

    n_rows, n_cols = sub.shape
    if figsize is None:
        figsize = (max(4, n_cols * 0.7), max(3, n_rows * 0.25))

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    ax_.imshow(sub.astype(float).values, aspect="auto", cmap="Reds", interpolation="none", vmin=0, vmax=1)
    col_labels = _rtl_safe_labels(list(sub.columns), ax=ax_)
    ax_.set_xticks(range(n_cols))
    ax_.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    ax_.set_yticks(range(n_rows))
    ax_.set_yticklabels([str(i) for i in sub.index], fontsize=fontsize - 1)
    ax_.set_title("Which values are missing?")
    return ax_


# ===========================================================================
# upset
# ===========================================================================

@_guarded(keeps=_KEEP_NO_ROWS_COLS_OR_MISSING, result="upset")
def upset(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    min_subset_size: int = 1,
    backend: str = "matplotlib",
    interactive: bool = False,
    missing_values=None,
    **kwargs: Any,
) -> dict | Any:
    """UpSet plot of missing-value co-occurrence patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose joint missingness patterns are plotted.
    figsize : tuple of float, optional
        Matplotlib figure size.  A readable default is selected when omitted.
    min_subset_size : int, default 1
        Minimum number of rows required for an intersection to be displayed.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.  ``interactive=True`` also selects Plotly.
    interactive : bool, default False
        Whether to return an interactive Plotly figure.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Backend-specific options.  The native Matplotlib backend accepts
        ``show_pct``; Plotly options are forwarded to its implementation.

    Returns
    -------
    dict or plotly.graph_objects.Figure
        Matplotlib axes under ``intersections``, ``matrix``, and ``totals``;
        or an interactive Plotly figure.

    Raises
    ------
    ImportError
        If Plotly is requested but not installed.
    TypeError
        If an unsupported Matplotlib keyword argument is supplied.

    Warns
    -----
    FutureWarning
        If *backend* is neither ``"matplotlib"`` nor ``"plotly"``; the call
        falls back to Matplotlib and a future release will raise.
    UserWarning
        If ``figsize`` is given together with the Plotly backend, which does
        not honour it.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from missingly.visualisation.static import upset
    >>> frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan]})
    >>> axes = upset(frame)
    >>> sorted(axes)
    ['intersections', 'matrix', 'totals']
    """
    if _use_plotly("upset", backend, interactive):
        _require_plotly()
        _warn_ignored_on_plotly(upset, figsize=figsize)
        from missingly.visualisation.interactive import _upset_plotly
        return _upset_plotly(
            _apply_sentinels(df, missing_values),
            min_subset_size=min_subset_size,
            **kwargs,
        )

    show_pct = bool(kwargs.pop("show_pct", False))
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported UpSet keyword arguments: {unknown}")

    df = _apply_sentinels(df, missing_values)
    null_mat = _null_matrix(df)
    if not null_mat.any(axis=None):
        return _empty_upset_axes(figsize, "No missingness patterns to display")

    cols_with_missing = [c for c in null_mat.columns if null_mat[c].any()]
    if not cols_with_missing:
        return _empty_upset_axes(figsize, "No missingness patterns to display")

    patterns = (
        null_mat[cols_with_missing]
        .value_counts(sort=True)
        .loc[lambda counts: counts >= min_subset_size]
    )
    if patterns.empty:
        return _empty_upset_axes(figsize, "No missingness patterns to display")

    figure = plt.figure(figsize=figsize or (9, 5), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.2, 4.0),
        height_ratios=(2.2, 1.4),
    )
    blank_ax = figure.add_subplot(grid[0, 0])
    blank_ax.axis("off")
    intersections_ax = figure.add_subplot(grid[0, 1])
    totals_ax = figure.add_subplot(grid[1, 0])
    matrix_ax = figure.add_subplot(grid[1, 1])

    x_positions = np.arange(len(patterns))
    intersection_values = patterns.to_numpy(dtype=float)
    intersections_ax.bar(x_positions, intersection_values, color="#4C72B0")
    intersections_ax.set_ylabel("Rows")
    intersections_ax.set_xticks([])
    intersections_ax.set_title("Missingness pattern intersections")
    if show_pct:
        total_rows = float(len(df))
        for x_pos, value in zip(x_positions, intersection_values):
            intersections_ax.text(
                x_pos,
                value,
                f"{100 * value / total_rows:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    pattern_values = [
        tuple(bool(value) for value in pattern)
        if isinstance(pattern, tuple)
        else (bool(pattern),)
        for pattern in patterns.index.tolist()
    ]
    for x_pos, pattern in zip(x_positions, pattern_values):
        active = [idx for idx, value in enumerate(pattern) if value]
        if len(active) > 1:
            matrix_ax.plot([x_pos, x_pos], [min(active), max(active)], color="#333333")
        for y_pos, value in enumerate(pattern):
            matrix_ax.scatter(
                x_pos,
                y_pos,
                color="#333333" if value else "#D9D9D9",
                s=36,
                zorder=3,
            )
    matrix_ax.set_yticks(range(len(cols_with_missing)))
    matrix_ax.set_yticklabels(_rtl_safe_labels(cols_with_missing, ax=matrix_ax))
    matrix_ax.set_xticks(x_positions)
    matrix_ax.set_xticklabels([str(index + 1) for index in x_positions])
    matrix_ax.set_xlabel("Pattern")
    matrix_ax.invert_yaxis()

    totals = null_mat[cols_with_missing].sum().to_numpy(dtype=float)
    y_positions = np.arange(len(cols_with_missing))
    totals_ax.barh(y_positions, totals, color="#7A9CC6")
    totals_ax.set_yticks([])
    totals_ax.set_xlabel("Missing")
    totals_ax.invert_xaxis()
    totals_ax.invert_yaxis()

    return {
        "intersections": intersections_ax,
        "matrix": matrix_ax,
        "totals": totals_ax,
    }


# ===========================================================================
# miss_patterns
# ===========================================================================

@_guarded()
def miss_patterns(
    df: pd.DataFrame,
    *,
    top_n: int = 10,
    figsize: tuple[float, float] | None = None,
    color: str = "#4C72B0",
    fontsize: int = 11,
    ax: Any = None,
    backend: str = "matplotlib",
    interactive: bool = False,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Bar chart of the most common missingness patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose row-wise missingness patterns are counted.
    top_n : int, default 10
        Maximum number of patterns to display.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    color : str, default "#4C72B0"
        Bar colour for the Matplotlib backend.
    fontsize : int, default 11
        Font size for tick and axis labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.  ``interactive=True`` also selects Plotly.
    interactive : bool, default False
        Whether to return an interactive Plotly figure.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Forwarded to the Plotly backend.  On the Matplotlib backend they are
        ignored.

    Returns
    -------
    matplotlib.axes.Axes or plotly Figure
        A horizontal bar chart of pattern frequencies.

    Raises
    ------
    ImportError
        If Plotly output is requested but Plotly is not installed.

    Warns
    -----
    FutureWarning
        If an unsupported keyword argument or an unknown *backend* is given;
        a future release will raise instead.
    UserWarning
        If styling arguments are given together with the Plotly backend,
        which does not honour them.

    Examples
    --------
    >>> import pandas as pd
    >>> ax = miss_patterns(pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]}))
    >>> ax.get_xlabel()
    'Row count'
    """
    if _use_plotly("miss_patterns", backend, interactive):
        _require_plotly()
        _warn_ignored_on_plotly(
            miss_patterns, figsize=figsize, color=color, fontsize=fontsize, ax=ax,
        )
        from missingly.visualisation.interactive import _miss_patterns_plotly
        return _miss_patterns_plotly(_apply_sentinels(df, missing_values), top_n=top_n, **kwargs)

    _warn_unsupported_kwargs("miss_patterns", kwargs)
    df = _apply_sentinels(df, missing_values)
    null_mat = _null_matrix(df)

    def _label(row: pd.Series) -> str:
        cols = [str(c) for c, v in row.items() if v]
        return "(complete)" if not cols else " + ".join(cols)

    patterns = null_mat.apply(_label, axis=1)
    counts = patterns.value_counts().head(top_n).sort_values()

    n = len(counts)
    if figsize is None:
        figsize = (7, max(3, n * 0.45))

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    y = np.arange(n)
    ax_.barh(y, counts.values, color=color)
    ax_.set_yticks(y)
    ax_.set_yticklabels(counts.index.tolist(), fontsize=fontsize)
    ax_.set_xlabel("Row count", fontsize=fontsize)
    ax_.set_title(f"Top-{top_n} Missingness Patterns")
    _clean_ax(ax_)
    return ax_


# ===========================================================================
# miss_cooccurrence
# ===========================================================================

@_guarded(raises=_RAISE_NO_COLS)
def miss_cooccurrence(
    df: pd.DataFrame,
    *,
    normalize: bool = True,
    figsize: tuple[float, float] | None = None,
    cmap: str = "Blues",
    fontsize: int = 11,
    annot: bool = True,
    fmt: str = ".2f",
    ax: Any = None,
    backend: str = "matplotlib",
    interactive: bool = False,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Co-occurrence heatmap of column missingness.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose columns' joint missingness is counted.
    normalize : bool, default True
        Whether to divide counts by the number of rows.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    cmap : str, default "Blues"
        Matplotlib colormap.
    fontsize : int, default 11
        Tick-label font size.
    annot : bool, default True
        Whether to annotate cells with their values.
    fmt : str, default ".2f"
        Numeric format used for annotations.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.  ``interactive=True`` also selects Plotly.
    interactive : bool, default False
        Whether to return an interactive Plotly figure.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Forwarded to :func:`seaborn.heatmap` (or to the Plotly backend).

    Returns
    -------
    matplotlib.axes.Axes or plotly Figure
        The heatmap axes, or an interactive Plotly figure.

    Raises
    ------
    ValueError
        If the frame has no columns.
    ImportError
        If Plotly output is requested but Plotly is not installed.

    Warns
    -----
    FutureWarning
        If an unknown *backend* is given; the call falls back to Matplotlib
        and a future release will raise.
    UserWarning
        If styling arguments are given together with the Plotly backend,
        which does not honour them.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
    >>> ax = miss_cooccurrence(frame, fontsize=9)
    >>> ax.get_title()
    'Missingness co-occurrence (fraction)'
    """
    if _use_plotly("miss_cooccurrence", backend, interactive):
        _require_plotly()
        _warn_ignored_on_plotly(
            miss_cooccurrence, figsize=figsize, cmap=cmap, fontsize=fontsize,
            annot=annot, fmt=fmt, ax=ax,
        )
        from missingly.visualisation.interactive import _miss_cooccurrence_plotly
        return _miss_cooccurrence_plotly(_apply_sentinels(df, missing_values), normalize=normalize, **kwargs)

    df = _apply_sentinels(df, missing_values)
    null_mat = _null_matrix(df).astype(int)
    cooc = null_mat.T.dot(null_mat)
    if normalize:
        cooc = cooc / len(df)

    n = len(cooc)
    if figsize is None:
        figsize = (max(4, n * 0.8), max(3, n * 0.7))

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    col_labels = _rtl_safe_labels(list(cooc.columns), ax=ax_)
    cooc.columns = col_labels
    cooc.index = col_labels
    sns.heatmap(cooc, ax=ax_, cmap=cmap, annot=annot, fmt=fmt, linewidths=0.4, **kwargs)
    ax_.tick_params(axis="both", labelsize=fontsize)
    ax_.set_title("Missingness co-occurrence" + (" (fraction)" if normalize else " (count)"))
    return ax_


# ===========================================================================
# heatmap
# ===========================================================================

@_guarded(keeps=_KEEP_NO_ROWS_COLS_OR_MISSING)
def heatmap(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "RdYlGn",
    fontsize: int = 12,
    annot: bool = True,
    fmt: str = ".2f",
    vmin: float = -1.0,
    vmax: float = 1.0,
    center: float = 0.0,
    linewidths: float = 0.5,
    ax: Any = None,
    backend: str = "matplotlib",
    interactive: bool = False,
    mask_insignificant: bool = True,
    significance: float = 0.05,
    method: str = "pearson",
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Plot correlations among per-column missingness indicators.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose missingness indicators are correlated.
    figsize : tuple of float, optional
        Matplotlib figure size. A readable default is selected when omitted.
    cmap : str, default "RdYlGn"
        Matplotlib colormap for the correlation matrix.
    fontsize : int, default 12
        Tick-label font size.
    annot : bool, default True
        Whether to annotate visible matrix cells with correlation values.
    fmt : str, default ".2f"
        Numeric format used for annotations.
    vmin, vmax, center : float
        Colormap bounds and midpoint forwarded to seaborn.
    linewidths : float, default 0.5
        Width of grid lines between matrix cells.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend. ``interactive=True`` also selects Plotly.
    interactive : bool, default False
        Whether to return an interactive Plotly figure.
    mask_insignificant : bool, default True
        Mask off-diagonal correlations whose two-sided Pearson p-value is at
        least ``significance``. Constant nullity indicators remain masked.
    significance : float, default 0.05
        Strictly positive p-value cutoff no greater than one.
    method : {"pearson", "phi"}, default "pearson"
        Correlation name. Both calculate Pearson correlation on binary
        missingness indicators, which is the phi coefficient.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`seaborn.heatmap`
        (or to the Plotly backend).

    Returns
    -------
    matplotlib.axes.Axes or plotly Figure
        The supplied or newly created axes for static output, or a Plotly
        figure when interactive output is selected.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or ``significance`` is outside ``(0, 1]``
        while significance masking is enabled.
    ImportError
        If Plotly output is requested but Plotly is not installed.

    Warns
    -----
    FutureWarning
        If an unknown *backend* is given; the call falls back to Matplotlib
        and a future release will raise.
    UserWarning
        If styling arguments are given together with the Plotly backend,
        which does not honour them.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    >>> heatmap(frame, annot=False).get_title()
    ''
    """
    if method not in {"pearson", "phi"}:
        raise ValueError("method must be either 'pearson' or 'phi'")
    if mask_insignificant and not 0 < significance <= 1:
        raise ValueError("significance must be in the interval (0, 1]")

    if _use_plotly("heatmap", backend, interactive):
        _require_plotly()
        _warn_ignored_on_plotly(
            heatmap, figsize=figsize, cmap=cmap, fontsize=fontsize, annot=annot,
            fmt=fmt, vmin=vmin, vmax=vmax, center=center, linewidths=linewidths,
            ax=ax,
        )
        from missingly.visualisation.interactive import _heatmap_plotly
        return _heatmap_plotly(
            _apply_sentinels(df, missing_values),
            method=method,
            mask_insignificant=mask_insignificant,
            significance=significance,
            **kwargs,
        )

    df = _apply_sentinels(df, missing_values)
    null_mat = _null_matrix(df)
    cols_with_missing = [c for c in null_mat.columns if null_mat[c].any()]
    if len(cols_with_missing) < 2:
        if ax is not None:
            ax_ = ax
        else:
            _, ax_ = plt.subplots(figsize=figsize or (4, 3), constrained_layout=True)
        ax_.text(0.5, 0.5, "Not enough columns with missing values",
                  ha="center", va="center", transform=ax_.transAxes)
        return ax_

    sub = null_mat[cols_with_missing].astype(float)
    # Pearson correlation between two binary indicators is the phi
    # coefficient, so both public names intentionally use this calculation.
    corr = sub.corr(method="pearson")
    diag_mask: NDArray[np.bool_] = np.eye(len(cols_with_missing), dtype=bool)
    final_mask = diag_mask | corr.isna().to_numpy()
    if mask_insignificant:
        from scipy.stats import pearsonr

        p_values: NDArray[np.float64] = np.ones(corr.shape, dtype=float)
        for left_idx, left_col in enumerate(cols_with_missing):
            for right_idx in range(left_idx):
                right_col = cols_with_missing[right_idx]
                left = sub[left_col]
                right = sub[right_col]
                if left.nunique() < 2 or right.nunique() < 2:
                    continue
                _, p_value = pearsonr(left, right)
                if np.isfinite(p_value):
                    p_values[left_idx, right_idx] = p_value
                    p_values[right_idx, left_idx] = p_value
        final_mask |= p_values >= significance

    if figsize is None:
        n = len(cols_with_missing)
        figsize = (max(4, n * 0.8), max(3, n * 0.7))

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    col_labels = _rtl_safe_labels(cols_with_missing, ax=ax_)
    corr.columns = col_labels
    corr.index = col_labels

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*set_bad.*")
        sns.heatmap(
            corr, mask=final_mask, ax=ax_,
            cmap=cmap, annot=annot, fmt=fmt,
            vmin=vmin, vmax=vmax, center=center, linewidths=linewidths,
            **kwargs,
        )
    ax_.tick_params(axis="both", labelsize=fontsize)
    return ax_


# ===========================================================================
# dendrogram
# ===========================================================================

@_guarded(keeps=_KEEP_NO_ROWS_COLS_OR_MISSING)
def dendrogram(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10, 4),
    fontsize: int = 12,
    orientation: str = "bottom",
    method: str = "average",
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Hierarchical-clustering dendrogram of column nullity correlation.

    Parameters
    ----------
    df : pandas.DataFrame
        Data whose columns are clustered by nullity correlation.
    figsize : tuple of float, default (10, 4)
        Figure size used when ``ax`` is not given.
    fontsize : int, default 12
        Size of the leaf labels (the x axis for ``"top"``/``"bottom"``, the
        y axis for ``"left"``/``"right"``).
    orientation : str, default "bottom"
        Dendrogram orientation passed to :func:`scipy.cluster.hierarchy.dendrogram`.
    method : str, default "average"
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`
        (for example ``"average"``, ``"complete"`` or ``"single"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    missing_values : list, optional
        Extra sentinel values to treat as missing.
    **kwargs
        Forwarded to :func:`scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``method`` is not a valid linkage method.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from missingly.visualisation.static import dendrogram
    >>> frame = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [np.nan, 2, 3, 4],
    ...                       "c": [1, 2, np.nan, np.nan]})
    >>> ax = dendrogram(frame, method="complete")
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    df = _apply_sentinels(df, missing_values)
    null_mat = _null_matrix(df)
    cols_with_missing = [c for c in null_mat.columns if null_mat[c].any()]
    if len(cols_with_missing) < 2:
        if ax is not None:
            ax_ = ax
        else:
            _, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)
        ax_.text(0.5, 0.5, "Not enough columns with missing values",
                  ha="center", va="center", transform=ax_.transAxes)
        ax_.set_title("Dendrogram of Variables by Missing Data Patterns")
        return ax_

    sub = null_mat[cols_with_missing].astype(float)
    corr = sub.corr(method="pearson").fillna(0)
    dist = 1 - corr.abs()
    # copy to avoid read-only array issue
    dist_values = np.array(dist.values, copy=True)
    np.fill_diagonal(dist_values, 0)
    linkage = hierarchy.linkage(
        squareform(np.clip(dist_values, 0, None)), method=method
    )

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    col_labels = _rtl_safe_labels(cols_with_missing, ax=ax_)
    hierarchy.dendrogram(linkage, labels=col_labels, orientation=orientation, ax=ax_, **kwargs)
    ax_.set_title("Dendrogram of Variables by Missing Data Patterns")
    label_axis = "y" if orientation in {"left", "right"} else "x"
    ax_.tick_params(axis=label_axis, labelsize=fontsize)
    _clean_ax(ax_)
    return ax_


# ===========================================================================
# vis_miss
# ===========================================================================

@_guarded()
def vis_miss(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 12,
    sort: bool = False,
    cluster: bool = False,
    ax: Any = None,
    backend: str = "matplotlib",
    interactive: bool = False,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Pixel-level heatmap of present / missing values (naniar-style).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose missing cells are drawn.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    fontsize : int, default 12
        Font size for tick and axis labels.
    sort : bool, default False
        Whether to order columns by descending missing share.
    cluster : bool, default False
        Whether to order rows by hierarchical clustering.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.  ``interactive=True`` also selects Plotly.
    interactive : bool, default False
        Whether to return an interactive Plotly figure.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Forwarded to the Plotly backend.  On the Matplotlib backend they are
        ignored.

    Returns
    -------
    matplotlib.axes.Axes or plotly Figure
        The heatmap axes, or an interactive Plotly figure.

    Raises
    ------
    ImportError
        If Plotly output is requested but Plotly is not installed.

    Warns
    -----
    FutureWarning
        If an unsupported keyword argument or an unknown *backend* is given;
        a future release will raise instead.
    UserWarning
        If styling arguments are given together with the Plotly backend,
        which does not honour them.

    Examples
    --------
    >>> import pandas as pd
    >>> ax = vis_miss(pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]}))
    >>> ax.get_xlabel()
    '25.0% missing'
    """
    if _use_plotly("vis_miss", backend, interactive):
        _require_plotly()
        _warn_ignored_on_plotly(vis_miss, figsize=figsize, fontsize=fontsize, ax=ax)
        from missingly.visualisation.interactive import _vis_miss_plotly
        return _vis_miss_plotly(
            _apply_sentinels(df, missing_values),
            sort=sort,
            cluster=cluster,
            **kwargs,
        )

    _warn_unsupported_kwargs("vis_miss", kwargs)
    df = _apply_sentinels(df, missing_values)
    null_mat = _null_matrix(df)

    if sort:
        order = null_mat.mean().sort_values(ascending=False, kind="stable").index
        null_mat = null_mat[order]

    if cluster and len(df) > 1:
        from scipy.cluster import hierarchy
        data = null_mat.astype(float).values
        try:
            lnk = hierarchy.linkage(data, method="ward", metric="euclidean")
            null_mat = null_mat.iloc[hierarchy.leaves_list(lnk)]
        except ValueError as exc:
            warnings.warn(
                "vis_miss: clustering could not form valid distances "
                f"({exc}); using original order.",
                UserWarning,
                stacklevel=2,
            )

    n_rows, n_cols = null_mat.shape
    if figsize is None:
        figsize = (min(12, max(4, n_cols * 0.6)), min(8, max(3, n_rows * 0.06)))

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    ax_.imshow(null_mat.astype(float).values, aspect="auto", interpolation="none",
               cmap=plt.get_cmap("gray_r"), vmin=0, vmax=1)
    ax_.set_xlim(-0.5, n_cols - 0.5)
    ax_.set_ylim(n_rows - 0.5, -0.5)

    col_labels = _rtl_safe_labels(list(null_mat.columns), ax=ax_)
    ax_.set_xticks(range(n_cols))
    ax_.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    ax_.set_yticks([])
    for spine in ax_.spines.values():
        spine.set_visible(False)
    ax_.set_xlabel(f"{null_mat.values.mean() * 100:.1f}% missing", fontsize=fontsize, labelpad=6)
    return ax_


# ===========================================================================
# miss_cluster  – heatmap after row+col clustering
# ===========================================================================

@_guarded()
def miss_cluster(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "gray_r",
    fontsize: int = 10,
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Clustered heatmap of missing indicator matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data whose missing indicators are clustered by rows and columns.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    cmap : str, default "gray_r"
        Matplotlib colormap.
    fontsize : int, default 10
        Font size for column labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Unsupported; this function has no interactive backend.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that were drawn on.

    Warns
    -----
    FutureWarning
        If any keyword argument (including ``interactive`` or ``method``) is
        given; it is ignored and a future release will raise ``TypeError``.
    UserWarning
        If clustering cannot form valid distances and the original order is
        used.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, 3.0]})
    >>> ax = miss_cluster(frame)
    >>> ax.get_title()
    'Clustered missing indicator matrix'
    """
    _warn_unsupported_kwargs("miss_cluster", kwargs)
    df = _apply_sentinels(df, missing_values)
    null_mat = _null_matrix(df).astype(float)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        row_order = leaves_list(linkage(null_mat.values, method="ward", metric="euclidean"))
        col_order = leaves_list(linkage(null_mat.values.T, method="ward", metric="euclidean"))
        null_mat = null_mat.iloc[row_order, col_order]
    except ValueError as exc:
        warnings.warn(
            "miss_cluster: clustering could not form valid distances "
            f"({exc}); using original order.",
            UserWarning,
            stacklevel=2,
        )

    n_rows, n_cols = null_mat.shape
    if figsize is None:
        figsize = (min(12, max(4, n_cols * 0.6)), min(8, max(3, n_rows * 0.08)))

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    ax_.imshow(null_mat.values, aspect="auto", interpolation="none",
               cmap=plt.get_cmap(cmap), vmin=0, vmax=1)
    col_labels = _rtl_safe_labels(list(null_mat.columns), ax=ax_)
    ax_.set_xticks(range(n_cols))
    ax_.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    ax_.set_yticks([])
    ax_.set_title("Clustered missing indicator matrix")
    return ax_


# ===========================================================================
# miss_row_profile  – per-row completeness summary
# ===========================================================================

@_guarded(keeps=_KEEP_NO_ROWS_OR_COLS)
def miss_row_profile(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 4),
    color: str = "#4C72B0",
    fontsize: int = 11,
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Histogram of per-row completeness fractions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. The function does not mutate it.
    figsize : tuple of float, default (8, 4)
        Figure size used only when *ax* is not supplied.
    color : str, default "#4C72B0"
        Histogram bar colour.
    fontsize : int, default 11
        Font size for axis labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    missing_values : sequence, optional
        Additional sentinel values treated as missing.
    **kwargs : Any
        Unsupported.

    Returns
    -------
    matplotlib.axes.Axes
        The supplied or newly created axes. Empty row or column inputs return
        annotated axes instead of attempting an undefined histogram.

    Warns
    -----
    FutureWarning
        If any keyword argument is given; it is ignored and a future release
        will raise ``TypeError``.

    Examples
    --------
    >>> import pandas as pd
    >>> ax = miss_row_profile(pd.DataFrame({"score": [1.0, None]}))
    >>> ax.get_title()
    'Row Missingness Profile'
    """
    _warn_unsupported_kwargs("miss_row_profile", kwargs)
    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    df = _apply_sentinels(df, missing_values)
    if df.empty or df.shape[1] == 0:
        ax_.text(
            0.5,
            0.5,
            "No row completeness data to display",
            ha="center",
            va="center",
            transform=ax_.transAxes,
        )
        ax_.set_title("Row Missingness Profile")
        _clean_ax(ax_)
        return ax_

    completeness = 1 - df.isnull().mean(axis=1)

    ax_.hist(completeness.values, bins=min(30, len(df)), color=color, edgecolor="white")
    ax_.set_xlabel("Row completeness", fontsize=fontsize)
    ax_.set_ylabel("Count", fontsize=fontsize)
    ax_.set_title("Row Missingness Profile")
    ax_.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    _clean_ax(ax_)
    return ax_


# ===========================================================================
# shadow_scatter  – scatter with missingness shadow
# ===========================================================================

@_guarded(raises=_RAISE_NO_COLS)
def shadow_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    shadow_col: Optional[str] = None,
    hue: Optional[str] = None,
    figsize: tuple[float, float] = (7, 5),
    color_present: str = "#4C72B0",
    color_missing: str = "#dd8452",
    alpha: float = 0.7,
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Scatter with margin rug for rows missing *x* or *y*.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    x, y : str
        Columns plotted on the horizontal and vertical axes.
    shadow_col : str, optional
        Column to use for colouring shadow points (alias for hue).
    hue : str, optional
        Column whose categories colour the present points.
    figsize : tuple of float, default (7, 5)
        Figure size used when *ax* is not given.
    color_present, color_missing : str
        Colours of present points and of the missing-value rug.
    alpha : float, default 0.7
        Point opacity.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Unsupported; this function has no interactive backend.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that were drawn on.

    Warns
    -----
    FutureWarning
        If any keyword argument (including ``interactive``) is given; it is
        ignored and a future release will raise ``TypeError``.

    Raises
    ------
    ValueError
        If the frame has no columns.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"x": [1.0, None, 3.0], "y": [2.0, 2.0, None]})
    >>> ax = shadow_scatter(frame, "x", "y")
    >>> ax.get_xlabel()
    'x'
    """
    _warn_unsupported_kwargs("shadow_scatter", kwargs)
    df = _apply_sentinels(df, missing_values)
    # shadow_col is an alias for hue
    if shadow_col is not None and hue is None:
        hue = shadow_col

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    xv = pd.to_numeric(df[x], errors="coerce")
    yv = pd.to_numeric(df[y], errors="coerce")
    present = xv.notna() & yv.notna()

    if hue is not None and hue in df.columns:
        categories = df.loc[present, hue].astype(str).unique()
        for cat in categories:
            mask = present & (df[hue].astype(str) == cat)
            ax_.scatter(xv[mask], yv[mask], alpha=alpha, label=str(cat))
    else:
        ax_.scatter(xv[present], yv[present], c=color_present, alpha=alpha, label="Present")

    x_rng = (xv.max() - xv.min()) if present.any() else 1.0
    y_rng = (yv.max() - yv.min()) if present.any() else 1.0
    j = 0.02

    xm = xv.isna() & yv.notna()
    if xm.any():
        ax_.scatter(np.full(xm.sum(), xv.min() - j * x_rng), yv[xm],
                    c=color_missing, alpha=alpha, marker="|", label=f"{x} missing")

    ym = yv.isna() & xv.notna()
    if ym.any():
        ax_.scatter(xv[ym], np.full(ym.sum(), yv.min() - j * y_rng),
                    c=color_missing, alpha=alpha, marker="_", label=f"{y} missing")

    xl, yl = _rtl_safe(str(x)), _rtl_safe(str(y))
    if _is_rtl(str(x)) or _is_rtl(str(y)):
        _apply_rtl_font(ax_)
    ax_.set_xlabel(xl); ax_.set_ylabel(yl)
    title = f"Shadow scatter: {xl} vs {yl}"
    if hue is not None:
        title += f" by {_rtl_safe(str(hue))}"
    ax_.set_title(title)
    ax_.legend(fontsize=9)
    _clean_ax(ax_)
    return ax_


# ===========================================================================
# scatter_miss  – alias kept for API compatibility
# ===========================================================================

def scatter_miss(
    df: pd.DataFrame,
    x: str,
    y: str,
    **kwargs: Any,
) -> Any:
    """Alias for :func:`shadow_scatter`."""
    return shadow_scatter(df, x, y, **kwargs)


# ===========================================================================
# vis_miss_fct  – vis_miss faceted by a grouping variable
# ===========================================================================

@_guarded(raises=_RAISE_NO_ROWS_OR_COLS)
def vis_miss_fct(
    df: pd.DataFrame,
    fct: str,
    *,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Faceted vis_miss split by a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    fct : str
        Column whose distinct values define the facets.
    figsize : tuple of float, optional
        Figure size; a size proportional to the facet count is used if omitted.
    fontsize : int, default 10
        Font size for labels and facet titles.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Forwarded to :func:`vis_miss` for every facet.

    Returns
    -------
    matplotlib.axes.Axes
        The last facet's axes; the others are reachable through
        ``ax.figure.axes``.  This function has no *ax* keyword.

    Warns
    -----
    FutureWarning
        If :func:`vis_miss` does not support a keyword argument; it is ignored
        and a future release will raise ``TypeError``.

    Raises
    ------
    ValueError
        If the frame has no rows or no columns.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"g": ["a", "b"], "x": [1.0, None]})
    >>> ax = vis_miss_fct(frame, "g")
    """
    df = _apply_sentinels(df, missing_values)
    groups = df[fct].unique()
    n_groups = len(groups)
    data_cols = [c for c in df.columns if c != fct]

    if figsize is None:
        figsize = (max(6, len(data_cols) * 0.7), 3 * n_groups)

    fig, axes = plt.subplots(n_groups, 1, figsize=figsize, constrained_layout=True)
    if n_groups == 1:
        axes = [axes]

    for ax_, grp in zip(axes, groups):
        sub = df.loc[df[fct] == grp, data_cols]
        vis_miss(sub, ax=ax_, fontsize=fontsize, **kwargs)
        ax_.set_title(f"{fct} = {grp}", fontsize=fontsize)

    return axes[-1]


# ===========================================================================
# vis_miss_by_group  – aggregate % missing per group
# ===========================================================================

@_guarded(raises=_RAISE_NO_COLS)
def vis_miss_by_group(
    df: pd.DataFrame,
    group_col: str,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "YlOrRd",
    fontsize: int = 11,
    annot: bool = True,
    fmt: str = ".0f",
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Heatmap of % missing per column per group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    group_col : str
        Column to group by.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    cmap : str, default "YlOrRd"
        Matplotlib colormap.
    fontsize : int, default 11
        Tick-label font size.
    annot : bool, default True
        Whether to annotate cells with their values.
    fmt : str, default ".0f"
        Numeric format used for annotations.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Forwarded to :func:`seaborn.heatmap`.

    Returns
    -------
    matplotlib.axes.Axes
        The heatmap axes.

    Raises
    ------
    ValueError
        If the frame has no columns.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"g": ["a", "b"], "x": [1.0, None]})
    >>> ax = vis_miss_by_group(frame, "g", fontsize=9)
    """
    df = _apply_sentinels(df, missing_values)
    data_cols = [c for c in df.columns if c != group_col]
    pct = df.groupby(group_col)[data_cols].apply(lambda g: g.isnull().mean() * 100)

    n_grp, n_col = pct.shape
    if figsize is None:
        figsize = (max(4, n_col * 0.8), max(3, n_grp * 0.6))

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    col_labels = _rtl_safe_labels(list(pct.columns), ax=ax_)
    pct.columns = col_labels
    sns.heatmap(pct, ax=ax_, cmap=cmap, annot=annot, fmt=fmt,
                vmin=0, vmax=100, linewidths=0.3, **kwargs)
    ax_.tick_params(axis="both", labelsize=fontsize)
    ax_.set_title(f"% missing per column by '{group_col}'")
    return ax_


# ===========================================================================
# vis_impute_dist  – before/after imputation distributions
# ===========================================================================

@_guarded(raises=_RAISE_NO_COLS)
def vis_impute_dist(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    column: str,
    *,
    figsize: tuple[float, float] = (8, 4),
    bins: int = 30,
    color_before: str = "#4C72B0",
    color_after: str = "#dd8452",
    fontsize: int = 11,
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Overlay histograms comparing a column before and after imputation.

    Parameters
    ----------
    column : str
        Column to compare.

    Returns
    -------
    matplotlib.axes.Axes
    Raises
    ------
    ValueError
        If the frame has no columns.

    """
    df_before = _apply_sentinels(df_before, missing_values)
    df_after = _apply_sentinels(df_after, missing_values)

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    before_vals = pd.to_numeric(df_before[column].dropna(), errors="coerce").dropna()
    after_vals = pd.to_numeric(df_after[column].dropna(), errors="coerce").dropna()

    all_vals = pd.concat([before_vals, after_vals])
    edges = np.histogram_bin_edges(all_vals.dropna(), bins=bins)

    ax_.hist(before_vals, bins=edges, alpha=0.6, color=color_before, label="Before", **kwargs)
    ax_.hist(after_vals, bins=edges, alpha=0.6, color=color_after, label="After", **kwargs)
    col_label = _rtl_safe(str(column))
    ax_.set_xlabel(col_label, fontsize=fontsize)
    ax_.set_ylabel("Count", fontsize=fontsize)
    ax_.set_title(f"Distribution of '{col_label}': before vs after imputation")
    ax_.legend(fontsize=fontsize)
    _clean_ax(ax_)
    return ax_


# ===========================================================================
# miss_impute_compare  – multiple imputers side-by-side
# ===========================================================================

@_guarded(keeps=_KEEP_NO_ROWS_COLS_OR_MISSING, result="figure")
def miss_impute_compare(
    original: pd.DataFrame,
    imputed: pd.DataFrame | Dict[str, pd.DataFrame],
    column: Optional[str] = None,
    *,
    columns: Optional[list[str]] = None,
    figsize: tuple[float, float] | None = None,
    bins: int = 30,
    fontsize: int = 11,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Grid of histograms comparing columns across multiple imputation results.

    Parameters
    ----------
    original : pd.DataFrame
        Data before imputation.
    imputed : pd.DataFrame or dict[str, pd.DataFrame]
        Either a single imputed DataFrame or a mapping of label -> DataFrame.
    column : str, optional
        Single column to compare.  If None and imputed is a DataFrame, all
        numeric columns with missing values in *original* are compared.
    columns : list of str, optional
        Explicit ordered subset of numeric columns to compare. This cannot
        be combined with *column*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    original = _apply_sentinels(original, missing_values)

    # Normalise imputed to a dict
    if isinstance(imputed, pd.DataFrame):
        imputed_dict: Dict[str, pd.DataFrame] = {"Imputed": imputed}
    else:
        imputed_dict = imputed

    # Determine columns to compare
    if column is not None and columns is not None:
        raise ValueError("Use either column or columns, not both")
    if columns is not None:
        cols_to_plot = list(columns)
    elif column is not None:
        cols_to_plot = [column]
    else:
        cols_to_plot = [
            c for c in original.columns
            if original[c].isnull().any() and pd.api.types.is_numeric_dtype(original[c])
        ]
        if not cols_to_plot:
            raise ValueError(
                "No numeric columns with missing values are available to compare"
            )

    unknown = [name for name in cols_to_plot if name not in original.columns]
    if unknown:
        raise ValueError(f"Columns not found in original data: {unknown}")

    n_cols_plot = len(cols_to_plot)
    n_methods = len(imputed_dict) + 1  # original + each method

    if figsize is None:
        figsize = (4 * n_methods, 4 * n_cols_plot)

    fig, axes = plt.subplots(
        n_cols_plot, n_methods,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )

    method_items = [("Original", original)] + list(imputed_dict.items())

    for row_i, col in enumerate(cols_to_plot):
        col_label = _rtl_safe(str(col))
        all_vals_col = pd.concat([d[col] for _, d in method_items if col in d.columns])
        edges = np.histogram_bin_edges(
            pd.to_numeric(all_vals_col, errors="coerce").dropna(), bins=bins
        )
        for col_j, (label, data) in enumerate(method_items):
            ax_ = axes[row_i, col_j]
            if col in data.columns:
                vals = pd.to_numeric(data[col], errors="coerce").dropna()
                ax_.hist(vals, bins=edges, color="#4C72B0", edgecolor="white", **kwargs)
            ax_.set_title(label, fontsize=fontsize)
            ax_.set_xlabel(col_label, fontsize=fontsize)
            _clean_ax(ax_)
        axes[row_i, 0].set_ylabel(col_label, fontsize=fontsize)

    fig.suptitle("Imputation comparison", fontsize=fontsize + 1)
    return fig


# ===========================================================================
# vis_miss_cumsum_var  – cumulative missingness by variable
# ===========================================================================

@_guarded()
def vis_miss_cumsum_var(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 4),
    color: str = "#4C72B0",
    fontsize: int = 11,
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Line plot of cumulative % missing as variables are added.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = _apply_sentinels(df, missing_values)
    pct = df.isnull().mean().sort_values(ascending=False) * 100
    cumsum = pct.cumsum() / pct.sum() * 100

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    x = np.arange(1, len(cumsum) + 1)
    ax_.plot(x, cumsum.values, color=color, marker="o", linewidth=2, **kwargs)
    col_labels = _rtl_safe_labels(list(pct.index), ax=ax_)
    ax_.set_xticks(x)
    ax_.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize - 1)
    ax_.set_ylabel("Cumulative % of total missingness", fontsize=fontsize)
    ax_.set_title("Cumulative missingness by variable")
    ax_.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax_.set_ylim(0, 105)
    _clean_ax(ax_)
    return ax_


# ===========================================================================
# vis_miss_cumsum_case  – cumulative missingness by case
# ===========================================================================

@_guarded()
def vis_miss_cumsum_case(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 4),
    color: str = "#dd8452",
    fontsize: int = 11,
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Line plot of cumulative % missing as rows are added.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = _apply_sentinels(df, missing_values)
    row_miss = df.isnull().mean(axis=1).sort_values(ascending=False)
    cumsum = row_miss.cumsum() / row_miss.sum() * 100

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    ax_.plot(np.arange(1, len(cumsum) + 1), cumsum.values, color=color, linewidth=2, **kwargs)
    ax_.set_xlabel("Cases (sorted by missingness)", fontsize=fontsize)
    ax_.set_ylabel("Cumulative % of total missingness", fontsize=fontsize)
    ax_.set_title("Cumulative missingness by case")
    ax_.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax_.set_ylim(0, 105)
    _clean_ax(ax_)
    return ax_


# ===========================================================================
# vis_miss_span  – rolling window missingness over time
# ===========================================================================

@_guarded(raises=_RAISE_NO_COLS)
def vis_miss_span(
    df: pd.DataFrame,
    column: str,
    span_every: int = 20,
    *,
    span: Optional[int] = None,
    figsize: tuple[float, float] = (10, 4),
    color: str = "#4C72B0",
    fontsize: int = 11,
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Rolling-window missing rate for a single column over the row index.

    Parameters
    ----------
    column : str
        Column name to analyse.
    span_every : int
        Rolling window size (preferred parameter name).
    span : int, optional
        Alias for *span_every* (kept for backwards compatibility).

    Returns
    -------
    matplotlib.axes.Axes
    Raises
    ------
    ValueError
        If the frame has no columns.

    """
    df = _apply_sentinels(df, missing_values)
    window = span if span is not None else span_every
    miss = df[column].isnull().astype(float)
    rolling = miss.rolling(window=window, min_periods=1).mean() * 100

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    ax_.plot(rolling.index, rolling.values, color=color, linewidth=1.5, **kwargs)
    ax_.fill_between(rolling.index, rolling.values, alpha=0.15, color=color)
    col_label = _rtl_safe(str(column))
    ax_.set_xlabel("Row index", fontsize=fontsize)
    ax_.set_ylabel(f"Rolling {window}-row miss rate (%)", fontsize=fontsize)
    ax_.set_title(f"Missingness span: '{col_label}' (window={window})")
    ax_.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    _clean_ax(ax_)
    return ax_


# ===========================================================================
# vis_parallel_coords  – parallel coordinates coloured by missingness
# ===========================================================================

@_guarded(keeps=_KEEP_NO_ROWS_COLS_OR_MISSING)
def vis_parallel_coords(
    df: pd.DataFrame,
    *,
    color_complete: str = "#4C72B0",
    color_missing: str = "#dd8452",
    alpha: float = 0.3,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 11,
    ax: Any = None,
    missing_values=None,
    **kwargs: Any,
) -> Any:
    """Parallel coordinates plot coloured by whether any value is missing.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data; only numeric columns are drawn.
    color_complete, color_missing : str
        Line colours for rows without and with a missing value; the legend
        names them.
    alpha : float, default 0.3
        Line opacity.
    figsize : tuple of float, optional
        Figure size for a newly-created figure.
    fontsize : int, default 11
        Font size for column labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    missing_values : optional
        Additional sentinel value or values treated as missing.
    **kwargs : Any
        Unsupported; this function has no interactive backend.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that were drawn on.

    Warns
    -----
    FutureWarning
        If any keyword argument (including ``interactive``) is given; it is
        ignored and a future release will raise ``TypeError``.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": [2.0, 2.5, 1.0]})
    >>> ax = vis_parallel_coords(frame)
    >>> ax.get_title()
    'Parallel coordinates'
    """
    _warn_unsupported_kwargs("vis_parallel_coords", kwargs)
    df = _apply_sentinels(df, missing_values)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        if ax is not None:
            ax_ = ax
        else:
            _, ax_ = plt.subplots(figsize=figsize or (6, 4), constrained_layout=True)
        ax_.text(0.5, 0.5, "No numeric columns available",
                  ha="center", va="center", transform=ax_.transAxes)
        return ax_

    norm = (num_df - num_df.min()) / (num_df.max() - num_df.min()).replace(0, 1)
    has_missing = df.isnull().any(axis=1)

    n_cols = norm.shape[1]
    if figsize is None:
        figsize = (max(6, n_cols * 1.2), 4)

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    xs = np.arange(n_cols)
    for idx in range(len(norm)):
        row = norm.iloc[idx].values
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        color = color_missing if has_missing.iloc[idx] else color_complete
        ax_.plot(xs[valid], row[valid], color=color, alpha=alpha, linewidth=0.8)

    col_labels = _rtl_safe_labels(list(num_df.columns), ax=ax_)
    ax_.set_xticks(xs)
    ax_.set_xticklabels(col_labels, fontsize=fontsize)
    ax_.set_yticks([])
    ax_.set_title("Parallel coordinates")
    _clean_ax(ax_)

    from matplotlib.lines import Line2D
    ax_.legend(
        handles=[
            Line2D([0], [0], color=color_complete, label="Complete"),
            Line2D([0], [0], color=color_missing, label="Has missing"),
        ],
        fontsize=fontsize - 1,
    )
    return ax_
