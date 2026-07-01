"""Static (matplotlib / seaborn) visualisations for missing-data analysis.

All public functions return a :class:`matplotlib.figure.Figure` unless
otherwise noted.  Every function accepts an optional *ax* keyword that,
when provided, draws into that :class:`~matplotlib.axes.Axes` instead of
creating a new figure.

RTL support
-----------
Column names and index values that contain Arabic / Persian characters are
automatically reshaped and bidi-reordered via :func:`_rtl_safe_labels`
before being passed to matplotlib.
"""

from __future__ import annotations

import warnings
from typing import (
    Any,
    Sequence,
)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from missingly.visualisation._base import (
    _apply_rtl_font,
    _is_rtl,
    _rtl_safe,
    _rtl_safe_labels,
)
from missingly.visualisation.interactive import (
    _bar_plotly,
    _heatmap_plotly,
    _matrix_plotly,
    _vis_miss_plotly,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _null_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean DataFrame: True where *df* has a missing value."""
    return df.isnull()


def _pct_missing(df: pd.DataFrame) -> pd.Series:
    """Return the percentage of missing values per column."""
    return df.isnull().mean() * 100


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------

def matrix(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    color: tuple[float, float, float] = (0.25, 0.25, 0.25),
    fontsize: int = 12,
    labels: bool = True,
    sparkline: bool = True,
    freq: str | None = None,
    ax: Any = None,
    backend: str = "matplotlib",
    **kwargs: Any,
) -> Figure | Any:
    """Nullity matrix: a column-aligned grid of present / absent indicators.

    Each cell is filled (value present) or blank (value missing).  A
    sparkline on the right shows the row-wise completeness trend.

    Parameters
    ----------
    df:
        Input DataFrame.  Only the nullity structure is used.
    figsize:
        ``(width, height)`` in inches.  Defaults to ``(10, 5)``.
    color:
        RGB tuple for present-value cells.  Default is dark grey.
    fontsize:
        Axis tick label font size.
    labels:
        Whether to draw column-name tick labels on the x-axis.
    sparkline:
        Whether to draw the row-completeness sparkline.
    freq:
        Optional time-series frequency string for the x-axis (e.g.
        ``"MS"`` for month-start).
    ax:
        Optional :class:`~matplotlib.axes.Axes` to draw into.
    backend:
        ``"matplotlib"`` (default) or ``"plotly"``.
    **kwargs:
        Additional keyword arguments forwarded to the backend.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        The rendered figure.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import matrix
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = matrix(df)
    """
    if backend == "plotly":
        return _matrix_plotly(df, **kwargs)

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
            1, 2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [20, 1]},
            constrained_layout=True,
        )
    else:
        fig, main_ax = plt.subplots(figsize=figsize, constrained_layout=True)
        spark_ax = None

    # ------------------------------------------------------------------
    # Main heatmap
    # ------------------------------------------------------------------
    data = null_mat.to_numpy().astype(float)
    # present → color, missing → white
    rgba = np.ones((n_rows, n_cols, 4))
    present = ~null_mat.to_numpy()
    for c in range(3):
        rgba[:, :, c] = np.where(present, color[c], 1.0)

    main_ax.imshow(rgba, aspect="auto", interpolation="none")
    main_ax.set_xlim(-0.5, n_cols - 0.5)
    main_ax.set_ylim(n_rows - 0.5, -0.5)

    # column labels
    if labels:
        col_labels = _rtl_safe_labels(list(df.columns), ax=main_ax)
        main_ax.set_xticks(range(n_cols))
        main_ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    else:
        main_ax.set_xticks([])

    # y-axis: time or integer index
    if freq and isinstance(df.index, pd.DatetimeIndex):
        _set_datetime_yticks(main_ax, df.index, freq, fontsize)
    else:
        main_ax.set_yticks([])

    main_ax.tick_params(axis="x", which="both", bottom=False)
    main_ax.set_facecolor("white")
    for spine in main_ax.spines.values():
        spine.set_visible(False)

    # ------------------------------------------------------------------
    # Sparkline
    # ------------------------------------------------------------------
    if spark_ax is not None:
        completeness = present.sum(axis=1) / n_cols
        spark_ax.plot(completeness, range(n_rows), color=color, linewidth=1)
        spark_ax.set_ylim(n_rows - 0.5, -0.5)
        spark_ax.set_xlim(0, 1)
        spark_ax.set_xticks([0, 1])
        spark_ax.set_xticklabels(["0", str(n_cols)], fontsize=fontsize - 2)
        spark_ax.set_yticks([])
        for spine in spark_ax.spines.values():
            spine.set_visible(False)

    return fig


def _set_datetime_yticks(
    ax: Any,
    index: pd.DatetimeIndex,
    freq: str,
    fontsize: int,
) -> None:
    """Set y-axis ticks to period boundaries of *freq* for datetime indices."""
    boundaries = pd.date_range(index.min(), index.max(), freq=freq)
    tick_positions = [
        np.searchsorted(index, dt) for dt in boundaries
    ]
    tick_labels = [dt.strftime("%Y-%m-%d") for dt in boundaries]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=fontsize)


# ---------------------------------------------------------------------------
# bar
# ---------------------------------------------------------------------------

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
    **kwargs: Any,
) -> Figure | Any:
    """Bar chart of per-column missing-value counts.

    Parameters
    ----------
    df:
        Input DataFrame.
    figsize:
        ``(width, height)`` in inches.  Defaults to ``(10, 4)``.
    color:
        Bar fill colour.  Default is a muted blue.
    log:
        Use a logarithmic y-axis.
    labels:
        Whether to draw column-name tick labels on the x-axis.
    fontsize:
        Tick label font size.
    sort:
        Sort bars descending by missing count.
    ax:
        Optional :class:`~matplotlib.axes.Axes` to draw into.
    backend:
        ``"matplotlib"`` (default) or ``"plotly"``.
    **kwargs:
        Additional keyword arguments forwarded to the backend.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import bar
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = bar(df)
    """
    if backend == "plotly":
        return _bar_plotly(df, **kwargs)

    pct = _pct_missing(df)
    if sort:
        pct = pct.sort_values(ascending=False)

    n = len(pct)
    if figsize is None:
        figsize = (min(12, max(4, n * 0.6)), 4)

    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    col_labels = _rtl_safe_labels(list(pct.index), ax=ax)
    x = np.arange(n)
    ax.bar(x, pct.values, color=color, **kwargs)

    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    else:
        ax.set_xticks([])

    ax.set_ylabel("% Missing", fontsize=fontsize)
    ax.set_ylim(0, 100)
    if log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    return fig


# ---------------------------------------------------------------------------
# upset
# ---------------------------------------------------------------------------

def upset(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    min_subset_size: int = 1,
    **kwargs: Any,
) -> dict[str, Figure] | {}:
    """UpSet plot of missing-value co-occurrence patterns.

    Parameters
    ----------
    df:
        Input DataFrame.
    figsize:
        Figure size passed to :mod:`upsetplot`.
    min_subset_size:
        Minimum number of rows in a pattern to include.
    **kwargs:
        Additional keyword arguments forwarded to
        :class:`upsetplot.UpSet`.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        A dict with a single key ``"upset"`` mapped to the figure, or an
        empty dict when there are no missing values.

    Raises
    ------
    ImportError
        When the ``upsetplot`` package is not installed.  Install it with
        ``pip install upsetplot`` or ``pip install missingly[upset]``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import upset
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> figs = upset(df)
    """
    try:
        from upsetplot import UpSet, from_indicators  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'upsetplot' package is required for upset plots.  "
            "Install it with:  pip install upsetplot\n"
            "or:               pip install missingly[upset]"
        ) from exc

    null_mat = _null_matrix(df)
    if not null_mat.any(axis=None):
        return {}

    # upsetplot wants columns with at least one True
    cols_with_missing = [c for c in null_mat.columns if null_mat[c].any()]
    if not cols_with_missing:
        return {}

    data = from_indicators(null_mat[cols_with_missing])
    data = data[data >= min_subset_size]
    if data.empty:
        return {}

    us = UpSet(data, **kwargs)
    if figsize:
        us.figure_width = figsize[0]
        us.figure_height = figsize[1]
    fig_dict = us.plot()
    return {"upset": fig_dict["matrix"].figure}


# ---------------------------------------------------------------------------
# dendrogram
# ---------------------------------------------------------------------------

def dendrogram(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10, 4),
    fontsize: int = 12,
    orientation: str = "bottom",
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Hierarchical-clustering dendrogram of column nullity correlation.

    Parameters
    ----------
    df:
        Input DataFrame.
    figsize:
        Figure size in inches.
    fontsize:
        Tick label font size.
    orientation:
        Dendrogram orientation: ``"top"``, ``"bottom"``, ``"left"``,
        or ``"right"``.
    ax:
        Optional axes to draw into.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import dendrogram
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan], "c": [1, 2, np.nan]})
    >>> fig = dendrogram(df)
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    null_mat = _null_matrix(df)
    cols_with_missing = [c for c in null_mat.columns if null_mat[c].any()]
    if len(cols_with_missing) < 2:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)
        ax_.text(0.5, 0.5, "Not enough columns with missing values",
                 ha="center", va="center", transform=ax_.transAxes)
        return fig

    sub = null_mat[cols_with_missing].astype(float)
    corr = sub.corr(method="pearson").fillna(0)
    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0)
    linkage = hierarchy.linkage(
        squareform(np.clip(dist.values, 0, None)),
        method="average",
    )

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    col_labels = _rtl_safe_labels(cols_with_missing, ax=ax_)
    hierarchy.dendrogram(
        linkage,
        labels=col_labels,
        orientation=orientation,
        ax=ax_,
        **kwargs,
    )
    ax_.tick_params(axis="x", labelsize=fontsize)
    for spine in ["top", "right"]:
        ax_.spines[spine].set_visible(False)

    return fig


# ---------------------------------------------------------------------------
# heatmap
# ---------------------------------------------------------------------------

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
    mask_insignificant: bool = True,
    significance: float = 0.05,
    **kwargs: Any,
) -> Figure | Any:
    """Nullity correlation heatmap.

    Computes pairwise Pearson correlations between the boolean missingness
    indicators of each column pair and visualises them as a colour-coded
    heatmap.

    Parameters
    ----------
    df:
        Input DataFrame.
    figsize:
        ``(width, height)`` in inches.  Defaults to a size proportional
        to the number of columns.
    cmap:
        Matplotlib / seaborn colour map name.
    fontsize:
        Tick label font size.
    annot:
        Annotate cells with the correlation coefficient.
    fmt:
        Format string for cell annotations.
    vmin, vmax:
        Colour scale bounds.
    center:
        Colour scale centre value.
    linewidths:
        Width of the grid lines between cells.
    ax:
        Optional :class:`~matplotlib.axes.Axes` to draw into.
    backend:
        ``"matplotlib"`` (default) or ``"plotly"``.
    mask_insignificant:
        Mask cells where the correlation is not statistically significant
        (requires ``data_quality_toolkit``).
    significance:
        Significance threshold for ``mask_insignificant``.
    **kwargs:
        Additional keyword arguments forwarded to the backend.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import heatmap
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan], "c": [1, 2, np.nan]})
    >>> fig = heatmap(df)
    """
    if backend == "plotly":
        return _heatmap_plotly(df, **kwargs)

    null_mat = _null_matrix(df)
    cols_with_missing = [c for c in null_mat.columns if null_mat[c].any()]
    if len(cols_with_missing) < 2:
        fig, ax_ = plt.subplots(figsize=figsize or (4, 3), constrained_layout=True)
        ax_.text(0.5, 0.5, "Not enough columns with missing values",
                 ha="center", va="center", transform=ax_.transAxes)
        return fig

    sub = null_mat[cols_with_missing].astype(float)
    corr = sub.corr(method="pearson")

    # build mask: hide diagonal + perfect self-correlation
    diag_mask = np.eye(len(cols_with_missing), dtype=bool)
    final_mask = diag_mask

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

    # --- single DQT glue point ---
    try:
        import warnings as _w
        import matplotlib as _mpl
        from data_quality_toolkit.visualization import correlation_heatmap as _dqt_hm
        with _w.catch_warnings():
            _w.filterwarnings("ignore", category=_mpl.MatplotlibDeprecationWarning)
            _w.filterwarnings("ignore", message=r".*set_bad.*")
            _dqt_hm(
                corr,
                ax=ax_,
                mask_insignificant=mask_insignificant,
                significance=significance,
                n_obs=len(null_mat),
                cmap=cmap,
                annot=annot,
                fmt=fmt,
                vmin=vmin,
                vmax=vmax,
            )
    except ImportError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*set_bad.*")
            sns.heatmap(
                corr, mask=final_mask, ax=ax_,
                cmap=cmap, annot=annot, fmt=fmt,
                vmin=vmin, vmax=vmax, center=center, linewidths=linewidths,
                **kwargs,
            )

    return fig


# ---------------------------------------------------------------------------
# vis_miss  (tidy grid, inspired by naniar::vis_miss)
# ---------------------------------------------------------------------------

def vis_miss(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 12,
    sort: bool = False,
    cluster: bool = False,
    ax: Any = None,
    backend: str = "matplotlib",
    **kwargs: Any,
) -> Figure | Any:
    """Visualise the full missingness pattern of a DataFrame.

    Renders a pixel-level heatmap: each cell is coloured by whether the
    value is present (light) or missing (dark).  Inspired by
    ``naniar::vis_miss`` in R.

    Parameters
    ----------
    df:
        Input DataFrame.
    figsize:
        ``(width, height)`` in inches.
    fontsize:
        Tick label font size.
    sort:
        Sort columns by descending missingness fraction.
    cluster:
        Cluster rows by similarity of missingness pattern via
        hierarchical clustering.
    ax:
        Optional :class:`~matplotlib.axes.Axes` to draw into.
    backend:
        ``"matplotlib"`` (default) or ``"plotly"``.
    **kwargs:
        Additional keyword arguments forwarded to the backend.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import vis_miss
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = vis_miss(df)
    """
    if backend == "plotly":
        return _vis_miss_plotly(df, **kwargs)

    null_mat = _null_matrix(df)

    if sort:
        order = null_mat.mean().sort_values(ascending=False).index
        null_mat = null_mat[order]

    if cluster and len(df) > 1:
        from scipy.cluster import hierarchy
        data = null_mat.astype(float).values
        try:
            linkage = hierarchy.linkage(data, method="ward", metric="euclidean")
            row_order = hierarchy.leaves_list(linkage)
            null_mat = null_mat.iloc[row_order]
        except Exception:  # noqa: BLE001
            pass

    n_rows, n_cols = null_mat.shape
    if figsize is None:
        figsize = (min(12, max(4, n_cols * 0.6)), min(8, max(3, n_rows * 0.06)))

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    cmap_ = plt.get_cmap("gray_r")
    ax_.imshow(
        null_mat.astype(float).values,
        aspect="auto",
        interpolation="none",
        cmap=cmap_,
        vmin=0,
        vmax=1,
    )
    ax_.set_xlim(-0.5, n_cols - 0.5)
    ax_.set_ylim(n_rows - 0.5, -0.5)

    col_labels = _rtl_safe_labels(list(null_mat.columns), ax=ax_)
    ax_.set_xticks(range(n_cols))
    ax_.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=fontsize)
    ax_.set_yticks([])

    for spine in ax_.spines.values():
        spine.set_visible(False)

    # summary line below
    pct_total = null_mat.values.mean() * 100
    ax_.set_xlabel(
        f"{pct_total:.1f}% missing",
        fontsize=fontsize,
        labelpad=6,
    )

    return fig


# ---------------------------------------------------------------------------
# scatter  (missingness-aware scatter plot)
# ---------------------------------------------------------------------------

def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    figsize: tuple[float, float] = (7, 5),
    color_present: str = "#4C72B0",
    color_missing: str = "#dd8452",
    alpha: float = 0.7,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Scatter plot that highlights rows with missing values in *x* or *y*.

    Points where at least one of *x* / *y* is missing are jittered to
    the margin and coloured differently so patterns in missingness can be
    compared against the complete-case scatter.

    Parameters
    ----------
    df:
        Input DataFrame.
    x, y:
        Column names for the two axes.
    figsize:
        Figure size in inches.
    color_present:
        Colour for rows where both *x* and *y* are observed.
    color_missing:
        Colour for rows where at least one value is missing.
    alpha:
        Point opacity.
    ax:
        Optional axes.
    **kwargs:
        Forwarded to :func:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import scatter
    >>> df = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [2, 3, np.nan, 4]})
    >>> fig = scatter(df, "a", "b")
    """
    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    x_vals = pd.to_numeric(df[x], errors="coerce")
    y_vals = pd.to_numeric(df[y], errors="coerce")

    present = x_vals.notna() & y_vals.notna()

    ax_.scatter(
        x_vals[present], y_vals[present],
        c=color_present, alpha=alpha, label="Both present", **kwargs,
    )

    # jitter missing rows to the axes' margins
    x_range = x_vals.max() - x_vals.min() if present.any() else 1.0
    y_range = y_vals.max() - y_vals.min() if present.any() else 1.0
    jitter = 0.02

    x_miss = x_vals.isna() & y_vals.notna()
    if x_miss.any():
        jittered_x = np.full(x_miss.sum(), x_vals.min() - jitter * x_range)
        ax_.scatter(
            jittered_x, y_vals[x_miss],
            c=color_missing, alpha=alpha, marker="|", label=f"{x} missing",
        )

    y_miss = y_vals.isna() & x_vals.notna()
    if y_miss.any():
        jittered_y = np.full(y_miss.sum(), y_vals.min() - jitter * y_range)
        ax_.scatter(
            x_vals[y_miss], jittered_y,
            c=color_missing, alpha=alpha, marker="_", label=f"{y} missing",
        )

    x_label = _rtl_safe(str(x))
    y_label = _rtl_safe(str(y))
    if _is_rtl(str(x)) or _is_rtl(str(y)):
        _apply_rtl_font(ax_)

    ax_.set_xlabel(x_label)
    ax_.set_ylabel(y_label)
    ax_.set_title(f"Scatter: {x_label} vs {y_label}")
    ax_.legend(fontsize=9)
    for spine in ["top", "right"]:
        ax_.spines[spine].set_visible(False)

    return fig
