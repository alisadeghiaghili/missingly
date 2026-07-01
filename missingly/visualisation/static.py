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
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from missingly.visualisation._base import (
    _apply_rtl_font,
    _is_rtl,
    _nullity,
    _rtl_safe,
    _rtl_safe_labels,
)
from missingly.visualisation.interactive import (
    _bar_plotly,
    _heatmap_plotly,
    _matrix_plotly,
    _miss_case_plotly,
    _miss_cooccurrence_plotly,
    _miss_patterns_plotly,
    _miss_var_pct_plotly,
    _upset_plotly,
    _vis_miss_plotly,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _null_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean DataFrame: True where *df* has a missing value."""
    return df.isnull()


def _pct_missing(df: pd.DataFrame) -> pd.Series:
    """Return the percentage of missing values per column."""
    return df.isnull().mean() * 100


def _set_datetime_yticks(
    ax: Any,
    index: pd.DatetimeIndex,
    freq: str,
    fontsize: int,
) -> None:
    """Set y-axis ticks to period boundaries of *freq* for datetime indices."""
    boundaries = pd.date_range(index.min(), index.max(), freq=freq)
    tick_positions = [np.searchsorted(index, dt) for dt in boundaries]
    tick_labels = [dt.strftime("%Y-%m-%d") for dt in boundaries]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=fontsize)


def _clean_ax(ax: Any, spines: Sequence[str] = ("top", "right")) -> None:
    for spine in spines:
        ax.spines[spine].set_visible(False)


# ===========================================================================
# matrix
# ===========================================================================

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

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple, optional
    color : tuple
        RGB for present cells.
    fontsize : int
    labels : bool
    sparkline : bool
    freq : str, optional
    ax : Axes, optional
    backend : str

    Returns
    -------
    Figure

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
        rgba[:, :, c] = np.where(present, color[c], 1.0)

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
        spark_ax.plot(completeness, range(n_rows), color=color, linewidth=1)
        spark_ax.set_ylim(n_rows - 0.5, -0.5)
        spark_ax.set_xlim(0, 1)
        spark_ax.set_xticks([0, 1])
        spark_ax.set_xticklabels(["0", str(n_cols)], fontsize=fontsize - 2)
        spark_ax.set_yticks([])
        for spine in spark_ax.spines.values():
            spine.set_visible(False)

    return fig


# ===========================================================================
# bar
# ===========================================================================

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
    df : pd.DataFrame
    figsize : tuple, optional
    color : str
    log : bool
    labels : bool
    fontsize : int
    sort : bool
    ax : Axes, optional
    backend : str

    Returns
    -------
    Figure

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
    ax.bar(x, pct.values, color=color)

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
    _clean_ax(ax)
    return fig


# ===========================================================================
# miss_case  – missing values per row
# ===========================================================================

def miss_case(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    color: str = "#4C72B0",
    fontsize: int = 12,
    sort: bool = True,
    ax: Any = None,
    backend: str = "matplotlib",
    **kwargs: Any,
) -> Figure | Any:
    """Bar chart of missing-value counts per row (case).

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple, optional
    color : str
    fontsize : int
    sort : bool
        Sort rows by descending missing count.
    ax : Axes, optional
    backend : str

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import miss_case
    >>> df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 2]})
    >>> fig = miss_case(df)
    """
    if backend == "plotly":
        return _miss_case_plotly(df, **kwargs)

    counts = df.isnull().sum(axis=1)
    if sort:
        counts = counts.sort_values(ascending=False)

    n = len(counts)
    if figsize is None:
        figsize = (min(14, max(4, n * 0.15)), 4)

    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.bar(np.arange(n), counts.values, color=color)
    ax.set_xlabel("Row index", fontsize=fontsize)
    ax.set_ylabel("# Missing", fontsize=fontsize)
    ax.set_title("Missing values per case (row)")
    if n <= 50:
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels([str(i) for i in counts.index], rotation=45, ha="right", fontsize=fontsize - 2)
    else:
        ax.set_xticks([])
    _clean_ax(ax)
    return fig


# ===========================================================================
# miss_var_pct  – % missing per variable
# ===========================================================================

def miss_var_pct(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    color: str = "#4C72B0",
    fontsize: int = 12,
    sort: bool = True,
    ax: Any = None,
    backend: str = "matplotlib",
    **kwargs: Any,
) -> Figure | Any:
    """Horizontal bar chart of percentage missing per variable.

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple, optional
    color : str
    fontsize : int
    sort : bool
    ax : Axes, optional
    backend : str

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import miss_var_pct
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = miss_var_pct(df)
    """
    if backend == "plotly":
        return _miss_var_pct_plotly(df, **kwargs)

    pct = df.isnull().mean() * 100
    if sort:
        pct = pct.sort_values(ascending=True)

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
    return fig


# ===========================================================================
# miss_which  – indicator grid (rows x missing cols)
# ===========================================================================

def miss_which(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Heatmap showing *which* cells are missing (rows × columns).

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple, optional
    fontsize : int
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import miss_which
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = miss_which(df)
    """
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
    return fig


# ===========================================================================
# upset
# ===========================================================================

def upset(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    min_subset_size: int = 1,
    backend: str = "matplotlib",
    **kwargs: Any,
) -> dict[str, Figure] | Any:
    """UpSet plot of missing-value co-occurrence patterns.

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple, optional
    min_subset_size : int
    backend : str
    **kwargs
        Forwarded to :class:`upsetplot.UpSet` (matplotlib) or plotly backend.

    Returns
    -------
    dict[str, Figure] or plotly Figure

    Raises
    ------
    ImportError
        When ``upsetplot`` is not installed (matplotlib backend only).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import upset
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> figs = upset(df)
    """
    if backend == "plotly":
        return _upset_plotly(df, **kwargs)

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


# ===========================================================================
# miss_patterns
# ===========================================================================

def miss_patterns(
    df: pd.DataFrame,
    *,
    top_n: int = 10,
    figsize: tuple[float, float] | None = None,
    color: str = "#4C72B0",
    fontsize: int = 11,
    ax: Any = None,
    backend: str = "matplotlib",
    **kwargs: Any,
) -> Figure | Any:
    """Bar chart of the most common missingness patterns.

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int
    figsize : tuple, optional
    color : str
    fontsize : int
    ax : Axes, optional
    backend : str

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import miss_patterns
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = miss_patterns(df)
    """
    if backend == "plotly":
        return _miss_patterns_plotly(df, top_n=top_n, **kwargs)

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
    ax_.set_title(f"Top-{top_n} missingness patterns")
    _clean_ax(ax_)
    return fig


# ===========================================================================
# miss_cooccurrence
# ===========================================================================

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
    **kwargs: Any,
) -> Figure | Any:
    """Co-occurrence heatmap of column missingness.

    Parameters
    ----------
    df : pd.DataFrame
    normalize : bool
        Normalise counts to fractions of total rows.
    figsize : tuple, optional
    cmap : str
    fontsize : int
    annot : bool
    fmt : str
    ax : Axes, optional
    backend : str

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import miss_cooccurrence
    >>> df = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [np.nan, np.nan, 3]})
    >>> fig = miss_cooccurrence(df)
    """
    if backend == "plotly":
        return _miss_cooccurrence_plotly(df, normalize=normalize, **kwargs)

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
    ax_.set_title("Missingness co-occurrence" + (" (fraction)" if normalize else " (count)"))
    return fig


# ===========================================================================
# heatmap
# ===========================================================================

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
) -> Any:
    """Nullity correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple, optional
    cmap : str
    fontsize : int
    annot : bool
    fmt : str
    vmin, vmax : float
    center : float
    linewidths : float
    ax : Axes, optional
    backend : str
    mask_insignificant : bool
    significance : float

    Returns
    -------
    matplotlib.axes.Axes
        The axes the heatmap was drawn into.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import heatmap
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan], "c": [1, 2, np.nan]})
    >>> ax = heatmap(df)
    """
    if backend == "plotly":
        return _heatmap_plotly(df, **kwargs)

    null_mat = _null_matrix(df)
    cols_with_missing = [c for c in null_mat.columns if null_mat[c].any()]
    if len(cols_with_missing) < 2:
        fig, ax_ = plt.subplots(figsize=figsize or (4, 3), constrained_layout=True)
        ax_.text(0.5, 0.5, "Not enough columns with missing values",
                 ha="center", va="center", transform=ax_.transAxes)
        return ax_

    sub = null_mat[cols_with_missing].astype(float)
    corr = sub.corr(method="pearson")
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

    try:
        import warnings as _w
        import matplotlib as _mpl
        from data_quality_toolkit.visualization import correlation_heatmap as _dqt_hm
        with _w.catch_warnings():
            _w.filterwarnings("ignore", category=_mpl.MatplotlibDeprecationWarning)
            _w.filterwarnings("ignore", message=r".*set_bad.*")
            _dqt_hm(
                corr, ax=ax_,
                mask_insignificant=mask_insignificant,
                significance=significance,
                n_obs=len(null_mat),
                cmap=cmap, annot=annot, fmt=fmt,
                vmin=vmin, vmax=vmax,
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
    return ax_


# ===========================================================================
# dendrogram
# ===========================================================================

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
    df : pd.DataFrame
    figsize : tuple
    fontsize : int
    orientation : str
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import dendrogram
    >>> df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 2], "c": [1, np.nan]})
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
    linkage = hierarchy.linkage(squareform(np.clip(dist.values, 0, None)), method="average")

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    col_labels = _rtl_safe_labels(cols_with_missing, ax=ax_)
    hierarchy.dendrogram(linkage, labels=col_labels, orientation=orientation, ax=ax_, **kwargs)
    ax_.tick_params(axis="x", labelsize=fontsize)
    _clean_ax(ax_)
    return fig


# ===========================================================================
# vis_miss
# ===========================================================================

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
    """Pixel-level heatmap of present / missing values (naniar-style).

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple, optional
    fontsize : int
    sort : bool
    cluster : bool
    ax : Axes, optional
    backend : str

    Returns
    -------
    Figure

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
            lnk = hierarchy.linkage(data, method="ward", metric="euclidean")
            null_mat = null_mat.iloc[hierarchy.leaves_list(lnk)]
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
    return fig


# ===========================================================================
# miss_cluster  – heatmap after row+col clustering
# ===========================================================================

def miss_cluster(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "gray_r",
    fontsize: int = 10,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Clustered heatmap of missing indicator matrix (rows and columns).

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple, optional
    cmap : str
    fontsize : int
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import miss_cluster
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = miss_cluster(df)
    """
    null_mat = _null_matrix(df).astype(float)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        row_order = leaves_list(linkage(null_mat.values, method="ward", metric="euclidean"))
        col_order = leaves_list(linkage(null_mat.values.T, method="ward", metric="euclidean"))
        null_mat = null_mat.iloc[row_order, col_order]
    except Exception:  # noqa: BLE001
        pass

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
    return fig


# ===========================================================================
# miss_row_profile  – per-row completeness summary
# ===========================================================================

def miss_row_profile(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 4),
    color: str = "#4C72B0",
    fontsize: int = 11,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Histogram of per-row completeness fractions.

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple
    color : str
    fontsize : int
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import miss_row_profile
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = miss_row_profile(df)
    """
    completeness = 1 - df.isnull().mean(axis=1)

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    ax_.hist(completeness.values, bins=min(30, len(df)), color=color, edgecolor="white")
    ax_.set_xlabel("Row completeness", fontsize=fontsize)
    ax_.set_ylabel("Count", fontsize=fontsize)
    ax_.set_title("Per-row completeness profile")
    ax_.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    _clean_ax(ax_)
    return fig


# ===========================================================================
# shadow_scatter  – scatter with missingness shadow
# ===========================================================================

def shadow_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: Optional[str] = None,
    figsize: tuple[float, float] = (7, 5),
    color_present: str = "#4C72B0",
    color_missing: str = "#dd8452",
    alpha: float = 0.7,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Scatter with margin rug for rows missing *x* or *y*.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str
        Column names.
    hue : str, optional
        Column used to colour complete-case points.
    figsize : tuple
    color_present : str
    color_missing : str
    alpha : float
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import shadow_scatter
    >>> df = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [2, 3, np.nan, 4]})
    >>> fig = shadow_scatter(df, "a", "b")
    """
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
            ax_.scatter(xv[mask], yv[mask], alpha=alpha, label=str(cat), **kwargs)
    else:
        ax_.scatter(xv[present], yv[present], c=color_present, alpha=alpha, label="Present", **kwargs)

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
    ax_.set_title(f"Shadow scatter: {xl} vs {yl}")
    ax_.legend(fontsize=9)
    _clean_ax(ax_)
    return fig


# ===========================================================================
# scatter_miss  – alias kept for API compatibility
# ===========================================================================

def scatter_miss(
    df: pd.DataFrame,
    x: str,
    y: str,
    **kwargs: Any,
) -> Figure:
    """Alias for :func:`shadow_scatter`.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str
    **kwargs
        Forwarded to :func:`shadow_scatter`.

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import scatter_miss
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [2, np.nan, 3]})
    >>> fig = scatter_miss(df, "a", "b")
    """
    return shadow_scatter(df, x, y, **kwargs)


# ===========================================================================
# vis_miss_fct  – vis_miss faceted by a grouping variable
# ===========================================================================

def vis_miss_fct(
    df: pd.DataFrame,
    fct: str,
    *,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    **kwargs: Any,
) -> Figure:
    """Faceted :func:`vis_miss` split by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
    fct : str
        Column to facet by.
    figsize : tuple, optional
    fontsize : int

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import vis_miss_fct
    >>> df = pd.DataFrame({"a": [1, np.nan, 3, np.nan], "g": ["A", "A", "B", "B"]})
    >>> fig = vis_miss_fct(df, "g")
    """
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

    return fig


# ===========================================================================
# vis_miss_by_group  – aggregate % missing per group
# ===========================================================================

def vis_miss_by_group(
    df: pd.DataFrame,
    group: str,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "YlOrRd",
    fontsize: int = 11,
    annot: bool = True,
    fmt: str = ".0f",
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Heatmap of % missing per column per group.

    Parameters
    ----------
    df : pd.DataFrame
    group : str
        Column to group by.
    figsize : tuple, optional
    cmap : str
    fontsize : int
    annot : bool
    fmt : str
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import vis_miss_by_group
    >>> df = pd.DataFrame({"a": [1, np.nan, 3, np.nan], "g": ["A", "A", "B", "B"]})
    >>> fig = vis_miss_by_group(df, "g")
    """
    data_cols = [c for c in df.columns if c != group]
    pct = df.groupby(group)[data_cols].apply(lambda g: g.isnull().mean() * 100)

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
    ax_.set_title(f"% missing per column by '{group}'")
    return fig


# ===========================================================================
# vis_impute_dist  – before/after imputation distributions
# ===========================================================================

def vis_impute_dist(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    col: str,
    *,
    figsize: tuple[float, float] = (8, 4),
    bins: int = 30,
    color_before: str = "#4C72B0",
    color_after: str = "#dd8452",
    fontsize: int = 11,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Overlay histograms comparing a column before and after imputation.

    Parameters
    ----------
    df_before, df_after : pd.DataFrame
    col : str
    figsize : tuple
    bins : int
    color_before, color_after : str
    fontsize : int
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import vis_impute_dist
    >>> before = pd.DataFrame({"a": [1, np.nan, 3, np.nan, 5]})
    >>> after  = pd.DataFrame({"a": [1, 2.5,    3, 2.5,    5]})
    >>> fig = vis_impute_dist(before, after, "a")
    """
    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    before_vals = pd.to_numeric(df_before[col].dropna(), errors="coerce").dropna()
    after_vals = pd.to_numeric(df_after[col].dropna(), errors="coerce").dropna()

    all_vals = pd.concat([before_vals, after_vals])
    edges = np.histogram_bin_edges(all_vals.dropna(), bins=bins)

    ax_.hist(before_vals, bins=edges, alpha=0.6, color=color_before, label="Before", **kwargs)
    ax_.hist(after_vals, bins=edges, alpha=0.6, color=color_after, label="After", **kwargs)
    col_label = _rtl_safe(str(col))
    ax_.set_xlabel(col_label, fontsize=fontsize)
    ax_.set_ylabel("Count", fontsize=fontsize)
    ax_.set_title(f"Distribution of '{col_label}': before vs after imputation")
    ax_.legend(fontsize=fontsize)
    _clean_ax(ax_)
    return fig


# ===========================================================================
# miss_impute_compare  – multiple imputers side-by-side
# ===========================================================================

def miss_impute_compare(
    original: pd.DataFrame,
    imputed_dict: Dict[str, pd.DataFrame],
    col: str,
    *,
    figsize: tuple[float, float] | None = None,
    bins: int = 30,
    fontsize: int = 11,
    **kwargs: Any,
) -> Figure:
    """Grid of histograms comparing *col* across multiple imputation results.

    Parameters
    ----------
    original : pd.DataFrame
        Data before imputation.
    imputed_dict : dict[str, pd.DataFrame]
        Mapping of label → imputed DataFrame.
    col : str
        Column to compare.
    figsize : tuple, optional
    bins : int
    fontsize : int

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import miss_impute_compare
    >>> orig = pd.DataFrame({"a": [1, np.nan, 3, np.nan, 5]})
    >>> d = {"mean": pd.DataFrame({"a": [1, 3, 3, 3, 5]}), "zero": pd.DataFrame({"a": [1, 0, 3, 0, 5]})}
    >>> fig = miss_impute_compare(orig, d, "a")
    """
    n = len(imputed_dict) + 1
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True, sharey=True)
    col_label = _rtl_safe(str(col))
    all_vals = pd.concat(
        [original[col]] + [d[col] for d in imputed_dict.values()]
    )
    edges = np.histogram_bin_edges(pd.to_numeric(all_vals, errors="coerce").dropna(), bins=bins)

    for ax_, (label, data) in zip(axes, [("Original", original)] + list(imputed_dict.items())):
        vals = pd.to_numeric(data[col], errors="coerce").dropna()
        ax_.hist(vals, bins=edges, color="#4C72B0", edgecolor="white", **kwargs)
        ax_.set_title(label, fontsize=fontsize)
        ax_.set_xlabel(col_label, fontsize=fontsize)
        _clean_ax(ax_)

    axes[0].set_ylabel("Count", fontsize=fontsize)
    fig.suptitle(f"Imputation comparison: '{col_label}'", fontsize=fontsize + 1)
    return fig


# ===========================================================================
# vis_miss_cumsum_var  – cumulative missingness by variable
# ===========================================================================

def vis_miss_cumsum_var(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 4),
    color: str = "#4C72B0",
    fontsize: int = 11,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Line plot of cumulative % missing as variables are added (sorted desc).

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple
    color : str
    fontsize : int
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import vis_miss_cumsum_var
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = vis_miss_cumsum_var(df)
    """
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
    return fig


# ===========================================================================
# vis_miss_cumsum_case  – cumulative missingness by case
# ===========================================================================

def vis_miss_cumsum_case(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 4),
    color: str = "#dd8452",
    fontsize: int = 11,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Line plot of cumulative % missing as rows are added (sorted desc).

    Parameters
    ----------
    df : pd.DataFrame
    figsize : tuple
    color : str
    fontsize : int
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import vis_miss_cumsum_case
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
    >>> fig = vis_miss_cumsum_case(df)
    """
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
    return fig


# ===========================================================================
# vis_miss_span  – rolling window missingness over time
# ===========================================================================

def vis_miss_span(
    df: pd.DataFrame,
    col: str,
    *,
    span: int = 20,
    figsize: tuple[float, float] = (10, 4),
    color: str = "#4C72B0",
    fontsize: int = 11,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Rolling-window missing rate for a single column over the row index.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    span : int
        Rolling window size.
    figsize : tuple
    color : str
    fontsize : int
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import vis_miss_span
    >>> df = pd.DataFrame({"a": [np.nan if i % 3 == 0 else i for i in range(30)]})
    >>> fig = vis_miss_span(df, "a")
    """
    miss = df[col].isnull().astype(float)
    rolling = miss.rolling(window=span, min_periods=1).mean() * 100

    if ax is not None:
        fig = ax.figure
        ax_ = ax
    else:
        fig, ax_ = plt.subplots(figsize=figsize, constrained_layout=True)

    ax_.plot(rolling.index, rolling.values, color=color, linewidth=1.5, **kwargs)
    ax_.fill_between(rolling.index, rolling.values, alpha=0.15, color=color)
    col_label = _rtl_safe(str(col))
    ax_.set_xlabel("Row index", fontsize=fontsize)
    ax_.set_ylabel(f"Rolling {span}-row miss rate (%)", fontsize=fontsize)
    ax_.set_title(f"Missingness span: '{col_label}' (window={span})")
    ax_.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    _clean_ax(ax_)
    return fig


# ===========================================================================
# vis_parallel_coords  – parallel coordinates coloured by missingness
# ===========================================================================

def vis_parallel_coords(
    df: pd.DataFrame,
    *,
    color_complete: str = "#4C72B0",
    color_missing: str = "#dd8452",
    alpha: float = 0.3,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 11,
    ax: Any = None,
    **kwargs: Any,
) -> Figure:
    """Parallel coordinates plot coloured by whether any value is missing.

    Parameters
    ----------
    df : pd.DataFrame
    color_complete : str
    color_missing : str
    alpha : float
    figsize : tuple, optional
    fontsize : int
    ax : Axes, optional

    Returns
    -------
    Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.visualisation.static import vis_parallel_coords
    >>> df = pd.DataFrame({"a": [1, np.nan, 3], "b": [2, 3, np.nan], "c": [1, 2, 3]})
    >>> fig = vis_parallel_coords(df)
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        fig, ax_ = plt.subplots(figsize=figsize or (6, 4), constrained_layout=True)
        ax_.text(0.5, 0.5, "No numeric columns available",
                 ha="center", va="center", transform=ax_.transAxes)
        return fig

    # normalise each column to [0, 1]
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
        ax_.plot(xs[valid], row[valid], color=color, alpha=alpha, linewidth=0.8, **kwargs)

    col_labels = _rtl_safe_labels(list(num_df.columns), ax=ax_)
    ax_.set_xticks(xs)
    ax_.set_xticklabels(col_labels, fontsize=fontsize)
    ax_.set_yticks([])
    ax_.set_title("Parallel coordinates (orange = any missing)")
    _clean_ax(ax_)

    from matplotlib.lines import Line2D
    ax_.legend(
        handles=[
            Line2D([0], [0], color=color_complete, label="Complete"),
            Line2D([0], [0], color=color_missing, label="Has missing"),
        ],
        fontsize=fontsize - 1,
    )
    return fig
