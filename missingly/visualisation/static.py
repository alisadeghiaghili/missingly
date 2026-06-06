"""Matplotlib-based static visualisations for missing data analysis.

Every public function follows these conventions:

* ``ax`` (optional) – embed the plot in an existing
  :class:`matplotlib.axes.Axes`.  If ``None``, a new figure is created.
* ``missing_values`` (optional) – extra sentinel values treated as missing
  (e.g. ``[-99, "N/A"]``).  Extends standard ``NaN`` / ``None`` detection.
* ``interactive`` (optional, default ``False``) – when ``True`` the function
  delegates to a Plotly backend in
  :mod:`missingly.visualisation.interactive` and returns a
  :class:`plotly.graph_objects.Figure` instead of
  :class:`matplotlib.axes.Axes`.  The ``ax`` parameter is ignored.
* ``**kwargs`` – forwarded to the underlying seaborn / matplotlib call.

Single-point DQT glue
---------------------
The *only* integration point with ``data_quality_toolkit`` lives in
:func:`heatmap` (static path).  When the package is available the
nullity-correlation matrix is rendered through
``data_quality_toolkit.visualization.correlation_heatmap``.  All other
functions are self-contained.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram, leaves_list
from scipy.spatial.distance import squareform, pdist

from missingly.visualisation._base import (
    _nullity,
    _pct_labels,
    _rtl_safe,
    _safe_labels,
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
# Basic visualisations
# ---------------------------------------------------------------------------

def matrix(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    interactive: bool = False,
    **kwargs,
):
    """Tile matrix visualising the *location* of missing data.

    Each cell is coloured to distinguish present (light) from missing (dark)
    values.  Useful as a first-pass "where is the data missing?" overview.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  May contain any dtypes.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  A new figure is created when ``None``.
    missing_values : list, optional
        Additional sentinel values treated as missing
        (e.g. ``[-99, "N/A"]``).  Combined with standard ``NaN`` detection.
    interactive : bool, default False
        When ``True`` returns a :class:`plotly.graph_objects.Figure`
        instead of :class:`matplotlib.axes.Axes`.  ``ax`` is ignored.
    **kwargs
        Forwarded to :func:`seaborn.heatmap`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot (static mode).
    plotly.graph_objects.Figure
        Interactive figure (when ``interactive=True``).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({
    ...     "age":    [25, np.nan, 30, np.nan, 45],
    ...     "income": [50000, 60000, np.nan, 70000, np.nan],
    ...     "city":   ["Berlin", "Paris", "Berlin", None, "Rome"],
    ... })
    >>> ax = visualise.matrix(df)                  # static
    >>> fig = visualise.matrix(df, interactive=True)  # Plotly
    """
    if interactive:
        return _matrix_plotly(df, missing_values)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    null_mat = _nullity(df, missing_values)
    sns.heatmap(null_mat, cbar=False, ax=ax, **kwargs)
    if df.shape[0] < 50:
        ax.set_yticks(np.arange(len(df)) + 0.5)
        ax.set_yticklabels(_safe_labels(df.index), rotation=0)
    else:
        ax.set_yticks([])
    ax.set_xticks(np.arange(len(df.columns)) + 0.5)
    ax.set_xticklabels(_safe_labels(df.columns), rotation=45, ha="right")
    ax.set_title("Missing Data Matrix")
    return ax


def vis_miss(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    show_pct: bool = True,
    cluster: bool = False,
    interactive: bool = False,
    **kwargs,
):
    """Annotated missingness matrix with per-column percentage labels.

    Similar to :func:`matrix` but adds missingness percentages to column
    labels and optionally clusters rows by their missing-data pattern.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
        Extra sentinel values (see :func:`matrix`).
    show_pct : bool, default True
        Append ``(x.x%)`` to each column label.
    cluster : bool, default False
        Reorder rows using Ward hierarchical clustering on hamming distances
        between their missingness bit-vectors.  Helps reveal structure in
        patterns across rows.
    interactive : bool, default False
        Return a :class:`plotly.graph_objects.Figure` when ``True``.
    **kwargs
        Forwarded to :func:`seaborn.heatmap`.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({
    ...     "a": [1, np.nan, 3, np.nan, 5],
    ...     "b": ["x", "y", None, "w", "v"],
    ...     "c": [np.nan, 2.0, 3.0, 4.0, np.nan],
    ... })
    >>> ax  = visualise.vis_miss(df)                       # pct labels
    >>> ax2 = visualise.vis_miss(df, cluster=True)         # clustered rows
    >>> fig = visualise.vis_miss(df, interactive=True)     # Plotly
    """
    if interactive:
        return _vis_miss_plotly(df, missing_values, show_pct, cluster)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, df.shape[1] * 1.2), 6))
    null_df = _nullity(df, missing_values).astype(float)
    if cluster and null_df.shape[0] > 1:
        row_dist = pdist(null_df.values, metric="hamming")
        row_order = leaves_list(linkage(row_dist, method="ward"))
        null_df = null_df.iloc[row_order]
    col_labels = _pct_labels(df, missing_values) if show_pct else _safe_labels(df.columns)
    cmap = kwargs.pop("cmap", ["#f0f0f0", "#d62728"])
    cbar = kwargs.pop("cbar", False)
    sns.heatmap(
        null_df, ax=ax, cmap=cmap, cbar=cbar,
        xticklabels=col_labels, yticklabels=False, **kwargs,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Missing Data Overview")
    plt.tight_layout()
    return ax


def bar(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    interactive: bool = False,
    **kwargs,
):
    """Bar chart of missing-value counts per column.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
        Extra sentinel values (see :func:`matrix`).
    interactive : bool, default False
        Return a Plotly figure with hover tooltips when ``True``.
    **kwargs
        Forwarded to :meth:`pandas.Series.plot` (``kind="bar"``).

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({
    ...     "a": [1, np.nan, 3],
    ...     "b": [np.nan, np.nan, 3],
    ...     "c": ["x", "y", "z"],
    ... })
    >>> ax  = visualise.bar(df)
    >>> fig = visualise.bar(df, interactive=True)
    """
    if interactive:
        return _bar_plotly(df, missing_values)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    miss_counts = _nullity(df, missing_values).sum()
    miss_counts.index = pd.Index(_safe_labels(miss_counts.index))
    miss_counts.plot(kind="bar", ax=ax, **kwargs)
    ax.set_title("Missing Values per Column")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of Missing Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return ax


def miss_case(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    interactive: bool = False,
    **kwargs,
):
    """Bar chart of missing-value counts per *row* (case).

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    interactive : bool, default False
        Plotly figure with hover info when ``True``.
    **kwargs
        Forwarded to :meth:`pandas.Series.plot` (``kind="bar"``).

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({
    ...     "a": [1, np.nan, 3, 4, np.nan],
    ...     "b": [np.nan, 2, np.nan, 4, 5],
    ...     "c": [1, 2, 3, np.nan, np.nan],
    ... })
    >>> ax  = visualise.miss_case(df)
    >>> fig = visualise.miss_case(df, interactive=True)
    """
    if interactive:
        return _miss_case_plotly(df, missing_values)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    miss_counts = _nullity(df, missing_values).sum(axis=1)
    miss_counts.plot(kind="bar", ax=ax, **kwargs)
    ax.set_title("Missing Values per Case")
    ax.set_xlabel("Cases (Rows)")
    ax.set_ylabel("Number of Missing Values")
    plt.tight_layout()
    return ax


def miss_var_pct(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    sort: bool = True,
    interactive: bool = False,
    **kwargs,
):
    """Horizontal bar chart of missingness *percentage* per variable.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    sort : bool, default True
        Sort bars from lowest to highest missing rate.
    interactive : bool, default False
        Return a Plotly figure when ``True``.
    **kwargs
        Forwarded to :func:`matplotlib.axes.Axes.barh`.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({
    ...     "age":    [25, np.nan, 30, np.nan, 45],
    ...     "income": [50000, 60000, np.nan, 70000, np.nan],
    ...     "city":   ["Berlin", "Paris", "Berlin", None, "Rome"],
    ... })
    >>> ax  = visualise.miss_var_pct(df)
    >>> fig = visualise.miss_var_pct(df, interactive=True)
    """
    if interactive:
        return _miss_var_pct_plotly(df, missing_values, sort)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, df.shape[1] * 0.5)))
    pct = _nullity(df, missing_values).mean() * 100
    if sort:
        pct = pct.sort_values(ascending=True)
    safe_idx = _safe_labels(pct.index)
    color = kwargs.pop("color", "steelblue")
    ax.barh(safe_idx, pct.values, color=color, **kwargs)
    ax.set_xlabel("% Missing")
    ax.set_xlim(0, 100)
    ax.axvline(x=0, color="black", linewidth=0.8)
    for i, val in enumerate(pct.values):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center", ha="left", fontsize=8)
    ax.set_title("Missing Values per Variable (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return ax


def miss_which(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Binary tile plot: which variables contain *any* missing values?"""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.8), 2.5))
    null_df = _nullity(df, missing_values)
    has_missing = null_df.any().astype(float).to_frame(name="has_missing").T
    pct = null_df.mean() * 100
    col_labels = [_rtl_safe(f"{col}\n({pct[col]:.1f}%)") for col in df.columns]
    annot_arr = np.where(has_missing.values.astype(bool), "Missing", "Complete")
    cmap = kwargs.pop("cmap", ["#f0f0f0", "#d62728"])
    cbar = kwargs.pop("cbar", False)
    sns.heatmap(
        has_missing, ax=ax, cmap=cmap, cbar=cbar,
        annot=annot_arr, fmt="",
        xticklabels=col_labels, yticklabels=False, **kwargs,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Which Variables Have Missing Data?")
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Pattern analysis
# ---------------------------------------------------------------------------

def upset(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    max_patterns: int = 20,
    show_pct: bool = True,
    color: str = "steelblue",
    interactive: bool = False,
    **kwargs,
):
    """UpSet plot for missing-value intersection analysis.

    Shows the *intersection sizes* of missing-value patterns: how many rows
    miss column A only, how many miss A+B together, etc.  More expressive
    than a Venn diagram for > 3 variables.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
    max_patterns : int, default 20
        Maximum number of intersections to display (sorted by frequency).
    show_pct : bool, default True
        Annotate bars with the percentage of total rows.
    color : str, default ``"steelblue"``
        Colour for active dots and bars.
    interactive : bool, default False
        Return a Plotly figure when ``True``.
    **kwargs
        Not forwarded (kept for API consistency).

    Returns
    -------
    dict
        Static mode: ``{"intersections": ax_bar, "matrix": ax_mat,
        "totals": ax_tot}`` — three Axes in a shared figure.
        When there are no missing values an empty-state figure is returned
        with the same dict shape so callers never need to branch on ``{}``.
    plotly.graph_objects.Figure
        Interactive mode.

    Notes
    -----
    When ``interactive=False`` the return value is a *dict of Axes*, not a
    single Axes.  Access individual panels via the keys above.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.standard_normal((50, 4)), columns=list("abcd"))
    >>> df.iloc[:10, 0] = np.nan
    >>> df.iloc[5:15, 1] = np.nan
    >>> axes = visualise.upset(df)           # static → dict of Axes
    >>> fig  = visualise.upset(df, interactive=True)
    """
    # NOTE: show_pct, color, and max_patterns are explicit params;
    # do NOT forward them (or any remaining kwargs) to matplotlib bar().
    if interactive:
        return _upset_plotly(df, missing_values, max_patterns, show_pct, color)
    null_mat = _nullity(df, missing_values)
    missing_cols = list(null_mat.columns[null_mat.any()])
    if not missing_cols:
        fig, ax_empty = plt.subplots(figsize=(6, 3))
        ax_empty.text(
            0.5, 0.5, "No missing values to plot.",
            ha="center", va="center", fontsize=13, color="#888888",
            transform=ax_empty.transAxes,
        )
        ax_empty.set_axis_off()
        fig.suptitle("UpSet Plot: Missing-Data Patterns", y=1.01)
        plt.tight_layout()
        return {"intersections": ax_empty, "matrix": ax_empty, "totals": ax_empty}
    null_mat = null_mat[missing_cols].astype(bool)
    n_rows_total = len(df)
    n_cols = len(missing_cols)
    combos: Dict = {}
    for row in null_mat.itertuples(index=False):
        key = tuple(row)
        combos[key] = combos.get(key, 0) + 1
    combos = {k: v for k, v in combos.items() if any(k)}
    if not combos:
        fig, ax_empty = plt.subplots(figsize=(6, 3))
        ax_empty.text(
            0.5, 0.5, "No missing combinations to plot.",
            ha="center", va="center", fontsize=13, color="#888888",
            transform=ax_empty.transAxes,
        )
        ax_empty.set_axis_off()
        fig.suptitle("UpSet Plot: Missing-Data Patterns", y=1.01)
        plt.tight_layout()
        return {"intersections": ax_empty, "matrix": ax_empty, "totals": ax_empty}
    sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)[:max_patterns]
    combo_keys = [c[0] for c in sorted_combos]
    combo_counts = [c[1] for c in sorted_combos]
    n_combos = len(combo_keys)
    col_totals = [null_mat[c].sum() for c in missing_cols]
    fig = plt.figure(figsize=(max(10, n_combos * 1.2), max(6, n_cols * 0.9 + 3)))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1, n_combos],
        height_ratios=[2, n_cols],
        hspace=0.05, wspace=0.05,
    )
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 1])
    ax_tot = fig.add_subplot(gs[1, 0])
    fig.add_subplot(gs[0, 0]).set_visible(False)
    x_pos = np.arange(n_combos)
    y_pos = np.arange(n_cols)
    bars = ax_bar.bar(x_pos, combo_counts, color=color, edgecolor="white")
    ax_bar.set_xlim(-0.5, n_combos - 0.5)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("Frequency")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    if show_pct:
        for bar_patch, count in zip(bars, combo_counts):
            pct = count / n_rows_total * 100
            ax_bar.text(
                bar_patch.get_x() + bar_patch.get_width() / 2,
                bar_patch.get_height() + ax_bar.get_ylim()[1] * 0.01,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=7, color="#333333",
            )
    for xi, key in enumerate(combo_keys):
        active_rows = [yi for yi, active in enumerate(key) if active]
        if len(active_rows) > 1:
            ax_mat.plot(
                [xi, xi], [min(active_rows), max(active_rows)],
                color=color, linewidth=2.5, zorder=1,
            )
        for yi, active in enumerate(key):
            ax_mat.scatter(
                xi, yi, s=160,
                color=color if active else "#dddddd", zorder=2,
                edgecolors="white" if active else "#cccccc", linewidths=0.5,
            )
    ax_mat.set_xlim(-0.5, n_combos - 0.5)
    ax_mat.set_ylim(-0.5, n_cols - 0.5)
    ax_mat.set_xticks([])
    ax_mat.set_yticks(y_pos)
    ax_mat.set_yticklabels(_safe_labels(missing_cols))
    ax_mat.spines["top"].set_visible(False)
    ax_mat.spines["right"].set_visible(False)
    ax_mat.spines["bottom"].set_visible(False)
    ax_tot.barh(y_pos, col_totals, color=color, edgecolor="white")
    ax_tot.set_ylim(-0.5, n_cols - 0.5)
    ax_tot.set_yticks([])
    ax_tot.set_xlabel("Set size")
    ax_tot.invert_xaxis()
    ax_tot.spines["top"].set_visible(False)
    ax_tot.spines["left"].set_visible(False)
    fig.suptitle("UpSet Plot: Missing-Data Patterns", y=1.01)
    plt.tight_layout()
    return {"intersections": ax_bar, "matrix": ax_mat, "totals": ax_tot}


def miss_patterns(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    top_n: int = 10,
    interactive: bool = False,
    **kwargs,
):
    """Horizontal bar chart of the most frequent missingness patterns.

    A *pattern* is the set of columns missing together in a row, e.g.
    ``"age + income"`` means rows where both columns are ``NaN``.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    top_n : int, default 10
        Number of most frequent patterns to show.
    interactive : bool, default False
        Return a Plotly figure when ``True``.
    **kwargs
        Forwarded to :func:`matplotlib.axes.Axes.barh`.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({
    ...     "a": [1, np.nan, 3, np.nan, 5],
    ...     "b": [np.nan, 2, np.nan, 4, np.nan],
    ...     "c": [1, 2, 3, np.nan, np.nan],
    ... })
    >>> ax  = visualise.miss_patterns(df, top_n=5)
    >>> fig = visualise.miss_patterns(df, interactive=True)
    """
    if interactive:
        return _miss_patterns_plotly(df, missing_values, top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(4, top_n * 0.55)))
    null_mat = _nullity(df, missing_values)

    def _pattern_label(row):
        cols = [c for c, v in row.items() if v]
        return "(complete)" if not cols else " + ".join(_rtl_safe(str(c)) for c in cols)

    patterns = null_mat.apply(_pattern_label, axis=1)
    counts = patterns.value_counts().head(top_n).sort_values(ascending=True)
    pct = counts / len(df) * 100
    color = kwargs.pop("color", "#4C72B0")
    ax.barh(range(len(counts)), counts.values, color=color, **kwargs)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel("Row count")
    ax.set_title(f"Top-{top_n} Missingness Patterns")
    for i, (count, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(count + len(df) * 0.002, i, f"  {p:.1f}%",
                va="center", fontsize=8, color="#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return ax


def miss_cooccurrence(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    normalize: bool = True,
    cmap: str = "Blues",
    annot: bool = True,
    interactive: bool = False,
    **kwargs,
):
    """Co-occurrence heatmap: how often do pairs of columns miss together?

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    normalize : bool, default True
        Divide co-occurrence counts by ``len(df)`` to obtain fractions.
        When ``False``, raw counts are shown.
    cmap : str, default ``"Blues"``
        Matplotlib / seaborn colormap name.
    annot : bool, default True
        Annotate cells with numeric values.
    interactive : bool, default False
        Return a Plotly figure when ``True``.
    **kwargs
        Forwarded to :func:`seaborn.heatmap`.

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({
    ...     "a": [np.nan, np.nan, 3, 4, 5],
    ...     "b": [np.nan, np.nan, np.nan, 4, 5],
    ...     "c": [1, 2, 3, 4, np.nan],
    ... })
    >>> ax  = visualise.miss_cooccurrence(df)
    >>> ax2 = visualise.miss_cooccurrence(df, normalize=False)  # raw counts
    >>> fig = visualise.miss_cooccurrence(df, interactive=True)
    """
    if interactive:
        return _miss_cooccurrence_plotly(df, missing_values, normalize)
    null_mat = _nullity(df, missing_values).astype(int)
    if ax is None:
        n = null_mat.shape[1]
        _, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    cooc = null_mat.T.dot(null_mat)
    if normalize:
        cooc = cooc / len(df)
        fmt = kwargs.pop("fmt", ".2f")
    else:
        fmt = kwargs.pop("fmt", "d")
    labels = _safe_labels(cooc.columns)
    cooc.columns = pd.Index(labels)
    cooc.index = pd.Index(labels)
    sns.heatmap(
        cooc, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
        linewidths=0.5, square=True, **kwargs,
    )
    title = "Missingness Co-occurrence (fraction)" if normalize else "Missingness Co-occurrence (count)"
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Correlation / clustering
# ---------------------------------------------------------------------------

def heatmap(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    method: str = "pearson",
    mask_insignificant: bool = False,
    significance: float = 0.05,
    interactive: bool = False,
    **kwargs,
):
    """Nullity-correlation heatmap between columns.

    Computes the pairwise Pearson (or phi) correlation between the binary
    missingness indicators of each column pair, then renders a colour-coded
    heatmap.  Values close to +1 or -1 indicate that two columns tend to
    be missing (or present) together.

    DQT integration
    ---------------
    When ``data_quality_toolkit`` is installed and ``interactive=False``,
    the seaborn rendering is delegated to
    ``data_quality_toolkit.visualization.correlation_heatmap``.  This is the
    **single glue point** between *missingly* and *data-quality-toolkit*.
    If DQT is not installed, a pure-seaborn fallback is used transparently.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    method : {"pearson", "phi"}, default ``"pearson"``
        Correlation method.  ``"phi"`` is an alias for Pearson on binary data.
    mask_insignificant : bool, default False
        Grey out cells where the correlation is not statistically significant
        at the *significance* level.
    significance : float, default 0.05
        Alpha threshold used when ``mask_insignificant=True``.
    interactive : bool, default False
        Return a :class:`plotly.graph_objects.Figure` when ``True``.
    **kwargs
        Forwarded to :func:`seaborn.heatmap` (or DQT's wrapper).

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame(rng.standard_normal((50, 4)), columns=list("abcd"))
    >>> df.iloc[:10, 0] = np.nan
    >>> df.iloc[5:15, 1] = np.nan
    >>> ax  = visualise.heatmap(df)                        # static
    >>> ax2 = visualise.heatmap(df, mask_insignificant=True)
    >>> fig = visualise.heatmap(df, interactive=True)      # Plotly
    """
    if interactive:
        return _heatmap_plotly(df, missing_values, method, mask_insignificant, significance)
    if ax is None:
        n = df.shape[1]
        _, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    null_mat = _nullity(df, missing_values).astype(float)
    corr = null_mat.corr(method="pearson")
    nan_mask = np.isnan(corr.values)
    sig_mask = np.zeros_like(nan_mask)
    if mask_insignificant:
        from scipy import stats
        n_obs = len(null_mat)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if i == j or nan_mask[i, j]:
                    continue
                r = corr.values[i, j]
                if abs(r) < 1.0:
                    t_stat = r * np.sqrt(n_obs - 2) / np.sqrt(1 - r**2)
                    p = 2 * stats.t.sf(abs(t_stat), df=n_obs - 2)
                    if p > significance:
                        sig_mask[i, j] = True
    final_mask = nan_mask | sig_mask
    labels = _safe_labels(corr.columns)
    corr.columns = pd.Index(labels)
    corr.index = pd.Index(labels)
    cmap = kwargs.pop("cmap", "RdBu")
    annot = kwargs.pop("annot", True)
    fmt = kwargs.pop("fmt", ".2f")
    vmin = kwargs.pop("vmin", -1)
    vmax = kwargs.pop("vmax", 1)
    center = kwargs.pop("center", 0)
    linewidths = kwargs.pop("linewidths", 0.5)
    method_label = "Phi" if method == "phi" else "Pearson"
    # --- single DQT glue point ---
    try:
        from data_quality_toolkit.visualization import correlation_heatmap as _dqt_hm
        _dqt_hm(
            corr,
            ax=ax,
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
        sns.heatmap(
            corr, mask=final_mask, ax=ax,
            cmap=cmap, annot=annot, fmt=fmt,
            vmin=vmin, vmax=vmax, center=center, linewidths=linewidths,
            **kwargs,
        )
    # Always set title after the rendering call (DQT may not set it)
    ax.set_title(f"Nullity Correlation Heatmap ({method_label})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    return ax


def dendrogram(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    method: str = "ward",
    **kwargs,
):
    """Dendrogram clustering variables by their nullity correlation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    null_mat = _nullity(df, missing_values).astype(int)
    variable_cols = null_mat.columns[null_mat.var() > 0]
    if len(variable_cols) < 2:
        raise ValueError(
            f"dendrogram requires at least 2 columns with variable missingness; "
            f"only {len(variable_cols)} found."
        )
    null_mat = null_mat[variable_cols]
    dist = squareform(1 - null_mat.corr().abs().values, checks=False)
    scipy_dendrogram(
        linkage(dist, method=method),
        labels=_safe_labels(variable_cols),
        ax=ax, orientation="top", **kwargs,
    )
    ax.set_title("Dendrogram of Variables by Missing Data Patterns")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Distance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return ax


def miss_cluster(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    method: str = "ward",
    **kwargs,
):
    """Heatmap with rows reordered by hierarchical clustering on missingness.

    Rows with similar missing-data patterns are placed adjacent to each
    other, making block structure visible at a glance.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    method : str, default ``"ward"``
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`.
        Common choices: ``"ward"``, ``"complete"``, ``"average"``.
    **kwargs
        Forwarded to :func:`seaborn.heatmap`.

    Returns
    -------
    matplotlib.axes.Axes
        A single Axes containing the clustered heatmap.  The y-axis shows
        row labels when ``len(df) < 50``, otherwise tick labels are hidden
        for readability.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame(rng.standard_normal((80, 5)), columns=list("abcde"))
    >>> df.iloc[:20, 0] = np.nan
    >>> df.iloc[10:30, 2] = np.nan
    >>> ax = visualise.miss_cluster(df)
    >>> # ax is a single matplotlib.axes.Axes (not a dict)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, df.shape[1] * 1.2), 6))
    null_df = _nullity(df, missing_values).astype(float)
    if null_df.shape[0] > 1:
        row_dist = pdist(null_df.values, metric="hamming")
        if row_dist.max() > 0:
            null_df = null_df.iloc[leaves_list(linkage(row_dist, method=method))]
    yticklabels = _safe_labels(null_df.index) if null_df.shape[0] < 50 else False
    cmap = kwargs.pop("cmap", ["#f0f0f0", "#d62728"])
    cbar = kwargs.pop("cbar", False)
    # 'method' is consumed above; must NOT be forwarded to sns.heatmap
    kwargs.pop("method", None)
    sns.heatmap(null_df, ax=ax, cmap=cmap, cbar=cbar, yticklabels=yticklabels, **kwargs)
    ax.set_xticklabels(_safe_labels(df.columns), rotation=45, ha="right")
    ax.set_title("Clustered Missing Data Matrix")
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Row / variable profiles
# ---------------------------------------------------------------------------

def miss_row_profile(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    bins: int = None,
    color: str = "steelblue",
    **kwargs,
):
    """Histogram of per-row missing-value counts."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    row_miss = _nullity(df, missing_values).sum(axis=1)
    n_cols = df.shape[1]
    bins = bins if bins is not None else n_cols + 1
    ax.hist(row_miss, bins=np.arange(-0.5, n_cols + 1.5, 1), color=color,
            edgecolor="white", **kwargs)
    mean_val = row_miss.mean()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_val:.1f}")
    ax.set_xlabel("Number of missing columns per row")
    ax.set_ylabel("Row count")
    ax.set_title("Row Missingness Profile")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Shadow / MAR detection
# ---------------------------------------------------------------------------

def shadow_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    shadow_col: str,
    ax=None,
    missing_values: Optional[List] = None,
    palette: Optional[Dict] = None,
    **kwargs,
):
    """Scatter plot coloured by whether a third variable is missing."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    null_col = _nullity(df[[shadow_col]], missing_values)[shadow_col]
    plot_df = df[[x, y]].copy()
    for col in [x, y]:
        mask = _nullity(plot_df[[col]], missing_values)[col]
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df.loc[mask, col] = plot_df[col].median()
    plot_df["__shadow__"] = null_col.map(
        {True: f"{shadow_col} missing", False: f"{shadow_col} observed"}
    )
    if palette is None:
        palette = {
            f"{shadow_col} missing": "#d62728",
            f"{shadow_col} observed": "#1f77b4",
        }
    sns.scatterplot(
        data=plot_df, x=x, y=y,
        hue="__shadow__", palette=palette,
        ax=ax, alpha=0.7, **kwargs,
    )
    ax.set_xlabel(_rtl_safe(x))
    ax.set_ylabel(_rtl_safe(y))
    ax.set_title(
        f"Shadow scatter: {_rtl_safe(x)} vs {_rtl_safe(y)}"
        f" — hue: {_rtl_safe(shadow_col)} missingness"
    )
    ax.legend(title="", fontsize=9)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Factor / group breakdown
# ---------------------------------------------------------------------------

def vis_miss_fct(
    df: pd.DataFrame,
    fct: str,
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Stacked bar chart of missing values per factor level."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    null_indicator = _nullity(df, missing_values)
    miss_by_fct = (
        null_indicator
        .drop(columns=fct, errors="ignore")
        .groupby(df[fct])
        .sum()
    )
    miss_by_fct.index = pd.Index(_safe_labels(miss_by_fct.index))
    miss_by_fct.columns = pd.Index(_safe_labels(miss_by_fct.columns))
    miss_by_fct.plot(kind="bar", stacked=True, ax=ax, **kwargs)
    ax.set_title(_rtl_safe(f"Missing Values by {fct}"))
    ax.set_xlabel(_rtl_safe(fct))
    ax.set_ylabel("Number of Missing Values")
    plt.tight_layout()
    return ax


def vis_miss_by_group(
    df: pd.DataFrame,
    group_col: str,
    ax=None,
    missing_values: Optional[List] = None,
    cmap: str = "YlOrRd",
    annot: bool = True,
    **kwargs,
):
    """Per-group missingness heatmap."""
    null_mat = _nullity(df, missing_values)
    miss_pct = (
        null_mat
        .drop(columns=group_col, errors="ignore")
        .groupby(df[group_col])
        .mean() * 100
    )
    n_groups, n_vars = miss_pct.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, n_vars * 0.8), max(4, n_groups * 0.6)))
    miss_pct.columns = pd.Index(_safe_labels(miss_pct.columns))
    miss_pct.index = pd.Index(_safe_labels(miss_pct.index))
    fmt = kwargs.pop("fmt", ".1f")
    linewidths = kwargs.pop("linewidths", 0.5)
    vmin = kwargs.pop("vmin", 0)
    vmax = kwargs.pop("vmax", 100)
    sns.heatmap(
        miss_pct, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
        linewidths=linewidths, vmin=vmin, vmax=vmax, **kwargs,
    )
    ax.set_title(_rtl_safe(f"Missingness Rate (%) by {group_col}"))
    ax.set_xlabel("Variable")
    ax.set_ylabel(_rtl_safe(group_col))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Imputation diagnostics
# ---------------------------------------------------------------------------

def vis_impute_dist(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    column: Optional[str] = None,
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """KDE comparison of original vs. imputed distribution.

    When *column* is given, plots a single-column KDE overlay on *ax*.
    When *column* is ``None``, plots a grid of KDEs for every numeric column
    that had missing values in *original_df* and returns a
    :class:`matplotlib.figure.Figure`.

    Parameters
    ----------
    original_df : pd.DataFrame
        Pre-imputation DataFrame (contains ``NaN`` values).
    imputed_df : pd.DataFrame
        Post-imputation DataFrame.
    column : str, optional
        A single column to compare.  When ``None`` all numeric columns with
        missing values are plotted in a grid (same as
        :func:`miss_impute_compare`).
    ax : matplotlib.axes.Axes, optional
        Target axes for the single-column mode.  Ignored in multi-column mode.
    missing_values : list, optional
        Extra sentinel values for :func:`_nullity`.
    **kwargs
        Forwarded to :func:`seaborn.kdeplot`.

    Returns
    -------
    matplotlib.axes.Axes
        Single-column mode.
    matplotlib.figure.Figure
        Multi-column mode (``column=None``).

    Raises
    ------
    ValueError
        Multi-column mode: no numeric columns with missing values found.
    KeyError
        Single-column mode: *column* not present in either DataFrame.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise, impute
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0],
    ...                    "b": [np.nan, 2.0, 3.0, 4.0, np.nan]})
    >>> imputed = impute.impute_mean(df)
    >>> ax  = visualise.vis_impute_dist(df, imputed, column="a")  # single col
    >>> fig = visualise.vis_impute_dist(df, imputed)              # all cols
    """
    if column is not None:
        # --- single-column mode ---
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(original_df[column].dropna(), ax=ax, label="Original", **kwargs)
        sns.kdeplot(imputed_df[column], ax=ax, label="Imputed", **kwargs)
        ax.set_title(
            f"Distribution of Original vs. Imputed Data for {_rtl_safe(column)}"
        )
        ax.legend()
        return ax

    # --- multi-column mode (column=None): delegate to miss_impute_compare ---
    return miss_impute_compare(
        original_df, imputed_df,
        columns=None, missing_values=missing_values, **kwargs,
    )


def miss_impute_compare(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Multi-column grid comparing original and imputed distributions."""
    null_mat = _nullity(original_df, missing_values)
    if columns is None:
        num_cols = original_df.select_dtypes(include=[np.number]).columns
        columns = [c for c in num_cols if null_mat[c].any()]
    if not columns:
        raise ValueError("No numeric columns with missing values found.")
    n = len(columns)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, col in enumerate(columns):
        ax = axes[i // ncols][i % ncols]
        orig_vals = original_df[col].dropna()
        imp_vals = imputed_df[col]
        if len(orig_vals) > 1:
            sns.kdeplot(orig_vals, ax=ax, label="Observed", color="steelblue", **kwargs)
        if len(imp_vals.dropna()) > 1:
            sns.kdeplot(imp_vals, ax=ax, label="Imputed",
                        color="#d62728", linestyle="--", **kwargs)
        ax.set_title(_rtl_safe(col), fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Original vs. Imputed Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scatter / MAR
# ---------------------------------------------------------------------------

def scatter_miss(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Scatter plot highlighting missing values in either axis variable.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        Column name for the horizontal axis.
    y : str
        Column name for the vertical axis.
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    **kwargs
        Forwarded to :func:`seaborn.scatterplot`.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly import visualise
    >>> df = pd.DataFrame({
    ...     "A": [1, 2, np.nan, 4, 5],
    ...     "B": [5, np.nan, 3, 4, 5],
    ... })
    >>> ax = visualise.scatter_miss(df, x="A", y="B")
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    plot_df = df[[x, y]].copy()
    null_df = _nullity(plot_df, missing_values)
    plot_df[f"{x}_NA"] = null_df[x]
    plot_df[f"{y}_NA"] = null_df[y]
    any_x_na = plot_df[f"{x}_NA"].any()
    any_y_na = plot_df[f"{y}_NA"].any()
    for col, flag in [(x, any_x_na), (y, any_y_na)]:
        if flag:
            num = pd.to_numeric(plot_df[col], errors="coerce")
            mn = num.min()
            plot_df[col] = num.fillna(mn - abs(mn) * 0.1)
    hue = None
    if any_x_na and any_y_na:
        hue = (plot_df[f"{x}_NA"].astype(str) + "_" + plot_df[f"{y}_NA"].astype(str))
        hue.name = "Missingness"
    elif any_x_na:
        hue = plot_df[f"{x}_NA"]
        hue.name = f"Missing {_rtl_safe(x)}"
    elif any_y_na:
        hue = plot_df[f"{y}_NA"]
        hue.name = f"Missing {_rtl_safe(y)}"
    sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, ax=ax, **kwargs)
    ax.set_xlabel(_rtl_safe(x))
    ax.set_ylabel(_rtl_safe(y))
    ax.set_title(f"Scatter Plot of {_rtl_safe(x)} vs {_rtl_safe(y)} with Missing Values")
    return ax


# ---------------------------------------------------------------------------
# Cumulative / span
# ---------------------------------------------------------------------------

def vis_miss_cumsum_var(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Cumulative sum of missing values across variables."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    cumsum = _nullity(df, missing_values).sum().cumsum()
    cumsum.index = pd.Index(_safe_labels(cumsum.index))
    cumsum.plot(kind="line", ax=ax, **kwargs)
    ax.set_title("Cumulative Sum of Missing Values per Variable")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Cumulative Sum of Missing Values")
    plt.tight_layout()
    return ax


def vis_miss_cumsum_case(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Cumulative sum of missing values across cases (rows)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    cumsum = _nullity(df, missing_values).sum(axis=1).cumsum()
    cumsum.plot(kind="line", ax=ax, **kwargs)
    ax.set_title("Cumulative Sum of Missing Values per Case")
    ax.set_xlabel("Cases (Rows)")
    ax.set_ylabel("Cumulative Sum of Missing Values")
    plt.tight_layout()
    return ax


def vis_miss_span(
    df: pd.DataFrame,
    column: str,
    span_every: int,
    ax=None,
    missing_values: Optional[List] = None,
    color: str = "steelblue",
    **kwargs,
):
    """Rolling-window missingness rate for a single column."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    null_series = _nullity(df[[column]], missing_values)[column].astype(float)
    n = len(null_series)
    span_indices = range(0, n, span_every)
    span_rates = [null_series.iloc[i: i + span_every].mean() for i in span_indices]
    x_vals = [i + span_every // 2 for i in span_indices]
    ax.plot(x_vals, span_rates, color=color, marker="o", linewidth=1.8, **kwargs)
    ax.set_xlabel("Row index (span midpoint)")
    ax.set_ylabel("Missingness rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"Missingness span: {_rtl_safe(column)} (span={span_every})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return ax


def vis_parallel_coords(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    color_complete: str = "steelblue",
    color_missing: str = "#d62728",
    alpha: float = 0.4,
    **kwargs,
):
    """Parallel coordinates plot coloured by row-level missingness."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, df.shape[1] * 1.2), 6))
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        ax.set_title("Parallel Coordinates Plot of Missingness")
        ax.text(0.5, 0.5, "No numeric columns", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        return ax
    null_mat = _nullity(df, missing_values)
    has_missing = null_mat.any(axis=1)
    norm_df = num_df.copy().astype(float)
    for col in norm_df.columns:
        col_min = norm_df[col].min()
        col_max = norm_df[col].max()
        rng = col_max - col_min
        norm_df[col] = (norm_df[col] - col_min) / rng if rng > 0 else 0.5
    x_vals = list(range(len(norm_df.columns)))
    for idx in norm_df.index:
        row_vals = norm_df.loc[idx].values
        c = color_missing if has_missing.loc[idx] else color_complete
        ax.plot(x_vals, row_vals, color=c, alpha=alpha, linewidth=0.8, **kwargs)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(_safe_labels(norm_df.columns), rotation=45, ha="right")
    ax.set_ylabel("Normalised value")
    ax.set_title("Parallel Coordinates Plot of Missingness")
    complete_patch = mpatches.Patch(color=color_complete, label="Complete row")
    missing_patch = mpatches.Patch(color=color_missing, label="Row with missing")
    ax.legend(handles=[complete_patch, missing_patch], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return ax
