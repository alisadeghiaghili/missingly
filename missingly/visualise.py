"""Visualization utilities for missing data analysis.

Every public function follows the same conventions:

* Accepts an optional ``ax`` (or ``fig`` for multi-panel plots) so callers
  can embed plots inside existing figures.
* Accepts an optional ``missing_values`` list to treat arbitrary sentinel
  values (e.g. ``-99``, ``"N/A"``) as missing.
* Returns the Axes object (or a dict of Axes for multi-panel plots) so
  callers can further customise the output.
* Titles, axis labels, and annotations all pass through
  :func:`_rtl_safe` which wraps any string containing Arabic/Persian
  characters in a Unicode RLM marker so matplotlib renders them
  left-to-right on the canvas while preserving correct letter ordering.

Visualization catalogue
-----------------------
Basic
  matrix, bar, miss_case, miss_var_pct, vis_miss, miss_which

Pattern analysis
  upset, miss_patterns, miss_cooccurrence

Correlation / clustering
  heatmap, dendrogram, miss_cluster

Row / variable profiles
  miss_row_profile, miss_impute_compare

Shadow / MAR detection
  shadow_scatter

Factor / group breakdown
  vis_miss_fct, vis_miss_by_group

Imputation diagnostics
  vis_impute_dist

Miscellaneous
  scatter_miss, vis_miss_cumsum_var, vis_miss_cumsum_case,
  vis_miss_span, vis_miss_fct, vis_parallel_coords

Compatibility
-------------
Requires Python 3.9+, pandas >= 1.5, matplotlib >= 3.6, seaborn >= 0.12.
Uses ``from __future__ import annotations`` for lazy evaluation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram, leaves_list
from scipy.spatial.distance import squareform, pdist


# ---------------------------------------------------------------------------
# RTL / Persian helpers
# ---------------------------------------------------------------------------

_RTL_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_RLM = "\u200F"  # Right-to-Left Mark
_LRM = "\u200E"  # Left-to-Right Mark


def _rtl_safe(text: str) -> str:
    """Wrap RTL (Arabic/Persian) text so matplotlib renders it correctly.

    matplotlib's text engine is LTR. Wrapping a Persian string with
    RLM + string + LRM forces the Unicode bidirectional algorithm to
    keep the character order intact while the overall text direction
    stays left-to-right on the canvas, which is what we want for
    axis labels on a standard Western-layout plot.

    Parameters
    ----------
    text : str
        Any string; if it contains no RTL characters it is returned
        unchanged.

    Returns
    -------
    str
        The original string, or the string wrapped in RLM/LRM markers.
    """
    if _RTL_PATTERN.search(str(text)):
        return f"{_RLM}{text}{_LRM}"
    return str(text)


def _safe_labels(labels: Sequence) -> List[str]:
    """Apply :func:`_rtl_safe` to every label in a sequence.

    Parameters
    ----------
    labels : sequence
        Any iterable of label values (strings, numbers, Timestamps).

    Returns
    -------
    list of str
        RTL-safe string representations.
    """
    return [_rtl_safe(str(lbl)) for lbl in labels]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nullity(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """Return a boolean DataFrame indicating missing positions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    missing_values : list or None
        Sentinel values to treat as missing in addition to ``NaN``.
        Pass ``None`` to treat only true ``NaN`` / ``None`` as missing.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame — ``True`` means the cell is considered missing.
    """
    if missing_values is None:
        return df.isnull()
    return df.isnull() | df.isin(missing_values)


def _pct_labels(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> List[str]:
    """Return RTL-safe column labels with missingness percentage appended.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    missing_values : list, optional
        Extra sentinel values.

    Returns
    -------
    list of str
        Labels in the form ``"col (12.3%)"`` with RTL wrapping applied.
    """
    pct = _nullity(df, missing_values).mean() * 100
    return [_rtl_safe(f"{col} ({pct[col]:.1f}%)") for col in df.columns]


# ---------------------------------------------------------------------------
# Basic visualisations
# ---------------------------------------------------------------------------

def matrix(df: pd.DataFrame, ax=None, missing_values: Optional[List] = None, **kwargs):
    """A matrix plot to visualize the location of missing data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  Created automatically if omitted.
    missing_values : list, optional
        Sentinel values treated as missing in addition to ``NaN``.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

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


def bar(df: pd.DataFrame, ax=None, missing_values: Optional[List] = None, **kwargs):
    """A bar plot to visualize the number of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        Created automatically if omitted.
    missing_values : list, optional
        Sentinel values treated as missing.
    **kwargs
        Forwarded to ``DataFrame.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    miss_counts = _nullity(df, missing_values).sum()
    miss_counts.index = pd.Index(_safe_labels(miss_counts.index))
    miss_counts.plot(kind="bar", ax=ax, **kwargs)
    ax.set_title("Missing Values per Column")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of Missing Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return ax


def miss_case(df: pd.DataFrame, ax=None, missing_values: Optional[List] = None, **kwargs):
    """Bar plot of missing value count per row (case).

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    **kwargs
        Forwarded to ``Series.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

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
    **kwargs,
):
    """Horizontal bar chart of missingness percentage per variable.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    sort : bool
        Sort by descending missingness.  Default ``True``.
    **kwargs
        Forwarded to ``ax.barh``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, df.shape[1] * 0.5)))

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


def vis_miss(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    show_pct: bool = True,
    cluster: bool = False,
    **kwargs,
):
    """Annotated missingness matrix with per-column percentage labels.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    show_pct : bool
        Append missingness % to column tick labels.  Default ``True``.
    cluster : bool
        Reorder rows by hierarchical clustering.  Default ``False``.
    **kwargs
        Forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 1.2), 6))

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


def miss_which(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Binary tile plot showing which variables contain missing values.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    **kwargs
        Forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.8), 2.5))

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
    **kwargs,
):
    """Enhanced UpSet plot for missing value pattern analysis.

    Improvements over the original implementation:

    * Percentage labels on intersection bars.
    * Limits to the *max_patterns* most common combinations so the plot
      stays readable for wide DataFrames.
    * Color-coded dots: active (missing) dots are coloured, inactive are
      light grey; a connecting line links all active dots in a combo.
    * x-axis label shows the pattern rank and count.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
    max_patterns : int
        Maximum number of intersection patterns to display.  Default 20.
    show_pct : bool
        Annotate intersection bars with percentage of total rows.
        Default ``True``.
    color : str
        Bar and dot colour.  Default ``"steelblue"``.
    **kwargs
        Ignored; present for API compatibility.

    Returns
    -------
    dict
        Keys ``'intersections'``, ``'matrix'``, ``'totals'`` mapping to
        the corresponding Axes objects.
    """
    null_mat = _nullity(df, missing_values)
    missing_cols = list(null_mat.columns[null_mat.any()])
    if not missing_cols:
        print("No missing values to plot.")
        return {}

    null_mat = null_mat[missing_cols].astype(bool)
    n_rows_total = len(df)
    n_cols = len(missing_cols)

    # Count combinations
    combos: Dict = {}
    for row in null_mat.itertuples(index=False):
        key = tuple(row)
        combos[key] = combos.get(key, 0) + 1

    combos = {k: v for k, v in combos.items() if any(k)}
    if not combos:
        print("No missing combinations to plot.")
        return {}

    sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)[:max_patterns]
    combo_keys = [c[0] for c in sorted_combos]
    combo_counts = [c[1] for c in sorted_combos]
    n_combos = len(combo_keys)
    col_totals = [null_mat[c].sum() for c in missing_cols]

    fig = plt.figure(figsize=(max(8, n_combos * 1.2), max(6, n_cols * 0.9 + 3)))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1, n_combos],
        height_ratios=[2, n_cols],
        hspace=0.05,
        wspace=0.05,
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
    ax_bar.set_ylabel("Intersection size")
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

    dot_on = color
    dot_off = "#dddddd"
    for xi, key in enumerate(combo_keys):
        active_rows = [yi for yi, active in enumerate(key) if active]
        if len(active_rows) > 1:
            ax_mat.plot(
                [xi, xi], [min(active_rows), max(active_rows)],
                color=dot_on, linewidth=2.5, zorder=1,
            )
        for yi, active in enumerate(key):
            ax_mat.scatter(
                xi, yi, s=160,
                color=dot_on if active else dot_off, zorder=2,
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

    fig.suptitle("UpSet Plot of Missing Value Combinations", y=1.01)
    plt.tight_layout()
    return {"intersections": ax_bar, "matrix": ax_mat, "totals": ax_tot}


def miss_patterns(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    top_n: int = 10,
    **kwargs,
):
    """Horizontal bar chart of the most frequent missingness patterns.

    A "pattern" is the unique combination of which columns are missing
    in a row (e.g. "A + C missing").  This plot answers: "What are the
    most common missing data patterns and how frequent are they?"

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    top_n : int
        Show only the top-*n* most frequent patterns.  Default 10.
    **kwargs
        Forwarded to ``ax.barh``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.55)))

    null_mat = _nullity(df, missing_values)

    def _pattern_label(row):
        cols = [c for c, v in row.items() if v]
        if not cols:
            return "(complete)"
        return " + ".join(_rtl_safe(str(c)) for c in cols)

    patterns = null_mat.apply(_pattern_label, axis=1)
    counts = patterns.value_counts().head(top_n).sort_values(ascending=True)
    pct = counts / len(df) * 100

    color = kwargs.pop("color", "#4C72B0")
    bars = ax.barh(range(len(counts)), counts.values, color=color, **kwargs)

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
    **kwargs,
):
    """Co-occurrence heatmap: how often do pairs of columns miss together?

    Cell (i, j) shows how many (or what fraction of) rows have both
    column *i* and column *j* missing simultaneously.  The diagonal
    shows each column's own missingness count/rate.

    This is the correct tool for diagnosing structural missingness
    patterns — if two sensors always fail together, this will show it.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    normalize : bool
        If ``True`` (default), values are fractions of total rows.
        If ``False``, raw co-occurrence counts are shown.
    cmap : str
        Seaborn/matplotlib colormap name.  Default ``"Blues"``.
    annot : bool
        Annotate each cell.  Default ``True``.
    **kwargs
        Forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    null_mat = _nullity(df, missing_values).astype(int)

    if ax is None:
        n = null_mat.shape[1]
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))

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
        cooc, ax=ax,
        cmap=cmap, annot=annot, fmt=fmt,
        linewidths=0.5, square=True,
        **kwargs,
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
    **kwargs,
):
    """Nullity correlation heatmap between columns.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    method : {'pearson', 'phi'}
        Correlation method.  ``'phi'`` computes the Matthews/phi
        coefficient for binary variables, which is more appropriate
        than Pearson for 0/1 missingness indicators.  Default
        ``'pearson'``.
    mask_insignificant : bool
        If ``True``, cells whose p-value exceeds *significance* are
        masked (shown in grey).  Requires ``scipy``.  Default ``False``.
    significance : float
        p-value threshold used when *mask_insignificant* is ``True``.
        Default ``0.05``.
    **kwargs
        Forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        n = df.shape[1]
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))

    null_mat = _nullity(df, missing_values).astype(float)

    if method == "phi":
        # phi coefficient = Pearson on binary variables (identical formula)
        corr = null_mat.corr(method="pearson")
    else:
        corr = null_mat.corr()

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

    sns.heatmap(
        corr, mask=final_mask, ax=ax,
        cmap=cmap, annot=annot, fmt=fmt,
        vmin=vmin, vmax=vmax, center=center, linewidths=linewidths,
        **kwargs,
    )
    method_label = "Phi" if method == "phi" else "Pearson"
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
    """Dendrogram clustering variables by their nullity correlation.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    method : str
        Linkage method.  Default ``'ward'``.
    **kwargs
        Forwarded to ``scipy.cluster.hierarchy.dendrogram``.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If fewer than two columns have variable missingness patterns.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

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

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    method : str
        Linkage method.  Default ``'ward'``.
    **kwargs
        Forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 1.2), 6))

    null_df = _nullity(df, missing_values).astype(float)

    if null_df.shape[0] > 1:
        row_dist = pdist(null_df.values, metric="hamming")
        if row_dist.max() > 0:
            null_df = null_df.iloc[leaves_list(linkage(row_dist, method=method))]

    yticklabels = (
        _safe_labels(null_df.index) if null_df.shape[0] < 50 else False
    )
    cmap = kwargs.pop("cmap", ["#f0f0f0", "#d62728"])
    cbar = kwargs.pop("cbar", False)

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
    """Distribution of per-row missing value counts.

    Shows the histogram of how many columns are missing in each row.
    The vertical dashed line marks the mean.  Useful for spotting
    rows that are systematically incomplete vs. randomly affected.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    bins : int, optional
        Number of histogram bins.  Defaults to the number of columns + 1.
    color : str
        Bar fill colour.  Default ``"steelblue"``.
    **kwargs
        Forwarded to ``ax.hist``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

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
    """Scatter plot coloured by whether a third variable is missing.

    This is the primary visual tool for diagnosing Missing At Random
    (MAR) patterns.  If the distribution of *x* vs *y* differs
    systematically between the two coloured groups, missingness in
    *shadow_col* is likely MAR (dependent on the observed variables)
    rather than MCAR.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        Column name for the x-axis.  Must be numeric.
    y : str
        Column name for the y-axis.  Must be numeric.
    shadow_col : str
        The column whose missingness indicator is used as the hue.
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    palette : dict, optional
        Mapping ``{True: color, False: color}`` for the hue.
        Defaults to ``{True: "#d62728", False: "#1f77b4"}``.
    **kwargs
        Forwarded to ``seaborn.scatterplot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    null_col = _nullity(df[[shadow_col]], missing_values)[shadow_col]
    plot_df = df[[x, y]].copy()
    # Replace sentinel/NaN in x,y with column median for plotting
    for col in [x, y]:
        mask = _nullity(plot_df[[col]], missing_values)[col]
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df.loc[mask, col] = plot_df[col].median()

    plot_df["__shadow__"] = null_col.map({True: f"{shadow_col} missing", False: f"{shadow_col} observed"})

    if palette is None:
        palette = {f"{shadow_col} missing": "#d62728", f"{shadow_col} observed": "#1f77b4"}

    sns.scatterplot(
        data=plot_df, x=x, y=y,
        hue="__shadow__",
        palette=palette,
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
    """Stacked bar chart of missing values per factor level.

    Parameters
    ----------
    df : pd.DataFrame
    fct : str
        Name of the categorical column to group by.
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    **kwargs
        Forwarded to ``DataFrame.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

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
    """Per-group missingness heatmap.

    Unlike ``vis_miss_fct`` (which stacks bars), this shows a heatmap
    where each row is a group level and each column is a variable,
    making it easy to compare missingness profiles across groups.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str
        Categorical column to group by.
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    cmap : str
        Colormap.  Default ``"YlOrRd"``.
    annot : bool
        Annotate cells with percentage.  Default ``True``.
    **kwargs
        Forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    null_mat = _nullity(df, missing_values)
    miss_pct = (
        null_mat
        .drop(columns=group_col, errors="ignore")
        .groupby(df[group_col])
        .mean() * 100
    )

    n_groups, n_vars = miss_pct.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_vars * 0.8), max(4, n_groups * 0.6)))

    # Apply RTL-safe labels
    miss_pct.columns = pd.Index(_safe_labels(miss_pct.columns))
    miss_pct.index = pd.Index(_safe_labels(miss_pct.index))

    fmt = kwargs.pop("fmt", ".1f")
    linewidths = kwargs.pop("linewidths", 0.5)
    vmin = kwargs.pop("vmin", 0)
    vmax = kwargs.pop("vmax", 100)

    sns.heatmap(
        miss_pct, ax=ax,
        cmap=cmap, annot=annot, fmt=fmt,
        linewidths=linewidths, vmin=vmin, vmax=vmax,
        **kwargs,
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
    column: str,
    ax=None,
    **kwargs,
):
    """KDE comparison of original vs. imputed distribution for one column.

    Parameters
    ----------
    original_df : pd.DataFrame
        DataFrame before imputation (may contain NaN).
    imputed_df : pd.DataFrame
        DataFrame after imputation.
    column : str
        Column to compare.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Forwarded to ``seaborn.kdeplot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(original_df[column].dropna(), ax=ax, label="Original", **kwargs)
    sns.kdeplot(imputed_df[column], ax=ax, label="Imputed", **kwargs)
    ax.set_title(f"Distribution of Original vs. Imputed Data for {_rtl_safe(column)}")
    ax.legend()
    return ax


def miss_impute_compare(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Multi-column grid comparing original and imputed distributions.

    Plots one KDE panel per column, overlaying the full imputed
    distribution against the observed (non-missing) original values.
    Only columns that had missing values are shown.

    Parameters
    ----------
    original_df : pd.DataFrame
    imputed_df : pd.DataFrame
    columns : list of str, optional
        Subset of columns to plot.  Defaults to all numeric columns
        that had at least one missing value.
    missing_values : list, optional
    **kwargs
        Forwarded to ``seaborn.kdeplot``.

    Returns
    -------
    matplotlib.figure.Figure
    """
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
            sns.kdeplot(imp_vals, ax=ax, label="Imputed", color="#d62728", linestyle="--", **kwargs)
        ax.set_title(_rtl_safe(col), fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide empty subplots
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
    """Scatter plot that highlights missing values in either axis variable.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        Column for the x-axis.
    y : str
        Column for the y-axis.
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    **kwargs
        Forwarded to ``seaborn.scatterplot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

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
    """Cumulative sum of missing values across variables.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    **kwargs
        Forwarded to ``Series.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

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
    """Cumulative sum of missing values across cases (rows).

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    **kwargs
        Forwarded to ``Series.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

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
    span: int,
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Rolling-span missing count for a single variable.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column to analyse.
    span : int
        Rolling window size (rows).
    ax : matplotlib.axes.Axes, optional
    missing_values : list, optional
    **kwargs
        Forwarded to ``Series.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    miss_span = _nullity(df[[column]], missing_values)[column].rolling(span).sum()
    miss_span.plot(kind="line", ax=ax, **kwargs)
    ax.set_title(f"Missing Values in Spans of {span} for {_rtl_safe(column)}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Number of Missing Values in Span")
    plt.tight_layout()
    return ax


def vis_parallel_coords(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Parallel coordinates plot of row missingness patterns.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
    **kwargs
        Forwarded to ``pandas.plotting.parallel_coordinates``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df_miss = _nullity(df, missing_values).astype(int)
    df_miss["missing_count"] = df_miss.sum(axis=1)
    df_miss.columns = pd.Index(_safe_labels(df_miss.columns))

    fig, ax = plt.subplots(figsize=(12, 8))
    pd.plotting.parallel_coordinates(
        df_miss,
        _rtl_safe("missing_count"),
        ax=ax, **kwargs,
    )
    ax.set_title("Parallel Coordinates Plot of Missingness")
    plt.tight_layout()
    return ax
