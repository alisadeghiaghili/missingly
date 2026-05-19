"""Visualization utilities for missing data analysis.

Every public function follows the same conventions:

* Accepts an optional ``ax`` (or ``fig`` for multi-panel plots) so callers
  can embed plots inside existing figures.
* Accepts an optional ``missing_values`` list to treat arbitrary sentinel
  values (e.g. ``-99``, ``"N/A"``) as missing.
* Accepts an optional ``interactive`` boolean (default ``False``). When
  ``True``, returns a :class:`plotly.graph_objects.Figure` instead of a
  matplotlib Axes — the ``ax`` parameter is ignored in that case.
* Returns the Axes object (or a dict of Axes for multi-panel plots) so
  callers can further customise the static output.
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

Interactive mode (Phase 1)
--------------------------
The following five functions support ``interactive=True`` via Plotly:

* :func:`vis_miss`
* :func:`heatmap`
* :func:`matrix`
* :func:`miss_var_pct`
* :func:`miss_cooccurrence`

Pass ``interactive=True`` to receive a :class:`plotly.graph_objects.Figure`
that can be rendered in Jupyter notebooks with ``.show()`` or saved as HTML
with ``.write_html(path)``.

Compatibility
-------------
Requires Python 3.9+, pandas >= 1.5, matplotlib >= 3.6, seaborn >= 0.12.
Interactive mode additionally requires plotly >= 5.0.
Uses ``from __future__ import annotations`` for lazy evaluation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

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


def _require_plotly():
    """Import and return plotly.graph_objects, raising ImportError with a
    helpful message if plotly is not installed.

    Returns
    -------
    module
        ``plotly.graph_objects``

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as exc:
        raise ImportError(
            "Interactive mode requires plotly >= 5.0. "
            "Install it with: pip install plotly"
        ) from exc


# ---------------------------------------------------------------------------
# Interactive backends (Phase 1)
# ---------------------------------------------------------------------------

def _vis_miss_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    show_pct: bool = True,
    cluster: bool = False,
) -> "go.Figure":  # type: ignore[name-defined]
    """Plotly backend for :func:`vis_miss`.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
    show_pct : bool
    cluster : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    null_df = _nullity(df, missing_values).astype(float)

    if cluster and null_df.shape[0] > 1:
        row_dist = pdist(null_df.values, metric="hamming")
        row_order = leaves_list(linkage(row_dist, method="ward"))
        null_df = null_df.iloc[row_order]

    col_labels = _pct_labels(df, missing_values) if show_pct else _safe_labels(df.columns)
    pct = _nullity(df, missing_values).mean() * 100

    fig = go.Figure(
        data=go.Heatmap(
            z=null_df.values,
            x=col_labels,
            y=[str(i) for i in null_df.index],
            colorscale=[[0, "#f0f0f0"], [1, "#d62728"]],
            showscale=False,
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Missing: %{z:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Missing Data Overview",
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed", showticklabels=null_df.shape[0] < 50),
        template="plotly_white",
        margin=dict(l=80, r=40, t=60, b=120),
    )
    return fig


def _heatmap_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    method: str = "pearson",
    mask_insignificant: bool = False,
    significance: float = 0.05,
) -> "go.Figure":  # type: ignore[name-defined]
    """Plotly backend for :func:`heatmap`.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
    method : {'pearson', 'phi'}
    mask_insignificant : bool
    significance : float

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    null_mat = _nullity(df, missing_values).astype(float)
    corr = null_mat.corr(method="pearson")  # phi == pearson on binary

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
                    t_stat = r * np.sqrt(n_obs - 2) / np.sqrt(1 - r ** 2)
                    p = 2 * stats.t.sf(abs(t_stat), df=n_obs - 2)
                    if p > significance:
                        sig_mask[i, j] = True

    z = corr.values.copy()
    z[nan_mask | sig_mask] = None  # masked cells shown as blank
    labels = _safe_labels(corr.columns)
    method_label = "Phi" if method == "phi" else "Pearson"

    text = np.where(
        np.isnan(z),
        "",
        np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(z)),
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Nullity Correlation Heatmap ({method_label})",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=100, r=40, t=60, b=120),
    )
    return fig


def _matrix_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> "go.Figure":  # type: ignore[name-defined]
    """Plotly backend for :func:`matrix`.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    null_mat = _nullity(df, missing_values).astype(float)
    labels = _safe_labels(df.columns)

    fig = go.Figure(
        data=go.Heatmap(
            z=null_mat.values,
            x=labels,
            y=[str(i) for i in df.index],
            colorscale=[[0, "#f0f0f0"], [1, "#d62728"]],
            showscale=False,
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Missing: %{z:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Missing Data Matrix",
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed", showticklabels=df.shape[0] < 50),
        template="plotly_white",
        margin=dict(l=80, r=40, t=60, b=120),
    )
    return fig


def _miss_var_pct_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    sort: bool = True,
) -> "go.Figure":  # type: ignore[name-defined]
    """Plotly backend for :func:`miss_var_pct`.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
    sort : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    pct = _nullity(df, missing_values).mean() * 100
    if sort:
        pct = pct.sort_values(ascending=True)

    labels = _safe_labels(pct.index)
    fig = go.Figure(
        data=go.Bar(
            x=pct.values,
            y=labels,
            orientation="h",
            marker_color="steelblue",
            text=[f"{v:.1f}%" for v in pct.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Missing Values per Variable (%)",
        xaxis=dict(title="% Missing", range=[0, 110]),
        yaxis=dict(title=""),
        template="plotly_white",
        margin=dict(l=120, r=60, t=60, b=60),
    )
    return fig


def _miss_cooccurrence_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    normalize: bool = True,
) -> "go.Figure":  # type: ignore[name-defined]
    """Plotly backend for :func:`miss_cooccurrence`.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
    normalize : bool

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    null_mat = _nullity(df, missing_values).astype(int)
    cooc = null_mat.T.dot(null_mat)
    if normalize:
        cooc = cooc / len(df)
        fmt_fn = lambda v: f"{v:.2f}"
        title = "Missingness Co-occurrence (fraction)"
    else:
        fmt_fn = lambda v: f"{int(v)}"
        title = "Missingness Co-occurrence (count)"

    labels = _safe_labels(cooc.columns)
    text = np.vectorize(fmt_fn)(cooc.values)

    fig = go.Figure(
        data=go.Heatmap(
            z=cooc.values,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=100, r=40, t=60, b=120),
    )
    return fig


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
    """A matrix plot to visualize the location of missing data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object.  Created automatically if omitted.
        Ignored when ``interactive=True``.
    missing_values : list, optional
        Sentinel values treated as missing in addition to ``NaN``.
    interactive : bool, optional
        If ``True``, return a :class:`plotly.graph_objects.Figure`
        instead of a matplotlib Axes.  Default ``False``.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.heatmap``
        (static mode only).

    Returns
    -------
    matplotlib.axes.Axes
        When ``interactive=False`` (default).
    plotly.graph_objects.Figure
        When ``interactive=True``.
    """
    if interactive:
        return _matrix_plotly(df, missing_values)

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
    """Nullity correlation heatmap between columns.

    Parameters
    ----------
    df : pd.DataFrame
    ax : matplotlib.axes.Axes, optional
        Ignored when ``interactive=True``.
    missing_values : list, optional
    method : {'pearson', 'phi'}
        Correlation method.  ``'phi'`` computes the Matthews/phi
        coefficient for binary variables.  Default ``'pearson'``.
    mask_insignificant : bool
        Mask cells whose p-value exceeds *significance*.  Default ``False``.
    significance : float
        p-value threshold for masking.  Default ``0.05``.
    interactive : bool, optional
        If ``True``, return a :class:`plotly.graph_objects.Figure`.
        Default ``False``.
    **kwargs
        Forwarded to ``data_quality_toolkit.visualization.correlation_heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
        When ``interactive=False`` (default).
    plotly.graph_objects.Figure
        When ``interactive=True``.
    """
    if interactive:
        return _heatmap_plotly(df, missing_values, method, mask_insignificant, significance)

    from data_quality_toolkit.visualization import correlation_heatmap

    if ax is None:
        n = df.shape[1]
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))

    null_mat = _nullity(df, missing_values).astype(float)

    if method == "phi":
        corr = null_mat.corr(method="pearson")
    else:
        corr = null_mat.corr()

    labels = _safe_labels(corr.columns)
    corr.columns = pd.Index(labels)
    corr.index = pd.Index(labels)

    ax = correlation_heatmap(
        corr,
        ax=ax,
        mask_insignificant=mask_insignificant,
        significance=significance,
        n_obs=len(null_mat),
        **kwargs,
    )
    method_label = "Phi" if method == "phi" else "Pearson"
    ax.set_title(f"Nullity Correlation Heatmap ({method_label})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Row / variable profiles, pattern analysis, etc.
# ---------------------------------------------------------------------------

# (The rest of the file, including bar, miss_case, miss_var_pct, vis_miss,
# miss_which, upset, miss_patterns, miss_cooccurrence, dendrogram,
# miss_cluster, miss_row_profile, shadow_scatter, vis_miss_fct,
# vis_miss_by_group, vis_impute_dist, miss_impute_compare, scatter_miss,
# vis_miss_cumsum_var, vis_miss_cumsum_case, vis_miss_span, and
# vis_parallel_coords) remains identical to commit 9ec292a.
