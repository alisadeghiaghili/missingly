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
from typing import Dict, List, Optional, Sequence

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


_RTL_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_RLM = "\u200F"
_LRM = "\u200E"


def _rtl_safe(text: str) -> str:
    if _RTL_PATTERN.search(str(text)):
        return f"{_RLM}{text}{_LRM}"
    return str(text)


def _safe_labels(labels: Sequence) -> List[str]:
    return [_rtl_safe(str(lbl)) for lbl in labels]


def _nullity(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    if missing_values is None:
        return df.isnull()
    return df.isnull() | df.isin(missing_values)


# ---------------------------------------------------------------------------
# Plotly helpers and backends
# ---------------------------------------------------------------------------


def _require_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as exc:
        raise ImportError(
            "Interactive mode requires plotly >= 5.0. "
            "Install it with: pip install plotly"
        ) from exc


def _matrix_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
):
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


def _heatmap_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    method: str = "pearson",
    mask_insignificant: bool = False,
    significance: float = 0.05,
):
    go = _require_plotly()
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
                    t_stat = r * np.sqrt(n_obs - 2) / np.sqrt(1 - r ** 2)
                    p = 2 * stats.t.sf(abs(t_stat), df=n_obs - 2)
                    if p > significance:
                        sig_mask[i, j] = True

    z = corr.values.copy()
    z[nan_mask | sig_mask] = None
    labels = _safe_labels(corr.columns)

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
        title="Nullity Correlation Heatmap",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=100, r=40, t=60, b=120),
    )
    return fig


# (بقیه backendهای plotly مثل _vis_miss_plotly, _miss_var_pct_plotly,
#  _miss_cooccurrence_plotly فرض می‌کنیم سر جایشان هستند.)


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
    """A matrix plot to visualize the location of missing data."""
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
    """Nullity correlation heatmap between columns."""
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


def vis_parallel_coords(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Parallel coordinates plot of row missingness patterns."""
    from data_quality_toolkit.visualization import parallel_coordinates

    df_miss = _nullity(df, missing_values).astype(int)
    df_miss["missing_count"] = df_miss.sum(axis=1)
    df_miss.columns = pd.Index(_safe_labels(df_miss.columns))

    ax = parallel_coordinates(df_miss, _rtl_safe("missing_count"), **kwargs)
    ax.set_title("Parallel Coordinates Plot of Missingness")
    plt.tight_layout()
    return ax
