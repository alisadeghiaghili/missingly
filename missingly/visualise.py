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


# ... (تمام سایر توابع بدون تغییر تا بخش heatmap و vis_parallel_coords)


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
        Forwarded to ``seaborn.heatmap`` (static mode only).
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
        Forwarded to ``data_quality_toolkit.visualization.parallel_coordinates``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from data_quality_toolkit.visualization import parallel_coordinates

    df_miss = _nullity(df, missing_values).astype(int)
    df_miss["missing_count"] = df_miss.sum(axis=1)
    df_miss.columns = pd.Index(_safe_labels(df_miss.columns))

    ax = parallel_coordinates(df_miss, _rtl_safe("missing_count"), **kwargs)
    ax.set_title("Parallel Coordinates Plot of Missingness")
    plt.tight_layout()
    return ax
