"""Plotly-based interactive backends for all public visualisation functions.

Each function in this module is a **private backend** called by the
corresponding public wrapper in :mod:`missingly.visualisation.static` when
``interactive=True`` is passed.

Do **not** call these functions directly – use the public API::

    from missingly import visualise
    fig = visualise.heatmap(df, interactive=True)   # returns go.Figure
    fig.show()

All functions return a :class:`plotly.graph_objects.Figure`.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from missingly.visualisation._base import (
    _nullity,
    _pct_labels,
    _require_plotly,
    _rtl_plotly_layout,
    _rtl_safe,
    _safe_labels,
)

if TYPE_CHECKING:
    from plotly import graph_objects as go


def _vis_miss_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    show_pct: bool = True,
    sort: bool = False,
    cluster: bool = False,
) -> "go.Figure":
    """Plotly backend for :func:`~missingly.visualisation.static.vis_miss`."""
    go = _require_plotly()
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    display_df = df
    if sort:
        ordered_columns = _nullity(df, missing_values).mean().sort_values(
            ascending=False
        ).index
        display_df = df.loc[:, ordered_columns]

    null_df = _nullity(display_df, missing_values).astype(float)
    if cluster and null_df.shape[0] > 1:
        try:
            row_dist = pdist(null_df.values, metric="hamming")
            row_order = leaves_list(linkage(row_dist, method="ward"))
            null_df = null_df.iloc[row_order]
        except ValueError as exc:
            warnings.warn(
                "_vis_miss_plotly: clustering could not form valid distances "
                f"({exc}); using original order.",
                UserWarning,
                stacklevel=2,
            )

    col_labels = (
        _pct_labels(display_df, missing_values)
        if show_pct
        else _safe_labels(display_df.columns)
    )
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
    fig.update_layout(**_rtl_plotly_layout(col_labels))
    return fig


def _heatmap_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    method: str = "pearson",
    mask_insignificant: bool = False,
    significance: float = 0.05,
) -> "go.Figure":
    """Plotly backend for :func:`~missingly.visualisation.static.heatmap`."""
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
    method_label = "Phi" if method == "phi" else "Pearson"
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            hovertemplate="%{y} \u00d7 %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Nullity Correlation Heatmap ({method_label})",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=100, r=40, t=60, b=120),
    )
    fig.update_layout(**_rtl_plotly_layout(labels))
    return fig


def _matrix_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> "go.Figure":
    """Plotly backend for :func:`~missingly.visualisation.static.matrix`."""
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
    fig.update_layout(**_rtl_plotly_layout(labels))
    return fig


def _miss_var_pct_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    sort: bool = True,
) -> "go.Figure":
    """Plotly backend for :func:`~missingly.visualisation.static.miss_var_pct`."""
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
        template="plotly_white",
        margin=dict(l=120, r=60, t=60, b=60),
    )
    fig.update_layout(**_rtl_plotly_layout(labels))
    return fig


def _miss_cooccurrence_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    normalize: bool = True,
) -> "go.Figure":
    """Plotly backend for :func:`~missingly.visualisation.static.miss_cooccurrence`."""
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
            hovertemplate="%{y} \u00d7 %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=100, r=40, t=60, b=120),
    )
    fig.update_layout(**_rtl_plotly_layout(labels))
    return fig


def _bar_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    sort: bool = False,
) -> "go.Figure":
    """Plotly backend for :func:`~missingly.visualisation.static.bar`."""
    go = _require_plotly()
    miss_counts = _nullity(df, missing_values).sum()
    if sort:
        miss_counts = miss_counts.sort_values(ascending=False)
    labels = _safe_labels(miss_counts.index)
    total_rows = len(df)
    pct_vals = miss_counts.values / total_rows * 100
    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=miss_counts.values,
            marker_color="steelblue",
            text=[f"{c} ({p:.1f}%)" for c, p in zip(miss_counts.values, pct_vals)],
            textposition="outside",
            hovertemplate="%{x}<br>Missing: %{y}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Missing Values per Column",
        xaxis=dict(title="Columns", tickangle=-45),
        yaxis=dict(title="Number of Missing Values"),
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=120),
    )
    fig.update_layout(**_rtl_plotly_layout(labels))
    return fig


def _miss_case_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    sort: bool = True,
) -> "go.Figure":
    """Plotly backend for :func:`~missingly.visualisation.static.miss_case`."""
    go = _require_plotly()
    miss_counts = _nullity(df, missing_values).sum(axis=1)
    if sort:
        miss_counts = miss_counts.sort_values(ascending=False)
    row_labels = _safe_labels(miss_counts.index)
    n_cols = df.shape[1]
    pct_vals = miss_counts.values / n_cols * 100
    fig = go.Figure(
        data=go.Bar(
            x=row_labels,
            y=miss_counts.values,
            marker_color="steelblue",
            text=[f"{p:.1f}%" for p in pct_vals],
            textposition="outside",
            hovertemplate="Row: %{x}<br>Missing: %{y} cols (%{text})<extra></extra>",
        )
    )
    fig.update_layout(
        title="Missing Values per Case (Row)",
        xaxis=dict(
            title="Cases (Rows)",
            tickangle=-45,
            showticklabels=len(df) < 100,
        ),
        yaxis=dict(title="Number of Missing Values"),
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=100),
    )
    fig.update_layout(**_rtl_plotly_layout(row_labels))
    return fig


def _upset_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    min_subset_size: int = 1,
    max_patterns: int = 20,
    show_pct: bool = True,
    color: str = "steelblue",
) -> "go.Figure":
    """Plotly backend for :func:`~missingly.visualisation.static.upset`."""
    go = _require_plotly()
    from plotly.subplots import make_subplots

    null_mat = _nullity(df, missing_values)
    missing_cols = list(null_mat.columns[null_mat.any()])
    if not missing_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values to plot.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template="plotly_white")
        return fig

    null_mat = null_mat[missing_cols].astype(bool)
    n_rows_total = len(df)
    n_cols = len(missing_cols)
    combos: Dict = {}
    for row in null_mat.itertuples(index=False):
        key = tuple(row)
        combos[key] = combos.get(key, 0) + 1
    combos = {
        key: count
        for key, count in combos.items()
        if any(key) and count >= min_subset_size
    }
    if not combos:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing combinations to plot.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template="plotly_white")
        return fig

    sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)[:max_patterns]
    combo_keys = [c[0] for c in sorted_combos]
    combo_counts = [c[1] for c in sorted_combos]
    n_combos = len(combo_keys)
    col_labels = _safe_labels(missing_cols)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )
    bar_text = (
        [f"{c/n_rows_total*100:.1f}%" for c in combo_counts]
        if show_pct else [""] * n_combos
    )
    fig.add_trace(
        go.Bar(
            x=list(range(n_combos)),
            y=combo_counts,
            marker_color=color,
            text=bar_text,
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=1,
    )
    for xi, key in enumerate(combo_keys):
        active_rows = [yi for yi, active in enumerate(key) if active]
        if len(active_rows) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[xi, xi],
                    y=[min(active_rows), max(active_rows)],
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2, col=1,
            )
        for yi, active in enumerate(key):
            fig.add_trace(
                go.Scatter(
                    x=[xi],
                    y=[yi],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=color if active else "#dddddd",
                        line=dict(color="white" if active else "#cccccc", width=1),
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"{col_labels[yi]}: {'missing' if active else 'present'}"
                        "<extra></extra>"
                    ),
                ),
                row=2, col=1,
            )
    fig.update_layout(
        title="UpSet Plot of Missing Value Combinations",
        template="plotly_white",
        margin=dict(l=80, r=40, t=70, b=60),
        height=400 + n_cols * 30,
    )
    fig.update_yaxes(title_text="Intersection size", row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(
        tickvals=list(range(n_cols)),
        ticktext=col_labels,
        autorange="reversed",
        row=2, col=1,
    )
    fig.update_layout(**_rtl_plotly_layout(col_labels))
    return fig


def _miss_patterns_plotly(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    top_n: int = 10,
) -> "go.Figure":
    """Plotly backend for :func:`~missingly.visualisation.static.miss_patterns`."""
    go = _require_plotly()
    null_mat = _nullity(df, missing_values)

    def _pattern_label(row):
        cols = [c for c, v in row.items() if v]
        return "(complete)" if not cols else " + ".join(_rtl_safe(str(c)) for c in cols)

    patterns = null_mat.apply(_pattern_label, axis=1)
    counts = patterns.value_counts().head(top_n).sort_values(ascending=True)
    pct = counts / len(df) * 100
    pattern_labels = counts.index.tolist()
    fig = go.Figure(
        data=go.Bar(
            x=counts.values,
            y=pattern_labels,
            orientation="h",
            marker_color="#4C72B0",
            text=[f"{p:.1f}%" for p in pct.values],
            textposition="outside",
            hovertemplate="%{y}<br>Count: %{x} (%{text})<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top-{top_n} Missingness Patterns",
        xaxis=dict(title="Row count"),
        template="plotly_white",
        margin=dict(l=200, r=80, t=60, b=60),
        height=max(300, top_n * 40 + 100),
    )
    fig.update_layout(**_rtl_plotly_layout(pattern_labels))
    return fig
