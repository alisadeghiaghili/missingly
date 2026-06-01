"""Thin facade for the visualisation layer.

Implementation has been moved to the :mod:`missingly.visualisation`
sub-package:

* :mod:`missingly.visualisation._base`       – shared helpers
  (``_rtl_safe``, ``_safe_labels``, ``_nullity``, ``_pct_labels``)
* :mod:`missingly.visualisation.static`      – all matplotlib-based functions
* :mod:`missingly.visualisation.interactive` – all Plotly backends and public
  wrappers (called when ``interactive=True``)

This module re-exports every public name unchanged so that existing code
continues to work without modification::

    from missingly import visualise
    visualise.heatmap(df)
    # or
    from missingly.visualise import heatmap
    heatmap(df)

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
  vis_miss_span, vis_parallel_coords

Helper utilities (also accessible via this module)
  _rtl_safe, _safe_labels
"""
from __future__ import annotations

from missingly.visualisation import (
    matrix,
    bar,
    miss_case,
    miss_var_pct,
    vis_miss,
    miss_which,
    upset,
    miss_patterns,
    miss_cooccurrence,
    heatmap,
    dendrogram,
    miss_cluster,
    miss_row_profile,
    shadow_scatter,
    vis_miss_fct,
    vis_miss_by_group,
    vis_impute_dist,
    miss_impute_compare,
    scatter_miss,
    vis_miss_cumsum_var,
    vis_miss_cumsum_case,
    vis_miss_span,
    vis_parallel_coords,
)
from missingly.visualisation._base import _rtl_safe, _safe_labels

__all__ = [
    "matrix",
    "bar",
    "miss_case",
    "miss_var_pct",
    "vis_miss",
    "miss_which",
    "upset",
    "miss_patterns",
    "miss_cooccurrence",
    "heatmap",
    "dendrogram",
    "miss_cluster",
    "miss_row_profile",
    "shadow_scatter",
    "vis_miss_fct",
    "vis_miss_by_group",
    "vis_impute_dist",
    "miss_impute_compare",
    "scatter_miss",
    "vis_miss_cumsum_var",
    "vis_miss_cumsum_case",
    "vis_miss_span",
    "vis_parallel_coords",
    "_rtl_safe",
    "_safe_labels",
]
