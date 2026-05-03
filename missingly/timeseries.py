"""Time series missing data analysis for missingly.

This module provides diagnostic and imputation utilities designed
specifically for time-indexed DataFrames.

All public functions assume the DataFrame's index is either a
``DatetimIndex`` or an ordinal integer index representing ordered
observations (rows must be in chronological order).

Functions
---------
miss_ts_summary
    Per-column summary of gaps: count, mean length, max length, etc.
gap_table
    Tidy table of every individual gap per column (start, end, length).
vis_ts_miss
    Tile heatmap of missingness along the time axis.
vis_gap_lengths
    Distribution (histogram or box) of gap lengths per column.
vis_miss_over_time
    Rolling-window missingness rate line chart for all columns.
impute_ts
    Time-series-aware imputation: forward-fill, backward-fill,
    linear interpolation, time interpolation, and spline.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_sorted(df: pd.DataFrame) -> None:
    """Raise ValueError if a DatetimeIndex is not monotonically increasing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose index is checked.

    Raises
    ------
    ValueError
        If *df* has a DatetimeIndex that is not monotonically increasing.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "DataFrame index must be sorted in ascending order. "
                "Call df.sort_index() before using timeseries functions."
            )


def _nullity(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """Boolean missingness mask, optionally extended with sentinel values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional sentinel values treated as missing.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame, ``True`` where a value is considered missing.
    """
    if missing_values is None:
        return df.isnull()
    return df.isnull() | df.isin(missing_values)


def _gaps_for_column(
    mask: pd.Series,
) -> list[dict]:
    """Extract contiguous missing runs (gaps) from a boolean missingness mask.

    Parameters
    ----------
    mask : pd.Series
        Boolean Series where ``True`` indicates a missing value.
        The Series must be in time order.

    Returns
    -------
    list of dict
        Each dict has keys ``'start_idx'``, ``'end_idx'``,
        ``'start_label'``, ``'end_label'``, and ``'length'``.
        ``start_label`` / ``end_label`` are the actual index labels
        (timestamps or integers).
    """
    gaps = []
    in_gap = False
    gap_start = None

    for pos, (label, val) in enumerate(mask.items()):
        if val and not in_gap:
            in_gap = True
            gap_start = (pos, label)
        elif not val and in_gap:
            end_pos = pos - 1
            end_label = mask.index[end_pos]
            gaps.append({
                "start_idx": gap_start[0],
                "end_idx": end_pos,
                "start_label": gap_start[1],
                "end_label": end_label,
                "length": end_pos - gap_start[0] + 1,
            })
            in_gap = False

    if in_gap:
        end_pos = len(mask) - 1
        end_label = mask.index[end_pos]
        gaps.append({
            "start_idx": gap_start[0],
            "end_idx": end_pos,
            "start_label": gap_start[1],
            "end_label": end_label,
            "length": end_pos - gap_start[0] + 1,
        })

    return gaps


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def miss_ts_summary(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Per-column summary statistics for time series missingness.

    Provides gap-aware metrics that are more informative than a simple
    count of missing values for time-indexed data.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame (DatetimeIndex or integer index).
    missing_values : list, optional
        Additional sentinel values treated as missing.

    Returns
    -------
    pd.DataFrame
        One row per column with the following columns:

        - ``n_miss``        : total number of missing observations.
        - ``pct_miss``      : percentage of observations that are missing.
        - ``n_gaps``        : number of contiguous missing runs.
        - ``mean_gap_len``  : mean length of a gap (rows).
        - ``max_gap_len``   : longest single gap (rows).
        - ``longest_gap_start`` : index label where the longest gap starts.
        - ``longest_gap_end``   : index label where the longest gap ends.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> dates = pd.date_range('2024-01-01', periods=10, freq='D')
    >>> df = pd.DataFrame({'temp': [1, np.nan, np.nan, 4, 5, np.nan, 7, 8, 9, 10]},
    ...                   index=dates)
    >>> miss_ts_summary(df)
    """
    _require_sorted(df)
    null_df = _nullity(df, missing_values)
    n = len(df)
    records = []

    for col in df.columns:
        mask = null_df[col]
        n_miss = int(mask.sum())
        pct_miss = n_miss / n * 100
        gaps = _gaps_for_column(mask)
        n_gaps = len(gaps)

        if gaps:
            lengths = [g["length"] for g in gaps]
            mean_gap = float(np.mean(lengths))
            max_gap = max(lengths)
            longest = max(gaps, key=lambda g: g["length"])
            longest_start = longest["start_label"]
            longest_end = longest["end_label"]
        else:
            mean_gap = 0.0
            max_gap = 0
            longest_start = None
            longest_end = None

        records.append({
            "column": col,
            "n_miss": n_miss,
            "pct_miss": round(pct_miss, 2),
            "n_gaps": n_gaps,
            "mean_gap_len": round(mean_gap, 2),
            "max_gap_len": max_gap,
            "longest_gap_start": longest_start,
            "longest_gap_end": longest_end,
        })

    return pd.DataFrame(records).set_index("column")


def gap_table(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Return a tidy table of every individual contiguous gap per column.

    Useful for filtering (e.g. ``gap_table(df)[gap_table(df).length > 24]``)
    or for feeding into downstream repair logic.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.
    missing_values : list, optional
        Additional sentinel values treated as missing.

    Returns
    -------
    pd.DataFrame
        Columns: ``'column'``, ``'gap_id'``, ``'start'``, ``'end'``,
        ``'length'``.
        Sorted by column then gap start position.

    Example
    -------
    >>> big_gaps = gap_table(df).query('length > 5')
    """
    _require_sorted(df)
    null_df = _nullity(df, missing_values)
    rows = []

    for col in df.columns:
        gaps = _gaps_for_column(null_df[col])
        for i, g in enumerate(gaps):
            rows.append({
                "column": col,
                "gap_id": i + 1,
                "start": g["start_label"],
                "end": g["end_label"],
                "length": g["length"],
            })

    if not rows:
        return pd.DataFrame(columns=["column", "gap_id", "start", "end", "length"])

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def vis_ts_miss(
    df: pd.DataFrame,
    ax=None,
    missing_values: Optional[List] = None,
    time_format: Optional[str] = None,
    n_xticks: int = 8,
    **kwargs,
):
    """Tile heatmap of missingness along the time axis.

    Each column of the DataFrame is a row in the heatmap; each time
    step is a column.  Red tiles are missing, light grey tiles are
    observed.  Unlike the generic ``vis_miss``, this function formats
    the x-axis as time labels and highlights the temporal structure.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  Created automatically if omitted.
    missing_values : list, optional
        Additional sentinel values treated as missing.
    time_format : str, optional
        ``strftime`` format string for x-axis labels.
        If ``None``, a sensible default is chosen based on the index
        frequency (daily: ``"%Y-%m-%d"``, sub-daily: ``"%m-%d %H:%M"``,
        integer: ``str``).
    n_xticks : int, optional
        Approximate number of x-axis tick labels to display.
        Default is 8.
    **kwargs
        Additional keyword arguments forwarded to ``seaborn.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    _require_sorted(df)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10, min(df.shape[0] // 10, 20)), max(3, df.shape[1] * 0.7)))

    null_df = _nullity(df, missing_values).astype(float).T

    cmap = kwargs.pop("cmap", ["#f0f0f0", "#d62728"])
    cbar = kwargs.pop("cbar", False)

    sns.heatmap(null_df, ax=ax, cmap=cmap, cbar=cbar, yticklabels=True, xticklabels=False, **kwargs)
    ax.set_title("Time Series Missingness Heatmap")
    ax.set_ylabel("Variable")
    ax.set_xlabel("Time")

    # X-axis tick formatting
    n = null_df.shape[1]
    tick_positions = np.linspace(0, n - 1, min(n_xticks, n), dtype=int)
    index_labels = df.index

    if isinstance(index_labels, pd.DatetimeIndex):
        if time_format is None:
            if (index_labels[-1] - index_labels[0]) <= pd.Timedelta(days=3):
                time_format = "%m-%d %H:%M"
            else:
                time_format = "%Y-%m-%d"
        tick_labels = [index_labels[i].strftime(time_format) for i in tick_positions]
    else:
        tick_labels = [str(index_labels[i]) for i in tick_positions]

    ax.set_xticks(tick_positions + 0.5)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    # Missingness rate legend
    pct = _nullity(df, missing_values).mean() * 100
    ytick_labels = [f"{col} ({pct[col]:.1f}%)" for col in df.columns]
    ax.set_yticklabels(ytick_labels, rotation=0)

    missing_patch = mpatches.Patch(color="#d62728", label="Missing")
    obs_patch = mpatches.Patch(color="#f0f0f0", label="Observed")
    ax.legend(handles=[missing_patch, obs_patch], loc="upper right",
              fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return ax


def vis_gap_lengths(
    df: pd.DataFrame,
    kind: Literal["hist", "box"] = "hist",
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Distribution of gap lengths per column.

    Helps answer: "Is missingness concentrated in a few long gaps
    (sensor failure) or scattered in many short ones (random dropout)?"

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.
    kind : {'hist', 'box'}, optional
        Plot type.  ``'hist'`` overlays per-column histograms.
        ``'box'`` draws a box plot with one box per column.
        Default is ``'hist'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  Created automatically if omitted.
    missing_values : list, optional
        Additional sentinel values treated as missing.
    **kwargs
        Additional keyword arguments forwarded to the underlying
        matplotlib / seaborn call.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.

    Raises
    ------
    ValueError
        If no gaps are found in the DataFrame.
    """
    _require_sorted(df)
    gt = gap_table(df, missing_values=missing_values)

    if gt.empty:
        raise ValueError("No gaps found in the DataFrame.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, df.shape[1] * 1.5), 5))

    if kind == "box":
        data = [gt.loc[gt["column"] == col, "length"].values for col in df.columns
                if col in gt["column"].values]
        labels = [col for col in df.columns if col in gt["column"].values]
        ax.boxplot(data, labels=labels, **kwargs)
        ax.set_ylabel("Gap length (rows)")
        ax.set_title("Distribution of Gap Lengths per Variable")
        plt.xticks(rotation=45, ha="right")
    else:
        cols_with_gaps = gt["column"].unique()
        for col in cols_with_gaps:
            lengths = gt.loc[gt["column"] == col, "length"].values
            ax.hist(lengths, bins="auto", alpha=0.6, label=col, **kwargs)
        ax.set_xlabel("Gap length (rows)")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of Gap Lengths")
        ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return ax


def vis_miss_over_time(
    df: pd.DataFrame,
    window: int = 10,
    ax=None,
    missing_values: Optional[List] = None,
    **kwargs,
):
    """Rolling-window missingness rate line chart for all columns.

    Shows how the fraction of missing values evolves over time in a
    rolling window, making it easy to spot periods of degraded data
    quality (e.g. sensor outages, batch processing failures).

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.
    window : int, optional
        Size of the rolling window in rows.  Default is 10.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  Created automatically if omitted.
    missing_values : list, optional
        Additional sentinel values treated as missing.
    **kwargs
        Additional keyword arguments forwarded to ``Series.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    _require_sorted(df)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    null_df = _nullity(df, missing_values).astype(float)
    rolling_rate = null_df.rolling(window=window, min_periods=1).mean() * 100

    for col in rolling_rate.columns:
        rolling_rate[col].plot(ax=ax, label=col, **kwargs)

    ax.set_title(f"Rolling Missingness Rate (window={window})", fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("% Missing")
    ax.set_ylim(0, 100)
    ax.axhline(y=5, color="orange", linestyle="--", linewidth=0.8, alpha=0.7, label="5% threshold")
    ax.axhline(y=20, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="20% threshold")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


_TS_STRATEGIES = frozenset({"ffill", "bfill", "linear", "time", "spline"})


def impute_ts(
    df: pd.DataFrame,
    strategy: Literal["ffill", "bfill", "linear", "time", "spline"] = "linear",
    limit: Optional[int] = None,
    order: int = 3,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Time-series-aware imputation for numeric columns.

    Unlike ``impute_mean`` or ``impute_knn``, this function respects
    the temporal ordering of observations.  It is the correct choice
    when the gap structure matters (e.g. sensor readings, stock prices,
    weather data).

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.  The index should be a
        ``DatetimeIndex`` for the ``'time'`` strategy; other strategies
        work with any ordered index.
    strategy : {'ffill', 'bfill', 'linear', 'time', 'spline'}, optional
        Imputation method:

        * ``'ffill'``   — propagate last valid observation forward.
        * ``'bfill'``   — propagate next valid observation backward.
        * ``'linear'``  — linear interpolation by row position.
        * ``'time'``    — linear interpolation weighted by time distance
          (requires DatetimeIndex).
        * ``'spline'``  — spline interpolation of given *order*.

        Default is ``'linear'``.
    limit : int, optional
        Maximum number of consecutive missing values to fill.
        ``None`` means no limit (fill all gaps).  Useful for preventing
        imputation of very long gaps (e.g. sensor failures > 1 hour).
    order : int, optional
        Spline order.  Only used when ``strategy='spline'``.
        Default is 3 (cubic spline).
    missing_values : list, optional
        Additional sentinel values replaced with ``NaN`` before
        imputation.

    Returns
    -------
    pd.DataFrame
        Imputed copy of *df*.  The original is not modified.
        Categorical / non-numeric columns are forward-filled regardless
        of the chosen strategy (the only sensible temporal fill for
        nominal data).

    Raises
    ------
    ValueError
        If ``strategy`` is not one of the supported values.
    ValueError
        If ``strategy='time'`` is used with a non-DatetimeIndex.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> dates = pd.date_range('2024-01-01', periods=6, freq='h')
    >>> df = pd.DataFrame({'temp': [10.0, np.nan, np.nan, 13.0, np.nan, 15.0]}, index=dates)
    >>> impute_ts(df, strategy='linear', limit=2)
         temp
    ...  10.0
    ...  11.0
    ...  12.0
    ...  13.0
    ...  14.0
    ...  15.0
    """
    _require_sorted(df)

    if strategy not in _TS_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(_TS_STRATEGIES)}; got {strategy!r}"
        )

    if strategy == "time" and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "strategy='time' requires a DatetimeIndex. "
            "Convert your index with pd.to_datetime() first."
        )

    result = df.copy()

    # Replace extra sentinels
    if missing_values:
        result = result.replace(missing_values, np.nan)

    num_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = result.select_dtypes(exclude=[np.number]).columns.tolist()

    # Numeric columns — use the chosen strategy
    if num_cols:
        if strategy in ("ffill", "bfill"):
            result[num_cols] = result[num_cols].fillna(
                method=strategy, limit=limit  # type: ignore[arg-type]
            )
        elif strategy == "linear":
            result[num_cols] = result[num_cols].interpolate(
                method="linear", limit=limit, limit_direction="forward"
            )
        elif strategy == "time":
            result[num_cols] = result[num_cols].interpolate(
                method="time", limit=limit, limit_direction="forward"
            )
        elif strategy == "spline":
            result[num_cols] = result[num_cols].interpolate(
                method="spline", order=order, limit=limit, limit_direction="forward"
            )

    # Categorical columns — always forward-fill (only sensible temporal fill)
    if cat_cols:
        result[cat_cols] = result[cat_cols].ffill(limit=limit)

    return result
