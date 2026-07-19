"""Time series missing data analysis for missingly."""
from __future__ import annotations

import inspect
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from ._deprecation import deprecated_api


def _require_sorted(df: pd.DataFrame) -> None:
    """Validate that a DataFrame's index is sorted for time-series ops.

    Duplicate timestamps are permitted as long as the index is
    non-decreasing; only strictly out-of-order labels raise.

    Raises
    ------
    ValueError
        If ``df.index`` is not monotonically increasing.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index must be monotonically increasing (sorted). "
            "Duplicate timestamps are allowed, but out-of-order timestamps "
            "are not. Call df.sort_index() before using time-series functions."
        )


def _nullity(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    null_mask = df.isnull()
    if missing_values:
        null_mask = null_mask | df.isin(missing_values)
    return null_mask


def _gaps_for_column(
    series: pd.Series,
    missing_values: Optional[List] = None,
) -> List[dict]:
    """Detect contiguous missing-value gaps in a time-indexed Series.

    Uses positional (integer) tracking instead of ``Index.get_loc()`` on
    index *values*, because ``get_loc()`` returns a ``slice`` (not an
    ``int``) when the index contains duplicate labels. Relying on
    positions makes gap length computation correct and crash-free
    regardless of whether the ``DatetimeIndex`` has duplicate timestamps.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series to scan for missing-value gaps. The index
        must be monotonically increasing (sorted); duplicate timestamps
        are allowed.
    missing_values : list, optional
        Additional sentinel values to treat as missing, on top of
        ``NaN``/``None``.

    Returns
    -------
    list of dict
        Each dict has keys ``start`` (index label where gap begins),
        ``end`` (index label where gap ends), and ``length`` (int,
        number of consecutive rows in the gap).

    Notes
    -----
    ``length`` counts rows (positions), not elapsed calendar time. If the
    index has duplicate timestamps, a run of missing rows sharing the same
    timestamp is counted correctly by position, not double-counted or
    dropped.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    >>> s = pd.Series([1.0, np.nan, np.nan, 4.0], index=idx)
    >>> _gaps_for_column(s)
    [{'start': Timestamp('2024-01-02 00:00:00'), 'end': Timestamp('2024-01-03 00:00:00'), 'length': 2}]
    """
    null_mask = series.isnull()
    if missing_values:
        null_mask = null_mask | series.isin(missing_values)

    gaps: List[dict] = []
    in_gap = False
    gap_start = None
    gap_start_pos = None
    index_vals = series.index

    for pos, (idx, is_null) in enumerate(zip(index_vals, null_mask)):
        if is_null and not in_gap:
            in_gap = True
            gap_start = idx
            gap_start_pos = pos
        elif not is_null and in_gap:
            in_gap = False
            gaps.append({"start": gap_start, "end": idx, "length": pos - gap_start_pos})

    if in_gap:
        end_pos = len(series) - 1
        gaps.append(
            {
                "start": gap_start,
                "end": index_vals[-1],
                "length": end_pos - gap_start_pos + 1,
            }
        )

    return gaps


def _boxplot_compat(ax, data: list, labels: list, **kwargs) -> None:
    sig = inspect.signature(ax.boxplot)
    if "tick_labels" in sig.parameters:
        ax.boxplot(data, tick_labels=labels, **kwargs)
    else:
        ax.boxplot(data, labels=labels, **kwargs)


def miss_ts_summary(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Per-column summary of missing gaps in a time-indexed DataFrame."""
    _require_sorted(df)
    records = []
    for col in df.columns:
        null_mask = df[col].isnull()
        if missing_values:
            null_mask = null_mask | df[col].isin(missing_values)

        n_miss = int(null_mask.sum())
        pct_miss = round(100.0 * n_miss / len(df), 2) if len(df) > 0 else 0.0
        gaps = _gaps_for_column(df[col], missing_values=missing_values)
        lengths = [g["length"] for g in gaps if g["length"] is not None]
        max_gap = int(max(lengths)) if lengths else 0

        records.append(
            {
                "variable": col,
                "n_miss": n_miss,
                "pct_miss": pct_miss,
                "n_gaps": len(gaps),
                "mean_gap": round(float(np.mean(lengths)), 2) if lengths else 0.0,
                "max_gap": max_gap,
                "max_gap_len": max_gap,
                "median_gap": float(np.median(lengths)) if lengths else 0.0,
            }
        )
    return pd.DataFrame(records).set_index("variable")


@deprecated_api(
    "gap_table is an experimental utility and may be moved to a separate timeseries package in a future release.",
    since="0.2.0",
)
def gap_table(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Tidy table of every individual missing gap per column (experimental).

    Returns
    -------
    pd.DataFrame
        Columns: ``column``, ``gap_id``, ``start``, ``end``, ``length``.
    """
    _require_sorted(df)
    target_cols = columns if columns is not None else df.columns.tolist()
    rows = []
    for col in target_cols:
        gaps = _gaps_for_column(df[col], missing_values=missing_values)
        for i, gap in enumerate(gaps, start=1):
            rows.append(
                {
                    "column": col,
                    "gap_id": i,
                    "start": gap["start"],
                    "end": gap["end"],
                    "length": gap["length"],
                }
            )
    if rows:
        return pd.DataFrame(rows, columns=["column", "gap_id", "start", "end", "length"])
    return pd.DataFrame(columns=["column", "gap_id", "start", "end", "length"])


def vis_ts_miss(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
    figsize: Optional[tuple] = None,
    cmap: str = "RdYlGn",
    title: str = "Time Series Missingness",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Tile heatmap showing missingness along the time axis."""
    _require_sorted(df)
    null_mask = _nullity(df, missing_values=missing_values)
    present = (~null_mask).astype(int)

    n_cols = len(df.columns)
    if figsize is None:
        figsize = (12, max(3, n_cols * 0.5))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.imshow(
        present.T.values,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    ax.set_yticks(range(n_cols))
    ax.set_yticklabels(df.columns)

    n_ticks = min(10, len(df))
    tick_indices = np.linspace(0, len(df) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [str(df.index[i])[:10] for i in tick_indices],
        rotation=45,
        ha="right",
    )

    present_patch = mpatches.Patch(color=plt.get_cmap(cmap)(1.0), label="Present")
    missing_patch = mpatches.Patch(color=plt.get_cmap(cmap)(0.0), label="Missing")
    ax.legend(handles=[present_patch, missing_patch], loc="upper right", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    return ax


@deprecated_api(
    "vis_gap_lengths is an experimental utility and may be moved to a separate timeseries package in a future release.",
    since="0.2.0",
)
def vis_gap_lengths(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
    kind: Literal["hist", "box"] = "hist",
    columns: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    title: str = "Gap Length Distribution",
) -> matplotlib.figure.Figure:
    """Distribution of gap lengths per column (experimental).

    Returns
    -------
    matplotlib.figure.Figure
        When no gaps are found, returns a Figure with an informational message.
    """
    _require_sorted(df)
    target_cols = columns if columns is not None else df.columns.tolist()

    col_gaps: dict = {}
    for col in target_cols:
        gaps = _gaps_for_column(df[col], missing_values=missing_values)
        lengths = [g["length"] for g in gaps if g["length"] is not None and g["length"] > 0]
        if lengths:
            col_gaps[col] = lengths

    if not col_gaps:
        fig, ax = plt.subplots(figsize=figsize or (6, 4))
        ax.text(
            0.5, 0.5,
            "No gaps found in the provided columns.",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        fig.suptitle(title)
        return fig

    n = len(col_gaps)
    if figsize is None:
        figsize = (max(6, n * 2), 4)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    for ax, (col, lengths) in zip(axes[0], col_gaps.items()):
        if kind == "hist":
            ax.hist(lengths, bins="auto", color="steelblue", edgecolor="white")
            ax.set_xlabel("Gap length")
            ax.set_ylabel("Count")
        else:
            _boxplot_compat(ax, [lengths], [col])
            ax.set_ylabel("Gap length")
        ax.set_title(col)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def vis_miss_over_time(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
    window: int = 10,
    figsize: Optional[tuple] = None,
    title: str = "Rolling Missingness Rate",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Rolling-window missingness rate line chart."""
    _require_sorted(df)
    null_mask = _nullity(df, missing_values=missing_values).astype(float)
    rolling_rate = null_mask.rolling(window=window, min_periods=1).mean() * 100

    if figsize is None:
        figsize = (12, 4)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for col in rolling_rate.columns:
        ax.plot(rolling_rate.index, rolling_rate[col], label=col)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"% missing (window={window})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    return ax


def impute_ts(
    df: pd.DataFrame,
    method: Optional[Literal["ffill", "bfill", "linear", "time", "spline"]] = None,
    *,
    strategy: Optional[Literal["ffill", "bfill", "linear", "time", "spline"]] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
    spline_order: int = 3,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Time-series-aware imputation."""
    if method is None and strategy is None:
        method = "linear"
    elif method is None:
        method = strategy
    elif strategy is not None and strategy != method:
        raise ValueError(
            f"Conflicting values for 'method' ({method!r}) and 'strategy' ({strategy!r}). "
            "Pass only one."
        )

    _require_sorted(df)
    valid_methods = ["bfill", "ffill", "linear", "spline", "time"]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid strategy: method must be one of {valid_methods}; got {method!r}."
        )

    result = df.copy()

    if missing_values:
        result = result.replace(missing_values, np.nan)

    if columns is not None:
        missing_cols = [c for c in columns if c not in result.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
        target_cols = columns
    else:
        target_cols = list(result.columns)

    for col in target_cols:
        if not pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].ffill(limit=limit)
            continue

        if not result[col].isnull().any():
            continue

        if method == "ffill":
            result[col] = result[col].ffill(limit=limit)
        elif method == "bfill":
            result[col] = result[col].bfill(limit=limit)
        elif method == "spline":
            result[col] = result[col].interpolate(
                method="spline", order=spline_order, limit=limit
            )
        else:
            result[col] = result[col].interpolate(method=method, limit=limit)

    return result
