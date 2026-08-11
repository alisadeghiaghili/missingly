"""Tidy missingness exploration API (naniar parity).

This module provides tidy data structures and functions for exploring
missing values, similar to R's naniar package. It enables:

- Shadow matrix creation and manipulation
- Tidy long-format missingness data
- Special missing value support
- ggplot2-style visualisation functions

The API is designed to work seamlessly with pandas and matplotlib,
following the tidy data philosophy of naniar.

References
----------
- Tierney, N. J., & Cook, D. H. (2023). naniar: Data Structures,
  Summaries, and Visualisations for Missing Data. Journal of
  Statistical Software, 105(7).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def shadow_matrix(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Create a shadow matrix indicating missing values.

    The shadow matrix has the same shape as the input DataFrame, with
    each cell containing "NA" if the original value is missing, or
    "!" if the value is present.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional values to treat as missing (e.g., [-99, "N/A"]).

    Returns
    -------
    pd.DataFrame
        Shadow matrix with same shape and index as *df*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    >>> shadow = shadow_matrix(df)
    >>> shadow.loc[0, "a"]
    '!'
    >>> shadow.loc[0, "b"]
    'NA'
    """
    shadow = pd.DataFrame(
        np.where(df.isna(), "NA", "!"),
        index=df.index,
        columns=df.columns,
    )

    if missing_values is not None:
        for val in missing_values:
            shadow = shadow.where(df != val, "NA")

    return shadow


def shadow_long(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Convert DataFrame to tidy long-format with missingness indicators.

    Returns a long-format DataFrame with columns:
    - ``variable``: column name
    - ``row``: row index
    - ``value``: original value
    - ``missing``: boolean indicating if value is missing

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional values to treat as missing.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with missingness indicators.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    >>> long_df = shadow_long(df)
    >>> long_df.columns.tolist()
    ['variable', 'row', 'value', 'missing']
    """
    records = []
    for col in df.columns:
        for idx in df.index:
            val = df.loc[idx, col]
            is_missing = pd.isna(val)
            if missing_values is not None and not is_missing:
                is_missing = val in missing_values
            records.append({
                "variable": col,
                "row": idx,
                "value": val,
                "missing": is_missing,
            })

    return pd.DataFrame(records)


def shadow_wide(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Create wide-format missingness indicator DataFrame.

    Returns a DataFrame with same shape as input, where each cell
    contains True if the value is missing, False otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional values to treat as missing.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame indicating missingness.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    >>> wide = shadow_wide(df)
    >>> wide.loc[0, "a"]
    False
    >>> wide.loc[0, "b"]
    True
    """
    result = df.isna()

    if missing_values is not None:
        for val in missing_values:
            result = result | (df == val)

    return result


def special_missing(
    df: pd.DataFrame,
    replacements: Dict[str, Any],
) -> pd.DataFrame:
    """Replace special values with NaN to mark as missing.

    This function enables handling of special missing values like
    "don't know", "refused", "not applicable", etc.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    replacements : dict
        Dictionary mapping column names to values that should be
        treated as missing. Can also be a global mapping if keys
        are values (not column names).

    Returns
    -------
    pd.DataFrame
        DataFrame with special values replaced by NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"response": ["yes", "no", "dk", "na"]})
    >>> result = special_missing(df, {"response": ["dk", "na"]})
    >>> result["response"].isna().sum()
    2
    """
    result = df.copy()
    for col, vals in replacements.items():
        if col in result.columns:
            if isinstance(vals, list):
                result[col] = result[col].replace(vals, np.nan)
            else:
                result[col] = result[col].replace([vals], np.nan)
    return result


def n_miss_var(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.Series:
    """Count missing values per variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional values to treat as missing.

    Returns
    -------
    pd.Series
        Series with variable names as index and missing counts as values.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    >>> n_miss_var(df)
    a    1
    b    1
    dtype: int64
    """
    if missing_values is None:
        return df.isna().sum()
    return (df.isna() | df.isin(missing_values)).sum()


def n_miss_case(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.Series:
    """Count missing values per case (row).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional values to treat as missing.

    Returns
    -------
    pd.Series
        Series with row indices as index and missing counts as values.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    >>> n_miss_case(df)
    0    1
    1    1
    2    0
    dtype: int64
    """
    if missing_values is None:
        return df.isna().sum(axis=1)
    return (df.isna() | df.isin(missing_values)).sum(axis=1)


def pct_miss_var(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.Series:
    """Calculate percentage of missing values per variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional values to treat as missing.

    Returns
    -------
    pd.Series
        Series with variable names as index and missing percentages.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    >>> pct_miss_var(df)
    a    33.333333
    b    33.333333
    dtype: float64
    """
    return (n_miss_var(df, missing_values) / len(df)) * 100


def gg_miss_var(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    show_pct: bool = True,
    figsize: tuple = (8, 6),
    ax: Optional[Any] = None,
) -> Any:
    """Bar chart of missing values per variable (ggplot2 style).

    This function creates a horizontal bar chart showing the number
    or percentage of missing values for each variable, similar to
    naniar's ``gg_miss_var()``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional values to treat as missing.
    show_pct : bool, default True
        If True, show percentage labels on bars.
    figsize : tuple, default (8, 6)
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    >>> ax = gg_miss_var(df)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    miss_counts = n_miss_var(df, missing_values)
    miss_pct = pct_miss_var(df, missing_values)

    # Sort by missing count
    sorted_idx = miss_counts.sort_values(ascending=True).index
    sorted_counts = miss_counts[sorted_idx]
    sorted_pct = miss_pct[sorted_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    y_pos = range(len(sorted_idx))
    ax.barh(y_pos, sorted_counts.values, color="#4C72B0", alpha=0.8)

    if show_pct:
        for i, (count, pct) in enumerate(zip(sorted_counts, sorted_pct)):
            ax.text(count + 0.1, i, f"{pct:.1f}%", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_idx)
    ax.set_xlabel("Number missing")
    ax.set_title("Missing values per variable")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def gg_miss_case(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    show_pct: bool = True,
    figsize: tuple = (8, 6),
    ax: Optional[Any] = None,
) -> Any:
    """Bar chart of missing values per case (row).

    This function creates a bar chart showing the number of missing
    values for each row, similar to naniar's ``gg_miss_case()``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional values to treat as missing.
    show_pct : bool, default True
        If True, show percentage labels on bars.
    figsize : tuple, default (8, 6)
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    >>> ax = gg_miss_case(df)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    miss_counts = n_miss_case(df, missing_values)
    miss_pct = (miss_counts / df.shape[1]) * 100

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x_pos = range(len(df))
    ax.bar(x_pos, miss_counts.values, color="#4C72B0", alpha=0.8)

    if show_pct:
        for i, (count, pct) in enumerate(zip(miss_counts, miss_pct)):
            if count > 0:
                ax.text(i, count + 0.1, f"{pct:.1f}%", ha="center", fontsize=8)

    ax.set_xlabel("Case (row)")
    ax.set_ylabel("Number missing")
    ax.set_title("Missing values per case")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def gg_miss_fct(
    df: pd.DataFrame,
    fct: str,
    missing_values: Optional[List] = None,
    figsize: tuple = (8, 6),
    ax: Optional[Any] = None,
) -> Any:
    """Missing values grouped by a factor variable.

    This function creates a heatmap showing missingness patterns
    grouped by a categorical factor, similar to naniar's
    ``gg_miss_fct()``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    fct : str
        Name of the factor (categorical) variable to group by.
    missing_values : list, optional
        Additional values to treat as missing.
    figsize : tuple, default (8, 6)
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     "group": ["A", "A", "B", "B", "A"],
    ...     "value": [1.0, np.nan, 3.0, np.nan, 5.0]
    ... })
    >>> ax = gg_miss_fct(df, fct="group")  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    if fct not in df.columns:
        raise ValueError(f"Factor column '{fct}' not found in DataFrame.")

    # Calculate missing percentage by group
    groups = df[fct].unique()
    other_cols = [c for c in df.columns if c != fct]

    miss_data = []
    for group in groups:
        group_df = df[df[fct] == group]
        for col in other_cols:
            miss_pct = group_df[col].isna().mean() * 100
            miss_data.append({
                "group": group,
                "variable": col,
                "pct_missing": miss_pct,
            })

    miss_df = pd.DataFrame(miss_data)
    pivot = miss_df.pivot(index="variable", columns="group", values="pct_missing")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Missing values by {fct}")

    plt.colorbar(im, ax=ax, label="% Missing")

    return ax


def gg_miss_upset(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    n_sets: int = 10,
    figsize: tuple = (10, 6),
    ax: Optional[Any] = None,
) -> Any:
    """UpSet plot of missing value combinations.

    This function creates an UpSet plot showing the most common
    combinations of missing values, similar to naniar's
    ``gg_miss_upset()``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Additional values to treat as missing.
    n_sets : int, default 10
        Number of top combinations to show.
    figsize : tuple, default (10, 6)
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     "a": [1.0, np.nan, 3.0, np.nan],
    ...     "b": [np.nan, 2.0, np.nan, 4.0]
    ... })
    >>> ax = gg_miss_upset(df)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    # Create binary missingness matrix
    if missing_values is None:
        miss_matrix = df.isna()
    else:
        miss_matrix = df.isna() | df.isin(missing_values)

    # Count combinations
    combos = miss_matrix.apply(lambda row: tuple(row), axis=1)
    combo_counts = combos.value_counts().head(n_sets)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot as horizontal bar chart
    labels = []
    for combo in combo_counts.index:
        label = " + ".join([col for col, is_miss in zip(df.columns, combo) if is_miss])
        if not label:
            label = "(complete)"
        labels.append(label)

    y_pos = range(len(labels))
    ax.barh(y_pos, combo_counts.values, color="#4C72B0", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.set_title("Missing value combinations (UpSet)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax
