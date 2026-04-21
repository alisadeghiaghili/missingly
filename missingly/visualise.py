import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from upsetplot import UpSet
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
from scipy.spatial.distance import pdist

def matrix(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
    """A matrix plot to visualize the location of missing data.

    This function creates a heatmap of the nullity of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the seaborn heatmap.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Create a nullity matrix
    if missing_values is None:
        nullity_matrix = df.isnull()
    else:
        nullity_matrix = df.isin(missing_values)

    # Use seaborn's heatmap to plot the nullity matrix
    sns.heatmap(nullity_matrix, cbar=False, ax=ax, **kwargs)

    # Improve labels for readability
    if df.shape[0] < 50:
        ax.set_yticks(np.arange(len(df)) + 0.5)
        ax.set_yticklabels(df.index)
    else:
        ax.set_yticks([])

    if df.shape[1] < 50:
        ax.set_xticks(np.arange(len(df.columns)) + 0.5)
        ax.set_xticklabels(df.columns, rotation=90)
    else:
        ax.set_xticks([])


    ax.set_title("Missing Data Matrix")

    return ax

def bar(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
    """A bar plot to visualize the number of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the matplotlib bar plot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if missing_values is None:
        miss_counts = df.isnull().sum()
    else:
        miss_counts = df.isin(missing_values).sum()
        
    miss_counts.plot(kind='bar', ax=ax, **kwargs)

    ax.set_title("Missing Values per Column")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of Missing Values")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return ax


def upset(df: pd.DataFrame, missing_values: list = None, **kwargs):
    """An upset plot to visualize combinations of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the upsetplot function.

    Returns
    -------
    upsetplot.UpSet
        An upsetplot object.
    """
    # Get the nullity matrix
    if missing_values is None:
        nullity_matrix = df.isnull()
    else:
        nullity_matrix = df.isin(missing_values)

    # Get the columns with at least one missing value
    missing_cols = nullity_matrix.columns[nullity_matrix.any()]
    if missing_cols.empty:
        print("No missing values to plot.")
        return

    nullity_matrix = nullity_matrix[missing_cols]

    # Get the combinations of missing values
    miss_combinations = nullity_matrix.groupby(list(missing_cols)).size()

    # Use upsetplot
    return UpSet(miss_combinations, sort_by='cardinality', **kwargs).plot()


def scatter_miss(df: pd.DataFrame, x: str, y: str, ax=None, missing_values: list = None, **kwargs):
    """A scatter plot that visualizes missing values.

    This function creates a scatter plot of two variables, with missing
    values highlighted.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    x : str
        The name of the column for the x-axis.
    y : str
        The name of the column for the y-axis.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the seaborn scatterplot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    plot_df = df[[x, y]].copy()
    
    if missing_values is None:
        plot_df[f'{x}_NA'] = plot_df[x].isnull()
        plot_df[f'{y}_NA'] = plot_df[y].isnull()
    else:
        plot_df[f'{x}_NA'] = plot_df[x].isin(missing_values)
        plot_df[f'{y}_NA'] = plot_df[y].isin(missing_values)

    any_x_na = plot_df[f'{x}_NA'].any()
    any_y_na = plot_df[f'{y}_NA'].any()

    # Impute values for plotting
    if any_x_na:
        x_min = plot_df[x].min()
        plot_df[x] = plot_df[x].fillna(x_min - (abs(x_min) * 0.1))
    if any_y_na:
        y_min = plot_df[y].min()
        plot_df[y] = plot_df[y].fillna(y_min - (abs(y_min) * 0.1))

    hue = None
    if any_x_na and any_y_na:
        hue = plot_df[f'{x}_NA'].astype(str) + "_" + plot_df[f'{y}_NA'].astype(str)
        hue.name = "Missingness"
    elif any_x_na:
        hue = plot_df[f'{x}_NA']
        hue.name = f"Missing {x}"
    elif any_y_na:
        hue = plot_df[f'{y}_NA']
        hue.name = f"Missing {y}"


    # Plot all points
    sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, ax=ax, **kwargs)

    ax.set_title(f"Scatter Plot of {x} vs {y} with Missing Values")
    return ax

def miss_case(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
    """A bar plot to visualize the number of missing values per case (row).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the matplotlib bar plot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if missing_values is None:
        miss_counts = df.isnull().sum(axis=1)
    else:
        miss_counts = df.isin(missing_values).sum(axis=1)
        
    miss_counts.plot(kind='bar', ax=ax, **kwargs)

    ax.set_title("Missing Values per Case")
    ax.set_xlabel("Cases (Rows)")
    ax.set_ylabel("Number of Missing Values")
    plt.tight_layout()

    return ax

def vis_impute_dist(original_df: pd.DataFrame, imputed_df: pd.DataFrame, column: str, ax=None, **kwargs):
    """Visualize the distribution of original and imputed data for a column.

    Parameters
    ----------
    original_df : pd.DataFrame
        The original dataframe with missing values.
    imputed_df : pd.DataFrame
        The dataframe after imputation.
    column : str
        The column to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    kwargs
        Additional keyword arguments to pass to the seaborn kdeplot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(original_df[column].dropna(), ax=ax, label='Original', **kwargs)
    sns.kdeplot(imputed_df[column], ax=ax, label='Imputed', **kwargs)
    ax.set_title(f"Distribution of Original vs. Imputed Data for {column}")
    ax.legend()
    return ax

def vis_miss_fct(df: pd.DataFrame, fct: str, ax=None, missing_values: list = None, **kwargs):
    """Visualize missingness by a factor variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    fct : str
        The name of the categorical variable to group by.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the seaborn barplot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if missing_values is None:
        miss_by_fct = df.drop(columns=fct).groupby(df[fct]).apply(lambda x: x.isnull().sum())
    else:
        miss_by_fct = df.drop(columns=fct).groupby(df[fct]).apply(lambda x: x.isin(missing_values).sum())
        
    miss_by_fct.plot(kind='bar', stacked=True, ax=ax, **kwargs)

    ax.set_title(f"Missing Values by {fct}")
    ax.set_xlabel(fct)
    ax.set_ylabel("Number of Missing Values")
    plt.tight_layout()

    return ax

def vis_miss_cumsum_var(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
    """Visualize the cumulative sum of missing values per variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the plot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if missing_values is None:
        miss_cumsum = df.isnull().sum().cumsum()
    else:
        miss_cumsum = df.isin(missing_values).sum().cumsum()
        
    miss_cumsum.plot(kind='line', ax=ax, **kwargs)

    ax.set_title("Cumulative Sum of Missing Values per Variable")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Cumulative Sum of Missing Values")
    plt.tight_layout()

    return ax

def vis_miss_cumsum_case(df: pd.DataFrame, ax=None, missing_values: list = None, **kwargs):
    """Visualize the cumulative sum of missing values per case.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the plot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if missing_values is None:
        miss_cumsum = df.isnull().sum(axis=1).cumsum()
    else:
        miss_cumsum = df.isin(missing_values).sum(axis=1).cumsum()
        
    miss_cumsum.plot(kind='line', ax=ax, **kwargs)

    ax.set_title("Cumulative Sum of Missing Values per Case")
    ax.set_xlabel("Cases (Rows)")
    ax.set_ylabel("Cumulative Sum of Missing Values")
    plt.tight_layout()

    return ax

def vis_miss_span(df: pd.DataFrame, column: str, span: int, ax=None, missing_values: list = None, **kwargs):
    """Visualize the number of missings in a given repeating span on a single variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    column : str
        The column to visualize.
    span : int
        The size of the span.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the plot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if missing_values is None:
        miss_span = df[column].isnull().rolling(span).sum()
    else:
        miss_span = df[column].isin(missing_values).rolling(span).sum()
        
    miss_span.plot(kind='line', ax=ax, **kwargs)

    ax.set_title(f"Missing Values in Spans of {span} for {column}")
    ax.set_xlabel("Index")
    ax.set_ylabel(f"Number of Missing Values in Span")
    plt.tight_layout()

    return ax

def vis_parallel_coords(df: pd.DataFrame, missing_values: list = None, **kwargs):
    """A parallel coordinates plot to visualize missingness patterns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    missing_values : list, optional
        A list of values to be considered as missing.
    kwargs
        Additional keyword arguments to pass to the plot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    
    if missing_values is None:
        df_miss = df.isnull().astype(int)
    else:
        df_miss = df.isin(missing_values).astype(int)
        
    df_miss['missing_count'] = df_miss.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    pd.plotting.parallel_coordinates(df_miss, 'missing_count', ax=ax, **kwargs)

    ax.set_title("Parallel Coordinates Plot of Missingness")
    plt.tight_layout()
    
    return ax

def dendrogram(df: pd.DataFrame, ax=None, missing_values: list = None, method='ward', **kwargs):
    """A dendrogram to cluster variables by their nullity correlation.

    This function creates a dendrogram that shows how variables cluster
    based on their missing data patterns, helping to identify groups
    of variables that tend to be missing together.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to visualize.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new one will be created.
    missing_values : list, optional
        A list of values to be considered as missing.
    method : str, optional
        The linkage method to use. Options: 'ward', 'single', 'complete', 
        'average', 'weighted', 'centroid', 'median'. Default is 'ward'.
    kwargs
        Additional keyword arguments to pass to scipy.cluster.hierarchy.dendrogram.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib axes object.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Create nullity matrix
    if missing_values is None:
        nullity_matrix = df.isnull().astype(int)
    else:
        nullity_matrix = df.isin(missing_values).astype(int)

    # Calculate correlation matrix of missingness patterns
    # We use 1 - correlation as distance
    corr_matrix = nullity_matrix.corr()
    
    # Convert to distance matrix (1 - correlation)
    distance_matrix = 1 - corr_matrix.abs()
    
    # Convert to condensed distance matrix for scipy
    condensed_distances = pdist(distance_matrix.values, metric='precomputed')
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method=method)
    
    # Create dendrogram
    dendro_result = scipy_dendrogram(
        linkage_matrix,
        labels=df.columns.tolist(),
        ax=ax,
        orientation='top',
        **kwargs
    )
    
    ax.set_title("Dendrogram of Variables by Missing Data Patterns")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Distance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return ax
