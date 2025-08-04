"""
Utility functions for data visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def set_visualization_style(style='whitegrid', context='notebook', palette='deep'):
    """
    Set the visualization style for seaborn and matplotlib.
    
    Parameters:
    -----------
    style : str, default='whitegrid'
        Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
    context : str, default='notebook'
        Seaborn context ('paper', 'notebook', 'talk', 'poster')
    palette : str, default='deep'
        Seaborn color palette
    """
    sns.set(style=style, context=context, palette=palette)
    plt.style.use('seaborn')


def plot_missing_values(df, figsize=(12, 8), title='Missing Values'):
    """
    Visualize missing values in a dataframe.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to visualize
    figsize : tuple, default=(12, 8)
        Figure size
    title : str, default='Missing Values'
        Plot title
    """
    # Calculate missing values
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_data = pd.DataFrame({
        'Count': missing,
        'Percent': missing_percent
    }).sort_values('Percent', ascending=False)
    
    # Only show columns with missing values
    missing_data = missing_data[missing_data['Count'] > 0]
    
    if len(missing_data) == 0:
        print("No missing values in the dataframe.")
        return
    
    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=missing_data.index, y='Percent', data=missing_data)
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Percent Missing')
    plt.xticks(rotation=90)
    
    # Add count labels on top of bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2.,
                height + 0.5,
                f"{missing_data['Count'].iloc[i]:,}",
                ha="center", rotation=0) 
    
    plt.tight_layout()
    plt.show()


def plot_distribution(df, column, bins=30, kde=True, figsize=(12, 6)):
    """
    Plot the distribution of a column.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the column
    column : str
        Column name to plot
    bins : int, default=30
        Number of bins for histogram
    kde : bool, default=True
        Whether to plot KDE
    figsize : tuple, default=(12, 6)
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if pd.api.types.is_numeric_dtype(df[column]):
        # For numeric columns
        sns.histplot(df[column], bins=bins, kde=kde)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        
        # Add statistics
        mean = df[column].mean()
        median = df[column].median()
        mode = df[column].mode()[0]
        std = df[column].std()
        
        plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.axvline(median, color='g', linestyle='-.', label=f'Median: {median:.2f}')
        plt.axvline(mode, color='y', linestyle=':', label=f'Mode: {mode:.2f}')
        
        plt.legend()
        
        # Print statistics
        print(f"Statistics for {column}:")
        print(f"Mean: {mean:.2f}")
        print(f"Median: {median:.2f}")
        print(f"Mode: {mode:.2f}")
        print(f"Std Dev: {std:.2f}")
        print(f"Min: {df[column].min():.2f}")
        print(f"Max: {df[column].max():.2f}")
        
    else:
        # For categorical columns
        value_counts = df[column].value_counts()
        ax = sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        
        # Add count labels on top of bars
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 0.5,
                    f"{int(height):,}",
                    ha="center", rotation=0)
        
        # Print statistics
        print(f"Top categories for {column}:")
        print(value_counts.head().to_string())
        print(f"\nTotal categories: {len(value_counts)}")
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, method='pearson', figsize=(12, 10), cmap='coolwarm', annot=True):
    """
    Plot correlation matrix for numeric columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to analyze
    method : str, default='pearson'
        Correlation method ('pearson', 'kendall', 'spearman')
    figsize : tuple, default=(12, 10)
        Figure size
    cmap : str, default='coolwarm'
        Colormap for the heatmap
    annot : bool, default=True
        Whether to annotate cells with correlation values
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation analysis.")
        return
    
    # Compute correlation matrix
    corr = numeric_df.corr(method=method)
    
    # Plot
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt='.2f', 
                linewidths=0.5, square=True, cbar_kws={"shrink": .5})
    plt.title(f'Correlation Matrix ({method.capitalize()})')
    plt.tight_layout()
    plt.show()


def plot_pairplot(df, hue=None, vars=None, diag_kind='kde', height=2.5, aspect=1):
    """
    Create a pairplot of selected variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to visualize
    hue : str, default=None
        Column name for coloring points
    vars : list, default=None
        List of columns to include (default: all numeric columns)
    diag_kind : str, default='kde'
        Kind of plot for diagonal ('hist', 'kde')
    height : float, default=2.5
        Height of each subplot in inches
    aspect : float, default=1
        Aspect ratio of each subplot
    """
    if vars is None:
        # Use all numeric columns, but limit to 6 to avoid overcrowding
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if numeric_df.shape[1] > 6:
            print(f"Limiting pairplot to first 6 of {numeric_df.shape[1]} numeric columns.")
            vars = numeric_df.columns[:6]
        else:
            vars = numeric_df.columns
    
    # Create pairplot
    g = sns.pairplot(df, hue=hue, vars=vars, diag_kind=diag_kind, 
                    height=height, aspect=aspect)
    g.fig.suptitle(f'Pairplot of Variables', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df, numeric_cols=None, figsize=(14, 10)):
    """
    Create box plots for numeric variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to visualize
    numeric_cols : list, default=None
        List of numeric columns to plot (default: all numeric columns)
    figsize : tuple, default=(14, 10)
        Figure size
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found.")
        return
    
    # Calculate optimal subplot layout
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    plt.figure(figsize=figsize)
    
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(y=df[col])
        plt.title(col)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df, title='Feature Importance', figsize=(12, 8)):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pandas DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    title : str, default='Feature Importance'
        Plot title
    figsize : tuple, default=(12, 8)
        Figure size
    """
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    
    # Add values to bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        plt.text(width + 0.01, p.get_y() + p.get_height()/2.,
                f"{width:.4f}",
                ha="left", va="center")
    
    plt.tight_layout()
    plt.show()


def plot_dimensionality_reduction(df, features=None, target=None, method='pca', n_components=2, 
                                  figsize=(10, 8), cmap='viridis'):
    """
    Perform dimensionality reduction and visualize the results.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the features
    features : list, default=None
        List of feature columns to use (default: all numeric columns)
    target : str, default=None
        Target column for coloring points
    method : str, default='pca'
        Dimensionality reduction method ('pca', 'tsne')
    n_components : int, default=2
        Number of components (2 or 3)
    figsize : tuple, default=(10, 8)
        Figure size
    cmap : str, default='viridis'
        Colormap for the plot
    """
    if features is None:
        # Use all numeric columns
        features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclude target if it's in the numeric columns
        if target in features:
            features.remove(target)
    
    if len(features) < 2:
        print("Not enough features for dimensionality reduction.")
        return
    
    # Prepare the data
    X = df[features].values
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        model = PCA(n_components=n_components)
        transformed = model.fit_transform(X)
        explained_var = model.explained_variance_ratio_
        method_name = 'PCA'
    elif method.lower() == 'tsne':
        model = TSNE(n_components=n_components, random_state=42)
        transformed = model.fit_transform(X)
        explained_var = None
        method_name = 't-SNE'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
    
    # Create plot
    if n_components == 2:
        plt.figure(figsize=figsize)
        
        if target is not None:
            scatter = plt.scatter(transformed[:, 0], transformed[:, 1], 
                                 c=df[target] if pd.api.types.is_numeric_dtype(df[target]) else pd.factorize(df[target])[0], 
                                 cmap=cmap, alpha=0.7, s=50)
            
            if pd.api.types.is_numeric_dtype(df[target]):
                plt.colorbar(label=target)
            else:
                categories = df[target].unique()
                handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                     markersize=8) for i in range(len(categories))]
                plt.legend(handles, categories, title=target)
        else:
            plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.7, s=50)
        
        plt.title(f'{method_name} Visualization')
        plt.xlabel(f'Component 1 {f"({explained_var[0]:.2%} variance)" if explained_var is not None else ""}')
        plt.ylabel(f'Component 2 {f"({explained_var[1]:.2%} variance)" if explained_var is not None else ""}')
        plt.grid(True, alpha=0.3)
        
    elif n_components == 3:
        # 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if target is not None:
            scatter = ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2],
                                c=df[target] if pd.api.types.is_numeric_dtype(df[target]) else pd.factorize(df[target])[0],
                                cmap=cmap, alpha=0.7, s=50)
            
            if pd.api.types.is_numeric_dtype(df[target]):
                fig.colorbar(scatter, ax=ax, label=target)
            else:
                categories = df[target].unique()
                handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                     markersize=8) for i in range(len(categories))]
                ax.legend(handles, categories, title=target)
        else:
            ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], alpha=0.7, s=50)
        
        ax.set_title(f'{method_name} Visualization (3D)')
        ax.set_xlabel(f'Component 1 {f"({explained_var[0]:.2%} variance)" if explained_var is not None else ""}')
        ax.set_ylabel(f'Component 2 {f"({explained_var[1]:.2%} variance)" if explained_var is not None else ""}')
        ax.set_zlabel(f'Component 3 {f"({explained_var[2]:.2%} variance)" if explained_var is not None else ""}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print explained variance for PCA
    if method.lower() == 'pca' and explained_var is not None:
        print(f"Explained variance by component:")
        for i, var in enumerate(explained_var):
            print(f"Component {i+1}: {var:.4f} ({var:.2%})")
        print(f"Total explained variance: {sum(explained_var):.2%}")
        
    return transformed 