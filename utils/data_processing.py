"""
Utility functions for data processing and feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def reduce_memory_usage(df, verbose=True):
    """
    Reduce memory usage of a dataframe by converting numeric columns to more memory-efficient types.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to optimize
    verbose : bool, default=True
        Whether to print memory reduction information
        
    Returns:
    --------
    pandas DataFrame
        Memory-optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)')
    
    return df


def handle_missing_values(df, numeric_strategy='median', categorical_strategy='most_frequent'):
    """
    Handle missing values in a dataframe.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with missing values
    numeric_strategy : str, default='median'
        Strategy for imputing numeric columns ('mean', 'median', 'most_frequent', 'constant')
    categorical_strategy : str, default='most_frequent'
        Strategy for imputing categorical columns ('most_frequent', 'constant')
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with imputed values
    """
    # Create copies of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
    
    # Impute numeric columns
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy=numeric_strategy)
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
    
    # Impute categorical columns
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy=categorical_strategy)
        df_copy[categorical_cols] = imputer.fit_transform(df_copy[categorical_cols])
    
    return df_copy


def encode_categorical_features(df, columns=None, method='onehot', drop_first=True):
    """
    Encode categorical features in a dataframe.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with categorical columns
    columns : list or None, default=None
        List of categorical columns to encode. If None, all object columns are encoded.
    method : str, default='onehot'
        Encoding method ('onehot', 'label')
    drop_first : bool, default=True
        Whether to drop the first category in one-hot encoding (to avoid multicollinearity)
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with encoded categorical features
    """
    df_copy = df.copy()
    
    # If columns not specified, use all object columns
    if columns is None:
        columns = df_copy.select_dtypes(include=['object', 'category']).columns
    
    # Apply encoding based on the specified method
    if method == 'onehot':
        df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=drop_first)
    elif method == 'label':
        for col in columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    
    return df_copy


def detect_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Detect outliers in specified columns of a dataframe.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to check for outliers
    columns : list or None, default=None
        List of columns to check. If None, all numeric columns are checked.
    method : str, default='iqr'
        Method to detect outliers ('iqr', 'zscore')
    threshold : float, default=1.5
        Threshold for outlier detection (1.5 for IQR, 3 for z-score)
        
    Returns:
    --------
    pandas DataFrame
        Boolean DataFrame indicating outliers
    """
    df_copy = df.copy()
    
    # If columns not specified, use all numeric columns
    if columns is None:
        columns = df_copy.select_dtypes(include=['int64', 'float64']).columns
    
    # Initialize outlier DataFrame
    outliers = pd.DataFrame(False, index=df_copy.index, columns=columns)
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
        elif method == 'zscore':
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            z_scores = np.abs((df_copy[col] - mean) / std)
            outliers[col] = z_scores > threshold
    
    return outliers


def create_date_features(df, date_column):
    """
    Create date-related features from a date column.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the date column
    date_column : str
        Name of the date column
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with new date features
    """
    df_copy = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    # Extract date components
    df_copy[f'{date_column}_year'] = df_copy[date_column].dt.year
    df_copy[f'{date_column}_month'] = df_copy[date_column].dt.month
    df_copy[f'{date_column}_day'] = df_copy[date_column].dt.day
    df_copy[f'{date_column}_dayofweek'] = df_copy[date_column].dt.dayofweek
    df_copy[f'{date_column}_dayofyear'] = df_copy[date_column].dt.dayofyear
    df_copy[f'{date_column}_quarter'] = df_copy[date_column].dt.quarter
    df_copy[f'{date_column}_is_month_start'] = df_copy[date_column].dt.is_month_start.astype(int)
    df_copy[f'{date_column}_is_month_end'] = df_copy[date_column].dt.is_month_end.astype(int)
    df_copy[f'{date_column}_is_weekend'] = (df_copy[date_column].dt.dayofweek >= 5).astype(int)
    
    return df_copy 