"""Data quality utilities for the DataForSEO project."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1
) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame for basic data quality.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if DataFrame is empty
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    # Check minimum rows
    if len(df) < min_rows:
        errors.append(f"DataFrame has {len(df)} rows, minimum required is {min_rows}")
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check for completely null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        errors.append(f"Columns with all null values: {null_columns}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def check_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, List[int]]:
    """
    Detect outliers in specified columns.
    
    Args:
        df: DataFrame to check
        columns: List of column names to check
        method: Method for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary mapping column names to lists of outlier indices
    """
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask = z_scores > threshold
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            continue
        
        outliers[col] = df[outlier_mask].index.tolist()
    
    return outliers


def check_data_consistency(
    df: pd.DataFrame,
    checks: Dict[str, callable]
) -> Dict[str, List[str]]:
    """
    Run custom data consistency checks.
    
    Args:
        df: DataFrame to check
        checks: Dictionary mapping check names to check functions
        
    Returns:
        Dictionary mapping check names to lists of error messages
    """
    results = {}
    
    for check_name, check_func in checks.items():
        try:
            errors = check_func(df)
            results[check_name] = errors if errors else []
        except Exception as e:
            results[check_name] = [f"Check failed with error: {str(e)}"]
    
    return results


def generate_data_quality_report(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    numeric_columns: List[str] = None
) -> Dict:
    """
    Generate a comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
        required_columns: List of required columns
        numeric_columns: List of numeric columns for outlier detection
        
    Returns:
        Dictionary containing data quality metrics
    """
    report = {
        "basic_info": {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "dtypes": df.dtypes.to_dict()
        },
        "missing_data": {
            "total_missing": df.isnull().sum().sum(),
            "missing_per_column": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
        },
        "duplicates": {
            "total_duplicates": df.duplicated().sum(),
            "duplicate_percentage": (df.duplicated().sum() / len(df) * 100)
        }
    }
    
    # Validation results
    if required_columns:
        is_valid, errors = validate_dataframe(df, required_columns)
        report["validation"] = {
            "is_valid": is_valid,
            "errors": errors
        }
    
    # Outlier detection
    if numeric_columns:
        outliers = check_outliers(df, numeric_columns)
        report["outliers"] = outliers
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report["statistics"] = df[numeric_cols].describe().to_dict()
    
    return report


def clean_dataframe(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    handle_missing: str = "drop",  # "drop", "fill", "keep"
    fill_value: Any = None
) -> pd.DataFrame:
    """
    Clean a DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: DataFrame to clean
        remove_duplicates: Whether to remove duplicate rows
        handle_missing: How to handle missing values
        fill_value: Value to use for filling missing values
        
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    if handle_missing == "drop":
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        removed_rows = initial_rows - len(cleaned_df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with missing values")
    elif handle_missing == "fill" and fill_value is not None:
        cleaned_df = cleaned_df.fillna(fill_value)
        logger.info(f"Filled missing values with {fill_value}")
    
    return cleaned_df
