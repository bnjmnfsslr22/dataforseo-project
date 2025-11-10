"""Caching utilities for the DataForSEO project."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Optional
import pandas as pd


def cache_to_file(
    cache_dir: str = "data/02_intermediate/cache",
    file_extension: str = ".pkl"
) -> Callable:
    """
    Decorator to cache function results to files.
    
    Args:
        cache_dir: Directory to store cache files
        file_extension: File extension for cache files
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache directory
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Generate cache key from function name and arguments
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            cache_file = cache_path / f"{cache_key}{file_extension}"
            
            # Try to load from cache
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    # If loading fails, continue with function execution
                    pass
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception:
                # If caching fails, continue without caching
                pass
            
            return result
        
        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from function name and arguments."""
    # Create a hash of the function name and arguments
    key_data = {
        'func_name': func_name,
        'args': args,
        'kwargs': kwargs
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def clear_cache(cache_dir: str = "data/02_intermediate/cache") -> None:
    """Clear all cache files."""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for cache_file in cache_path.glob("*"):
            cache_file.unlink()


def cache_dataframe(
    df: pd.DataFrame,
    cache_file: str,
    cache_dir: str = "data/02_intermediate/cache"
) -> None:
    """
    Cache a DataFrame to a file.
    
    Args:
        df: DataFrame to cache
        cache_file: Cache file name
        cache_dir: Cache directory
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    file_path = cache_path / cache_file
    df.to_parquet(file_path)


def load_cached_dataframe(
    cache_file: str,
    cache_dir: str = "data/02_intermediate/cache"
) -> Optional[pd.DataFrame]:
    """
    Load a cached DataFrame from a file.
    
    Args:
        cache_file: Cache file name
        cache_dir: Cache directory
        
    Returns:
        Cached DataFrame or None if not found
    """
    cache_path = Path(cache_dir)
    file_path = cache_path / cache_file
    
    if file_path.exists():
        try:
            return pd.read_parquet(file_path)
        except Exception:
            return None
    
    return None
