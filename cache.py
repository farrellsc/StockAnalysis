# TODO: consider improve cache to be database-based, not func param based

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, Callable
import os
import pickle
import hashlib
from functools import wraps
import tempfile
import time


def cache(cache_dir: str = None):
    """
    Simple cache decorator for methods. No size or TTL limits - just pure caching.

    Args:
        cache_dir (str): Directory to store cache files. Defaults to temp directory.

    Returns:
        Callable: Decorator function

    Example:
        @cache(cache_dir='./my_cache')
        def expensive_function(param1, param2):
            return expensive_result
    """

    def decorator(func: Callable) -> Callable:
        # Set up cache directory
        if cache_dir is None:
            func_cache_dir = os.path.join(tempfile.gettempdir(), f"cache_{func.__name__}")
        else:
            func_cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(func_cache_dir, exist_ok=True)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            cache_file = os.path.join(func_cache_dir, f"{cache_key}.pkl")

            # Try to load from cache
            cached_result = _load_from_cache(cache_file)
            if cached_result is not None:
                print(f"📂 Cache HIT for {func.__name__}({_format_args(args, kwargs)})")
                return cached_result

            print(f"📡 Cache MISS for {func.__name__}({_format_args(args, kwargs)})")

            # Call the original function
            result = func(*args, **kwargs)

            # Save result to cache
            _save_to_cache(cache_file, result)

            return result

        # Attach cache directory and management methods to the wrapper
        wrapper.cache_dir = func_cache_dir
        wrapper.cache_clear = lambda: _clear_cache_dir(func_cache_dir)
        wrapper.cache_info = lambda: _get_cache_info(func_cache_dir)

        return wrapper

    return decorator


def clean_cache(cache_dir: str, max_age_hours: int = None, max_size_mb: int = None, max_files: int = None) -> Dict[
    str, int]:
    """
    Clean cache directory based on specified criteria.

    Args:
        cache_dir (str): Cache directory to clean
        max_age_hours (int, optional): Remove files older than this many hours
        max_size_mb (int, optional): Keep total cache size under this limit (MB)
        max_files (int, optional): Keep only this many most recent files

    Returns:
        Dict[str, int]: Statistics about cleanup operation

    Example:
        # Remove files older than 24 hours
        clean_cache('./my_cache', max_age_hours=24)

        # Keep cache under 100MB
        clean_cache('./my_cache', max_size_mb=100)

        # Keep only 50 most recent files
        clean_cache('./my_cache', max_files=50)

        # Combine criteria
        clean_cache('./my_cache', max_age_hours=12, max_size_mb=50, max_files=100)
    """
    if not os.path.exists(cache_dir):
        return {"removed_files": 0, "removed_size_mb": 0, "remaining_files": 0}

    print(f"🧹 Cleaning cache directory: {cache_dir}")

    # Get all cache files with metadata
    files_info = []
    current_time = time.time()

    try:
        for filename in os.listdir(cache_dir):
            if not filename.endswith('.pkl'):
                continue

            file_path = os.path.join(cache_dir, filename)
            try:
                file_stat = os.stat(file_path)
                files_info.append({
                    'path': file_path,
                    'size': file_stat.st_size,
                    'mtime': file_stat.st_mtime,
                    'age_hours': (current_time - file_stat.st_mtime) / 3600
                })
            except OSError:
                # Handle corrupted or inaccessible files
                files_info.append({
                    'path': file_path,
                    'size': 0,
                    'mtime': 0,
                    'age_hours': float('inf')
                })
    except OSError:
        print(f"⚠️  Cannot access cache directory: {cache_dir}")
        return {"removed_files": 0, "removed_size_mb": 0, "remaining_files": 0}

    files_to_remove = set()

    # 1. Age-based cleanup
    if max_age_hours is not None:
        for file_info in files_info:
            if file_info['age_hours'] > max_age_hours:
                files_to_remove.add(file_info['path'])

        if files_to_remove:
            print(f"⏰ Found {len(files_to_remove)} files older than {max_age_hours} hours")

    # 2. Size-based cleanup (after age cleanup)
    if max_size_mb is not None:
        max_size_bytes = max_size_mb * 1024 * 1024

        # Calculate remaining files after age cleanup
        remaining_files = [f for f in files_info if f['path'] not in files_to_remove]
        current_size = sum(f['size'] for f in remaining_files)

        if current_size > max_size_bytes:
            # Sort by modification time (oldest first) and remove until under limit
            remaining_files.sort(key=lambda x: x['mtime'])

            for file_info in remaining_files:
                if current_size <= max_size_bytes:
                    break
                files_to_remove.add(file_info['path'])
                current_size -= file_info['size']

            print(f"📦 Size-based cleanup to stay under {max_size_mb} MB")

    # 3. File count-based cleanup (after age and size cleanup)
    if max_files is not None:
        remaining_files = [f for f in files_info if f['path'] not in files_to_remove]

        if len(remaining_files) > max_files:
            # Sort by modification time (oldest first) and keep only the most recent
            remaining_files.sort(key=lambda x: x['mtime'], reverse=True)
            files_to_keep = remaining_files[:max_files]
            files_to_remove_for_count = remaining_files[max_files:]

            for file_info in files_to_remove_for_count:
                files_to_remove.add(file_info['path'])

            print(f"📁 Keeping only {max_files} most recent files")

    # Perform the actual cleanup
    removed_count = 0
    removed_size = 0

    for file_path in files_to_remove:
        try:
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            removed_count += 1
            removed_size += file_size
        except OSError as e:
            print(f"⚠️  Failed to remove {os.path.basename(file_path)}: {e}")

    remaining_count = len(files_info) - removed_count
    removed_size_mb = removed_size / (1024 * 1024)

    if removed_count > 0:
        print(f"✓ Removed {removed_count} files ({removed_size_mb:.2f} MB), {remaining_count} files remaining")
    else:
        print("✓ No files needed to be removed")

    return {
        "removed_files": removed_count,
        "removed_size_mb": removed_size_mb,
        "remaining_files": remaining_count
    }


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a unique cache key from function name and arguments."""
    args_str = str(args) + str(sorted(kwargs.items()))
    cache_string = f"{func_name}_{args_str}"
    return cache_string


def _format_args(args: tuple, kwargs: dict) -> str:
    """Format arguments for display purposes."""
    arg_strs = [str(arg)[:20] + "..." if len(str(arg)) > 20 else str(arg) for arg in args]
    kwarg_strs = [f"{k}={str(v)[:20] + '...' if len(str(v)) > 20 else str(v)}" for k, v in kwargs.items()]
    all_args = arg_strs + kwarg_strs
    return ", ".join(all_args[:3]) + ("..." if len(all_args) > 3 else "")


def _load_from_cache(cache_file: str) -> Any:
    """Load data from cache file if it exists."""
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️  Cache file corrupted: {os.path.basename(cache_file)} ({e})")
        return None


def _save_to_cache(cache_file: str, data: Any) -> None:
    """Save data to cache file."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Cached: {os.path.basename(cache_file)}")
    except Exception as e:
        print(f"⚠️  Failed to save cache: {e}")


def _get_cache_info(cache_dir: str) -> Dict[str, Any]:
    """Get information about cache directory."""
    if not os.path.exists(cache_dir):
        return {"files": 0, "size_mb": 0, "oldest_hours": 0, "newest_hours": 0}

    files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    if not files:
        return {"files": 0, "size_mb": 0, "oldest_hours": 0, "newest_hours": 0}

    total_size = 0
    file_ages = []
    current_time = time.time()

    for filename in files:
        file_path = os.path.join(cache_dir, filename)
        try:
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)
            total_size += file_size
            file_ages.append((current_time - file_mtime) / 3600)  # Convert to hours
        except OSError:
            continue

    return {
        "files": len(files),
        "size_mb": total_size / (1024 * 1024),
        "oldest_hours": max(file_ages) if file_ages else 0,
        "newest_hours": min(file_ages) if file_ages else 0
    }


def _clear_cache_dir(cache_dir: str) -> int:
    """Clear all cache files in directory."""
    if not os.path.exists(cache_dir):
        return 0

    removed_count = 0
    for filename in os.listdir(cache_dir):
        if filename.endswith('.pkl'):
            try:
                os.remove(os.path.join(cache_dir, filename))
                removed_count += 1
            except OSError:
                pass

    print(f"🗑️  Cleared {removed_count} cache files from {cache_dir}")
    return removed_count