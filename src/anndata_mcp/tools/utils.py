import fnmatch
import os
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import zarr
from anndata._core.xarray import Dataset2D
from anndata.experimental import read_lazy


class AccessTrackingStore(zarr.storage.FsspecStore):
    """A store that tracks the keys that have been accessed."""

    _keys_hit = set()

    async def get(self, key, *args, **kwargs):
        """Get a key from the store."""
        try:
            res = await super().get(key, *args, **kwargs)
            if key not in self._keys_hit and res is not None:
                self._keys_hit.add(key)
            return res
        except (KeyError, OSError):
            # Key doesn't exist or filesystem error - return None
            return None


def _is_url(path: str | Path) -> bool:
    """Check if a string is a URL or a file system path.

    Parameters
    ----------
    path : str | Path
        The path or URL to check

    Returns
    -------
    bool
        True if the string appears to be a URL, False otherwise
    """
    # Convert Path objects to strings
    path_str = str(path)
    # Check for common URL schemes
    url_schemes = ("http://", "https://", "s3://", "gs://", "gcs://", "abfs://", "az://")
    return any(path_str.startswith(scheme) for scheme in url_schemes)


def read_lazy_general(path_or_url: str | Path):
    """Read an AnnData object lazily from either a file path or URL.

    This function automatically detects whether the input is a URL or a file system path
    and handles it appropriately. For URLs, it uses AccessTrackingStore.from_url() to
    create a zarr store, then reads it lazily. For file paths, it uses read_lazy directly.

    Parameters
    ----------
    path_or_url : str | Path
        Either a file system path (e.g., "data/test.h5ad" or "data/test.zarr") or
        a URL (e.g., "https://example.com/data.zarr/")

    Returns
    -------
    AnnData
        A lazily-loaded AnnData object
    """
    # Convert Path objects to strings
    path_str = str(path_or_url)
    if _is_url(path_str):
        # For URLs, use AccessTrackingStore.from_url() then read_lazy
        store = AccessTrackingStore.from_url(path_str, read_only=True)
        return read_lazy(store)
    else:
        # For file paths, use read_lazy directly
        return read_lazy(path_str)


def truncate_string(string: str, max_output_len: int | None = None) -> str:
    """Truncate a string to the maximum length."""
    max_output_len = max_output_len or int(os.getenv("MCP_MAX_OUTPUT_LEN", "1000"))
    if len(string) > max_output_len:
        return string[:max_output_len] + "..."
    return string


def get_shape_str(obj: Any) -> str:
    """Get the shape of an object as a string."""
    try:
        return str(obj.shape)
    except AttributeError:
        return "NA"


def class_string_to_type(class_string: str) -> str:
    """Convert a class string to a type."""
    return class_string.split("'")[1]


def raw_type_to_string(raw_type: type, full_name: bool = False) -> str:
    """Convert a raw type to a string."""
    if full_name:
        return class_string_to_type(str(raw_type))
    else:
        return raw_type.__name__


def extract_original_type(obj: Any) -> type:
    """Extract the original type of an object."""
    if isinstance(obj, dask.array.core.Array):
        return type(obj._meta)
    elif isinstance(obj, Dataset2D):
        return pd.DataFrame
    else:
        return type(obj)


def extract_original_type_string(obj: Any, full_name: bool = False) -> str:
    """Extract the original type of an object and convert it to a string."""
    return raw_type_to_string(extract_original_type(obj), full_name=full_name)


def parse_slice(slice_str: str | None) -> slice:
    """Parse a slice string like '0:10' or ':100' into a slice object.

    Parameters
    ----------
    slice_str : str, optional
        Slice string

    Returns
    -------
    slice
        Parsed slice object
    """
    if slice_str is None:
        return slice(None)

    if ":" not in slice_str:
        raise ValueError("Slice string must contain ':'")

    parts = slice_str.split(":")
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None

    return slice(start, stop, step)


def extract_slice_from_dask_array(array: dask.array.core.Array, row_slice: slice, col_slice: slice) -> np.ndarray:
    """Extract a slice from a dask array."""
    return array[row_slice, col_slice].compute()


def extract_indices_from_dask_array(
    array: dask.array.core.Array, row_slice: slice, col_indices: list[int]
) -> np.ndarray:
    """Extract data from a dask array using column indices."""
    return array[row_slice, col_indices].compute()


def array_to_csv(array: np.ndarray) -> str:
    """Convert a numpy array to a CSV string."""
    return truncate_string("\n".join(pd.DataFrame(array).to_csv(index=False).split("\n")[1::]))


def extract_data_from_dask_array(
    array: dask.array.core.Array, row_slice: slice, col_slice: slice, return_shape: bool = False
) -> tuple[str, str] | str:
    """Extract data from a dask array."""
    data = extract_slice_from_dask_array(array, row_slice, col_slice)
    if return_shape:
        return truncate_string(array_to_csv(data)), str(data.shape)
    else:
        return truncate_string(array_to_csv(data))


def extract_data_from_dask_array_with_indices(
    array: dask.array.core.Array, row_slice: slice, col_indices: list[int], return_shape: bool = False
) -> tuple[str, str] | str:
    """Extract data from a dask array using column indices."""
    data = extract_indices_from_dask_array(array, row_slice, col_indices)
    if return_shape:
        return truncate_string(array_to_csv(data)), str(data.shape)
    else:
        return truncate_string(array_to_csv(data))


def extract_data_from_dataset2d(
    dataset2d: Dataset2D,
    columns: list[str],
    row_slice: slice | None = None,
    index: bool = True,
    return_shape: bool = False,
) -> tuple[str, str] | str:
    """Extract data from a dataset2d."""
    if row_slice is not None:
        data = dataset2d.iloc[row_slice][columns].to_memory()
    else:
        data = dataset2d[columns].to_memory()
    if return_shape:
        return truncate_string(data.to_csv(index=index)), str(data.shape)
    else:
        return truncate_string(data.to_csv(index=index))


def select_by_glob(items: list[str] | pd.Index, pattern: str):
    """Select items from a list or index matching a glob pattern."""
    return fnmatch.filter(items, pattern)


def match_patterns(items: list[str] | pd.Index, pattern_list: list[str]) -> tuple[list[str], str | None]:
    """Match items to patterns and return the matched items and a message listing any patterns that were not found."""
    result = []
    errors = []
    for pattern in pattern_list:
        selected = select_by_glob(items, pattern)
        if len(selected) == 0:
            errors.append(pattern)
            continue
        result.extend(selected)
    # Remove duplicates while preserving order
    result = list(dict.fromkeys(result))
    return result, f"No matches found for: {', '.join(errors)}" if len(errors) > 0 else None


def get_nested_key(obj: Any, keys: list[str]) -> Any:
    """Retrieve a nested value from an object using a list of keys.

    This function traverses through nested structures (dicts, objects with attributes, etc.)
    using the provided list of keys. It uses `get()` for dict-like objects and `hasattr()`/`getattr()`
    for objects with attributes where possible.

    Parameters
    ----------
    obj : Any
        The object to traverse
    keys : list[str]
        List of keys to traverse through the nested structure

    Returns
    -------
    Any
        The value at the nested key path

    Raises
    ------
    KeyError
        If any key in the path is not found
    AttributeError
        If any attribute in the path is not found
    """
    current = obj
    path_traversed = []

    for key in keys:
        path_traversed.append(key)

        # Try dict-like access first (supports dict, Mapping, etc.)
        if hasattr(current, "get"):
            get_method = current.get
            if callable(get_method):
                if key in current:
                    current = current[key]
                    continue

        # Try attribute access
        if hasattr(current, key):
            current = getattr(current, key)
            continue

        # Try direct indexing (for list-like or other indexable objects)
        try:
            current = current[key]
            continue
        except (KeyError, TypeError, IndexError):
            pass

        # If we get here, the key was not found
        path_str = " -> ".join(path_traversed)
        raise KeyError(f"Key path '{path_str}' not found in object")

    return current
