from typing import Any

import dask
import numpy as np
import pandas as pd
from anndata._core.xarray import Dataset2D

MAX_STRING_LENGTH = 1000


def truncate_string(string: str) -> str:
    """Truncate a string to the maximum length."""
    if len(string) > MAX_STRING_LENGTH:
        return string[:MAX_STRING_LENGTH] + "..."
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
