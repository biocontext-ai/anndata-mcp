"""Utility functions for handling AnnData objects and their components."""

from typing import Any

import numpy as np
import pandas as pd


def safe_to_list(data: Any, max_items: int = 100) -> list:
    """Safely convert data to a list with size limit.

    Parameters
    ----------
    data : Any
        Data to convert to list
    max_items : int
        Maximum number of items to return

    Returns
    -------
    list
        Converted data as list
    """
    if hasattr(data, "tolist"):
        data = data.tolist()
    elif hasattr(data, "to_list"):
        data = data.to_list()
    elif isinstance(data, (list, tuple)):
        data = list(data)
    else:
        data = [data]

    if len(data) > max_items:
        return data[:max_items]
    return data


def convert_to_serializable(obj: Any) -> Any:
    """Convert various data types to JSON-serializable format.

    Parameters
    ----------
    obj : Any
        Object to convert

    Returns
    -------
    Any
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (str, int, float, bool)):
        # Basic types pass through
        return obj
    else:
        # Handle dask arrays and xarray DataArrays
        if hasattr(obj, "compute"):
            # Dask array - compute it first
            try:
                obj = obj.compute()
                return convert_to_serializable(obj)
            except Exception:
                pass

        if hasattr(obj, "values"):
            # xarray DataArray or similar - get the underlying values
            try:
                obj = obj.values
                return convert_to_serializable(obj)
            except Exception:
                pass

        # Check for NaN/None - this must come after container checks
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            # pd.isna() can fail on some objects, just pass through
            pass
        return obj


def get_dtype_string(dtype: Any) -> str:
    """Get a string representation of a dtype.

    Parameters
    ----------
    dtype : Any
        Dtype to convert to string

    Returns
    -------
    str
        String representation of dtype
    """
    if hasattr(dtype, "name"):
        return dtype.name
    return str(dtype)


def handle_dataset2d(data: Any) -> pd.DataFrame:
    """Convert Dataset2D or xarray.Dataset to pandas DataFrame.

    Parameters
    ----------
    data : Any
        Dataset2D or xarray.Dataset object

    Returns
    -------
    pd.DataFrame
        Converted dataframe
    """
    # If it's already a DataFrame, return as is
    if isinstance(data, pd.DataFrame):
        return data

    # Handle Dataset2D - access the underlying xarray Dataset
    if hasattr(data, "_ds"):
        # Dataset2D stores the xarray Dataset in _ds
        xr_ds = data._ds
        # Convert each data_var to a pandas Series and combine into DataFrame
        data_dict = {}
        for var_name in xr_ds.data_vars:
            data_dict[var_name] = xr_ds[var_name].to_pandas()
        return pd.DataFrame(data_dict)
    elif hasattr(data, "ds"):
        # Try ds attribute as well
        xr_ds = data.ds
        data_dict = {}
        for var_name in xr_ds.data_vars:
            data_dict[var_name] = xr_ds[var_name].to_pandas()
        return pd.DataFrame(data_dict)

    # If it's an xarray Dataset directly
    if hasattr(data, "data_vars"):
        data_dict = {}
        for var_name in data.data_vars:
            data_dict[var_name] = data[var_name].to_pandas()
        return pd.DataFrame(data_dict)

    # Try other conversion methods
    if hasattr(data, "to_dataframe"):
        return data.to_dataframe()
    elif hasattr(data, "to_pandas"):
        return data.to_pandas()
    else:
        raise ValueError(f"Cannot convert {type(data)} to DataFrame")


def get_sparse_info(sparse_matrix: Any) -> dict:
    """Get information about a sparse matrix.

    Parameters
    ----------
    sparse_matrix : Any
        Sparse matrix

    Returns
    -------
    dict
        Information about the sparse matrix
    """
    from scipy import sparse

    info = {
        "format": sparse_matrix.format if hasattr(sparse_matrix, "format") else "unknown",
        "shape": sparse_matrix.shape,
        "nnz": sparse_matrix.nnz if hasattr(sparse_matrix, "nnz") else "unknown",
        "dtype": get_dtype_string(sparse_matrix.dtype),
    }

    # Calculate sparsity if possible
    if hasattr(sparse_matrix, "nnz") and hasattr(sparse_matrix, "shape"):
        total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
        if total_elements > 0:
            info["sparsity"] = 1.0 - (sparse_matrix.nnz / total_elements)

    return info


def slice_data(data: Any, row_slice: slice | None = None, col_slice: slice | None = None) -> Any:
    """Slice data appropriately based on its type.

    Parameters
    ----------
    data : Any
        Data to slice
    row_slice : slice, optional
        Row slice
    col_slice : slice, optional
        Column slice

    Returns
    -------
    Any
        Sliced data
    """
    if row_slice is None and col_slice is None:
        return data

    # Handle 2D arrays and matrices
    if hasattr(data, "shape") and len(data.shape) == 2:
        if row_slice is not None and col_slice is not None:
            return data[row_slice, col_slice]
        elif row_slice is not None:
            return data[row_slice, :]
        elif col_slice is not None:
            return data[:, col_slice]

    # Handle 1D arrays
    elif hasattr(data, "shape") and len(data.shape) == 1:
        if row_slice is not None:
            return data[row_slice]

    # Handle DataFrames
    elif isinstance(data, pd.DataFrame):
        if row_slice is not None and col_slice is not None:
            return data.iloc[row_slice, col_slice]
        elif row_slice is not None:
            return data.iloc[row_slice]
        elif col_slice is not None:
            return data.iloc[:, col_slice]

    return data
