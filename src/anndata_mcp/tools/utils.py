from typing import Any

import anndata
import dask
import numpy as np
import pandas as pd

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
    elif isinstance(obj, anndata._core.xarray.Dataset2D):
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
    dataset2d: anndata._core.xarray.Dataset2D,
    row_slice: slice,
    columns: list[str],
    index: bool = True,
    return_shape: bool = False,
) -> tuple[str, str] | str:
    """Extract data from a dataset2d."""
    data = dataset2d.iloc[row_slice][columns].to_memory()
    if return_shape:
        return truncate_string(data.to_csv(index=index)), str(data.shape)
    else:
        return truncate_string(data.to_csv(index=index))


def describe_dataset2d(dataset2d: anndata._core.xarray.Dataset2D) -> pd.DataFrame:
    """Generate descriptive statistics for a Dataset2D object.

    This function provides a statistical summary similar to pandas DataFrame.describe(),
    including count, mean, std, min, quartiles, and max for numeric columns,
    and count, unique, top, and freq for object/categorical columns.

    Memory-efficient implementation that processes columns one at a time without
    loading the entire dataset into memory.

    Parameters
    ----------
    dataset2d : anndata._core.xarray.Dataset2D
        The Dataset2D object to describe

    Returns
    -------
    pd.DataFrame
        A DataFrame containing descriptive statistics for each column
    """
    columns = dataset2d.columns.tolist()
    stats_dict = {}
    is_numeric_list = []

    for col in columns:
        # Access column as DataArray - try to compute stats without loading into memory
        col_array = dataset2d[col]

        # Check if column is numeric by checking dtype (without loading data)
        is_numeric = pd.api.types.is_numeric_dtype(col_array.dtype) if hasattr(col_array, "dtype") else None
        is_numeric_list.append(is_numeric)

        if is_numeric:
            # For numeric columns, compute statistics directly on DataArray without loading
            # These operations are lazy and compute efficiently (e.g., with Dask chunks)
            # Compute statistics using DataArray methods - these may use lazy computation
            count_result = col_array.count()
            mean_result = col_array.mean()
            std_result = col_array.std()
            min_result = col_array.min()
            max_result = col_array.max()
            quantile_25 = col_array.quantile(0.25)
            quantile_50 = col_array.quantile(0.50)
            quantile_75 = col_array.quantile(0.75)

            # Extract scalar values - compute() triggers actual computation
            # but only aggregates, not loading full column
            def _extract_scalar(result):
                """Extract scalar value from computed result."""
                computed = result.compute()
                return computed.values if hasattr(computed, "values") else computed

            count_val = int(_extract_scalar(count_result))
            mean_val = float(_extract_scalar(mean_result))
            std_val = float(_extract_scalar(std_result))
            min_val = float(_extract_scalar(min_result))
            max_val = float(_extract_scalar(max_result))
            q25_val = float(_extract_scalar(quantile_25))
            q50_val = float(_extract_scalar(quantile_50))
            q75_val = float(_extract_scalar(quantile_75))

            if count_val > 0:
                stats_dict[col] = {
                    "count": count_val,
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "25%": q25_val,
                    "50%": q50_val,
                    "75%": q75_val,
                    "max": max_val,
                }
            else:
                # All null values
                stats_dict[col] = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "25%": np.nan,
                    "50%": np.nan,
                    "75%": np.nan,
                    "max": np.nan,
                }
        else:
            # For object/categorical columns, still need to load data for value_counts
            col_df = dataset2d[[col]].to_memory()
            col_data = col_df[col]
            non_null = col_data.dropna()
            count = len(non_null)

            if count > 0:
                value_counts = non_null.value_counts()
                unique_count = len(value_counts)
                top_value = value_counts.index[0] if len(value_counts) > 0 else None
                freq = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0

                stats_dict[col] = {
                    "count": count,
                    "unique": unique_count,
                    "top": top_value,
                    "freq": freq,
                }
            else:
                # All null values
                stats_dict[col] = {
                    "count": 0,
                    "unique": 0,
                    "top": None,
                    "freq": 0,
                }

    # Convert to DataFrame and reorder columns to match pandas describe() output order
    numeric_stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    object_stats = ["count", "unique", "top", "freq"]

    # Reorder stats for each column based on type
    final_stats = {}
    for col, stats in stats_dict.items():
        if "mean" in stats:
            # Numeric column
            final_stats[col] = {stat: stats[stat] for stat in numeric_stats if stat in stats}
        elif "unique" in stats:
            # Object column
            final_stats[col] = {stat: stats[stat] for stat in object_stats if stat in stats}

    result_df = pd.DataFrame(final_stats).T
    result_df.index.name = "column"

    # Ensure appropriate dtypes
    # Convert numeric statistics to float (compatible with NaN)
    numeric_stat_cols = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    for col in numeric_stat_cols:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    # Convert count and freq to int
    if "count" in result_df.columns:
        result_df["count"] = pd.to_numeric(result_df["count"], errors="coerce").fillna(0).astype(int)
    result_df["is_numeric"] = is_numeric_list
    return result_df
