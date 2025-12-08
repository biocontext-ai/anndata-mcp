import gc
from typing import Annotated, Literal

import dask
import numpy as np
import pandas as pd
import xarray as xr
from anndata._core.xarray import Dataset2D
from dask.array.core import Array
from pydantic import BaseModel, Field

from anndata_mcp.tools.utils import get_nested_key, match_patterns, read_lazy_general, truncate_string


class ExplorationResult(BaseModel):
    description: Annotated[str | None, Field(description="The description of the attribute value")]
    value_counts: Annotated[str | None, Field(description="The value counts for the attribute value")]
    error: Annotated[str | None, Field(description="Any error message")]


def create_dataframe_mask_from_tuple(
    df: pd.DataFrame | Dataset2D,
    filter_tuple: tuple[
        Annotated[str, Field(description="The column name to filter by")],
        Annotated[
            Literal["==", "!=", ">", ">=", "<", "<=", "isin", "notin"],
            Field(description="The operator to use for the filter"),
        ],
        Annotated[list[str | float | bool] | str | float | bool, Field(description="The value(s) to filter by")],
    ]
    | None,
) -> pd.Series:
    column_name, operator, value = filter_tuple

    if (column_name not in df.columns) and (column_name != "index"):
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    series = df[column_name] if column_name != "index" else df.index

    if operator == "==":
        return series == value
    elif operator == "!=":
        return series != value
    elif operator == ">":
        return series > float(value)
    elif operator == ">=":
        return series >= float(value)
    elif operator == "<":
        return series < float(value)
    elif operator == "<=":
        return series <= float(value)
    elif operator == "isin":
        value = [value] if not isinstance(value, list) else value
        return series.isin(value)
    elif operator == "notin":
        value = [value] if not isinstance(value, list) else value
        return ~series.isin(value)
    else:
        raise ValueError(f"Unknown operator: {operator}")


def get_descriptive_stats(
    path: Annotated[str, Field(description="Absolute path or URL to the AnnData file (.h5ad or .zarr)")],
    attribute: Annotated[
        Literal["X", "obs", "var", "obsm", "varm", "obsp", "varp", "uns", "layers"],
        Field(description="The attribute to describe"),
    ],
    key: Annotated[
        str | list[str] | None,
        Field(
            description="The key of the attribute value to explore. Can be a single string or a list of strings for nested key retrieval (e.g., ['key1', 'key2'] to access attr_obj['key1']['key2']). Should be None for attributes X, obs, and var.",
            default=None,
        ),
    ] = None,
    columns_or_genes: Annotated[
        list[str] | None,
        Field(
            description="The columns or genes to describe. For pandas.DataFrame attributes (e.g., obs, var), these are column names. For 'X' or 'layers' attributes, these are gene names (from var_names). If None, the entire dataset is considered. Also accepts glob-like patterns as input, e.g. ['RE*', 'CD4*']."
        ),
    ] = None,
    return_value_counts_for_categorical: Annotated[
        bool, Field(description="Whether to return the value counts for categorical columns.")
    ] = False,
    filter_attribute: Annotated[
        Literal["obs", "var"],
        Field(
            description="The attribute to filter by. One of 'obs' or 'var' or None for no filtering. Has to be provided TOGETHER with filter_column, filter_operator, and filter_value."
        ),
    ] = None,
    filter_column: Annotated[
        str | None,
        Field(description="The column name of the obs or var dataframe to filter by."),
    ] = None,
    filter_operator: Annotated[
        Literal["==", "!=", ">", ">=", "<", "<=", "isin", "notin"] | None,
        Field(description="The operator to use for the filter."),
    ] = None,
    filter_value: Annotated[
        list[str | float | bool] | str | float | bool | None, Field(description="The value(s) to filter by.")
    ] = None,
) -> ExplorationResult:
    """Provide basic descriptive statistics (e.g., count, mean, std, min, max, etc. or value counts) for an attribute or attribute value of an optionally filtered AnnData object."""
    error = None
    description = None
    value_counts = None
    adata = None

    try:
        adata = read_lazy_general(path)

        # Apply filter if provided
        if filter_attribute is not None:
            # Validate that all filter parameters are provided together
            if filter_column is None or filter_operator is None or filter_value is None:
                raise ValueError(
                    "If filter_attribute is set, filter_column, filter_operator, and filter_value must also be provided"
                )

            # Construct filter tuple
            filter_tuple = (filter_column, filter_operator, filter_value)

            # Get the appropriate dataframe based on filter_attribute
            if filter_attribute == "obs":
                df = adata.obs
            elif filter_attribute == "var":
                df = adata.var
            else:
                raise ValueError(f"filter_attribute must be 'obs' or 'var', got '{filter_attribute}'")

            # Create mask using the helper function
            mask = create_dataframe_mask_from_tuple(df, filter_tuple)

            # Convert mask if needed (for Dataset2D/lazy-loaded data)
            # AnnData accepts pandas Series or numpy arrays for boolean indexing
            if hasattr(mask, "compute"):
                # If mask is a dask array, compute it first
                mask = mask.compute()
            if isinstance(mask, pd.Series):
                # Keep pandas Series as-is (AnnData handles this)
                pass
            elif hasattr(mask, "values"):
                # Convert to pandas Series to preserve index alignment
                mask = pd.Series(mask.values, index=df.index, dtype=bool)
            elif not isinstance(mask, np.ndarray | pd.Series):
                # Convert other types to numpy array
                mask = np.asarray(mask, dtype=bool)

            # Apply mask to AnnData
            if filter_attribute == "obs":
                adata = adata[mask]
            else:  # filter_attribute == "var"
                adata = adata[:, mask]

            # Check if filtering resulted in zero-sized dimensions
            if adata.n_obs == 0:
                raise ValueError(
                    "Filtering resulted in an empty dataset: no observations (cells) remain after filtering"
                )
            if adata.n_vars == 0:
                raise ValueError("Filtering resulted in an empty dataset: no variables (genes) remain after filtering")

        attr_obj = getattr(adata, attribute, None)
        if attr_obj is None:
            raise KeyError(f"Attribute {attribute} not found")

        if key is not None:
            if attribute in ("obs", "var", "X"):
                raise ValueError(
                    "The 'key' argument is not supported for 'obs', 'var', or 'X' attributes, use 'columns_or_genes' instead"
                )
            try:
                # Convert single string to list for consistent handling
                key_list = [key] if isinstance(key, str) else key
                attr_obj = get_nested_key(attr_obj, key_list)
            except (KeyError, AttributeError) as err:
                key_str = key if isinstance(key, str) else " -> ".join(key)
                raise KeyError(f"Attribute {attribute} with key {key_str} not found") from err

        if columns_or_genes is not None and attribute in ("X", "layers"):
            columns_or_genes, _ = match_patterns(adata.var_names, columns_or_genes)
            if len(columns_or_genes) == 0:
                raise ValueError("None of the provided genes were found in var_names")
            indices = [adata.var_names.tolist().index(g) for g in columns_or_genes]
            attr_obj = dask_array_to_dataset2d(attr_obj[:, indices], columns_or_genes)

        if isinstance(attr_obj, Dataset2D):
            description = describe_dataset2d(attr_obj, columns_or_genes)
            description = truncate_string(description.to_csv())
            if return_value_counts_for_categorical:
                value_counts = value_counts_dataset2d(attr_obj, columns_or_genes)
                value_counts = truncate_string(value_counts.to_csv())
            else:
                value_counts = None
        elif isinstance(attr_obj, Array):
            description = describe_dask_array(attr_obj)
            description = truncate_string(description.to_csv())
            value_counts = None
        else:
            raise ValueError(
                f"Attribute {attribute} is not a dataframe or array"
                if key is None
                else f"Attribute value of {attribute} for key {key} is not a dataframe or array"
            )

    except Exception as e:  # noqa: BLE001
        # Catch all exceptions to ensure function always returns ExplorationResult
        # This is intentional for API stability - all errors are returned in the error field
        error = truncate_string(str(e), max_output_len=100)
    finally:
        if adata is not None:
            adata.file.close()
            del adata
            gc.collect()

    return ExplorationResult(description=description, value_counts=value_counts, error=error)


def _compute_categorical_stats(dataset2d: Dataset2D, col: str) -> dict:
    """Compute descriptive statistics for a categorical column in a Dataset2D.

    Parameters
    ----------
    dataset2d : Dataset2D
        The Dataset2D object containing the column
    col : str
        The column name to compute statistics for

    Returns
    -------
    dict
        A dictionary containing count, unique, top, freq, and #NaN statistics
    """
    # For object/categorical columns, still need to load data for value_counts
    col_df = dataset2d[[col]].to_memory()
    col_data = col_df[col]
    non_null = col_data.dropna()
    count = len(non_null)
    nan_count = int(col_data.isna().sum())

    if count > 0:
        value_counts = non_null.value_counts()
        unique_count = len(value_counts)
        top_value = value_counts.index[0] if len(value_counts) > 0 else None
        freq = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0

        return {
            "count": count,
            "unique": unique_count,
            "top": top_value,
            "freq": freq,
            "#NaN": nan_count,
        }
    else:
        # All null values
        return {
            "count": 0,
            "unique": 0,
            "top": None,
            "freq": 0,
            "#NaN": nan_count,
        }


def describe_dataset2d(
    dataset2d: Dataset2D, columns: list[str] | None = None, index_name: str | None = None
) -> pd.DataFrame | str:
    """Generate descriptive statistics for a Dataset2D object.

    This function provides a statistical summary similar to pandas DataFrame.describe(),
    including count, mean, std, min, quartiles, and max for numeric columns,
    and count, unique, top, and freq for object/categorical columns.

    Memory-efficient implementation that processes columns one at a time without
    loading the entire dataset into memory.

    Parameters
    ----------
    dataset2d : Dataset2D
        The Dataset2D object to describe

    Returns
    -------
    pd.DataFrame
        A DataFrame containing descriptive statistics for each column
    """
    available_columns = dataset2d.columns.tolist()
    columns = columns or available_columns

    missing_columns = [col for col in set(columns) if col not in set(available_columns)]
    if missing_columns:
        raise ValueError(f"The following columns are not present in the dataframe: {missing_columns}")

    stats_dict = {}
    is_numeric_list = []

    for col in columns:
        # Access column as DataArray - try to compute stats without loading into memory
        col_array = dataset2d[col]

        # Check if column is numeric by checking dtype (without loading data)
        # More robust check: explicitly exclude object, string, and categorical dtypes
        dtype = col_array.dtype if hasattr(col_array, "dtype") else None
        is_numeric = False
        if dtype is not None:
            # First check: exclude object, string, and categorical types explicitly
            dtype_str = str(dtype).lower()
            if any(x in dtype_str for x in ["object", "string", "str", "unicode", "category", "categorical"]):
                is_numeric = False
            elif hasattr(dtype, "kind") and dtype.kind in ["O", "U", "S"]:  # Object, Unicode string, byte string
                is_numeric = False
            elif hasattr(dtype, "categories"):  # Categorical dtype
                is_numeric = False
            # Then check if it's a numeric dtype
            elif pd.api.types.is_numeric_dtype(dtype):
                is_numeric = True

        is_numeric_list.append(is_numeric)

        if is_numeric:
            # For numeric columns, compute statistics directly on DataArray without loading
            # These operations are lazy and compute efficiently (e.g., with Dask chunks)
            # Compute statistics using DataArray methods - these may use lazy computation
            # Wrap quantile computation in try-except to catch any dtype issues
            try:
                count_result = col_array.count()
                mean_result = col_array.mean()
                std_result = col_array.std()
                min_result = col_array.min()
                max_result = col_array.max()
                quantile_25 = col_array.quantile(0.25)
                quantile_50 = col_array.quantile(0.50)
                quantile_75 = col_array.quantile(0.75)

                # Calculate NaN count: total size - non-null count
                isnull_result = col_array.isnull().sum()

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
                nan_count = int(_extract_scalar(isnull_result))

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
                        "#NaN": nan_count,
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
                        "#NaN": nan_count,
                    }
            except (ValueError, TypeError):
                # If quantile computation fails, treat as categorical
                # This can happen if dtype check was incorrect
                is_numeric = False
                is_numeric_list[-1] = False
                # Use helper function to compute categorical stats
                stats_dict[col] = _compute_categorical_stats(dataset2d, col)
        else:
            # For object/categorical columns, use helper function
            stats_dict[col] = _compute_categorical_stats(dataset2d, col)

    # Convert to DataFrame and reorder columns to match pandas describe() output order
    numeric_stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "#NaN"]
    object_stats = ["count", "unique", "top", "freq", "#NaN"]

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
    if index_name is not None:
        result_df.index.name = index_name

    # Ensure appropriate dtypes
    # Convert numeric statistics to float (compatible with NaN)
    numeric_stat_cols = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    for col in numeric_stat_cols:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    # Convert count, freq, and #NaN to int
    if "count" in result_df.columns:
        result_df["count"] = pd.to_numeric(result_df["count"], errors="coerce").fillna(0).astype(int)
    if "freq" in result_df.columns:
        result_df["freq"] = pd.to_numeric(result_df["freq"], errors="coerce").fillna(0).astype(int)
    if "#NaN" in result_df.columns:
        result_df["#NaN"] = pd.to_numeric(result_df["#NaN"], errors="coerce").fillna(0).astype(int)
    result_df["is_numeric"] = is_numeric_list
    return result_df


def value_counts_dataset2d(dataset2d: Dataset2D, columns: list[str] | None = None) -> pd.DataFrame:
    """Generate value counts for categorical columns in a dataset2d.

    Memory-efficient implementation that processes columns one at a time,
    only loading categorical (non-numeric) columns into memory.

    Parameters
    ----------
    dataset2d : Dataset2D
        The Dataset2D object to analyze
    columns : list[str]
        List of column names to check for value counts

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['column', 'value', 'count'] containing
        value counts for all categorical columns. Numeric columns are skipped.
    """
    # Validate columns exist
    available_columns = dataset2d.columns.tolist()
    columns = columns or available_columns
    missing_columns = [col for col in set(columns) if col not in set(available_columns)]
    if missing_columns:
        raise ValueError(f"The following columns are not present in the dataframe: {missing_columns}")

    # Identify categorical columns (non-numeric)
    categorical_columns = []
    for col in columns:
        col_array = dataset2d[col]
        # More robust check: explicitly exclude object, string, and categorical dtypes
        dtype = col_array.dtype if hasattr(col_array, "dtype") else None
        is_numeric = False
        if dtype is not None:
            # First check: exclude object, string, and categorical types explicitly
            dtype_str = str(dtype).lower()
            if any(x in dtype_str for x in ["object", "string", "str", "unicode", "category", "categorical"]):
                is_numeric = False
            elif hasattr(dtype, "kind") and dtype.kind in ["O", "U", "S"]:  # Object, Unicode string, byte string
                is_numeric = False
            elif hasattr(dtype, "categories"):  # Categorical dtype
                is_numeric = False
            # Then check if it's a numeric dtype
            elif pd.api.types.is_numeric_dtype(dtype):
                is_numeric = True
        if not is_numeric:
            categorical_columns.append(col)

    if not categorical_columns:
        return pd.DataFrame({"column": [], "value": [], "count": []})

    # Process each categorical column one at a time to be memory efficient
    all_value_counts = []

    for col in categorical_columns:
        # Load only this column into memory
        col_df = dataset2d[[col]].to_memory()
        col_data = col_df[col]
        value_counts = col_data.value_counts(dropna=False)

        # Convert to DataFrame format with column name
        for value, count in value_counts.items():
            all_value_counts.append({"column": col, "value": value, "count": int(count)})

    # Return as DataFrame
    if all_value_counts:
        return pd.DataFrame(all_value_counts)
    else:
        return pd.DataFrame({"column": [], "value": [], "count": []})


def describe_dask_array(array: Array, axis: int | None = None) -> pd.DataFrame:
    """Generate descriptive statistics for a numerical dask array.

    This function provides a statistical summary similar to pandas DataFrame.describe(),
    including count (non-NaN), mean, std, min, quartiles, and max.

    Memory-efficient implementation that uses lazy computation to compute statistics
    without loading the entire array into memory.

    Parameters
    ----------
    array : dask.array.core.Array
        The dask array to describe (must be numerical)
    axis : int | None, optional
        Axis along which to compute statistics. If None, statistics are computed
        over the entire flattened array. If an integer, statistics are computed
        along that axis (e.g., axis=0 for rows, axis=1 for columns).
        Default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing descriptive statistics. If axis is None, returns
        a single-row DataFrame. If axis is specified, returns a DataFrame with
        one row per slice along that axis.
    """
    # Compute statistics using dask array methods - these use lazy computation
    # For count, we need to count non-NaN values
    if axis is None:
        # Statistics over entire flattened array
        # Flatten array for quantile computation (quantile requires an axis)
        flattened = array.flatten()
        count_result = (~dask.array.isnan(array)).sum()
        mean_result = dask.array.nanmean(array)
        std_result = dask.array.nanstd(array)
        min_result = dask.array.nanmin(array)
        max_result = dask.array.nanmax(array)
        # Compute quantiles along axis 0 of flattened array using nanquantile
        quantile_25 = dask.array.nanquantile(flattened, 0.25, axis=0)
        quantile_50 = dask.array.nanquantile(flattened, 0.50, axis=0)
        quantile_75 = dask.array.nanquantile(flattened, 0.75, axis=0)

        # Extract scalar values - compute() triggers actual computation
        count_val = int(count_result.compute())
        mean_val = float(mean_result.compute())
        std_val = float(std_result.compute())
        min_val = float(min_result.compute())
        max_val = float(max_result.compute())
        q25_val = float(quantile_25.compute())
        q50_val = float(quantile_50.compute())
        q75_val = float(quantile_75.compute())

        if count_val > 0:
            stats_dict = {
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
            # All NaN values
            stats_dict = {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "25%": np.nan,
                "50%": np.nan,
                "75%": np.nan,
                "max": np.nan,
            }

        result_df = pd.DataFrame([stats_dict])
        result_df.index.name = "statistic"
    else:
        # Statistics along specified axis
        count_result = (~dask.array.isnan(array)).sum(axis=axis)
        mean_result = dask.array.nanmean(array, axis=axis)
        std_result = dask.array.nanstd(array, axis=axis)
        min_result = dask.array.nanmin(array, axis=axis)
        max_result = dask.array.nanmax(array, axis=axis)
        quantile_25 = dask.array.nanquantile(array, 0.25, axis=axis)
        quantile_50 = dask.array.nanquantile(array, 0.50, axis=axis)
        quantile_75 = dask.array.nanquantile(array, 0.75, axis=axis)

        # Compute all results
        count_vals = count_result.compute()
        mean_vals = mean_result.compute()
        std_vals = std_result.compute()
        min_vals = min_result.compute()
        max_vals = max_result.compute()
        q25_vals = quantile_25.compute()
        q50_vals = quantile_50.compute()
        q75_vals = quantile_75.compute()

        # Handle scalar results (when array is 1D)
        if count_vals.ndim == 0:
            count_vals = np.array([count_vals])
            mean_vals = np.array([mean_vals])
            std_vals = np.array([std_vals])
            min_vals = np.array([min_vals])
            max_vals = np.array([max_vals])
            q25_vals = np.array([q25_vals])
            q50_vals = np.array([q50_vals])
            q75_vals = np.array([q75_vals])

        # Create DataFrame
        stats_dict = {
            "count": count_vals.astype(int),
            "mean": mean_vals.astype(float),
            "std": std_vals.astype(float),
            "min": min_vals.astype(float),
            "25%": q25_vals.astype(float),
            "50%": q50_vals.astype(float),
            "75%": q75_vals.astype(float),
            "max": max_vals.astype(float),
        }

        result_df = pd.DataFrame(stats_dict)
        result_df.index.name = f"axis_{axis}_index"

    return result_df


def dask_array_to_dataset2d(
    dask_array: Array,
    column_names: list[str],
    index: pd.Index | None = None,
    index_name: str = "index",
) -> Dataset2D:
    """
    Convert a dask array to an AnnData Dataset2D with column names.

    Parameters
    ----------
    dask_array : dask.array.Array
        A 2D dask array with shape (n_obs, n_cols)
    column_names : list[str]
        List of column names. Must match n_cols.
    index : pd.Index, optional
        Custom index. If None, uses RangeIndex.
    index_name : str
        Name for the index dimension (default: "index")

    Returns
    -------
    Dataset2D
        A Dataset2D instance wrapping an xarray Dataset
    """
    if dask_array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {dask_array.ndim}D")

    n_obs, n_cols = dask_array.shape
    if len(column_names) != n_cols:
        raise ValueError(f"Number of column names ({len(column_names)}) must match number of columns ({n_cols})")

    # Create index coordinate
    if index is None:
        index = pd.RangeIndex(n_obs)
    index_coord = xr.IndexVariable(index_name, index)

    # Create DataArrays for each column
    data_vars = {}
    for i, col_name in enumerate(column_names):
        data_vars[col_name] = (index_name, dask_array[:, i])

    # Create xarray Dataset
    xr_dataset = xr.Dataset(data_vars, coords={index_name: index_coord})

    # Wrap in Dataset2D
    return Dataset2D(xr_dataset)
