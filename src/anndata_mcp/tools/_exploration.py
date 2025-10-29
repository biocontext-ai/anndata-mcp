"""Tools for exploring AnnData objects."""

from typing import Any

import pandas as pd
from anndata.experimental import read_lazy
from pydantic import BaseModel, Field

from anndata_mcp.mcp import mcp
from anndata_mcp.tools._utils import convert_to_serializable, get_dtype_string, handle_dataset2d, safe_to_list


class AttributeInfo(BaseModel):
    """Information about an AnnData attribute."""

    name: str
    type: str
    shape: list[int] | None = None
    keys: list[str] | None = None
    description: str


class ColumnInfo(BaseModel):
    """Information about a dataframe column."""

    name: str
    dtype: str
    n_unique: int | None = None
    has_nulls: bool
    description: str | None = None


class DataFrameInfo(BaseModel):
    """Detailed information about a dataframe-like attribute."""

    shape: list[int]
    columns: list[ColumnInfo]
    index_name: str | None = None
    n_rows: int
    n_cols: int


class UniqueValuesResult(BaseModel):
    """Result of unique values query."""

    column: str
    unique_values: list[Any]
    n_unique: int
    truncated: bool = Field(default=False, description="Whether the results were truncated")


class ColumnStats(BaseModel):
    """Statistics for a column."""

    column: str
    dtype: str
    n_unique: int | None = None
    n_nulls: int
    stats: dict[str, Any] = Field(default_factory=dict)


@mcp.tool
def get_attribute_info(path: str, attribute: str) -> AttributeInfo:
    """Get detailed information about a specific AnnData attribute.

    **Use this when you need to:**
    - Check the type and shape of a specific AnnData attribute
    - See what keys are available in mapping attributes (obsm, layers, etc.)
    - Verify an attribute exists before accessing it

    **Example usage:**
    - "What's in the X matrix?" → get_attribute_info(path, "X")
    - "What embeddings are in obsm?" → get_attribute_info(path, "obsm")
    - "What layers are available?" → get_attribute_info(path, "layers")

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    attribute : str
        Attribute name: 'obs', 'var', 'X', 'obsm', 'varm', 'obsp', 'varp', 'uns', 'layers', 'raw'

    Returns
    -------
    AttributeInfo
        Information including:
        - name: Attribute name
        - type: Python type name
        - shape: Shape if applicable (e.g., [2638, 1838] for X)
        - keys: Available keys if it's a mapping (e.g., ['X_pca', 'X_umap'] for obsm)

    Examples
    --------
    Check what's in obsm:
    >>> info = get_attribute_info("pbmc3k.h5ad", "obsm")
    >>> # Returns: AttributeInfo(name='obsm', type='AxisArrays', keys=['X_pca', 'X_umap'], ...)
    >>> # Next: Use get_obsm_data() to retrieve specific embeddings
    """
    adata = read_lazy(path)

    valid_attrs = ["obs", "var", "X", "obsm", "varm", "obsp", "varp", "uns", "layers", "raw"]

    if attribute not in valid_attrs:
        raise ValueError(f"Invalid attribute. Must be one of: {valid_attrs}")

    attr_obj = getattr(adata, attribute, None)
    if attr_obj is None:
        return AttributeInfo(name=attribute, type="None", description="Attribute is None")

    # Get type information
    type_str = type(attr_obj).__name__

    # Get shape if applicable
    shape = None
    if hasattr(attr_obj, "shape"):
        shape = list(attr_obj.shape)

    # Get keys for mapping-like attributes
    keys = None
    if hasattr(attr_obj, "keys"):
        keys = list(attr_obj.keys())

    # Generate description based on attribute type
    description = f"AnnData.{attribute}: {type_str}"
    if shape:
        description += f" with shape {shape}"
    if keys:
        description += f" with {len(keys)} keys"

    return AttributeInfo(name=attribute, type=type_str, shape=shape, keys=keys, description=description)


@mcp.tool
def get_dataframe_info(path: str, attribute: str) -> DataFrameInfo:
    """Get detailed information about cell or gene metadata structure.

    **Use this when you need to:**
    - Explore what columns are in .obs (cell metadata) or .var (gene metadata)
    - Check data types of each column before querying
    - See which columns have unique values or null values
    - Understand the metadata structure after getting the summary

    **Example usage:**
    - "What cell metadata is available?" → get_dataframe_info(path, "obs")
    - "Show me the gene metadata columns" → get_dataframe_info(path, "var")
    - "What's the structure of obs?" → get_dataframe_info(path, "obs")

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    attribute : str
        Attribute name ('obs' for cell metadata or 'var' for gene metadata)

    Returns
    -------
    DataFrameInfo
        Detailed information including:
        - shape: [n_rows, n_cols]
        - columns: List of ColumnInfo with name, dtype, n_unique, has_nulls
        - n_rows: Number of rows (cells or genes)
        - n_cols: Number of columns

    Examples
    --------
    Explore cell metadata structure:
    >>> info = get_dataframe_info("pbmc3k.h5ad", "obs")
    >>> # Returns: DataFrameInfo(shape=[2638, 5], columns=[ColumnInfo(name='n_genes', dtype='int64', n_unique=850, has_nulls=False), ...])
    >>> # Next: Use get_unique_values() to see specific values in a column
    """
    adata = read_lazy(path)

    if attribute not in ["obs", "var"]:
        raise ValueError("Attribute must be 'obs' or 'var'")

    df_obj = getattr(adata, attribute)

    # Check if it's a Dataset2D (from lazy reading)
    is_dataset2d = hasattr(df_obj, "_ds") or (hasattr(df_obj, "__class__") and "Dataset2D" in str(type(df_obj)))

    if is_dataset2d:
        # For Dataset2D, get info without loading into memory
        # Dataset2D wraps an xarray Dataset
        if hasattr(df_obj, "_ds"):
            xr_ds = df_obj._ds
        else:
            xr_ds = df_obj

        # Get shape from xarray Dataset
        coord_name = list(xr_ds.coords.keys())[0] if xr_ds.coords else None
        n_rows = len(xr_ds.coords[coord_name]) if coord_name else 0
        col_names = list(xr_ds.data_vars.keys())
        n_cols = len(col_names)
        shape = [n_rows, n_cols]

        # Get column information from xarray Dataset
        columns = []
        for col in col_names:
            try:
                data_var = xr_ds[col]
                dtype = get_dtype_string(data_var.dtype)

                # Convert to pandas to get unique values and null check
                # This is efficient as we only load one column at a time
                col_series = data_var.to_pandas()
                n_unique = None
                has_nulls = False

                if n_rows < 100000:
                    try:
                        n_unique = int(col_series.nunique())
                        has_nulls = bool(col_series.isna().any())
                    except Exception as _excp:  # noqa: BLE001
                        pass

                columns.append(
                    ColumnInfo(
                        name=col, dtype=dtype, n_unique=n_unique, has_nulls=has_nulls, description=f"Column {col}"
                    )
                )
            except Exception as _excp:  # noqa: BLE001
                columns.append(
                    ColumnInfo(name=col, dtype="unknown", n_unique=None, has_nulls=False, description=f"Column {col}")
                )

        index_name = coord_name

    else:
        # Regular DataFrame handling
        shape = list(df_obj.shape)
        n_rows, n_cols = df_obj.shape

        # Get column information
        columns = []
        for col in df_obj.columns:
            col_data = df_obj[col]
            dtype = get_dtype_string(col_data.dtype)
            n_unique = None
            has_nulls = bool(col_data.isna().any())

            # Calculate unique values for reasonable-sized columns
            if n_rows < 100000:
                try:
                    n_unique = int(col_data.nunique())
                except Exception as _excp:  # noqa: BLE001
                    n_unique = None

            columns.append(
                ColumnInfo(name=col, dtype=dtype, n_unique=n_unique, has_nulls=has_nulls, description=f"Column {col}")
            )

        index_name = df_obj.index.name

    return DataFrameInfo(shape=shape, columns=columns, index_name=index_name, n_rows=n_rows, n_cols=n_cols)


@mcp.tool
def get_unique_values(path: str, attribute: str, column: str, max_values: int = 100) -> UniqueValuesResult:
    """Get all unique values from a categorical column.

    **Use this when you need to:**
    - See all cluster names or cell types available
    - Check what categories exist in a metadata column
    - Identify possible values before filtering data
    - Answer questions like "What clusters are in this dataset?"

    **Example usage:**
    - "What cell types are there?" → get_unique_values(path, "obs", "cell_type")
    - "List all cluster IDs" → get_unique_values(path, "obs", "louvain")
    - "What are the unique batch IDs?" → get_unique_values(path, "obs", "batch")

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    attribute : str
        Attribute name ('obs' for cell metadata or 'var' for gene metadata)
    column : str
        Column name to get unique values from
    max_values : int
        Maximum number of unique values to return (default: 100)

    Returns
    -------
    UniqueValuesResult
        Contains:
        - unique_values: List of unique values
        - n_unique: Total count of unique values
        - truncated: Whether results were limited by max_values

    Examples
    --------
    Get all cluster names:
    >>> result = get_unique_values("pbmc3k.h5ad", "obs", "louvain")
    >>> # Returns: UniqueValuesResult(unique_values=['0', '1', '2', ...], n_unique=8, truncated=False)
    >>> # Next: Use get_value_counts() to see how many cells per cluster
    """
    adata = read_lazy(path)

    if attribute not in ["obs", "var"]:
        raise ValueError("Attribute must be 'obs' or 'var'")

    df_obj = getattr(adata, attribute)

    # For Dataset2D, convert just the column we need
    if not isinstance(df_obj, pd.DataFrame):
        # Check if it's Dataset2D
        if hasattr(df_obj, "_ds"):
            xr_ds = df_obj._ds
            if column not in xr_ds.data_vars:
                raise ValueError(f"Column '{column}' not found in {attribute}")
            # Convert just this column to pandas Series
            col_data = xr_ds[column].to_pandas()
        else:
            # Try to convert the whole thing
            df_obj = handle_dataset2d(df_obj)
            if column not in df_obj.columns:
                raise ValueError(f"Column '{column}' not found in {attribute}")
            col_data = df_obj[column]
    else:
        if column not in df_obj.columns:
            raise ValueError(f"Column '{column}' not found in {attribute}")
        col_data = df_obj[column]

    unique_vals = col_data.unique()

    n_unique = len(unique_vals)
    truncated = n_unique > max_values

    # Limit results
    if truncated:
        unique_vals = unique_vals[:max_values]

    # Convert to serializable format
    unique_vals = safe_to_list(unique_vals, max_items=max_values)
    unique_vals = [convert_to_serializable(v) for v in unique_vals]

    return UniqueValuesResult(column=column, unique_values=unique_vals, n_unique=n_unique, truncated=truncated)


@mcp.tool
def get_column_stats(path: str, attribute: str, column: str) -> ColumnStats:
    """Get statistical summary for a numeric or categorical column.

    **Use this when you need to:**
    - Get min/max/mean/median/std for numeric columns (e.g., n_genes, n_counts)
    - Check the distribution of a continuous variable
    - Get value counts for categorical columns with few unique values
    - Answer questions like "What's the average number of genes per cell?"

    **Example usage:**
    - "Statistics for gene counts" → get_column_stats(path, "obs", "n_genes")
    - "What's the range of UMI counts?" → get_column_stats(path, "obs", "n_counts")
    - "Distribution of batches" → get_column_stats(path, "obs", "batch")

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    attribute : str
        Attribute name ('obs' for cell metadata or 'var' for gene metadata)
    column : str
        Column name to get statistics for

    Returns
    -------
    ColumnStats
        Contains:
        - stats: Dict with min, max, mean, median, std (numeric) or value_counts (categorical)
        - dtype: Data type
        - n_unique: Number of unique values
        - n_nulls: Number of null/NA values

    Examples
    --------
    Get statistics for gene counts:
    >>> stats = get_column_stats("pbmc3k.h5ad", "obs", "n_genes")
    >>> # Returns: ColumnStats(stats={'min': 47, 'max': 2500, 'mean': 850.3, 'median': 820, 'std': 340}, ...)
    >>> # For grouped stats, use get_grouped_stats() instead
    """
    adata = read_lazy(path)

    if attribute not in ["obs", "var"]:
        raise ValueError("Attribute must be 'obs' or 'var'")

    df_obj = getattr(adata, attribute)

    # For Dataset2D, convert just the column we need
    if not isinstance(df_obj, pd.DataFrame):
        # Check if it's Dataset2D
        if hasattr(df_obj, "_ds"):
            xr_ds = df_obj._ds
            if column not in xr_ds.data_vars:
                raise ValueError(f"Column '{column}' not found in {attribute}")
            # Convert just this column to pandas Series
            col_data = xr_ds[column].to_pandas()
        else:
            # Try to convert the whole thing
            df_obj = handle_dataset2d(df_obj)
            if column not in df_obj.columns:
                raise ValueError(f"Column '{column}' not found in {attribute}")
            col_data = df_obj[column]
    else:
        if column not in df_obj.columns:
            raise ValueError(f"Column '{column}' not found in {attribute}")
        col_data = df_obj[column]

    dtype = get_dtype_string(col_data.dtype)
    n_nulls = int(col_data.isna().sum())

    stats = {}

    # Numeric statistics
    if pd.api.types.is_numeric_dtype(col_data):
        try:
            stats["min"] = convert_to_serializable(col_data.min())
            stats["max"] = convert_to_serializable(col_data.max())
            stats["mean"] = convert_to_serializable(col_data.mean())
            stats["median"] = convert_to_serializable(col_data.median())
            stats["std"] = convert_to_serializable(col_data.std())
        except Exception as _excp:  # noqa: BLE001
            pass

    # Categorical/string statistics
    else:
        try:
            n_unique = int(col_data.nunique())
            stats["n_unique"] = n_unique
            if n_unique <= 20:
                value_counts = col_data.value_counts().to_dict()
                stats["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
        except Exception as _excp:  # noqa: BLE001
            n_unique = None

    n_unique_total = None
    try:
        n_unique_total = int(col_data.nunique())
    except Exception as _excp:  # noqa: BLE001
        pass

    return ColumnStats(column=column, dtype=dtype, n_unique=n_unique_total, n_nulls=n_nulls, stats=stats)


@mcp.tool
def get_value_counts(path: str, attribute: str, column: str) -> dict[str, Any]:
    """Count occurrences of each unique value in a column (e.g., cells per cluster).

    **Use this when you need to:**
    - Count how many cells are in each cluster/cell type
    - Get the distribution of categories (e.g., number of observations per group)
    - Answer questions like "How many cells are in each louvain cluster?"

    **Example usage:**
    - "How many cells per cluster?" → get_value_counts(path, "obs", "louvain")
    - "Distribution of cell types" → get_value_counts(path, "obs", "cell_type")

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    attribute : str
        Attribute name ('obs' for cell metadata, 'var' for gene metadata)
    column : str
        Column name to count (e.g., 'louvain', 'cell_type')

    Returns
    -------
    dict[str, Any]
        Dictionary with 'counts' (value -> count mapping) and 'total' (total count).

    Examples
    --------
    Get number of cells in each cluster:
    >>> result = get_value_counts("data.h5ad", "obs", "louvain")
    >>> # Returns: {"counts": {"CD4 T cells": 1200, "B cells": 300, ...}, "total": 2638}
    """
    adata = read_lazy(path)

    if attribute not in ["obs", "var"]:
        raise ValueError("Attribute must be 'obs' or 'var'")

    df_obj = getattr(adata, attribute)

    # For Dataset2D, convert just the column we need
    if not isinstance(df_obj, pd.DataFrame):
        if hasattr(df_obj, "_ds"):
            xr_ds = df_obj._ds
            if column not in xr_ds.data_vars:
                raise ValueError(f"Column '{column}' not found in {attribute}")
            col_data = xr_ds[column].to_pandas()
        else:
            df_obj = handle_dataset2d(df_obj)
            if column not in df_obj.columns:
                raise ValueError(f"Column '{column}' not found in {attribute}")
            col_data = df_obj[column]
    else:
        if column not in df_obj.columns:
            raise ValueError(f"Column '{column}' not found in {attribute}")
        col_data = df_obj[column]

    # Get value counts
    value_counts = col_data.value_counts().to_dict()

    # Convert to serializable format
    counts = {str(k): int(v) for k, v in value_counts.items()}
    total = sum(counts.values())

    return {"counts": counts, "total": total, "column": column}


@mcp.tool
def get_grouped_stats(path: str, attribute: str, group_by: str, value_column: str) -> dict[str, Any]:
    """Calculate statistics for a numeric column grouped by a categorical column.

    **Use this when you need to:**
    - Get average gene count per cluster
    - Compare distributions across groups (e.g., mean UMI counts per cell type)
    - Answer questions like "What's the average number of genes per cluster?"

    **Example usage:**
    - "Average genes per cluster?" → get_grouped_stats(path, "obs", "louvain", "n_genes")
    - "Mean UMI counts by cell type" → get_grouped_stats(path, "obs", "cell_type", "n_counts")

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    attribute : str
        Attribute name ('obs' for cell metadata, 'var' for gene metadata)
    group_by : str
        Column to group by (e.g., 'louvain', 'cell_type')
    value_column : str
        Numeric column to calculate statistics for (e.g., 'n_genes', 'n_counts')

    Returns
    -------
    dict[str, Any]
        Dictionary with statistics per group (mean, median, std, min, max, count).

    Examples
    --------
    Get average number of genes per cluster:
    >>> result = get_grouped_stats("data.h5ad", "obs", "louvain", "n_genes")
    >>> # Returns: {"CD4 T cells": {"mean": 850, "median": 820, "count": 1200, ...}, ...}
    """
    adata = read_lazy(path)

    if attribute not in ["obs", "var"]:
        raise ValueError("Attribute must be 'obs' or 'var'")

    df_obj = getattr(adata, attribute)

    # Convert Dataset2D if necessary (we need both columns)
    if not isinstance(df_obj, pd.DataFrame):
        df_obj = handle_dataset2d(df_obj)

    if group_by not in df_obj.columns:
        raise ValueError(f"Column '{group_by}' not found in {attribute}")
    if value_column not in df_obj.columns:
        raise ValueError(f"Column '{value_column}' not found in {attribute}")

    # Check if value column is numeric
    if not pd.api.types.is_numeric_dtype(df_obj[value_column]):
        raise ValueError(f"Column '{value_column}' must be numeric for statistics")

    # Group and calculate statistics
    grouped = df_obj.groupby(group_by)[value_column]

    results = {}
    for group_name, group_data in grouped:
        stats = {
            "mean": convert_to_serializable(group_data.mean()),
            "median": convert_to_serializable(group_data.median()),
            "std": convert_to_serializable(group_data.std()),
            "min": convert_to_serializable(group_data.min()),
            "max": convert_to_serializable(group_data.max()),
            "count": int(len(group_data)),
        }
        results[str(group_name)] = stats

    return {
        "group_by": group_by,
        "value_column": value_column,
        "groups": results,
        "n_groups": len(results),
    }


@mcp.tool
def list_available_keys(path: str, attribute: str) -> list[str]:
    """List all available keys in a mapping attribute.

    **Use this when you need to:**
    - See what embeddings are available (obsm)
    - Check what layers exist (layers)
    - List unstructured annotations (uns)
    - Verify a key exists before accessing it

    **Example usage:**
    - "What embeddings are available?" → list_available_keys(path, "obsm")
    - "What layers exist?" → list_available_keys(path, "layers")
    - "What's in uns?" → list_available_keys(path, "uns")

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    attribute : str
        Attribute name: 'obsm', 'varm', 'layers', 'uns', 'obsp', 'varp'

    Returns
    -------
    list[str]
        List of available keys (e.g., ['X_pca', 'X_umap'] for obsm)

    Examples
    --------
    Check available embeddings:
    >>> keys = list_available_keys("pbmc3k.h5ad", "obsm")
    >>> # Returns: ['X_pca', 'X_umap', 'X_draw_graph_fr']
    >>> # Next: Use get_obsm_data() to retrieve a specific embedding
    """
    adata = read_lazy(path)

    valid_attrs = ["obsm", "varm", "layers", "uns", "obsp", "varp"]

    if attribute not in valid_attrs:
        raise ValueError(f"Invalid attribute. Must be one of: {valid_attrs}")

    attr_obj = getattr(adata, attribute, None)
    if attr_obj is None:
        return []

    if hasattr(attr_obj, "keys"):
        return list(attr_obj.keys())

    return []
