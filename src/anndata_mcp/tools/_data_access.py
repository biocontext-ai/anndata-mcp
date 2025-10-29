"""Tools for accessing data from AnnData objects."""

from typing import Any

import numpy as np
import pandas as pd
from anndata.experimental import read_lazy
from pydantic import BaseModel, Field
from scipy import sparse

from anndata_mcp.mcp import mcp
from anndata_mcp.tools._utils import convert_to_serializable, get_sparse_info, handle_dataset2d


class DataSlice(BaseModel):
    """A slice of data from AnnData."""

    data: Any
    shape: list[int] | None = None
    dtype: str | None = None
    rows_returned: int
    cols_returned: int | None = None
    total_rows: int
    total_cols: int | None = None
    is_sparse: bool = False
    sparse_info: dict | None = None


class MatrixSlice(BaseModel):
    """A slice of matrix data (X or layers)."""

    data: list[list[float]] | None = None
    indices: list[int] | None = None
    indptr: list[int] | None = None
    shape: list[int]
    dtype: str
    is_sparse: bool = False
    sparse_format: str | None = None
    sparsity: float | None = None
    rows_returned: int
    cols_returned: int
    total_rows: int
    total_cols: int


def _parse_slice(slice_str: str | None, max_val: int) -> slice:
    """Parse a slice string like '0:10' or ':100' into a slice object.

    Parameters
    ----------
    slice_str : str, optional
        Slice string
    max_val : int
        Maximum value for the slice

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


@mcp.tool
def get_obs_data(
    path: str,
    row_slice: str | None = None,
    columns: list[str] | None = None,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Retrieve cell metadata from .obs with optional slicing and column selection.

    **Use this when you need to:**
    - Get specific cell metadata for analysis (cluster assignments, QC metrics)
    - Retrieve data for specific cells by index range
    - Extract specific metadata columns only
    - Answer questions like "Show me the first 10 cells with their cluster labels"

    **Example usage:**
    - "Get first 10 cells" → get_obs_data(path, row_slice="0:10")
    - "Show cluster labels for all cells" → get_obs_data(path, columns=["louvain"], max_rows=1000)
    - "Get n_genes and n_counts for cells 100-200" → get_obs_data(path, row_slice="100:200", columns=["n_genes", "n_counts"])

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    row_slice : str, optional
        Row slice string (e.g., '0:10' for rows 0-9, ':100' for first 100, '50:' for row 50 onwards)
    columns : list[str], optional
        Specific column names to retrieve (e.g., ['louvain', 'n_genes'])
    max_rows : int
        Maximum number of rows to return if row_slice not specified (default: 100)

    Returns
    -------
    dict[str, Any]
        Dictionary with 'data' (list of dicts), 'index', 'columns', 'shape', 'total_rows', etc.

    Examples
    --------
    Get cluster assignments for first 10 cells:
    >>> result = get_obs_data("pbmc3k.h5ad", row_slice="0:10", columns=["louvain"])
    >>> # Returns: {'data': [{'louvain': '0'}, {'louvain': '2'}, ...], 'rows_returned': 10, ...}
    """
    adata = read_lazy(path)
    df_obj = adata.obs

    # Handle Dataset2D if necessary
    if not isinstance(df_obj, pd.DataFrame):
        df_obj = handle_dataset2d(df_obj)

    total_rows = df_obj.shape[0]
    total_cols = df_obj.shape[1]

    # Parse row slice
    if row_slice:
        row_slice_obj = _parse_slice(row_slice, total_rows)
        df_obj = df_obj.iloc[row_slice_obj]
    else:
        # Limit to max_rows by default
        df_obj = df_obj.head(max_rows)

    # Select columns
    if columns:
        missing_cols = set(columns) - set(df_obj.columns)
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        df_obj = df_obj[columns]

    rows_returned = df_obj.shape[0]
    cols_returned = df_obj.shape[1]

    # Convert to dict
    data_dict = df_obj.to_dict(orient="records")
    data_dict = convert_to_serializable(data_dict)

    return {
        "data": data_dict,
        "index": safe_to_list(df_obj.index),
        "columns": list(df_obj.columns),
        "shape": [rows_returned, cols_returned],
        "total_rows": total_rows,
        "total_cols": total_cols,
        "rows_returned": rows_returned,
        "cols_returned": cols_returned,
    }


@mcp.tool
def get_var_data(
    path: str,
    row_slice: str | None = None,
    columns: list[str] | None = None,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Retrieve gene metadata from .var with optional slicing and column selection.

    **Use this when you need to:**
    - Get specific gene metadata (gene names, highly variable genes, etc.)
    - Retrieve data for specific genes by index range
    - Extract specific gene annotation columns
    - Answer questions like "Show me highly variable genes"

    **Example usage:**
    - "Get first 20 genes" → get_var_data(path, row_slice="0:20")
    - "Show highly variable genes" → get_var_data(path, columns=["highly_variable"])
    - "Get gene names for top 100 genes" → get_var_data(path, row_slice="0:100", columns=["gene_name"])

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    row_slice : str, optional
        Row slice string (e.g., '0:10' for rows 0-9, ':100' for first 100)
    columns : list[str], optional
        Specific column names to retrieve
    max_rows : int
        Maximum number of rows to return if row_slice not specified (default: 100)

    Returns
    -------
    dict[str, Any]
        Dictionary with 'data' (list of dicts), 'index', 'columns', 'shape', 'total_rows', etc.

    Examples
    --------
    Get highly variable gene markers:
    >>> result = get_var_data("pbmc3k.h5ad", columns=["highly_variable", "means", "dispersions"])
    >>> # Returns: {'data': [{'highly_variable': True, 'means': 0.5, ...}, ...], ...}
    """
    adata = read_lazy(path)
    df_obj = adata.var

    # Handle Dataset2D if necessary
    if not isinstance(df_obj, pd.DataFrame):
        df_obj = handle_dataset2d(df_obj)

    total_rows = df_obj.shape[0]
    total_cols = df_obj.shape[1]

    # Parse row slice
    if row_slice:
        row_slice_obj = _parse_slice(row_slice, total_rows)
        df_obj = df_obj.iloc[row_slice_obj]
    else:
        # Limit to max_rows by default
        df_obj = df_obj.head(max_rows)

    # Select columns
    if columns:
        missing_cols = set(columns) - set(df_obj.columns)
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        df_obj = df_obj[columns]

    rows_returned = df_obj.shape[0]
    cols_returned = df_obj.shape[1]

    # Convert to dict
    data_dict = df_obj.to_dict(orient="records")
    data_dict = convert_to_serializable(data_dict)

    return {
        "data": data_dict,
        "index": safe_to_list(df_obj.index),
        "columns": list(df_obj.columns),
        "shape": [rows_returned, cols_returned],
        "total_rows": total_rows,
        "total_cols": total_cols,
        "rows_returned": rows_returned,
        "cols_returned": cols_returned,
    }


@mcp.tool
def get_X_data(
    path: str,
    row_slice: str | None = None,
    col_slice: str | None = None,
    max_rows: int = 10,
    max_cols: int = 10,
    return_dense: bool = True,
) -> MatrixSlice:
    """Retrieve expression data from the main .X matrix.

    **Use this when you need to:**
    - Get gene expression values for specific cells and genes
    - Sample the expression matrix to understand the data
    - Extract expression data for downstream analysis
    - Check if the matrix is sparse and what the sparsity is

    **Example usage:**
    - "Show expression for first 5 cells and 5 genes" → get_X_data(path, row_slice="0:5", col_slice="0:5")
    - "Get expression for cells 100-150, all genes" → get_X_data(path, row_slice="100:150", max_cols=1838)
    - "Sample the expression matrix" → get_X_data(path)

    **Note:** Expression matrices are often sparse (mostly zeros). The tool automatically converts to dense format by default.

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    row_slice : str, optional
        Row (cell) slice string (e.g., '0:10', ':100')
    col_slice : str, optional
        Column (gene) slice string (e.g., '0:10', ':100')
    max_rows : int
        Maximum number of rows (cells) to return if not sliced (default: 10)
    max_cols : int
        Maximum number of columns (genes) to return if not sliced (default: 10)
    return_dense : bool
        Convert sparse matrices to dense format (default: True)

    Returns
    -------
    MatrixSlice
        Contains 'data' (2D array), 'shape', 'dtype', 'is_sparse', 'sparsity', etc.

    Examples
    --------
    Get expression sample:
    >>> result = get_X_data("pbmc3k.h5ad", row_slice="0:5", col_slice="0:5")
    >>> # Returns: MatrixSlice(data=[[0.0, 1.2, ...], ...], shape=[5, 5], is_sparse=True, sparsity=0.87)
    """
    adata = read_lazy(path)
    X = adata.X

    if X is None:
        raise ValueError("X matrix is None")

    total_rows, total_cols = X.shape

    # Parse slices
    row_slice_obj = _parse_slice(row_slice, total_rows) if row_slice else slice(0, min(max_rows, total_rows))
    col_slice_obj = _parse_slice(col_slice, total_cols) if col_slice else slice(0, min(max_cols, total_cols))

    # Slice the data
    X_slice = X[row_slice_obj, col_slice_obj]

    # Check if sparse
    is_sparse = sparse.issparse(X_slice)

    result = MatrixSlice(
        shape=list(X_slice.shape),
        dtype=str(X_slice.dtype),
        is_sparse=is_sparse,
        rows_returned=X_slice.shape[0],
        cols_returned=X_slice.shape[1],
        total_rows=total_rows,
        total_cols=total_cols,
    )

    # Handle sparse matrices
    if is_sparse:
        sparse_info = get_sparse_info(X_slice)
        result.sparse_format = sparse_info["format"]
        result.sparsity = sparse_info.get("sparsity")

        if return_dense:
            # Convert to dense for easier consumption
            X_dense = X_slice.toarray()
            result.data = convert_to_serializable(X_dense)
        else:
            # Return sparse format (CSR/CSC)
            if hasattr(X_slice, "data") and hasattr(X_slice, "indices") and hasattr(X_slice, "indptr"):
                result.indices = convert_to_serializable(X_slice.indices)
                result.indptr = convert_to_serializable(X_slice.indptr)
                result.data = None  # Sparse representation provided via indices/indptr
    else:
        # Dense matrix
        result.data = convert_to_serializable(X_slice)

    return result


@mcp.tool
def get_layer_data(
    path: str,
    layer: str,
    row_slice: str | None = None,
    col_slice: str | None = None,
    max_rows: int = 10,
    max_cols: int = 10,
    return_dense: bool = True,
) -> MatrixSlice:
    """Retrieve data from a specific layer (e.g., raw counts, normalized data).

    **Use this when you need to:**
    - Access raw counts, normalized data, or other layer-specific data
    - Compare different data transformations
    - Get expression data from alternative processing
    - Check what's stored in a specific layer

    **Example usage:**
    - "Get raw counts for first 10 cells" → get_layer_data(path, "counts", row_slice="0:10")
    - "Show normalized data" → get_layer_data(path, "lognorm", row_slice="0:5", col_slice="0:5")
    - First use list_available_keys(path, "layers") to see available layers

    **Note:** Use get_X_data() for the main expression matrix, use this for alternative data layers.

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    layer : str
        Layer name (e.g., 'counts', 'lognorm'). Check available layers first with list_available_keys().
    row_slice : str, optional
        Row (cell) slice string
    col_slice : str, optional
        Column (gene) slice string
    max_rows : int
        Maximum number of rows if not sliced (default: 10)
    max_cols : int
        Maximum number of columns if not sliced (default: 10)
    return_dense : bool
        Convert sparse matrices to dense format (default: True)

    Returns
    -------
    MatrixSlice
        Contains 'data' (2D array), 'shape', 'dtype', 'is_sparse', etc.

    Examples
    --------
    Access raw count data:
    >>> result = get_layer_data("pbmc3k.h5ad", "counts", row_slice="0:5", col_slice="0:5")
    >>> # Returns: MatrixSlice(data=[[0, 3, 0, ...], ...], shape=[5, 5], dtype='int32')
    """
    adata = read_lazy(path)

    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}")

    layer_data = adata.layers[layer]
    total_rows, total_cols = layer_data.shape

    # Parse slices
    row_slice_obj = _parse_slice(row_slice, total_rows) if row_slice else slice(0, min(max_rows, total_rows))
    col_slice_obj = _parse_slice(col_slice, total_cols) if col_slice else slice(0, min(max_cols, total_cols))

    # Slice the data
    data_slice = layer_data[row_slice_obj, col_slice_obj]

    # Check if sparse
    is_sparse = sparse.issparse(data_slice)

    result = MatrixSlice(
        shape=list(data_slice.shape),
        dtype=str(data_slice.dtype),
        is_sparse=is_sparse,
        rows_returned=data_slice.shape[0],
        cols_returned=data_slice.shape[1],
        total_rows=total_rows,
        total_cols=total_cols,
    )

    # Handle sparse matrices
    if is_sparse:
        sparse_info = get_sparse_info(data_slice)
        result.sparse_format = sparse_info["format"]
        result.sparsity = sparse_info.get("sparsity")

        if return_dense:
            # Convert to dense for easier consumption
            data_dense = data_slice.toarray()
            result.data = convert_to_serializable(data_dense)
        else:
            # Return sparse format (CSR/CSC)
            if hasattr(data_slice, "data") and hasattr(data_slice, "indices") and hasattr(data_slice, "indptr"):
                result.indices = convert_to_serializable(data_slice.indices)
                result.indptr = convert_to_serializable(data_slice.indptr)
                result.data = None  # Sparse representation provided via indices/indptr
    else:
        # Dense matrix
        result.data = convert_to_serializable(data_slice)

    return result


@mcp.tool
def get_obsm_data(
    path: str,
    key: str,
    row_slice: str | None = None,
    col_slice: str | None = None,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Retrieve embedding or multi-dimensional cell annotations from .obsm.

    **Use this when you need to:**
    - Get UMAP or t-SNE coordinates for visualization
    - Retrieve PCA components for dimensionality reduction analysis
    - Access other multi-dimensional cell embeddings
    - Answer questions like "Get UMAP coordinates for all cells"

    **Example usage:**
    - "Get UMAP coordinates" → get_obsm_data(path, "X_umap")
    - "Get first 3 PCA components for first 100 cells" → get_obsm_data(path, "X_pca", row_slice="0:100", col_slice="0:3")
    - "Show t-SNE coordinates" → get_obsm_data(path, "X_tsne")

    **Note:** Use list_available_keys(path, "obsm") first to see available embeddings.

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    key : str
        Key in obsm (e.g., 'X_pca', 'X_umap', 'X_tsne'). Check available keys first.
    row_slice : str, optional
        Row (cell) slice string (e.g., '0:100' for first 100 cells)
    col_slice : str, optional
        Column slice for components (e.g., '0:2' for first 2 UMAP dimensions, '0:10' for first 10 PCs)
    max_rows : int
        Maximum number of rows if not sliced (default: 100)

    Returns
    -------
    dict[str, Any]
        Dictionary with 'data' (2D array), 'shape', 'dtype', 'total_shape', etc.

    Examples
    --------
    Get UMAP coordinates for visualization:
    >>> result = get_obsm_data("pbmc3k.h5ad", "X_umap", row_slice="0:100")
    >>> # Returns: {'data': [[3.2, -1.5], [3.4, -1.2], ...], 'shape': [100, 2], ...}
    >>> # Use with cluster labels from get_obs_data() to color points
    """
    adata = read_lazy(path)

    if key not in adata.obsm:
        raise ValueError(f"Key '{key}' not found in obsm. Available keys: {list(adata.obsm.keys())}")

    data = adata.obsm[key]
    total_shape = list(data.shape) if hasattr(data, "shape") else [len(data), 1]
    total_rows = total_shape[0]
    total_cols = total_shape[1] if len(total_shape) > 1 else 1

    # Parse slices
    row_slice_obj = _parse_slice(row_slice, total_rows) if row_slice else slice(0, min(max_rows, total_rows))

    # Slice the data
    if len(total_shape) > 1:
        col_slice_obj = _parse_slice(col_slice, total_cols) if col_slice else slice(None)
        data_slice = data[row_slice_obj, col_slice_obj]
    else:
        data_slice = data[row_slice_obj]

    result_shape = list(data_slice.shape) if hasattr(data_slice, "shape") else [len(data_slice), 1]

    return {
        "data": convert_to_serializable(data_slice),
        "shape": result_shape,
        "total_shape": total_shape,
        "dtype": str(data.dtype) if hasattr(data, "dtype") else "unknown",
        "rows_returned": result_shape[0],
        "cols_returned": result_shape[1] if len(result_shape) > 1 else 1,
        "total_rows": total_rows,
        "total_cols": total_cols,
    }


@mcp.tool
def get_varm_data(
    path: str,
    key: str,
    row_slice: str | None = None,
    col_slice: str | None = None,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Retrieve multi-dimensional gene annotations from .varm.

    **Use this when you need to:**
    - Get PCA loadings for genes
    - Access multi-dimensional gene embeddings
    - Retrieve gene-level dimensional reduction data
    - Check what multi-dimensional gene annotations are available

    **Example usage:**
    - "Get PCA loadings for genes" → get_varm_data(path, "PCs")
    - "Show gene embeddings" → get_varm_data(path, key, row_slice="0:100")

    **Note:** .varm is less commonly used than .obsm. Use list_available_keys(path, "varm") to check availability.

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    key : str
        Key in varm (check with list_available_keys first)
    row_slice : str, optional
        Row (gene) slice string
    col_slice : str, optional
        Column slice for components
    max_rows : int
        Maximum number of rows if not sliced (default: 100)

    Returns
    -------
    dict[str, Any]
        Dictionary with 'data' (2D array), 'shape', 'dtype', etc.

    Examples
    --------
    Get PCA loadings:
    >>> result = get_varm_data("pbmc3k.h5ad", "PCs", row_slice="0:50", col_slice="0:5")
    >>> # Returns: {'data': [[0.02, -0.01, ...], ...], 'shape': [50, 5], ...}
    """
    adata = read_lazy(path)

    if key not in adata.varm:
        raise ValueError(f"Key '{key}' not found in varm. Available keys: {list(adata.varm.keys())}")

    data = adata.varm[key]
    total_shape = list(data.shape) if hasattr(data, "shape") else [len(data), 1]
    total_rows = total_shape[0]
    total_cols = total_shape[1] if len(total_shape) > 1 else 1

    # Parse slices
    row_slice_obj = _parse_slice(row_slice, total_rows) if row_slice else slice(0, min(max_rows, total_rows))

    # Slice the data
    if len(total_shape) > 1:
        col_slice_obj = _parse_slice(col_slice, total_cols) if col_slice else slice(None)
        data_slice = data[row_slice_obj, col_slice_obj]
    else:
        data_slice = data[row_slice_obj]

    result_shape = list(data_slice.shape) if hasattr(data_slice, "shape") else [len(data_slice), 1]

    return {
        "data": convert_to_serializable(data_slice),
        "shape": result_shape,
        "total_shape": total_shape,
        "dtype": str(data.dtype) if hasattr(data, "dtype") else "unknown",
        "rows_returned": result_shape[0],
        "cols_returned": result_shape[1] if len(result_shape) > 1 else 1,
        "total_rows": total_rows,
        "total_cols": total_cols,
    }


@mcp.tool
def get_uns_data(path: str, key: str | None = None) -> dict[str, Any]:
    """Retrieve unstructured annotations from .uns.

    **Use this when you need to:**
    - Access analysis parameters (e.g., neighbor graph parameters)
    - Get color palettes for clusters
    - Retrieve any unstructured metadata
    - Check what additional information is stored

    **Example usage:**
    - "What's stored in uns?" → get_uns_data(path)  # Returns all keys
    - "Get cluster colors" → get_uns_data(path, "louvain_colors")
    - "Show analysis parameters" → get_uns_data(path, "neighbors")

    **Note:** .uns contains miscellaneous data that doesn't fit into other AnnData attributes.

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).
    key : str, optional
        Specific key to retrieve. If None, returns overview of all keys with their types.

    Returns
    -------
    dict[str, Any]
        If key=None: Dictionary of all keys with their types
        If key specified: Dictionary with 'key', 'type', and 'value'

    Examples
    --------
    List all unstructured data:
    >>> result = get_uns_data("pbmc3k.h5ad")
    >>> # Returns: {'louvain': {'type': 'dict', 'value': {...}}, 'louvain_colors': {'type': 'ndarray', ...}, ...}

    Get specific item:
    >>> result = get_uns_data("pbmc3k.h5ad", "louvain_colors")
    >>> # Returns: {'key': 'louvain_colors', 'type': 'ndarray', 'value': ['#1f77b4', '#ff7f0e', ...]}
    """
    adata = read_lazy(path)

    if key is None:
        # Return all keys and their types
        result = {}
        for k in adata.uns.keys():
            v = adata.uns[k]
            result[k] = {
                "type": type(v).__name__,
                "value": convert_to_serializable(v) if not hasattr(v, "shape") else f"Array with shape {v.shape}",
            }
        return result
    else:
        if key not in adata.uns:
            raise ValueError(f"Key '{key}' not found in uns. Available keys: {list(adata.uns.keys())}")

        value = adata.uns[key]
        return {
            "key": key,
            "type": type(value).__name__,
            "value": convert_to_serializable(value),
        }


def safe_to_list(data: Any, max_items: int = 1000) -> list:
    """Safely convert data to a list."""
    if hasattr(data, "tolist"):
        data = data.tolist()
    elif isinstance(data, (list, tuple)):
        data = list(data)
    else:
        data = [data]

    if len(data) > max_items:
        return data[:max_items]
    return data
