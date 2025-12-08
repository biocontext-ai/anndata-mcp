import gc
from typing import Annotated, Literal

import numpy as np
from anndata._core.xarray import Dataset2D
from dask.array.core import Array
from pydantic import BaseModel, Field

from anndata_mcp.tools._exploration import create_dataframe_mask_from_tuple
from anndata_mcp.tools.utils import (
    extract_data_from_dask_array,
    extract_data_from_dask_array_with_indices,
    extract_data_from_dataset2d,
    extract_original_type_string,
    get_nested_key,
    get_shape_str,
    match_patterns,
    read_lazy_general,
    truncate_string,
)


class DataView(BaseModel):
    data: Annotated[
        str | None,
        Field(
            description="The data to view, e.g. a slice of a pandas.DataFrame or a numpy array in csv format. Other data types are converted to a plain string."
        ),
    ] = None
    data_type: Annotated[str | None, Field(description="The original type of the data")] = None
    slice_shape: Annotated[
        str | None, Field(description="The shape of the data after slicing, if applicable, otherwise 'NA'")
    ] = None
    full_shape: Annotated[
        str | None, Field(description="The full shape of the data, before slicing, if applicable, otherwise 'NA'")
    ] = None
    error: Annotated[str | None, Field(description="Any error message")] = None


def view_raw_data(
    path: Annotated[str, Field(description="Absolute path or URL to the AnnData file")],
    attribute: Annotated[
        Literal["X", "obs", "var", "obsm", "varm", "obsp", "varp", "uns", "layers"],
        Field(description="The attribute to view"),
    ],
    key: Annotated[
        str | list[str] | None,
        Field(
            description="The key of the attribute value to view. Can be a single string or a list of strings for nested key retrieval (e.g., ['key1', 'key2'] to access attr_obj['key1']['key2'])."
        ),
    ] = None,
    columns_or_genes: Annotated[
        list[str] | None,
        Field(
            description="Column names or gene names to select. For pandas.DataFrame attributes (e.g., obs, var), these are column names. For 'X' or 'layers' attributes, these are gene names (from var_names) and are used instead of col_start_index/col_stop_index. If None, the entire attribute is considered or col_start_index/col_stop_index is used. Also accepts glob-like patterns as input, e.g. ['RE*', 'CD4*'].",
        ),
    ] = None,
    row_start_index: Annotated[
        int,
        Field(
            description="The start index for the row slice. Only applied to attributes or attribute values with a suitable type."
        ),
    ] = 0,
    row_stop_index: Annotated[
        int,
        Field(
            description="The stop index for the row slice. Only applied to attributes or attribute values with a suitable type."
        ),
    ] = 5,
    col_start_index: Annotated[
        int,
        Field(
            description="The start index for the column slice. Only applied to attributes or attribute values with a suitable type."
        ),
    ] = 0,
    col_stop_index: Annotated[
        int,
        Field(
            description="The stop index for the column slice. Only applied to attributes or attribute values with a suitable type."
        ),
    ] = 5,
    filter_column: Annotated[
        str | None,
        Field(
            description="The column name of the dataframe to filter by. Only applicable when the selected attribute (or attribute value) is a dataframe. Must be provided TOGETHER with filter_operator and filter_value."
        ),
    ] = None,
    filter_operator: Annotated[
        Literal["==", "!=", ">", ">=", "<", "<=", "isin", "notin"] | None,
        Field(description="The operator to use for the dataframe filter."),
    ] = None,
    filter_value: Annotated[
        list[str | float | bool] | str | float | bool | None,
        Field(description="The value(s) to filter the dataframe by."),
    ] = None,
) -> DataView:
    """View the raw data of an AnnData object."""
    error = None
    data = None
    data_type = None
    slice_shape = None
    full_shape = None
    adata = None

    try:
        row_slice = slice(row_start_index, row_stop_index, None)
        col_slice = slice(col_start_index, col_stop_index, None)

        adata = read_lazy_general(path)

        attr_obj = getattr(adata, attribute, None)
        if attr_obj is None:
            raise KeyError(f"Attribute {attribute} not found")

        if key is not None:
            try:
                # Convert single string to list for consistent handling
                key_list = [key] if isinstance(key, str) else key
                attr_obj = get_nested_key(attr_obj, key_list)
            except (KeyError, AttributeError) as err:
                key_str = key if isinstance(key, str) else " -> ".join(key)
                raise KeyError(f"Attribute {attribute} with key {key_str} not found") from err

        # Apply filter if provided - filters only the dataframe being viewed, not the whole AnnData
        # This allows showing specific parts of any dataframe attribute (e.g., filtered rows of obs or var)
        if any([filter_column is not None, filter_operator is not None, filter_value is not None]):
            # Validate that all filter parameters are provided together
            missing = [
                name
                for name, val in [
                    ("filter_column", filter_column),
                    ("filter_operator", filter_operator),
                    ("filter_value", filter_value),
                ]
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"If any filter parameter is provided, all must be provided. Missing: {', '.join(missing)}"
                )

            # Construct filter tuple
            filter_tuple = (filter_column, filter_operator, filter_value)

            if isinstance(attr_obj, Dataset2D):
                mask = create_dataframe_mask_from_tuple(attr_obj, filter_tuple)
                # Convert mask to numpy boolean array, then to integer indices for Dataset2D indexing
                if hasattr(mask, "compute"):
                    mask = mask.compute()
                if hasattr(mask, "values"):
                    mask_bool = np.asarray(mask.values, dtype=bool)
                else:
                    mask_bool = np.asarray(mask, dtype=bool)
                # Convert boolean mask to integer indices for .iloc indexing
                indices = np.where(mask_bool)[0]
                if len(indices) == 0:
                    raise ValueError("Filtering resulted in an empty dataframe: no rows remain after filtering")
                attr_obj = attr_obj.iloc[indices]  # Filter only this dataframe, not the whole AnnData
            else:
                raise ValueError(
                    "Filtering can only be applied to dataframe or dataframe-like attributes (e.g., obs, var)"
                )

        if isinstance(attr_obj, Dataset2D):
            # Use columns_or_genes as column names for dataframe, or all columns if None
            available_columns = attr_obj.columns.tolist()
            selected_columns = (
                match_patterns(available_columns, columns_or_genes)[0]
                if columns_or_genes is not None
                else available_columns
            )
            if len(selected_columns) == 0:
                raise ValueError("None of the provided columns were found in the attribute")

            data, slice_shape = extract_data_from_dataset2d(
                attr_obj, selected_columns, row_slice, index=True, return_shape=True
            )
            full_shape = str(attr_obj.shape)
        elif isinstance(attr_obj, Array):
            if attribute in ("X", "layers") and columns_or_genes is not None:
                # Convert gene names to indices for X and layers
                var_names = adata.var_names.tolist()
                columns_or_genes, _ = match_patterns(var_names, columns_or_genes)
                if len(columns_or_genes) == 0:
                    raise ValueError("None of the provided genes were found in var_names")

                gene_indices = [var_names.index(gene) for gene in columns_or_genes]
                data, slice_shape = extract_data_from_dask_array_with_indices(
                    attr_obj, row_slice, gene_indices, return_shape=True
                )
            else:
                data, slice_shape = extract_data_from_dask_array(attr_obj, row_slice, col_slice, return_shape=True)
            full_shape = str(attr_obj.shape)
        else:
            data = (
                "Entries: "
                + ", ".join(
                    [
                        f"{key}: {extract_original_type_string(attr_obj[key], full_name=True)} {get_shape_str(attr_obj[key])}"
                        for key in attr_obj.keys()
                    ]
                )
                if hasattr(attr_obj, "keys")
                else str(attr_obj)
            )

        data_type = extract_original_type_string(attr_obj, full_name=True)

    except Exception as e:  # noqa: BLE001
        # Catch all exceptions to ensure function always returns DataView
        # This is intentional for API stability - all errors are returned in the error field
        error = truncate_string(str(e), max_output_len=100)
    finally:
        if adata is not None:
            adata.file.close()
            del adata
            gc.collect()

    return DataView(data=data, data_type=data_type, slice_shape=slice_shape, full_shape=full_shape, error=error)
