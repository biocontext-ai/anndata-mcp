from pathlib import Path
from typing import Annotated, Literal

from anndata._core.xarray import Dataset2D
from dask.array.core import Array
from pydantic import BaseModel, Field

from anndata_mcp.cache import read_lazy_with_cache
from anndata_mcp.mcp import mcp
from anndata_mcp.tools.utils import (
    extract_data_from_dask_array,
    extract_data_from_dataset2d,
    extract_original_type_string,
    get_shape_str,
)


class DataView(BaseModel):
    data: Annotated[
        str,
        Field(
            description="The data to view, e.g. a slice of a pandas.DataFrame or a numpy array in csv format. Other data types are converted to a plain string."
        ),
    ]
    data_type: Annotated[str, Field(description="The original type of the data")]
    slice_shape: Annotated[str, Field(description="The shape of the data after slicing, if applicable, otherwise 'NA'")]
    full_shape: Annotated[
        str, Field(description="The full shape of the data, before slicing, if applicable, otherwise 'NA'")
    ]


@mcp.tool
def view_data(
    path: Annotated[Path, Field(description="Absolute path to the AnnData file")],
    attribute: Annotated[
        Literal["X", "obs", "var", "obsm", "varm", "obsp", "varp", "uns", "layers"],
        Field(description="The attribute to view"),
    ],
    key: Annotated[str | None, Field(description="The key of the attribute value to view.", default=None)] = None,
    columns: Annotated[
        list[str] | None,
        Field(
            description="The columns to view. If None, the entire attribute is considered. Only applied to pandas.DataFrame attributes or attribute values.",
            default=None,
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
    return_index: Annotated[
        bool, Field(description="Whether to return the index for dataframe output", default=True)
    ] = True,
) -> DataView | str:
    """View the data of an AnnData object."""
    row_slice = slice(row_start_index, row_stop_index, None)
    col_slice = slice(col_start_index, col_stop_index, None)

    adata = read_lazy_with_cache(path)
    attr_obj = getattr(adata, attribute, None)
    if key is not None and attr_obj is not None:
        try:
            attr_obj = attr_obj[key]
        except KeyError:
            adata.file.close()
            return f"Attribute {attribute} with key {key} not found"

    attr_obj_type = extract_original_type_string(attr_obj, full_name=True)

    if isinstance(attr_obj, Dataset2D):
        data, slice_shape = extract_data_from_dataset2d(
            attr_obj, row_slice, columns or attr_obj.columns.tolist(), return_index, return_shape=True
        )
        full_shape = str(attr_obj.shape)
    elif isinstance(attr_obj, Array):
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
        slice_shape = "NA"
        full_shape = "NA"

    adata.file.close()
    return DataView(data=data, data_type=attr_obj_type, slice_shape=slice_shape, full_shape=full_shape)
