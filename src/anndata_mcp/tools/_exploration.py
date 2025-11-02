from pathlib import Path
from typing import Annotated, Literal

from anndata._core.xarray import Dataset2D
from dask.array.core import Array
from pydantic import BaseModel, Field

from anndata_mcp.cache import read_lazy_with_cache
from anndata_mcp.mcp import mcp
from anndata_mcp.tools.utils import describe_dask_array, describe_dataset2d, truncate_string, value_counts_dataset2d


class ExplorationResult(BaseModel):
    description: Annotated[str | None, Field(description="The description of the attribute value")]
    value_counts: Annotated[str | None, Field(description="The value counts for the attribute value")]
    error: Annotated[str | None, Field(description="Any error message")]


@mcp.tool
def explore_data(
    path: Annotated[Path, Field(description="Absolute path to the AnnData file (.h5ad or .zarr)")],
    attribute: Annotated[Literal["obs", "var"], Field(description="The attribute to explore")] = "obs",
    key: Annotated[str | None, Field(description="The key of the attribute value to explore.", default=None)] = None,
    columns: Annotated[
        list[str] | None,
        Field(description="The columns to explore. If None, the entire dataset is considered."),
    ] = None,
    return_value_counts_for_categorical: Annotated[
        bool, Field(description="Whether to return the value counts for categorical columns.")
    ] = False,
) -> ExplorationResult:
    """Provide explorative information about the attribute values of an AnnData object."""
    adata = read_lazy_with_cache(path)

    attr_obj = getattr(adata, attribute, None)

    if key is not None and attr_obj is not None:
        try:
            attr_obj = attr_obj[key]
        except KeyError:
            adata.file.close()
            return f"Attribute {attribute} with key {key} not found"
    if isinstance(attr_obj, Dataset2D):
        description = describe_dataset2d(attr_obj, columns)
        description = truncate_string(description.to_csv())
        if return_value_counts_for_categorical:
            value_counts = value_counts_dataset2d(attr_obj, columns)
            value_counts = truncate_string(value_counts.to_csv())
        else:
            value_counts = None
        error = None
    elif isinstance(attr_obj, Array):
        description = describe_dask_array(attr_obj)
        description = truncate_string(description.to_csv())
        value_counts = None
        error = None
    else:
        adata.file.close()
        description = None
        value_counts = None
        error = (
            f"Attribute {attribute} is not a dataframe or array"
            if key is None
            else f"Attribute value of {attribute} for key {key} is not a dataframe or array"
        )
    adata.file.close()
    exploration_result = ExplorationResult(description=description, value_counts=value_counts, error=error)
    return exploration_result
