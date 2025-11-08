from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from anndata_mcp.tools.utils import truncate_string


class FilePaths(BaseModel):
    paths: Annotated[list[str] | None, Field(description="Absolute paths to the AnnData files (.h5ad or .zarr)")] = None
    error: Annotated[str | None, Field(description="Any error message")] = None


def locate_anndata_stores(
    data_dir: Annotated[Path, Field(description="Absolute path to the data directory")],
    recursive: Annotated[bool, Field(description="Whether to search recursively for AnnData stores", default=True)],
) -> FilePaths:
    """Locate all AnnData stores (.h5ad or .zarr) in a data directory."""
    try:
        prefix = "**/" if recursive else ""
        paths = list(data_dir.glob(f"{prefix}*.h5ad")) + list(data_dir.glob(f"{prefix}*.zarr"))
        # Convert Path objects to strings
        path_strings = [str(path) for path in paths]
        return FilePaths(paths=path_strings)
    except Exception as e:  # noqa: BLE001
        # Catch all exceptions to ensure function always returns FilePaths
        # This is intentional for API stability - all errors are returned in the error field
        error = truncate_string(str(e), max_output_len=100)
        return FilePaths(error=error)
