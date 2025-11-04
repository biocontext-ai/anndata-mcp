from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field


class FilePaths(BaseModel):
    paths: list[Annotated[Path, Field(description="Absolute paths to the AnnData files (.h5ad or .zarr)")]]


def locate_anndata_stores(
    data_dir: Annotated[Path, Field(description="Absolute path to the data directory")],
    recursive: Annotated[bool, Field(description="Whether to search recursively for AnnData stores", default=True)],
) -> FilePaths:
    """Locate all AnnData stores (.h5ad or .zarr) in a data directory."""
    prefix = "**/" if recursive else ""
    paths = list(data_dir.glob(f"{prefix}*.h5ad")) + list(data_dir.glob(f"{prefix}*.zarr"))
    return FilePaths(paths=paths)
