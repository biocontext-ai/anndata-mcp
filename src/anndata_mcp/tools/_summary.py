import gc
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from anndata_mcp.tools.utils import (
    _is_url,
    extract_original_type_string,
    get_shape_str,
    read_lazy_general,
    truncate_string,
)


class AnnDataSummary(BaseModel):
    n_obs: Annotated[int | None, Field(description="The number of observations (cells) in the AnnData object")] = None
    n_vars: Annotated[int | None, Field(description="The number of variables (genes) in the AnnData object")] = None
    X_type: Annotated[tuple[str, str] | None, Field(description="The type and dtype of the X attribute")] = None
    obs_columns: Annotated[
        list[tuple[str, str]] | None, Field(description="The columns of the obs dataframe and their dtypes")
    ] = None
    var_columns: Annotated[
        list[tuple[str, str]] | None, Field(description="The columns of the var dataframe and their dtypes")
    ] = None
    obsm_keys: Annotated[
        list[tuple[str, str, str]] | None,
        Field(description="The keys of the obsm attribute and their types and shapes"),
    ] = None
    varm_keys: Annotated[
        list[tuple[str, str, str]] | None,
        Field(description="The keys of the varm attribute and their types and shapes"),
    ] = None
    obsp_keys: Annotated[
        list[tuple[str, str]] | None,
        Field(description="The keys of the obsp attribute and their types, all with shape (n_obs, n_obs)"),
    ] = None
    varp_keys: Annotated[
        list[tuple[str, str]] | None,
        Field(description="The keys of the varp attribute and their types, all with shape (n_vars, n_vars)"),
    ] = None
    uns_keys: Annotated[
        list[tuple[str, str, str]] | None,
        Field(description="The keys of the uns attribute and their types and shapes (if available, otherwise 'NA')"),
    ] = None
    layers: Annotated[
        list[tuple[str, str]] | None,
        Field(description="The layers of the AnnData object (always arrays), all with shape (n_obs, n_vars)"),
    ] = None
    has_raw: Annotated[bool | None, Field(description="Whether the AnnData object has a raw attribute")] = None
    # attr_types: Annotated[dict[str, str], Field(description="The types of the attributes of the AnnData object")]
    last_modified: Annotated[
        datetime | None, Field(description="The last modified time of the AnnData file in UTC, or None for URLs")
    ] = None
    error: Annotated[str | None, Field(description="Any error message")] = None


def get_summary(
    path: Annotated[str, Field(description="Absolute path or URL to the AnnData file (.h5ad or .zarr)")],
) -> AnnDataSummary:
    """Get a summary of an AnnData object from a file or URL."""
    try:
        adata = read_lazy_general(path)

        # Get last_modified timestamp
        last_modified = None if _is_url(path) else datetime.fromtimestamp(Path(path).stat().st_mtime, tz=UTC)
        attributes = ["X", "obs", "var", "obsm", "varm", "obsp", "varp", "uns", "layers", "raw"]
        has_attribute = {attr: True if getattr(adata, attr, None) is not None else False for attr in attributes}

        # Close the file to release the file handle and allow the file to be edited
        summary = AnnDataSummary(
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
            X_type=(extract_original_type_string(adata.X, full_name=True), adata.X.dtype.name),
            obs_columns=[(col, adata.obs[col].dtype.name) for col in adata.obs.columns.tolist()],
            var_columns=[(col, adata.var[col].dtype.name) for col in adata.var.columns.tolist()],
            obsm_keys=[
                (key, extract_original_type_string(adata.obsm[key], full_name=True), get_shape_str(adata.obsm[key]))
                for key in adata.obsm.keys()
            ]
            if has_attribute["obsm"]
            else [],
            varm_keys=[
                (key, extract_original_type_string(adata.varm[key], full_name=True), get_shape_str(adata.varm[key]))
                for key in adata.varm.keys()
            ]
            if has_attribute["varm"]
            else [],
            obsp_keys=[
                (key, extract_original_type_string(adata.obsp[key], full_name=True)) for key in adata.obsp.keys()
            ]
            if has_attribute["obsp"]
            else [],
            varp_keys=[
                (key, extract_original_type_string(adata.varp[key], full_name=True)) for key in adata.varp.keys()
            ]
            if has_attribute["varp"]
            else [],
            uns_keys=[
                (key, extract_original_type_string(adata.uns[key], full_name=True), get_shape_str(adata.uns[key]))
                for key in adata.uns.keys()
            ]
            if has_attribute["uns"]
            else [],
            layers=[
                (key, extract_original_type_string(adata.layers[key], full_name=True)) for key in adata.layers.keys()
            ]
            if has_attribute["layers"]
            else [],
            # attr_types=attr_types,
            has_raw=has_attribute["raw"],
            last_modified=last_modified,
        )
        adata.file.close()
        del adata
        gc.collect()
    except Exception as e:  # noqa: BLE001
        # Catch all exceptions to ensure function always returns AnnDataSummary
        # This is intentional for API stability - all errors are returned in the error field
        error = truncate_string(str(e), max_output_len=100)
        summary = AnnDataSummary(error=error)
    return summary


def print_anndata_summary(summary: AnnDataSummary) -> str:
    """Return a nicely formatted string representation of the AnnData summary."""
    lines = [
        "AnnData Summary",
        "=" * 60,
        f"Observations (cells): {summary.n_obs:,}",
        f"Variables (genes): {summary.n_vars:,}",
        f"X type: {summary.X_type[0]} ({summary.X_type[1]})",
        f"Last modified: {summary.last_modified.strftime('%Y-%m-%d %H:%M:%S UTC') if summary.last_modified else 'N/A'}",
        f"Has raw: {summary.has_raw}",
        "",
    ]

    # obs columns
    if summary.obs_columns:
        lines.append(f"obs columns ({len(summary.obs_columns)}):")
        for col, dtype in summary.obs_columns:
            lines.append(f"  - {col}: {dtype}")
    else:
        lines.append("obs columns: (none)")

    # var columns
    if summary.var_columns:
        lines.append(f"\nvar columns ({len(summary.var_columns)}):")
        for col, dtype in summary.var_columns:
            lines.append(f"  - {col}: {dtype}")
    else:
        lines.append("\nvar columns: (none)")

    # obsm keys
    if summary.obsm_keys:
        lines.append(f"\nobsm keys ({len(summary.obsm_keys)}):")
        for key, dtype, shape in summary.obsm_keys:
            lines.append(f"  - {key}: {dtype} {shape}")
    else:
        lines.append("\nobsm keys: (none)")

    # varm keys
    if summary.varm_keys:
        lines.append(f"\nvarm keys ({len(summary.varm_keys)}):")
        for key, dtype, shape in summary.varm_keys:
            lines.append(f"  - {key}: {dtype} {shape}")
    else:
        lines.append("\nvarm keys: (none)")

    # obsp keys
    if summary.obsp_keys:
        lines.append(f"\nobsp keys ({len(summary.obsp_keys)}):")
        for key, dtype in summary.obsp_keys:
            lines.append(f"  - {key}: {dtype}")
    else:
        lines.append("\nobsp keys: (none)")

    # varp keys
    if summary.varp_keys:
        lines.append(f"\nvarp keys ({len(summary.varp_keys)}):")
        for key, dtype in summary.varp_keys:
            lines.append(f"  - {key}: {dtype}")
    else:
        lines.append("\nvarp keys: (none)")

    # uns keys
    if summary.uns_keys:
        lines.append(f"\nuns keys ({len(summary.uns_keys)}):")
        for key, dtype, shape in summary.uns_keys:
            lines.append(f"  - {key}: {dtype} {shape}")
    else:
        lines.append("\nuns keys: (none)")

    # layers
    if summary.layers:
        lines.append(f"\nlayers ({len(summary.layers)}):")
        for layer, dtype in summary.layers:
            lines.append(f"  - {layer}: {dtype}")
    else:
        lines.append("\nlayers: (none)")

    return "\n".join(lines)
