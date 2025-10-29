from anndata.experimental import read_lazy
from pydantic import BaseModel

from anndata_mcp.mcp import mcp


class AnnDataSummary(BaseModel):
    """Summary information about an AnnData object."""

    n_obs: int
    n_vars: int
    obs_columns: list[str]
    var_columns: list[str]
    uns_keys: list[str]
    obsm_keys: list[str]
    obsp_keys: list[str]
    varm_keys: list[str]
    varp_keys: list[str]
    layers: list[str]


@mcp.tool
def get_anndata_summary(path: str) -> AnnDataSummary:
    """Get a high-level overview of an AnnData file structure.

    **ALWAYS START HERE when exploring a new AnnData file!**

    **Use this when you need to:**
    - Understand what data is available in an AnnData file
    - See what metadata columns exist before querying specific data
    - Check what embeddings (UMAP, PCA) are available for visualization
    - List all available layers and keys before accessing them

    **Example usage:**
    - "What's in this AnnData file?" → get_anndata_summary(path)
    - "Show me the dataset structure" → get_anndata_summary(path)
    - "What embeddings are available?" → get_anndata_summary(path) then check obsm_keys

    Parameters
    ----------
    path : str
        Path to the AnnData file (.h5ad or .zarr).

    Returns
    -------
    AnnDataSummary
        Summary containing:
        - n_obs: Number of observations (cells)
        - n_vars: Number of variables (genes)
        - obs_columns: Cell metadata column names
        - var_columns: Gene metadata column names
        - obsm_keys: Available embeddings (e.g., 'X_umap', 'X_pca')
        - layers: Available data layers
        - uns_keys: Unstructured annotation keys

    Examples
    --------
    Start exploring a dataset:
    >>> summary = get_anndata_summary("pbmc3k.h5ad")
    >>> # Returns: AnnDataSummary(n_obs=2638, n_vars=1838, obs_columns=['n_genes', 'louvain', ...])
    >>> # Next: Use get_dataframe_info() to explore specific columns
    """
    adata = read_lazy(path)

    # Get columns - handle Dataset2D objects
    obs_cols = list(adata.obs.columns) if hasattr(adata.obs, "columns") else []
    var_cols = list(adata.var.columns) if hasattr(adata.var, "columns") else []

    return AnnDataSummary(
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
        obs_columns=obs_cols,
        var_columns=var_cols,
        uns_keys=list(adata.uns.keys()) if adata.uns else [],
        obsm_keys=list(adata.obsm.keys()) if adata.obsm else [],
        obsp_keys=list(adata.obsp.keys()) if adata.obsp else [],
        varm_keys=list(adata.varm.keys()) if adata.varm else [],
        varp_keys=list(adata.varp.keys()) if adata.varp else [],
        layers=list(adata.layers.keys()) if adata.layers else [],
    )
