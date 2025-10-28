from anndata.experimental import read_lazy
from pydantic import BaseModel

from anndata_mcp.mcp import mcp


class AnnDataSummary(BaseModel):
    n_obs: int
    n_vars: int
    obs_columns: list[str]
    var_columns: list[str]
    uns_keys: list[str]
    obsm_keys: list[str]
    obsp_keys: list[str]
    varm_keys: list[str]
    layers: list[str]


@mcp.tool
def get_anndata_summary(path: str) -> AnnDataSummary:
    """Get a summary of an AnnData object from a file.

    Parameters
    ----------
    path : str
        Path to the AnnData file.

    Returns
    -------
    AnnDataSummary
        Summary of the AnnData object.
    """
    adata = read_lazy(path)
    return AnnDataSummary(
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
        obs_columns=adata.obs.columns.tolist(),
        var_columns=adata.var.columns.tolist(),
        uns_keys=list(adata.uns.keys()),
        obsm_keys=list(adata.obsm.keys()),
        obsp_keys=list(adata.obsp.keys()),
        varm_keys=list(adata.varm.keys()),
        layers=list(adata.layers.keys()),
    )
