"""Test helper functions for creating dummy AnnData objects."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def create_dummy_anndata(n_obs: int = 100, n_vars: int = 50, seed: int = 42) -> ad.AnnData:
    """Create a dummy AnnData object with various attributes for testing.

    Parameters
    ----------
    n_obs : int
        Number of observations (cells)
    n_vars : int
        Number of variables (genes)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    ad.AnnData
        A dummy AnnData object with various attributes populated
    """
    rng = np.random.default_rng(seed)

    # Create main X matrix
    X = rng.random((n_obs, n_vars))

    # Create obs dataframe with some columns
    obs = pd.DataFrame(
        {
            "n_genes": rng.integers(10, n_vars, size=n_obs),
            "percent_mito": rng.random(n_obs),
            "n_counts": rng.integers(1000, 10000, size=n_obs),
            "cell_type": rng.choice(["TypeA", "TypeB", "TypeC"], size=n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    # Create var dataframe with some columns
    var = pd.DataFrame(
        {
            "n_cells": rng.integers(5, n_obs, size=n_vars),
            "mean_counts": rng.random(n_vars),
        },
        index=[f"gene_{i}" for i in range(n_vars)],
    )

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Add obsm (observation matrices)
    adata.obsm["X_pca"] = rng.random((n_obs, 10))
    adata.obsm["X_umap"] = rng.random((n_obs, 2))
    adata.obsm["spatial"] = rng.random((n_obs, 2))

    # Add varm (variable matrices)
    adata.varm["PCs"] = rng.random((n_vars, 10))
    adata.varm["embeddings"] = rng.random((n_vars, 5))

    # Add obsp (observation sparse matrices)
    # Create sparse matrices for neighbors - create a simple diagonal-like structure
    obsp_indices = []
    obsp_data = []
    for _ in range(n_obs):
        # Each row has a few connections
        neighbors = rng.choice(n_obs, size=min(5, n_obs), replace=False)
        obsp_indices.extend(neighbors)
        obsp_data.extend(rng.random(len(neighbors)))
    obsp_indptr = np.append([0], np.cumsum([min(5, n_obs)] * n_obs))
    adata.obsp["connectivities"] = csr_matrix((obsp_data, obsp_indices, obsp_indptr), shape=(n_obs, n_obs))
    adata.obsp["distances"] = csr_matrix((obsp_data, obsp_indices, obsp_indptr), shape=(n_obs, n_obs))

    # Add varp (variable sparse matrices)
    varp_indices = []
    varp_data = []
    for _ in range(n_vars):
        # Each row has a few connections
        neighbors = rng.choice(n_vars, size=min(3, n_vars), replace=False)
        varp_indices.extend(neighbors)
        varp_data.extend(rng.random(len(neighbors)))
    varp_indptr = np.append([0], np.cumsum([min(3, n_vars)] * n_vars))
    adata.varp["neighbors"] = csr_matrix((varp_data, varp_indices, varp_indptr), shape=(n_vars, n_vars))

    # Add uns (unstructured data)
    adata.uns["array"] = rng.random((n_obs, 5))
    adata.uns["df"] = pd.DataFrame(rng.random((n_obs, 3)), columns=["col1", "col2", "col3"])
    adata.uns["dict"] = {"key1": "value1", "key2": 42}
    adata.uns["string"] = "test_string"

    # Add layers
    adata.layers["counts"] = rng.integers(0, 100, size=(n_obs, n_vars))
    adata.layers["normalized"] = rng.random((n_obs, n_vars))

    # Add raw (optional)
    adata.raw = adata[:, : n_vars // 2].copy()

    return adata
