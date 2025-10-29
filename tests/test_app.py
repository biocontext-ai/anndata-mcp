"""Tests for AnnData MCP server."""

import pytest
from fastmcp import Client

import anndata_mcp
from anndata_mcp.tools import (
    get_anndata_summary,
    get_attribute_info,
    get_column_stats,
    get_dataframe_info,
    get_grouped_stats,
    get_obs_data,
    get_obsm_data,
    get_unique_values,
    get_value_counts,
    get_var_data,
    get_X_data,
    list_available_keys,
)


def test_package_has_version():
    """Test that package version exists."""
    assert anndata_mcp.__version__ is not None


# =============================================================================
# Direct function call tests (using .fn)
# =============================================================================


def test_get_anndata_summary_direct(test_data_path):
    """Test get_anndata_summary with direct function call."""
    summary = get_anndata_summary(test_data_path)

    assert summary.n_obs == 2638
    assert summary.n_vars == 1838
    assert "n_genes" in summary.obs_columns
    assert "louvain" in summary.obs_columns
    assert "n_counts" in summary.obs_columns
    assert "X_umap" in summary.obsm_keys
    assert "X_pca" in summary.obsm_keys


def test_get_dataframe_info_direct(test_data_path):
    """Test get_dataframe_info with direct function call."""
    # Test obs
    obs_info = get_dataframe_info(test_data_path, "obs")
    assert obs_info.n_rows == 2638
    assert obs_info.n_cols == 4
    assert any(col.name == "louvain" for col in obs_info.columns)
    assert any(col.name == "n_genes" for col in obs_info.columns)

    # Test var
    var_info = get_dataframe_info(test_data_path, "var")
    assert var_info.n_rows == 1838
    assert var_info.n_cols >= 1


def test_get_unique_values_direct(test_data_path):
    """Test get_unique_values with direct function call."""
    result = get_unique_values(test_data_path, "obs", "louvain")

    assert result.n_unique == 8
    assert len(result.unique_values) == 8
    assert "CD4 T cells" in result.unique_values
    assert "B cells" in result.unique_values
    assert not result.truncated


def test_get_column_stats_direct(test_data_path):
    """Test get_column_stats with direct function call."""
    stats = get_column_stats(test_data_path, "obs", "n_genes")

    assert stats.dtype == "int64"
    assert stats.n_nulls == 0
    assert "mean" in stats.stats
    assert "median" in stats.stats
    assert "min" in stats.stats
    assert "max" in stats.stats
    assert stats.stats["min"] > 0
    assert stats.stats["max"] > stats.stats["mean"]


def test_get_value_counts_direct(test_data_path):
    """Test get_value_counts with direct function call."""
    counts = get_value_counts(test_data_path, "obs", "louvain")

    assert "counts" in counts
    assert "total" in counts
    assert counts["total"] == 2638
    assert len(counts["counts"]) == 8
    assert "CD4 T cells" in counts["counts"]
    assert counts["counts"]["CD4 T cells"] > 0


def test_get_grouped_stats_direct(test_data_path):
    """Test get_grouped_stats with direct function call."""
    grouped = get_grouped_stats(test_data_path, "obs", "louvain", "n_genes")

    assert "groups" in grouped
    assert grouped["n_groups"] == 8
    assert "CD4 T cells" in grouped["groups"]

    cd4_stats = grouped["groups"]["CD4 T cells"]
    assert "mean" in cd4_stats
    assert "median" in cd4_stats
    assert "count" in cd4_stats
    assert cd4_stats["count"] > 0
    assert cd4_stats["mean"] > 0


def test_get_obs_data_direct(test_data_path):
    """Test get_obs_data with direct function call."""
    # Test with slice
    result = get_obs_data(test_data_path, row_slice="0:10", columns=["louvain", "n_genes"])

    assert result["rows_returned"] == 10
    assert result["cols_returned"] == 2
    assert len(result["data"]) == 10
    assert "louvain" in result["data"][0]
    assert "n_genes" in result["data"][0]

    # Test without slice (should use max_rows)
    result2 = get_obs_data(test_data_path, max_rows=5)
    assert result2["rows_returned"] == 5


def test_get_var_data_direct(test_data_path):
    """Test get_var_data with direct function call."""
    result = get_var_data(test_data_path, row_slice="0:20")

    assert result["rows_returned"] == 20
    assert result["total_rows"] == 1838
    assert len(result["data"]) == 20


def test_get_X_data_direct(test_data_path):
    """Test get_X_data with direct function call."""
    result = get_X_data(test_data_path, row_slice="0:5", col_slice="0:5")

    assert result.shape == [5, 5]
    assert result.rows_returned == 5
    assert result.cols_returned == 5
    assert result.total_rows == 2638
    assert result.total_cols == 1838
    assert result.data is not None
    assert len(result.data) == 5


def test_get_obsm_data_direct(test_data_path):
    """Test get_obsm_data with direct function call."""
    # Test UMAP
    umap_result = get_obsm_data(test_data_path, "X_umap", row_slice="0:10")

    assert umap_result["rows_returned"] == 10
    assert umap_result["shape"][1] == 2  # UMAP has 2 dimensions
    assert len(umap_result["data"]) == 10
    assert len(umap_result["data"][0]) == 2

    # Test PCA with column slice
    pca_result = get_obsm_data(test_data_path, "X_pca", row_slice="0:10", col_slice="0:3")

    assert pca_result["rows_returned"] == 10
    assert pca_result["cols_returned"] == 3
    assert len(pca_result["data"][0]) == 3


def test_list_available_keys_direct(test_data_path):
    """Test list_available_keys with direct function call."""
    # Test obsm
    obsm_keys = list_available_keys(test_data_path, "obsm")
    assert "X_umap" in obsm_keys
    assert "X_pca" in obsm_keys

    # Test layers
    layer_keys = list_available_keys(test_data_path, "layers")
    assert isinstance(layer_keys, list)


def test_get_attribute_info_direct(test_data_path):
    """Test get_attribute_info with direct function call."""
    # Test X
    x_info = get_attribute_info(test_data_path, "X")
    assert x_info.name == "X"
    assert x_info.shape == [2638, 1838]

    # Test obsm
    obsm_info = get_attribute_info(test_data_path, "obsm")
    assert obsm_info.name == "obsm"
    assert obsm_info.keys is not None
    assert "X_umap" in obsm_info.keys


# =============================================================================
# MCP Client tests (without .fn)
# =============================================================================


@pytest.mark.asyncio
async def test_get_anndata_summary_mcp(test_data_path):
    """Test get_anndata_summary through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        result = await client.call_tool("get_anndata_summary", {"path": test_data_path})

        # Access as attributes
        assert result.data.n_obs == 2638
        assert result.data.n_vars == 1838
        assert "n_genes" in result.data.obs_columns
        assert "louvain" in result.data.obs_columns


@pytest.mark.asyncio
async def test_get_dataframe_info_mcp(test_data_path):
    """Test get_dataframe_info through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        result = await client.call_tool("get_dataframe_info", {"path": test_data_path, "attribute": "obs"})

        # Access as attributes
        assert result.data.n_rows == 2638
        assert result.data.n_cols == 4
        assert any(col.name == "louvain" for col in result.data.columns)


@pytest.mark.asyncio
async def test_get_unique_values_mcp(test_data_path):
    """Test get_unique_values through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        result = await client.call_tool(
            "get_unique_values", {"path": test_data_path, "attribute": "obs", "column": "louvain"}
        )

        # Access as attributes
        assert result.data.n_unique == 8
        assert "CD4 T cells" in result.data.unique_values


@pytest.mark.asyncio
async def test_get_value_counts_mcp(test_data_path):
    """Test get_value_counts through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        result = await client.call_tool(
            "get_value_counts", {"path": test_data_path, "attribute": "obs", "column": "louvain"}
        )

        data = result.data
        assert data["total"] == 2638
        assert len(data["counts"]) == 8
        assert "CD4 T cells" in data["counts"]


@pytest.mark.asyncio
async def test_get_grouped_stats_mcp(test_data_path):
    """Test get_grouped_stats through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        result = await client.call_tool(
            "get_grouped_stats",
            {"path": test_data_path, "attribute": "obs", "group_by": "louvain", "value_column": "n_genes"},
        )

        data = result.data
        assert data["n_groups"] == 8
        assert "CD4 T cells" in data["groups"]
        assert "mean" in data["groups"]["CD4 T cells"]


@pytest.mark.asyncio
async def test_get_obs_data_mcp(test_data_path):
    """Test get_obs_data through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        result = await client.call_tool(
            "get_obs_data", {"path": test_data_path, "row_slice": "0:10", "columns": ["louvain", "n_genes"]}
        )

        data = result.data
        assert data["rows_returned"] == 10
        assert len(data["data"]) == 10


@pytest.mark.asyncio
async def test_get_X_data_mcp(test_data_path):
    """Test get_X_data through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        result = await client.call_tool(
            "get_X_data", {"path": test_data_path, "row_slice": "0:5", "col_slice": "0:5"}
        )

        # Access as attributes
        assert result.data.shape == [5, 5]
        assert result.data.rows_returned == 5
        assert result.data.cols_returned == 5


@pytest.mark.asyncio
async def test_get_obsm_data_mcp(test_data_path):
    """Test get_obsm_data through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        result = await client.call_tool(
            "get_obsm_data", {"path": test_data_path, "key": "X_umap", "row_slice": "0:10"}
        )

        data = result.data
        assert data["rows_returned"] == 10
        assert data["shape"][1] == 2


@pytest.mark.asyncio
async def test_list_available_keys_mcp(test_data_path):
    """Test list_available_keys through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        result = await client.call_tool("list_available_keys", {"path": test_data_path, "attribute": "obsm"})

        keys = result.data
        assert "X_umap" in keys
        assert "X_pca" in keys


# =============================================================================
# Error handling tests
# =============================================================================


def test_invalid_column_error(test_data_path):
    """Test that invalid column names raise errors."""
    with pytest.raises(ValueError, match="Column 'invalid_column' not found"):
        get_unique_values(test_data_path, "obs", "invalid_column")


def test_invalid_attribute_error(test_data_path):
    """Test that invalid attribute names raise errors."""
    with pytest.raises(ValueError, match="Attribute must be 'obs' or 'var'"):
        get_dataframe_info(test_data_path, "invalid_attr")


def test_invalid_key_error(test_data_path):
    """Test that invalid obsm keys raise errors."""
    with pytest.raises(ValueError, match="Key 'invalid_key' not found in obsm"):
        get_obsm_data(test_data_path, "invalid_key")


def test_numeric_column_required_error(test_data_path):
    """Test that grouped_stats requires numeric column."""
    with pytest.raises(ValueError, match="must be numeric"):
        get_grouped_stats(test_data_path, "obs", "louvain", "louvain")


# =============================================================================
# Edge case tests
# =============================================================================


def test_large_slice(test_data_path):
    """Test slicing with large ranges."""
    result = get_obs_data(test_data_path, row_slice="0:2638")
    assert result["rows_returned"] == 2638


def test_slice_with_step(test_data_path):
    """Test slicing with step parameter."""
    result = get_X_data(test_data_path, row_slice="0:10:2", col_slice="0:10:2")
    assert result.rows_returned == 5  # 0, 2, 4, 6, 8
    assert result.cols_returned == 5


def test_max_values_truncation(test_data_path):
    """Test that unique values can be truncated."""
    # n_genes has many unique values
    result = get_unique_values(test_data_path, "obs", "n_genes", max_values=10)
    assert len(result.unique_values) == 10
    assert result.truncated


def test_empty_columns_list(test_data_path):
    """Test with specific columns specified."""
    result = get_obs_data(test_data_path, row_slice="0:5", columns=["n_genes"])
    assert result["cols_returned"] == 1
    assert len(result["data"][0]) == 1


# =============================================================================
# Integration tests
# =============================================================================


def test_workflow_integration(test_data_path):
    """Test a typical analysis workflow."""
    # 1. Get summary
    summary = get_anndata_summary(test_data_path)
    assert summary.n_obs > 0

    # 2. Check what's in obs
    obs_info = get_dataframe_info(test_data_path, "obs")
    assert any(col.name == "louvain" for col in obs_info.columns)

    # 3. Get unique clusters
    clusters = get_unique_values(test_data_path, "obs", "louvain")
    assert clusters.n_unique > 0

    # 4. Count cells per cluster
    counts = get_value_counts(test_data_path, "obs", "louvain")
    assert counts["total"] == summary.n_obs

    # 5. Get stats per cluster
    grouped = get_grouped_stats(test_data_path, "obs", "louvain", "n_genes")
    assert grouped["n_groups"] == clusters.n_unique

    # 6. Get UMAP coordinates
    umap = get_obsm_data(test_data_path, "X_umap", row_slice="0:100")
    assert umap["rows_returned"] == 100


@pytest.mark.asyncio
async def test_mcp_workflow_integration(test_data_path):
    """Test a typical analysis workflow through MCP client."""
    async with Client(anndata_mcp.mcp) as client:
        # 1. Get summary
        summary_result = await client.call_tool("get_anndata_summary", {"path": test_data_path})
        assert summary_result.data.n_obs == 2638

        # 2. Get value counts (returns dict, not Pydantic model)
        counts_result = await client.call_tool(
            "get_value_counts", {"path": test_data_path, "attribute": "obs", "column": "louvain"}
        )
        assert counts_result.data["total"] == 2638

        # 3. Get grouped stats (returns dict, not Pydantic model)
        grouped_result = await client.call_tool(
            "get_grouped_stats",
            {"path": test_data_path, "attribute": "obs", "group_by": "louvain", "value_column": "n_genes"},
        )
        assert grouped_result.data["n_groups"] == 8
