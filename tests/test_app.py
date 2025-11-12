"""Tests for anndata_mcp MCP server application."""

import pytest
from fastmcp import Client

import anndata_mcp
from anndata_mcp.mcp import mcp

from .helpers import create_dummy_anndata


def test_package_has_version():
    """Test that the package has a version."""
    assert anndata_mcp.__version__ is not None
    assert isinstance(anndata_mcp.__version__, str)


def test_mcp_server_initialized():
    """Test that the MCP server is properly initialized."""
    assert mcp is not None
    assert mcp.name == "anndata-mcp"


def test_mcp_server_has_tools():
    """Test that the MCP server has tools registered."""
    # Tools are registered when the module is imported
    # Check that we can access tools through the module
    from anndata_mcp import tools

    assert hasattr(tools, "__all__")
    assert len(tools.__all__) > 0

    # Verify that expected tools are available
    expected_tools = ["get_summary", "get_descriptive_stats", "view_raw_data"]
    for tool_name in expected_tools:
        assert tool_name in tools.__all__


@pytest.mark.asyncio
async def test_mcp_server_tools_work_with_dummy_data(tmp_path):
    """Test that MCP server tools work correctly with dummy AnnData."""
    # Create a dummy AnnData file
    adata = create_dummy_anndata()
    test_file = tmp_path / "test_anndata.h5ad"
    adata.write_h5ad(test_file)

    # Register tools with mcp instance (similar to how main.py does it)
    from anndata_mcp import tools

    for name in tools.__all__:
        tool_func = getattr(tools, name)
        mcp.tool(tool_func)

    # Test tools via MCP Client
    async with Client(mcp) as client:
        # Test get_summary tool
        result = await client.call_tool("get_summary", {"path": str(test_file)})
        assert result.data is not None
        summary = result.data
        assert summary.n_obs == 100
        assert summary.n_vars == 50
        assert summary.X_type is not None
        assert len(summary.obs_columns) > 0
        assert len(summary.var_columns) > 0

        # Test get_descriptive_stats tool
        result = await client.call_tool("get_descriptive_stats", {"path": str(test_file), "attribute": "X"})
        assert result.data is not None
        stats_result = result.data
        assert stats_result.error is None
        assert stats_result.description is not None

        # Test view_raw_data tool
        result = await client.call_tool("view_raw_data", {"path": str(test_file), "attribute": "X"})
        assert result.data is not None
        view_result = result.data
        assert view_result.data is not None or isinstance(view_result.data, str)
        assert view_result.data_type is not None
