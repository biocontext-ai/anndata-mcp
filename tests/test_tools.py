"""Tests for anndata_mcp tools."""

from pathlib import Path

import pytest

from anndata_mcp.tools._exploration import get_descriptive_stats
from anndata_mcp.tools._file_system import locate_anndata_stores
from anndata_mcp.tools._summary import get_summary
from anndata_mcp.tools._view import DataView, view_raw_data

from .helpers import create_dummy_anndata


@pytest.fixture
def tmp_h5ad_path(tmp_path):
    """Create a temporary h5ad file with dummy AnnData and return its path."""
    adata = create_dummy_anndata()
    path = tmp_path / "test_anndata.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.fixture
def tmp_zarr_path(tmp_path):
    """Create a temporary zarr file with dummy AnnData and return its path."""
    adata = create_dummy_anndata()
    path = tmp_path / "test_anndata.zarr"
    adata.write_zarr(path)
    return path


@pytest.fixture
def test_h5ad_path(tmp_h5ad_path):
    """Alias for tmp_h5ad_path for backward compatibility."""
    return tmp_h5ad_path


@pytest.fixture
def test_zarr_path(tmp_zarr_path):
    """Alias for tmp_zarr_path for backward compatibility."""
    return tmp_zarr_path


class TestGetSummary:
    """Tests for get_summary tool."""

    def test_get_summary_h5ad(self, test_h5ad_path):
        """Test get_summary with h5ad file."""
        summary = get_summary(str(test_h5ad_path))

        # Check basic structure
        assert summary.n_obs > 0
        assert summary.n_vars > 0
        assert isinstance(summary.X_type, tuple)
        assert len(summary.X_type) == 2
        assert isinstance(summary.obs_columns, list)
        assert isinstance(summary.var_columns, list)
        assert isinstance(summary.obsm_keys, list)
        assert isinstance(summary.varm_keys, list)
        assert isinstance(summary.obsp_keys, list)
        assert isinstance(summary.varp_keys, list)
        assert isinstance(summary.uns_keys, list)
        assert isinstance(summary.layers, list)
        assert isinstance(summary.has_raw, bool)
        assert summary.last_modified is not None

    def test_get_summary_zarr(self, test_zarr_path):
        """Test get_summary with zarr file."""
        summary = get_summary(str(test_zarr_path))

        # Check basic structure
        assert summary.n_obs > 0
        assert summary.n_vars > 0
        assert isinstance(summary.X_type, tuple)
        assert len(summary.X_type) == 2

    def test_get_summary_obs_columns(self, test_h5ad_path):
        """Test that obs columns are correctly extracted."""
        summary = get_summary(str(test_h5ad_path))

        # Check that obs_columns is a list of tuples
        for col_info in summary.obs_columns:
            assert isinstance(col_info, tuple)
            assert len(col_info) == 2
            assert isinstance(col_info[0], str)  # column name
            assert isinstance(col_info[1], str)  # dtype

    def test_get_summary_var_columns(self, test_h5ad_path):
        """Test that var columns are correctly extracted."""
        summary = get_summary(str(test_h5ad_path))

        # Check that var_columns is a list of tuples
        for col_info in summary.var_columns:
            assert isinstance(col_info, tuple)
            assert len(col_info) == 2
            assert isinstance(col_info[0], str)  # column name
            assert isinstance(col_info[1], str)  # dtype

    def test_get_summary_obsm_keys(self, test_h5ad_path):
        """Test that obsm keys are correctly extracted."""
        summary = get_summary(str(test_h5ad_path))

        # Check that obsm_keys is a list of tuples
        for key_info in summary.obsm_keys:
            assert isinstance(key_info, tuple)
            assert len(key_info) == 3
            assert isinstance(key_info[0], str)  # key name
            assert isinstance(key_info[1], str)  # type
            assert isinstance(key_info[2], str)  # shape

    def test_get_summary_nonexistent_file(self):
        """Test get_summary with nonexistent file."""
        nonexistent_path = "/nonexistent/file.h5ad"
        summary = get_summary(nonexistent_path)
        assert summary.error is not None
        assert summary.n_obs is None
        assert summary.n_vars is None
        # Verify the error message indicates a file not found error
        error_lower = summary.error.lower()
        assert any(
            keyword in error_lower for keyword in ["file", "not found", "no such file", "cannot find", "does not exist"]
        )


class TestGetDescriptiveStats:
    """Tests for get_descriptive_stats tool."""

    def test_get_descriptive_stats_X(self, test_h5ad_path):
        """Test get_descriptive_stats for X attribute."""
        result = get_descriptive_stats(str(test_h5ad_path), attribute="X")

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_obs(self, test_h5ad_path):
        """Test get_descriptive_stats for obs attribute."""
        result = get_descriptive_stats(str(test_h5ad_path), attribute="obs")

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_var(self, test_h5ad_path):
        """Test get_descriptive_stats for var attribute."""
        result = get_descriptive_stats(str(test_h5ad_path), attribute="var")

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_obsm(self, test_h5ad_path):
        """Test get_descriptive_stats for obsm attribute with key."""
        # First get summary to find available obsm keys
        summary = get_summary(str(test_h5ad_path))
        if summary.obsm_keys:
            key = summary.obsm_keys[0][0]
            result = get_descriptive_stats(str(test_h5ad_path), attribute="obsm", key=key)

            assert result.error is None
            assert result.description is not None
            assert isinstance(result.description, str)

    def test_get_descriptive_stats_obsm_no_key(self, test_h5ad_path):
        """Test get_descriptive_stats for obsm attribute without key."""
        result = get_descriptive_stats(str(test_h5ad_path), attribute="obsm")

        # Should return error since obsm requires a key
        assert result.error is not None

    def test_get_descriptive_stats_uns(self, test_h5ad_path):
        """Test get_descriptive_stats for uns attribute."""
        result = get_descriptive_stats(str(test_h5ad_path), attribute="uns")

        # uns is a dict, so it should return an error
        assert result.error is not None

    def test_get_descriptive_stats_uns_with_key(self, test_h5ad_path):
        """Test get_descriptive_stats for uns attribute with key."""
        # First get summary to find available uns keys
        summary = get_summary(str(test_h5ad_path))
        if summary.uns_keys:
            # Try to find a key that might be an array or dataframe
            for key_info in summary.uns_keys:
                key = key_info[0]
                result = get_descriptive_stats(str(test_h5ad_path), attribute="uns", key=key)
                # May or may not have error depending on the type
                assert result is not None
                break

    def test_get_descriptive_stats_invalid_key(self, test_h5ad_path):
        """Test get_descriptive_stats with invalid key."""
        result = get_descriptive_stats(str(test_h5ad_path), attribute="obsm", key="nonexistent_key")

        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_get_descriptive_stats_with_columns(self, test_h5ad_path):
        """Test get_descriptive_stats with specific columns."""
        result = get_descriptive_stats(str(test_h5ad_path), attribute="obs", columns_or_genes=["n_genes"])

        assert result.error is None
        assert result.description is not None

    def test_get_descriptive_stats_with_value_counts(self, test_h5ad_path):
        """Test get_descriptive_stats with return_value_counts_for_categorical."""
        result = get_descriptive_stats(str(test_h5ad_path), attribute="obs", return_value_counts_for_categorical=True)

        assert result.error is None
        # value_counts may be None if no categorical columns
        assert result.description is not None

    def test_get_descriptive_stats_nonexistent_file(self):
        """Test get_descriptive_stats with nonexistent file."""
        nonexistent_path = "/nonexistent/file.h5ad"
        result = get_descriptive_stats(nonexistent_path, attribute="X")
        assert result.error is not None
        assert result.description is None
        assert result.value_counts is None
        # Verify the error message indicates a file not found error
        error_lower = result.error.lower()
        assert any(
            keyword in error_lower for keyword in ["file", "not found", "no such file", "cannot find", "does not exist"]
        )

    def test_get_descriptive_stats_with_obs_filter_equals(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using == operator."""
        # Filter for a specific cell_type
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="cell_type",
            filter_operator="==",
            filter_value="TypeA",
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_not_equals(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using != operator."""
        # Filter for cells that are not TypeA
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="cell_type",
            filter_operator="!=",
            filter_value="TypeA",
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_greater_than(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using > operator."""
        # Filter for cells with n_genes > 30
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="n_genes",
            filter_operator=">",
            filter_value=30,
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_greater_equal(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using >= operator."""
        # Filter for cells with n_genes >= 30
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="n_genes",
            filter_operator=">=",
            filter_value=30,
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_less_than(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using < operator."""
        # Filter for cells with n_genes < 30
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="n_genes",
            filter_operator="<",
            filter_value=30,
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_less_equal(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using <= operator."""
        # Filter for cells with n_genes <= 30
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="n_genes",
            filter_operator="<=",
            filter_value=30,
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_isin(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using isin operator."""
        # Filter for cells with cell_type in [TypeA, TypeB]
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="cell_type",
            filter_operator="isin",
            filter_value=["TypeA", "TypeB"],
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_isin_single_value(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using isin operator with single value."""
        # Filter for cells with cell_type in [TypeA] (single value, not a list)
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="cell_type",
            filter_operator="isin",
            filter_value="TypeA",
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_notin(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using notin operator."""
        # Filter for cells with cell_type not in [TypeA]
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="cell_type",
            filter_operator="notin",
            filter_value=["TypeA"],
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_and_X_attribute(self, test_h5ad_path):
        """Test get_descriptive_stats with filter applied to X attribute."""
        # Filter cells and then get stats for X
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="X",
            filter_attribute="obs",
            filter_column="n_genes",
            filter_operator=">",
            filter_value=30,
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_and_var_attribute(self, test_h5ad_path):
        """Test get_descriptive_stats with filter applied to var attribute."""
        # Filter cells and then get stats for var (should still work, filter affects obs)
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="var",
            filter_attribute="obs",
            filter_column="n_genes",
            filter_operator=">",
            filter_value=30,
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_float_column(self, test_h5ad_path):
        """Test get_descriptive_stats with filter on float column."""
        # Filter for cells with percent_mito > 0.5
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="percent_mito",
            filter_operator=">",
            filter_value=0.5,
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_invalid_column(self, test_h5ad_path):
        """Test get_descriptive_stats with filter using invalid column name."""
        # Filter with non-existent column
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="nonexistent_column",
            filter_operator="==",
            filter_value="value",
        )

        assert result.error is not None
        assert "not found" in result.error.lower() or "column" in result.error.lower()

    def test_get_descriptive_stats_with_obs_filter_and_columns(self, test_h5ad_path):
        """Test get_descriptive_stats with filter and columns_or_genes."""
        # Filter cells and get stats for specific columns
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="cell_type",
            filter_operator="==",
            filter_value="TypeA",
            columns_or_genes=["n_genes", "n_counts"],
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_obs_filter_empty_result(self, test_h5ad_path):
        """Test get_descriptive_stats with filter that results in empty dataset."""
        # Filter for a value that likely doesn't exist (very high n_genes)
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="obs",
            filter_column="n_genes",
            filter_operator=">",
            filter_value=1000000,
        )

        # Should return an error indicating empty dataset
        assert result is not None
        assert result.error is not None
        assert "empty dataset" in result.error.lower()
        assert "no observations" in result.error.lower() or "no cells" in result.error.lower()
        assert result.description is None

    def test_get_descriptive_stats_with_var_filter(self, test_h5ad_path):
        """Test get_descriptive_stats with var filter."""
        # Filter genes and then get stats for obs
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="obs",
            filter_attribute="var",
            filter_column="n_cells",
            filter_operator=">",
            filter_value=50,
        )

        assert result.error is None
        assert result.description is not None
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_get_descriptive_stats_with_var_filter_empty_result(self, test_h5ad_path):
        """Test get_descriptive_stats with var filter that results in empty dataset."""
        # Filter for a value that likely doesn't exist (very high n_cells)
        result = get_descriptive_stats(
            str(test_h5ad_path),
            attribute="var",
            filter_attribute="var",
            filter_column="n_cells",
            filter_operator=">",
            filter_value=1000000,
        )

        # Should return an error indicating empty dataset
        assert result is not None
        assert result.error is not None
        assert "empty dataset" in result.error.lower()
        assert "no variables" in result.error.lower() or "no genes" in result.error.lower()
        assert result.description is None


class TestViewRawData:
    """Tests for view_raw_data tool."""

    def test_view_raw_data_X(self, test_h5ad_path):
        """Test view_raw_data for X attribute."""
        result = view_raw_data(str(test_h5ad_path), attribute="X")

        assert isinstance(result, str | DataView)
        if isinstance(result, DataView):
            assert result.data is not None
            assert result.data_type is not None
            assert isinstance(result.data, str)
            assert isinstance(result.data_type, str)

    def test_view_raw_data_obs(self, test_h5ad_path):
        """Test view_raw_data for obs attribute."""
        result = view_raw_data(str(test_h5ad_path), attribute="obs")

        assert isinstance(result, str | DataView)
        if isinstance(result, DataView):
            assert result.data is not None
            assert result.data_type is not None

    def test_view_raw_data_var(self, test_h5ad_path):
        """Test view_raw_data for var attribute."""
        result = view_raw_data(str(test_h5ad_path), attribute="var")

        assert isinstance(result, str | DataView)
        if isinstance(result, DataView):
            assert result.data is not None
            assert result.data_type is not None

    def test_view_raw_data_with_slice(self, test_h5ad_path):
        """Test view_raw_data with row and column slices."""
        result = view_raw_data(
            str(test_h5ad_path), attribute="X", row_start_index=0, row_stop_index=5, col_start_index=0, col_stop_index=5
        )

        assert isinstance(result, str | DataView)
        if isinstance(result, DataView):
            assert result.data is not None
            assert result.slice_shape is not None
            assert result.full_shape is not None

    def test_view_raw_data_obsm(self, test_h5ad_path):
        """Test view_raw_data for obsm attribute with key."""
        # First get summary to find available obsm keys
        summary = get_summary(str(test_h5ad_path))
        if summary.obsm_keys:
            key = summary.obsm_keys[0][0]
            result = view_raw_data(str(test_h5ad_path), attribute="obsm", key=key)

            assert isinstance(result, str | DataView)
            if isinstance(result, DataView):
                assert result.data is not None

    def test_view_raw_data_obsm_no_key(self, test_h5ad_path):
        """Test view_raw_data for obsm attribute without key."""
        result = view_raw_data(str(test_h5ad_path), attribute="obsm")

        # Returns DataView with string representation of entries
        assert isinstance(result, str | DataView)

    def test_view_raw_data_invalid_key(self, test_h5ad_path):
        """Test view_raw_data with invalid key."""
        result = view_raw_data(str(test_h5ad_path), attribute="obsm", key="nonexistent_key")

        assert isinstance(result, DataView)
        assert result.error is not None
        assert "not found" in result.error.lower()
        assert result.data is None

    def test_view_raw_data_with_columns(self, test_h5ad_path):
        """Test view_raw_data with specific columns."""
        result = view_raw_data(str(test_h5ad_path), attribute="obs", columns_or_genes=["n_genes"])

        assert isinstance(result, str | DataView)
        if isinstance(result, DataView):
            assert result.data is not None

    def test_view_raw_data_uns(self, test_h5ad_path):
        """Test view_raw_data for uns attribute."""
        result = view_raw_data(str(test_h5ad_path), attribute="uns")

        # uns is a dict, should return DataView or string representation
        assert isinstance(result, str | DataView)

    def test_view_raw_data_nonexistent_file(self):
        """Test view_raw_data with nonexistent file."""
        nonexistent_path = "/nonexistent/file.h5ad"
        result = view_raw_data(nonexistent_path, attribute="X")
        assert isinstance(result, DataView)
        assert result.error is not None
        assert result.data is None
        assert result.data_type is None
        # Verify the error message indicates a file not found error
        error_lower = result.error.lower()
        assert any(
            keyword in error_lower for keyword in ["file", "not found", "no such file", "cannot find", "does not exist"]
        )

    def test_view_raw_data_with_df_filter_equals(self, test_h5ad_path):
        """Test view_raw_data with filter using == operator."""
        result = view_raw_data(
            str(test_h5ad_path),
            attribute="obs",
            filter_column="cell_type",
            filter_operator="==",
            filter_value="TypeA",
        )

        assert isinstance(result, DataView)
        assert result.error is None
        assert result.data is not None
        assert result.data_type is not None

    def test_view_raw_data_with_df_filter_greater_than(self, test_h5ad_path):
        """Test view_raw_data with filter using > operator."""
        result = view_raw_data(
            str(test_h5ad_path),
            attribute="obs",
            filter_column="n_genes",
            filter_operator=">",
            filter_value=30,
        )

        assert isinstance(result, DataView)
        assert result.error is None
        assert result.data is not None

    def test_view_raw_data_with_df_filter_and_X_attribute(self, test_h5ad_path):
        """Test view_raw_data with filter applied to X attribute - should error since X is not a dataframe."""
        result = view_raw_data(
            str(test_h5ad_path),
            attribute="X",
            filter_column="n_genes",
            filter_operator=">",
            filter_value=30,
        )

        assert isinstance(result, DataView)
        assert result.error is not None
        assert (
            "filter can only be applied to dataframe" in result.error.lower()
            or "dataframe-like" in result.error.lower()
        )

    def test_view_raw_data_with_df_filter_var(self, test_h5ad_path):
        """Test view_raw_data with filter applied to var attribute."""
        result = view_raw_data(
            str(test_h5ad_path),
            attribute="var",
            filter_column="n_cells",
            filter_operator=">",
            filter_value=0,
        )

        assert isinstance(result, DataView)
        assert result.error is None
        assert result.data is not None

    def test_view_raw_data_with_df_filter_isin(self, test_h5ad_path):
        """Test view_raw_data with filter using isin operator."""
        result = view_raw_data(
            str(test_h5ad_path),
            attribute="obs",
            filter_column="cell_type",
            filter_operator="isin",
            filter_value=["TypeA", "TypeB"],
        )

        assert isinstance(result, DataView)
        assert result.error is None
        assert result.data is not None

    def test_view_raw_data_with_df_filter_invalid_column(self, test_h5ad_path):
        """Test view_raw_data with filter using invalid column name."""
        result = view_raw_data(
            str(test_h5ad_path),
            attribute="obs",
            filter_column="nonexistent_column",
            filter_operator="==",
            filter_value="value",
        )

        assert isinstance(result, DataView)
        assert result.error is not None
        assert "not found" in result.error.lower() or "column" in result.error.lower()

    def test_view_raw_data_with_df_filter_empty_result(self, test_h5ad_path):
        """Test view_raw_data with filter that results in empty dataframe."""
        # Filter for a value that likely doesn't exist (very high n_genes)
        result = view_raw_data(
            str(test_h5ad_path),
            attribute="obs",
            filter_column="n_genes",
            filter_operator=">",
            filter_value=1000000,
        )

        # Should return an error indicating empty dataframe
        assert isinstance(result, DataView)
        assert result.error is not None
        assert "empty dataframe" in result.error.lower()
        assert "no rows" in result.error.lower()
        assert result.data is None


class TestLocateAnndataStores:
    """Tests for locate_anndata_stores tool."""

    def test_locate_anndata_stores_recursive(self, tmp_path):
        """Test locate_anndata_stores with recursive search."""
        # Create some test files in the temporary directory
        adata = create_dummy_anndata()
        adata.write_h5ad(tmp_path / "test1.h5ad")
        adata.write_zarr(tmp_path / "test1.zarr")
        # Create a subdirectory with more files
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        adata.write_h5ad(subdir / "test2.h5ad")

        result = locate_anndata_stores(tmp_path, recursive=True)

        assert result.paths is not None
        assert isinstance(result.paths, list)
        # Should find at least some files
        assert len(result.paths) >= 3  # At least 3 files created

        # Check that all paths are strings
        for path in result.paths:
            assert isinstance(path, str)
            # Check that files have correct extensions
            path_obj = Path(path)
            assert path_obj.suffix in (".h5ad", "") or path_obj.name.endswith(".zarr")

    def test_locate_anndata_stores_non_recursive(self, tmp_path):
        """Test locate_anndata_stores with non-recursive search."""
        # Create some test files in the temporary directory
        adata = create_dummy_anndata()
        adata.write_h5ad(tmp_path / "test1.h5ad")
        adata.write_zarr(tmp_path / "test1.zarr")
        # Create a subdirectory with more files (should not be found in non-recursive)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        adata.write_h5ad(subdir / "test2.h5ad")

        result = locate_anndata_stores(tmp_path, recursive=False)

        assert result.paths is not None
        assert isinstance(result.paths, list)
        # Should find files in the root directory but not in subdirectory
        assert len(result.paths) == 2  # Only the 2 files in root
        for path in result.paths:
            assert isinstance(path, str)

    def test_locate_anndata_stores_empty_directory(self, tmp_path):
        """Test locate_anndata_stores with empty directory."""
        result = locate_anndata_stores(tmp_path, recursive=True)

        assert result.paths is not None
        assert isinstance(result.paths, list)
        assert len(result.paths) == 0

    def test_locate_anndata_stores_nonexistent_directory(self):
        """Test locate_anndata_stores with nonexistent directory."""
        nonexistent_dir = Path("/nonexistent/directory")
        # glob doesn't raise an error for nonexistent directories, it returns empty list
        result = locate_anndata_stores(nonexistent_dir, recursive=True)
        assert result.paths is not None
        assert isinstance(result.paths, list)
        assert len(result.paths) == 0
