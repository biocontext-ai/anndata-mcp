"""Basic tests for anndata_mcp."""

import os

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from anndata._core.xarray import Dataset2D

from anndata_mcp import __version__
from anndata_mcp.tools.utils import (
    array_to_csv,
    class_string_to_type,
    extract_data_from_dask_array,
    extract_data_from_dask_array_with_indices,
    extract_data_from_dataset2d,
    extract_indices_from_dask_array,
    extract_original_type,
    extract_original_type_string,
    extract_slice_from_dask_array,
    get_nested_key,
    get_shape_str,
    match_patterns,
    parse_slice,
    raw_type_to_string,
    truncate_string,
)


def test_version():
    """Test that version is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_truncate_string():
    """Test string truncation."""
    # Test short string (no truncation)
    assert truncate_string("short") == "short"

    # Test long string (truncation)
    long_string = "a" * 2000
    result = truncate_string(long_string)
    assert len(result) <= 1000 + 3  # max_output_len + "..."
    assert result.endswith("...")

    # Test with custom max_output_len
    os.environ["MCP_MAX_OUTPUT_LEN"] = "10"
    result = truncate_string("a" * 20)
    assert len(result) <= 13  # 10 + "..."
    assert result.endswith("...")
    # Restore default
    del os.environ["MCP_MAX_OUTPUT_LEN"]


def test_get_shape_str():
    """Test getting shape string."""
    import numpy as np

    # Test with numpy array
    arr = np.array([[1, 2], [3, 4]])
    assert get_shape_str(arr) == "(2, 2)"

    # Test with object without shape
    obj = "no shape"
    assert get_shape_str(obj) == "NA"


def test_class_string_to_type():
    """Test converting class string to type."""
    assert class_string_to_type("<class 'str'>") == "str"
    assert class_string_to_type("<class 'int'>") == "int"


def test_raw_type_to_string():
    """Test converting raw type to string."""
    assert raw_type_to_string(str) == "str"
    assert raw_type_to_string(int) == "int"
    assert raw_type_to_string(str, full_name=True) == "str"
    assert raw_type_to_string(int, full_name=True) == "int"


def test_parse_slice():
    """Test parsing slice strings."""
    # Test full slice
    assert parse_slice("0:10") == slice(0, 10, None)
    assert parse_slice("0:10:2") == slice(0, 10, 2)

    # Test partial slice
    assert parse_slice(":10") == slice(None, 10, None)
    assert parse_slice("5:") == slice(5, None, None)

    # Test None
    assert parse_slice(None) == slice(None)

    # Test invalid slice
    with pytest.raises(ValueError, match="Slice string must contain"):
        parse_slice("invalid")


def test_extract_original_type():
    """Test extracting original type from objects."""
    # Test with regular Python types
    assert extract_original_type("string") is str
    assert extract_original_type(42) is int
    assert extract_original_type([1, 2, 3]) is list

    # Test with numpy array
    arr = np.array([1, 2, 3])
    assert extract_original_type(arr) is np.ndarray

    # Test with dask array
    dask_arr = da.from_array(np.array([1, 2, 3]), chunks=2)
    # The original type should be the type of the meta
    assert extract_original_type(dask_arr) is np.ndarray

    # Test with Dataset2D
    # Create Dataset2D from xarray Dataset
    dask_arr_2d = da.from_array(np.array([[1, 4], [2, 5], [3, 6]]), chunks=(2, 2))
    index_coord = xr.IndexVariable("index", pd.RangeIndex(3))
    data_vars = {"a": ("index", dask_arr_2d[:, 0]), "b": ("index", dask_arr_2d[:, 1])}
    xr_dataset = xr.Dataset(data_vars, coords={"index": index_coord})
    dataset2d = Dataset2D(xr_dataset)
    assert extract_original_type(dataset2d) is pd.DataFrame


def test_extract_original_type_string():
    """Test extracting original type string from objects."""
    # Test with regular types
    assert extract_original_type_string("string") == "str"
    assert extract_original_type_string(42) == "int"

    # Test with numpy array
    arr = np.array([1, 2, 3])
    assert extract_original_type_string(arr) == "ndarray"

    # Test with full_name=True
    assert extract_original_type_string("string", full_name=True) == "str"
    # When full_name=True, numpy types include the module name
    result = extract_original_type_string(arr, full_name=True)
    assert result in ("ndarray", "numpy.ndarray")  # Accept both formats

    # Test with dask array
    dask_arr = da.from_array(np.array([1, 2, 3]), chunks=2)
    assert extract_original_type_string(dask_arr) == "ndarray"

    # Test with Dataset2D
    # Create Dataset2D from xarray Dataset
    dask_arr_2d = da.from_array(np.array([[1, 4], [2, 5], [3, 6]]), chunks=(2, 2))
    index_coord = xr.IndexVariable("index", pd.RangeIndex(3))
    data_vars = {"a": ("index", dask_arr_2d[:, 0]), "b": ("index", dask_arr_2d[:, 1])}
    xr_dataset = xr.Dataset(data_vars, coords={"index": index_coord})
    dataset2d = Dataset2D(xr_dataset)
    assert extract_original_type_string(dataset2d) == "DataFrame"


def test_extract_slice_from_dask_array():
    """Test extracting slice from dask array."""
    # Create a test dask array
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dask_arr = da.from_array(data, chunks=(2, 2))

    # Test extracting a slice
    result = extract_slice_from_dask_array(dask_arr, slice(0, 2), slice(0, 2))
    expected = np.array([[1, 2], [4, 5]])
    np.testing.assert_array_equal(result, expected)

    # Test extracting full array
    result = extract_slice_from_dask_array(dask_arr, slice(None), slice(None))
    np.testing.assert_array_equal(result, data)

    # Test extracting single row
    result = extract_slice_from_dask_array(dask_arr, slice(0, 1), slice(None))
    expected = np.array([[1, 2, 3]])
    np.testing.assert_array_equal(result, expected)


def test_extract_indices_from_dask_array():
    """Test extracting data from dask array using column indices."""
    # Create a test dask array
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dask_arr = da.from_array(data, chunks=(2, 2))

    # Test extracting specific columns
    result = extract_indices_from_dask_array(dask_arr, slice(0, 2), [0, 2])
    expected = np.array([[1, 3], [4, 6]])
    np.testing.assert_array_equal(result, expected)

    # Test extracting single column
    result = extract_indices_from_dask_array(dask_arr, slice(None), [1])
    expected = np.array([[2], [5], [8]])
    np.testing.assert_array_equal(result, expected)

    # Test extracting with row slice
    result = extract_indices_from_dask_array(dask_arr, slice(1, 3), [0, 1])
    expected = np.array([[4, 5], [7, 8]])
    np.testing.assert_array_equal(result, expected)


def test_array_to_csv():
    """Test converting numpy array to CSV string."""
    # Test simple 2D array
    arr = np.array([[1, 2], [3, 4]])
    result = array_to_csv(arr)
    # Should contain the data without header
    assert "1,2" in result or "1.0,2.0" in result
    assert "3,4" in result or "3.0,4.0" in result

    # Test 1D array (will be converted to column)
    arr_1d = np.array([1, 2, 3])
    result = array_to_csv(arr_1d)
    assert isinstance(result, str)
    assert len(result) > 0

    # Test that truncation works for very large arrays
    large_arr = np.random.rand(1000, 100)
    result = array_to_csv(large_arr)
    # Should be truncated if it exceeds max_output_len
    max_output_len = int(os.getenv("MCP_MAX_OUTPUT_LEN", "1000"))
    assert len(result) <= max_output_len + 3  # +3 for "..."

    # Test with custom max_output_len
    os.environ["MCP_MAX_OUTPUT_LEN"] = "50"
    result = array_to_csv(arr)
    assert len(result) <= 53  # 50 + "..."
    del os.environ["MCP_MAX_OUTPUT_LEN"]


def test_extract_data_from_dask_array():
    """Test extracting data from dask array."""
    # Create a test dask array
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dask_arr = da.from_array(data, chunks=(2, 2))

    # Test without return_shape
    result = extract_data_from_dask_array(dask_arr, slice(0, 2), slice(0, 2))
    assert isinstance(result, str)
    assert len(result) > 0

    # Test with return_shape
    result, shape = extract_data_from_dask_array(dask_arr, slice(0, 2), slice(0, 2), return_shape=True)
    assert isinstance(result, str)
    assert isinstance(shape, str)
    assert shape == "(2, 2)"

    # Test full array
    result, shape = extract_data_from_dask_array(dask_arr, slice(None), slice(None), return_shape=True)
    assert shape == "(3, 3)"


def test_extract_data_from_dask_array_with_indices():
    """Test extracting data from dask array using column indices."""
    # Create a test dask array
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dask_arr = da.from_array(data, chunks=(2, 2))

    # Test without return_shape
    result = extract_data_from_dask_array_with_indices(dask_arr, slice(0, 2), [0, 2])
    assert isinstance(result, str)
    assert len(result) > 0

    # Test with return_shape
    result, shape = extract_data_from_dask_array_with_indices(dask_arr, slice(0, 2), [0, 2], return_shape=True)
    assert isinstance(result, str)
    assert isinstance(shape, str)
    assert shape == "(2, 2)"

    # Test single column
    result, shape = extract_data_from_dask_array_with_indices(dask_arr, slice(None), [1], return_shape=True)
    assert shape == "(3, 1)"


def test_extract_data_from_dataset2d():
    """Test extracting data from Dataset2D."""
    # Create Dataset2D from xarray Dataset
    dask_arr_2d = da.from_array(np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]), chunks=(2, 3))
    index_coord = xr.IndexVariable("index", pd.RangeIndex(3))
    data_vars = {
        "a": ("index", dask_arr_2d[:, 0]),
        "b": ("index", dask_arr_2d[:, 1]),
        "c": ("index", dask_arr_2d[:, 2]),
    }
    xr_dataset = xr.Dataset(data_vars, coords={"index": index_coord})
    dataset2d = Dataset2D(xr_dataset)

    # Test without row_slice, with index
    result = extract_data_from_dataset2d(dataset2d, ["a", "b"], index=True)
    assert isinstance(result, str)
    assert "a,b" in result or "a, b" in result

    # Test without row_slice, without index
    result = extract_data_from_dataset2d(dataset2d, ["a", "b"], index=False)
    assert isinstance(result, str)

    # Test with row_slice
    result = extract_data_from_dataset2d(dataset2d, ["a", "b"], row_slice=slice(0, 2), index=True)
    assert isinstance(result, str)

    # Test with return_shape
    result, shape = extract_data_from_dataset2d(dataset2d, ["a", "b"], return_shape=True)
    assert isinstance(result, str)
    assert isinstance(shape, str)
    assert shape == "(3, 2)"

    # Test with row_slice and return_shape
    result, shape = extract_data_from_dataset2d(dataset2d, ["a", "b"], row_slice=slice(0, 2), return_shape=True)
    assert isinstance(result, str)
    assert shape == "(2, 2)"

    # Test single column
    result, shape = extract_data_from_dataset2d(dataset2d, ["a"], return_shape=True)
    assert shape == "(3, 1)"


def test_match_patterns():
    """Test match_patterns function with various scenarios."""
    items = ["gene_1", "gene_2", "RE_1", "RE_2", "CD4", "CD8", "cell_marker"]

    # Test basic glob pattern matching
    result, error_msg = match_patterns(items, ["gene_*"])
    assert len(result) == 2
    assert "gene_1" in result
    assert "gene_2" in result
    assert error_msg is None

    # Test multiple patterns, all matching
    result, error_msg = match_patterns(items, ["RE_*", "CD*"])
    assert len(result) == 4  # RE_1, RE_2, CD4, CD8
    assert "RE_1" in result
    assert "RE_2" in result
    assert "CD4" in result
    assert "CD8" in result
    assert error_msg is None

    # Test multiple patterns, some not matching
    result, error_msg = match_patterns(items, ["RE_*", "nonexistent*", "CD*"])
    assert len(result) == 4  # RE_1, RE_2, CD4, CD8
    assert "nonexistent*" in error_msg
    assert error_msg == "No matches found for: nonexistent*"

    # Test exact match (no glob)
    result, error_msg = match_patterns(items, ["CD4"])
    assert len(result) == 1
    assert result == ["CD4"]
    assert error_msg is None

    # Test empty pattern list
    result, error_msg = match_patterns(items, [])
    assert len(result) == 0
    assert error_msg is None

    # Test no patterns matching
    result, error_msg = match_patterns(items, ["xyz*", "abc"])
    assert len(result) == 0
    assert "xyz*" in error_msg
    assert "abc" in error_msg

    # Test duplicate matches (multiple patterns matching same item)
    result, error_msg = match_patterns(items, ["gene_*", "gene_1"])
    # Both patterns match "gene_1", but duplicates are removed
    assert result.count("gene_1") == 1
    assert result.count("gene_2") == 1
    assert len(result) == 2  # gene_1 and gene_2, no duplicates
    assert error_msg is None

    # Test with pandas Index
    index = pd.Index(items)
    result, error_msg = match_patterns(index, ["RE_*"])
    assert len(result) == 2
    assert "RE_1" in result
    assert "RE_2" in result
    assert error_msg is None


def test_get_nested_key():
    """Test get_nested_key functionality with various data structures."""
    # Test simple dict access
    obj = {"a": 1}
    assert get_nested_key(obj, ["a"]) == 1

    # Test nested dict access
    obj = {"level1": {"level2": {"level3": "value"}}}
    assert get_nested_key(obj, ["level1", "level2", "level3"]) == "value"
    assert get_nested_key(obj, ["level1", "level2"]) == {"level3": "value"}
    assert get_nested_key(obj, ["level1"]) == {"level2": {"level3": "value"}}

    # Test mixed nested dict
    obj = {"a": {"b": 1, "c": {"d": 2}}}
    assert get_nested_key(obj, ["a", "b"]) == 1
    assert get_nested_key(obj, ["a", "c", "d"]) == 2

    # Test object attributes
    class TestObj:
        def __init__(self):
            self.attr1 = "value1"
            self.attr2 = TestObj2()

    class TestObj2:
        def __init__(self):
            self.nested_attr = "nested_value"

    obj = TestObj()
    assert get_nested_key(obj, ["attr1"]) == "value1"
    assert get_nested_key(obj, ["attr2", "nested_attr"]) == "nested_value"

    # Test mixed dict and attribute access
    class MixedObj:
        def __init__(self):
            self.data = {"nested": {"key": "value"}}

    obj = MixedObj()
    assert get_nested_key(obj, ["data", "nested", "key"]) == "value"

    # Test dict with nested object
    class NestedObj:
        def __init__(self):
            self.value = 42

    obj = {"level1": NestedObj()}
    assert get_nested_key(obj, ["level1", "value"]) == 42

    # Test dict-like object that supports indexing
    class DictLike:
        def __init__(self):
            self._data = {"key": "value"}

        def __getitem__(self, key):
            return self._data[key]

        def __contains__(self, key):
            return key in self._data

    obj = {"level1": DictLike()}
    assert get_nested_key(obj, ["level1", "key"]) == "value"

    # Test complex nested structure
    class ComplexObj:
        def __init__(self):
            self.metadata = {
                "info": {
                    "author": "test",
                    "version": {"major": 1, "minor": 0},
                }
            }

    obj = ComplexObj()
    assert get_nested_key(obj, ["metadata", "info", "author"]) == "test"
    assert get_nested_key(obj, ["metadata", "info", "version", "major"]) == 1
    assert get_nested_key(obj, ["metadata", "info", "version", "minor"]) == 0

    # Test error handling
    obj = {"a": {"b": 1}}
    with pytest.raises(KeyError, match="Key path 'missing' not found"):
        get_nested_key(obj, ["missing"])

    with pytest.raises(KeyError, match="Key path 'a -> missing' not found"):
        get_nested_key(obj, ["a", "missing"])

    with pytest.raises(KeyError, match="Key path 'a -> b -> missing' not found"):
        get_nested_key(obj, ["a", "b", "missing"])

    # Test empty keys list (should return the object itself)
    assert get_nested_key(obj, []) == obj
