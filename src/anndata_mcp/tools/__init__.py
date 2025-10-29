from ._data_access import (
    get_layer_data as _get_layer_data,
    get_obs_data as _get_obs_data,
    get_obsm_data as _get_obsm_data,
    get_uns_data as _get_uns_data,
    get_var_data as _get_var_data,
    get_varm_data as _get_varm_data,
    get_X_data as _get_X_data,
)
from ._exploration import (
    get_attribute_info as _get_attribute_info,
    get_column_stats as _get_column_stats,
    get_dataframe_info as _get_dataframe_info,
    get_grouped_stats as _get_grouped_stats,
    get_unique_values as _get_unique_values,
    get_value_counts as _get_value_counts,
    list_available_keys as _list_available_keys,
)
from ._read_anndata import get_anndata_summary as _get_anndata_summary

# Export the .fn versions (actual callable functions, not MCP tool wrappers)
get_anndata_summary = _get_anndata_summary.fn
get_attribute_info = _get_attribute_info.fn
get_dataframe_info = _get_dataframe_info.fn
get_unique_values = _get_unique_values.fn
get_column_stats = _get_column_stats.fn
get_value_counts = _get_value_counts.fn
get_grouped_stats = _get_grouped_stats.fn
list_available_keys = _list_available_keys.fn
get_obs_data = _get_obs_data.fn
get_var_data = _get_var_data.fn
get_X_data = _get_X_data.fn
get_layer_data = _get_layer_data.fn
get_obsm_data = _get_obsm_data.fn
get_varm_data = _get_varm_data.fn
get_uns_data = _get_uns_data.fn

__all__ = [
    # Summary
    "get_anndata_summary",
    # Exploration
    "get_attribute_info",
    "get_dataframe_info",
    "get_unique_values",
    "get_column_stats",
    "get_value_counts",
    "get_grouped_stats",
    "list_available_keys",
    # Data Access
    "get_obs_data",
    "get_var_data",
    "get_X_data",
    "get_layer_data",
    "get_obsm_data",
    "get_varm_data",
    "get_uns_data",
]
