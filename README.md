# anndata-mcp

[![BioContextAI - Registry](https://img.shields.io/badge/Registry-package?style=flat&label=BioContextAI&labelColor=%23fff&color=%233555a1&link=https%3A%2F%2Fbiocontext.ai%2Fregistry)](https://biocontext.ai/registry)
[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/dschaub95/anndata-mcp/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/anndata-mcp

Allows retrieval and lazy access to information from AnnData objects via MCP (Model Context Protocol)

## Features

This MCP server provides **lazy, memory-efficient access** to AnnData files for biomedical analysis agents. Key features include:

- **Lazy Loading**: Only loads requested data into memory using `anndata.experimental.read_lazy`
- **Comprehensive Exploration**: Get summaries, metadata, unique values, and statistics
- **Flexible Data Access**: Retrieve specific slices of data matrices, embeddings, and metadata
- **Dataset2D Support**: Handles both pandas DataFrames and AnnData's Dataset2D objects
- **Sparse Matrix Support**: Efficiently handles sparse expression matrices

## Tools Available

### Summary Tools
- `get_anndata_summary` - Get high-level overview of AnnData structure

### Exploration Tools
- `get_attribute_info` - Get detailed info about specific attributes
- `get_dataframe_info` - Get detailed info about dataframe-like attributes (obs, var)
- `get_unique_values` - Get unique values from a column
- `get_column_stats` - Get statistics for a column
- `get_value_counts` - Count occurrences of each unique value (e.g., cells per cluster)
- `get_grouped_stats` - Calculate statistics grouped by a categorical column
- `list_available_keys` - List keys in mapping attributes (obsm, varm, layers, etc.)

### Data Access Tools
- `get_obs_data` - Get cell/observation metadata slices
- `get_var_data` - Get gene/variable metadata slices
- `get_X_data` - Get expression matrix slices
- `get_layer_data` - Get data from specific layers
- `get_obsm_data` - Get multi-dimensional annotations (embeddings like PCA, UMAP)
- `get_varm_data` - Get variable annotations
- `get_uns_data` - Get unstructured data

## Installation

You need to have Python 3.13 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

### Option 1: Use with uvx (Recommended)

```bash
uvx anndata_mcp
```

### Option 2: Install via pip

```bash
pip install anndata_mcp
```

### Option 3: Install for development

```bash
git clone https://github.com/dschaub95/anndata-mcp.git
cd anndata-mcp

# Create virtual environment with Python 3.13
uv venv --python 3.13

# Install dependencies
uv sync
```

### Option 4: Install with examples

To run the examples, install with the examples extra:

```bash
pip install "anndata_mcp[examples]"
# or with uv:
uv sync --extra examples
```

## Configuration

Include in your MCP client configuration (e.g., Claude Desktop, Continue, etc.):

```json
{
  "mcpServers": {
    "anndata-mcp": {
      "command": "uvx",
      "args": ["anndata_mcp"],
      "env": {
        "UV_PYTHON": "3.13"
      }
    }
  }
}
```

Or if installed via pip:

```json
{
  "mcpServers": {
    "anndata-mcp": {
      "command": "python",
      "args": ["-m", "anndata_mcp"],
      "env": {}
    }
  }
}
```

## Usage

### For LLM Agents

Once configured, LLM agents can use the tools to explore AnnData files:

```
Agent: "What's in the pbmc3k.h5ad file?"
→ Calls: get_anndata_summary("pbmc3k.h5ad")

Agent: "Show me unique cell types"
→ Calls: get_unique_values("pbmc3k.h5ad", "obs", "cell_type")

Agent: "Get UMAP coordinates for the first 100 cells"
→ Calls: get_obsm_data("pbmc3k.h5ad", "X_umap", row_slice="0:100")
```

### Direct Python Usage

For testing or direct use in Python:

```python
from anndata_mcp.tools import get_anndata_summary

# Call the tool directly
summary = get_anndata_summary("path/to/data.h5ad")
print(f"Dataset has {summary.n_obs} cells and {summary.n_vars} genes")
```

## Examples

See the `examples/` directory for detailed usage examples:

- **`example_script.py`**: Comprehensive Python script showing all tools including:
  - Dataset exploration and metadata inspection
  - Cluster analysis with value counts and grouped statistics
  - Expression matrix access and embedding retrieval
  - Complete analysis workflows

To run the examples:

```bash
# Install with examples extra (includes scanpy for data download)
uv sync --extra examples

# Run the example script (downloads pbmc3k sample data automatically)
uv run examples/example_script.py
```

The example script demonstrates:
1. Dataset summary and structure exploration
2. Cell and gene metadata inspection
3. Cluster information and statistics
4. **NEW**: Value counts (cells per cluster)
5. **NEW**: Grouped statistics (average genes per cluster)
6. Expression matrix sampling
7. UMAP/PCA embedding retrieval
8. Complete analysis workflows

## Use Cases

This MCP server is designed for biomedical analysis agents that need to:

1. **Explore large single-cell datasets** without loading everything into memory
2. **Query specific metadata** (cell types, gene names, QC metrics)
3. **Extract embeddings** for visualization (UMAP, PCA, t-SNE)
4. **Access expression data** for specific genes or cells
5. **Work with multiple data layers** (raw counts, normalized, scaled)
6. **Handle datasets larger than RAM** through lazy loading

## Technical Details

### Lazy Reading

The server uses `anndata.experimental.read_lazy` to open files without loading data into memory. Data is only loaded when specifically requested through tools.

### Dataset2D Support

AnnData's lazy reading often returns `Dataset2D` objects instead of pandas DataFrames. This server handles both transparently.

### Sparse Matrix Handling

Expression matrices are often sparse. The server can return sparse data in sparse format or automatically densify small slices for easier consumption.

### Slicing Syntax

Use string-based slicing: `"0:10"`, `"100:200"`, `":50"` for rows and columns.

## Getting started

Please refer to the [documentation][], in particular, the [API documentation][].

You can also find the project on [BioContextAI](https://biocontext.ai), the community-hub for biomedical MCP servers: [anndata-mcp on BioContextAI](https://biocontext.ai/registry/dschaub95/anndata-mcp).

## Contact

If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/dschaub95/anndata-mcp/issues
[tests]: https://github.com/dschaub95/anndata-mcp/actions/workflows/test.yaml
[documentation]: https://anndata-mcp.readthedocs.io
[changelog]: https://anndata-mcp.readthedocs.io/en/latest/changelog.html
[api documentation]: https://anndata-mcp.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/anndata-mcp
