"""Example script demonstrating common use cases for the AnnData MCP server.

This script shows how to:
1. Download and prepare sample data
2. Use MCP tools to explore the data
3. Extract specific information for downstream analysis

Run this script after installing the package:
    python examples/example_script.py
"""

from pathlib import Path

import scanpy as sc

from anndata_mcp.tools import (
    get_anndata_summary,
    get_column_stats,
    get_dataframe_info,
    get_grouped_stats,
    get_obs_data,
    get_obsm_data,
    get_unique_values,
    get_value_counts,
    get_X_data,
    list_available_keys,
)


def prepare_sample_data():
    """Download and prepare sample AnnData dataset."""
    print("=== Preparing Sample Data ===")

    # Create data directory
    data_dir = Path("./examples/data")
    data_dir.mkdir(exist_ok=True, parents=True)

    output_path = data_dir / "pbmc3k_processed.h5ad"

    # Check if data already exists
    if output_path.exists():
        print(f"Data already exists at: {output_path}")
        return str(output_path)

    print("Downloading pbmc3k dataset...")
    adata = sc.datasets.pbmc3k_processed()

    # Save
    adata.write_h5ad(output_path)
    print(f"Saved dataset to: {output_path}")

    return str(output_path)


def explore_dataset_summary(file_path: str):
    """Example 1: Get an overview of the dataset structure."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Dataset Summary")
    print("=" * 60)

    summary = get_anndata_summary(file_path)

    print("\nDataset Overview:")
    print(f"  - Observations (cells): {summary.n_obs}")
    print(f"  - Variables (genes): {summary.n_vars}")
    print(f"  - Cell metadata columns: {', '.join(summary.obs_columns[:5])}...")
    print(f"  - Gene metadata columns: {', '.join(summary.var_columns[:5])}...")
    print(f"  - Available embeddings: {', '.join(summary.obsm_keys)}")
    print(f"  - Available layers: {', '.join(summary.layers)}")


def explore_cell_metadata(file_path: str):
    """Example 2: Explore cell metadata structure."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Cell Metadata Exploration")
    print("=" * 60)

    df_info = get_dataframe_info(file_path, "obs")

    print(f"\nCell metadata shape: {df_info.shape}")
    print(f"Number of columns: {df_info.n_cols}")
    print("\nColumn details:")
    for col in df_info.columns[:5]:  # Show first 5 columns
        print(f"  - {col.name}: {col.dtype} (unique values: {col.n_unique}, has nulls: {col.has_nulls})")


def find_cluster_information(file_path: str):
    """Example 3: Get cluster information."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Cluster Information")
    print("=" * 60)

    # Get unique cluster labels
    unique_clusters = get_unique_values(file_path, "obs", "louvain")

    print(f"\nFound {unique_clusters.n_unique} clusters:")
    print(f"Cluster IDs: {unique_clusters.unique_values}")

    # Get statistics about the number of genes per cell for each cluster
    stats = get_column_stats(file_path, "obs", "n_genes")

    print("\nGenes per cell statistics:")
    print(f"  - Mean: {stats.stats.get('mean', 'N/A'):.2f}")
    print(f"  - Median: {stats.stats.get('median', 'N/A'):.2f}")
    print(f"  - Min: {stats.stats.get('min', 'N/A'):.2f}")
    print(f"  - Max: {stats.stats.get('max', 'N/A'):.2f}")


def get_specific_cells(file_path: str):
    """Example 4: Get data for specific cells."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Retrieve Specific Cell Data")
    print("=" * 60)

    # Get first 10 cells with selected columns
    obs_data = get_obs_data(file_path, row_slice="0:10", columns=["louvain", "n_genes", "n_counts"])

    print(f"\nRetrieved {obs_data['rows_returned']} cells:")
    for i, cell_data in enumerate(obs_data["data"][:3]):  # Show first 3
        print(f"  Cell {i}: Cluster {cell_data['louvain']}, {cell_data['n_genes']} genes")


def get_expression_data(file_path: str):
    """Example 5: Get expression matrix data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Expression Matrix Data")
    print("=" * 60)

    # Get a small slice of the expression matrix
    X_data = get_X_data(file_path, row_slice="0:5", col_slice="0:5")

    print("\nExpression matrix slice:")
    print(f"  - Shape: {X_data.shape}")
    print(f"  - Is sparse: {X_data.is_sparse}")
    if X_data.is_sparse:
        print(f"  - Sparse format: {X_data.sparse_format}")
        print(f"  - Sparsity: {X_data.sparsity:.2%}")
    print(f"  - Data type: {X_data.dtype}")


def get_embeddings(file_path: str):
    """Example 6: Get UMAP/PCA embeddings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Embeddings for Visualization")
    print("=" * 60)

    # List available embeddings
    obsm_keys = list_available_keys(file_path, "obsm")
    print(f"\nAvailable embeddings: {obsm_keys}")

    # Get UMAP coordinates for first 10 cells
    umap_data = get_obsm_data(file_path, "X_umap", row_slice="0:10")

    print("\nUMAP coordinates (first 3 cells):")
    for i, coords in enumerate(umap_data["data"][:3]):
        print(f"  Cell {i}: [{coords[0]:.2f}, {coords[1]:.2f}]")

    # Get first 3 PCA components for first 10 cells
    pca_data = get_obsm_data(file_path, "X_pca", row_slice="0:10", col_slice="0:3")

    print("\nPCA coordinates (first 3 PCs, first 3 cells):")
    for i, coords in enumerate(pca_data["data"][:3]):
        print(f"  Cell {i}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")


def explore_grouping_operations(file_path: str):
    """Example 7: Grouping operations and value counts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Grouping Operations and Value Counts")
    print("=" * 60)

    # Get number of cells per cluster
    print("\nGet number of cells in each cluster:")
    counts = get_value_counts(file_path, "obs", "louvain")
    print(f"Total cells: {counts['total']}")
    print("Cells per cluster:")
    for cluster, count in sorted(counts["counts"].items())[:5]:
        print(f"  Cluster {cluster}: {count} cells")

    # Get average gene count per cluster
    print("\nGet average gene count per cluster:")
    grouped_stats = get_grouped_stats(file_path, "obs", "louvain", "n_genes")
    print(f"Number of clusters: {grouped_stats['n_groups']}")
    print("\nCluster statistics:")
    for cluster, stats in sorted(grouped_stats["groups"].items())[:3]:
        print(f"  Cluster {cluster}: mean={stats['mean']:.1f}, median={stats['median']:.1f}, count={stats['count']}")


def workflow_example(file_path: str):
    """Example 8: Complete analysis workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Complete Analysis Workflow")
    print("=" * 60)

    print("\nScenario: Find cells in cluster 0 with high gene counts")

    # Step 1: Get summary
    summary = get_anndata_summary(file_path)
    print(f"\n1. Dataset has {summary.n_obs} cells")

    # Step 2: Get cluster information
    obs_data = get_obs_data(file_path, row_slice="0:100", columns=["louvain", "n_genes"])

    # Filter for cluster 0
    cluster_0_cells = [cell for cell in obs_data["data"] if cell["louvain"] == "0"]
    print(f"2. Found {len(cluster_0_cells)} cells in cluster 0 (from first 100 cells)")

    # Find cells with high gene counts
    high_gene_cells = [cell for cell in cluster_0_cells if cell["n_genes"] > 1000]
    print(f"3. Found {len(high_gene_cells)} cells with >1000 genes")

    if high_gene_cells:
        avg_genes = sum(cell["n_genes"] for cell in high_gene_cells) / len(high_gene_cells)
        print(f"4. Average gene count: {avg_genes:.0f}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("AnnData MCP Server - Usage Examples")
    print("=" * 60)

    # Prepare sample data
    file_path = prepare_sample_data()

    # Run examples
    explore_dataset_summary(file_path)
    explore_cell_metadata(file_path)
    find_cluster_information(file_path)
    get_specific_cells(file_path)
    get_expression_data(file_path)
    get_embeddings(file_path)
    explore_grouping_operations(file_path)
    workflow_example(file_path)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print(
        "\nNote: In production, these tools would be called by an LLM agent"
        "\nthrough the MCP protocol to explore and analyze AnnData files."
    )


if __name__ == "__main__":
    main()
