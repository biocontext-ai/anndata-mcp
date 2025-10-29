"""Pytest configuration and fixtures for AnnData MCP tests."""

from pathlib import Path

import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def test_data_path():
    """Download and provide path to test AnnData file.

    Downloads pbmc3k_processed dataset if not already present.
    Returns path to the test file.
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    test_file = data_dir / "pbmc3k_processed.h5ad"

    if not test_file.exists():
        # Download the test data
        import scanpy as sc

        adata = sc.datasets.pbmc3k_processed()
        adata.write_h5ad(test_file)

    return str(test_file)
