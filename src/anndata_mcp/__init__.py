from importlib.metadata import version

from anndata_mcp.main import run_app
from anndata_mcp.mcp import mcp

__version__ = version("anndata_mcp")

__all__ = ["__version__", "mcp", "run_app"]


if __name__ == "__main__":
    run_app()
