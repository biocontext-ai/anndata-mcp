from importlib.metadata import version

from anndata_mcp.main import run_app
from anndata_mcp.mcp import mcp

__version__ = version("anndata_mcp")

__all__ = ["mcp", "run_app", "__version__"]


if __name__ == "__main__":
    run_app()
