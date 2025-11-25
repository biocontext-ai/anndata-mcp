import os

from ._exploration import get_descriptive_stats
from ._summary import get_summary
from ._view import view_raw_data

# Conditionally import file_system tool based on environment variable
if os.getenv("MCP_EXPOSE_FILE_SYSTEM_TOOLS", "false").lower() in ("true", "1", "yes"):
    from ._file_system import locate_anndata_stores

    __all__ = ["locate_anndata_stores", "view_raw_data", "get_summary", "get_descriptive_stats"]
else:
    __all__ = [
        "view_raw_data",
        "get_summary",
        "get_descriptive_stats",
    ]
