from pathlib import Path

from anndata.experimental import read_lazy

# Cache stores: path -> (anndata_object, mtime)
# mtime is used to detect if the file has changed and invalidate the cache
_cache: dict[str, tuple[object, float]] = {}


def read_lazy_with_cache(path: Path) -> object:
    """Get a lazily loaded AnnData object from cache or load it if not cached.

    The cache is invalidated if the file modification time has changed.

    Args:
        path: Path to the AnnData file (.h5ad or .zarr)

    Returns
    -------
        Lazily loaded AnnData object
    """
    # Resolve the path to ensure consistent cache keys
    resolved_path = str(path.resolve())

    # Check if file exists
    if not Path(resolved_path).exists():
        raise FileNotFoundError(f"AnnData file not found: {resolved_path}")

    # Get current modification time
    current_mtime = Path(resolved_path).stat().st_mtime

    # Check cache
    if resolved_path in _cache:
        cached_adata, cached_mtime = _cache[resolved_path]

        # If file hasn't changed, return cached object
        if cached_mtime == current_mtime:
            return cached_adata

        # File has changed, remove from cache
        del _cache[resolved_path]

    # Load and cache the object
    adata = read_lazy(resolved_path)
    _cache[resolved_path] = (adata, current_mtime)

    return adata


def clear_cache(path: Path | None = None) -> None:
    """Clear the cache for a specific path or all paths.

    Args:
        path: If provided, clear cache only for this path. If None, clear all cache.
    """
    if path is None:
        _cache.clear()
    else:
        resolved_path = str(path.resolve())
        _cache.pop(resolved_path, None)


def get_cache_size() -> int:
    """Get the number of cached AnnData objects.

    Returns
    -------
        Number of cached objects
    """
    return len(_cache)
