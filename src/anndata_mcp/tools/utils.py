import asyncio
import fnmatch
import os
import pickle
import tempfile
import uuid
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import anndata as ad
import dask
import numpy as np
import pandas as pd
import zarr
from anndata._core.xarray import Dataset2D
from anndata.experimental import read_lazy
from zarr.buffer import default_buffer_prototype
from zarr.buffer.cpu import Buffer
from zarr.storage import LocalStore, MemoryStore


class AccessTrackingStore(zarr.storage.FsspecStore):
    """A store that tracks the keys that have been accessed."""

    _keys_hit = set()

    async def get(self, key, *args, **kwargs):
        """Get a key from the store."""
        try:
            res = await super().get(key, *args, **kwargs)
            if key not in self._keys_hit and res is not None:
                self._keys_hit.add(key)
            return res
        except (KeyError, OSError):
            # Key doesn't exist or filesystem error - return None
            return None


class SharedMemoryStore(MemoryStore):
    """A zarr store backed by multiprocessing.shared_memory for cross-process sharing.

    This store allows sharing zarr data between processes without filesystem overhead.
    The data is stored in shared memory blocks that can be accessed by name.

    **Important**: After creating the store, you must call `get_shared_name()` to get
    the identifier that other processes can use to attach to this store. When done,
    call `cleanup()` to release the shared memory.

    Parameters
    ----------
    name : str, optional
        A name identifier for the shared memory. If None, a unique name is generated.
        This name is used to share the store between processes.

    Examples
    --------
    Process 1 (create):
    >>> store = SharedMemoryStore(name="my_data")
    >>> adata.write_zarr(store)
    >>> shared_name = store.get_shared_name()  # Share this name with Process 2

    Process 2 (attach):
    >>> store = SharedMemoryStore.attach(shared_name)
    >>> adata = read_lazy(store)
    """

    def __init__(self, name: str | None = None, read_only: bool = False, write_directly: bool = True):
        """Initialize a SharedMemoryStore.

        Parameters
        ----------
        name : str, optional
            A name identifier for the shared memory. If None, a unique name is generated.
        read_only : bool, default False
            Whether the store is read-only.
        write_directly : bool, default True
            If True, writes go directly to shared memory as they happen (more efficient).
            If False, writes go to MemoryStore first and must be copied via get_shared_name().
        """
        super().__init__(read_only=read_only)
        self._name = name or f"zarr_shm_{uuid.uuid4().hex[:12]}"
        self._shared_memories: dict[str, shared_memory.SharedMemory] = {}
        self._metadata_shm: shared_memory.SharedMemory | None = None
        self._write_directly = write_directly
        self._keys_written: set[str] = set()  # Track keys written to shared memory

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        """Set a key-value pair in the store.

        If write_directly is True, also writes directly to shared memory.
        """
        # Write to parent MemoryStore first (required for zarr to work)
        await super().set(key, value)

        # If write_directly is enabled, also write to shared memory immediately
        if self._write_directly and not self.read_only:
            # Convert Buffer to bytes
            if hasattr(value, "to_bytes"):
                value_bytes = value.to_bytes()
            elif hasattr(value, "tobytes"):
                value_bytes = value.tobytes()
            else:
                value_bytes = bytes(value)

            # Sanitize key name for shared memory
            safe_key = key.replace("/", "_").replace(".", "_")
            shm_name = f"{self._name}_{safe_key}"

            # Create or resize shared memory block
            if key not in self._shared_memories:
                # Create new shared memory block
                try:
                    shm = shared_memory.SharedMemory(create=True, size=len(value_bytes), name=shm_name)
                    shm.buf[:] = value_bytes
                    self._shared_memories[key] = shm
                except FileExistsError:
                    # Shared memory already exists, attach to it
                    shm = shared_memory.SharedMemory(name=shm_name)
                    if len(shm.buf) < len(value_bytes):
                        # Need to resize - close and recreate
                        shm.close()
                        shm.unlink()
                        shm = shared_memory.SharedMemory(create=True, size=len(value_bytes), name=shm_name)
                    shm.buf[: len(value_bytes)] = value_bytes
                    self._shared_memories[key] = shm
            else:
                # Update existing shared memory block
                shm = self._shared_memories[key]
                if len(shm.buf) < len(value_bytes):
                    # Need to resize - close, unlink, and recreate
                    shm.close()
                    shm.unlink()
                    shm = shared_memory.SharedMemory(create=True, size=len(value_bytes), name=shm_name)
                    self._shared_memories[key] = shm
                shm.buf[: len(value_bytes)] = value_bytes

            # Update metadata - check if this was a new key before we added it
            was_new_key = key not in self._keys_written
            if was_new_key:
                self._keys_written.add(key)

            if self._metadata_shm is None:
                # Create metadata shared memory
                keys_list = list(self._keys_written)
                serialized = pickle.dumps(keys_list)
                self._metadata_shm = shared_memory.SharedMemory(
                    create=True, size=len(serialized), name=f"{self._name}_metadata"
                )
                self._metadata_shm.buf[:] = serialized
            elif was_new_key:
                # Update metadata with new key
                keys_list = list(self._keys_written)
                serialized = pickle.dumps(keys_list)
                if len(self._metadata_shm.buf) < len(serialized):
                    # Resize metadata shared memory
                    old_shm = self._metadata_shm
                    old_shm.close()
                    old_shm.unlink()
                    self._metadata_shm = shared_memory.SharedMemory(
                        create=True, size=len(serialized), name=f"{self._name}_metadata"
                    )
                self._metadata_shm.buf[: len(serialized)] = serialized

    def with_read_only(self, read_only: bool = True):
        """Create a read-only version of this store."""
        # Create a new instance with the same name and shared memory references
        new_store = type(self)(name=self._name, read_only=read_only, write_directly=False)
        new_store._shared_memories = self._shared_memories.copy()
        new_store._metadata_shm = self._metadata_shm
        new_store._keys_written = self._keys_written.copy()
        # Copy the underlying store data from MemoryStore
        new_store._store_dict = self._store_dict.copy()
        return new_store

    def get_shared_name(self) -> str:
        """Get the shared memory name that other processes can use to attach.

        If write_directly is True, the data is already in shared memory and this
        just ensures metadata is up to date.

        If write_directly is False, this method copies all data from the MemoryStore
        into shared memory blocks.

        Returns
        -------
        str
            The shared memory name identifier
        """

        # Get all keys and values from the store
        # Both list_dir() and get() are async, so we need to handle them properly
        async def _get_all_items():
            keys_list = []
            async for key in self.list_dir(""):
                keys_list.append(key)

            if not keys_list:
                return {}

            # Get all values
            store_items = {}
            for key in keys_list:
                value = await self.get(key, default_buffer_prototype())
                if value is not None:
                    # Convert Buffer to bytes
                    if hasattr(value, "to_bytes"):
                        store_items[key] = value.to_bytes()
                    elif hasattr(value, "tobytes"):
                        store_items[key] = value.tobytes()
                    else:
                        store_items[key] = bytes(value)
            return store_items

        # If write_directly is enabled, data is already in shared memory
        if self._write_directly:
            # Just ensure metadata is up to date
            if self._metadata_shm is None:
                # Get all keys from MemoryStore to create initial metadata
                store_items = asyncio.run(_get_all_items())
                keys_list = list(store_items.keys())
                serialized = pickle.dumps(keys_list)
                self._metadata_shm = shared_memory.SharedMemory(
                    create=True, size=len(serialized), name=f"{self._name}_metadata"
                )
                self._metadata_shm.buf[:] = serialized
            else:
                # Update metadata with current keys
                keys_list = list(self._keys_written)
                serialized = pickle.dumps(keys_list)
                if len(self._metadata_shm.buf) < len(serialized):
                    # Resize metadata shared memory
                    old_shm = self._metadata_shm
                    old_shm.close()
                    old_shm.unlink()
                    self._metadata_shm = shared_memory.SharedMemory(
                        create=True, size=len(serialized), name=f"{self._name}_metadata"
                    )
                self._metadata_shm.buf[: len(serialized)] = serialized
            return self._name

        # Otherwise, copy from MemoryStore to shared memory (original behavior)
        store_items = asyncio.run(_get_all_items())
        if not store_items:
            # Store is empty, just create metadata
            serialized = pickle.dumps([])
            if self._metadata_shm is None:
                self._metadata_shm = shared_memory.SharedMemory(
                    create=True, size=len(serialized), name=f"{self._name}_metadata"
                )
                self._metadata_shm.buf[:] = serialized
            return self._name

        # Serialize the keys list for metadata
        serialized = pickle.dumps(list(store_items.keys()))

        # Create shared memory for metadata (list of keys)
        if self._metadata_shm is None:
            self._metadata_shm = shared_memory.SharedMemory(
                create=True, size=len(serialized), name=f"{self._name}_metadata"
            )
            self._metadata_shm.buf[:] = serialized

        # For each value in the store, create a shared memory block
        for key, value in store_items.items():
            if key not in self._shared_memories:
                # Sanitize key name for shared memory (shared memory names have restrictions)
                safe_key = key.replace("/", "_").replace(".", "_")
                shm_name = f"{self._name}_{safe_key}"
                shm = shared_memory.SharedMemory(create=True, size=len(value), name=shm_name)
                shm.buf[:] = value
                self._shared_memories[key] = shm
                self._keys_written.add(key)

        return self._name

    @classmethod
    def attach(cls, shared_name: str) -> "SharedMemoryStore":
        """Attach to an existing SharedMemoryStore by name.

        Parameters
        ----------
        shared_name : str
            The shared memory name from `get_shared_name()`

        Returns
        -------
        SharedMemoryStore
            A SharedMemoryStore attached to the shared memory
        """
        # Attach to metadata shared memory
        metadata_shm = shared_memory.SharedMemory(name=f"{shared_name}_metadata")
        keys = pickle.loads(bytes(metadata_shm.buf))

        # Create a new store and populate it
        store = cls(name=shared_name)
        store._metadata_shm = metadata_shm

        # Attach to each value's shared memory and copy to store
        async def _populate_store():
            for key in keys:
                try:
                    # Sanitize key name (same as in get_shared_name)
                    safe_key = key.replace("/", "_").replace(".", "_")
                    shm = shared_memory.SharedMemory(name=f"{shared_name}_{safe_key}")
                    # Copy data from shared memory to the store using async set()
                    buffer_value = Buffer.from_bytes(bytes(shm.buf))
                    await store.set(key, buffer_value)
                    store._shared_memories[key] = shm
                except FileNotFoundError:
                    # Key might not exist yet (store is being populated)
                    pass

        asyncio.run(_populate_store())
        return store

    def cleanup(self):
        """Clean up shared memory resources.

        Call this when you're done with the store to free shared memory.
        """
        # Close and unlink all shared memory blocks
        for shm in self._shared_memories.values():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass  # Already cleaned up

        if self._metadata_shm is not None:
            try:
                self._metadata_shm.close()
                self._metadata_shm.unlink()
            except FileNotFoundError:
                pass  # Already cleaned up

        self._shared_memories.clear()
        self._metadata_shm = None


def _is_url(path: str | Path) -> bool:
    """Check if a string is a URL or a file system path.

    Parameters
    ----------
    path : str | Path
        The path or URL to check

    Returns
    -------
    bool
        True if the string appears to be a URL, False otherwise
    """
    # Convert Path objects to strings
    path_str = str(path)
    # Check for common URL schemes
    url_schemes = ("http://", "https://", "s3://", "gs://", "gcs://", "abfs://", "az://")
    return any(path_str.startswith(scheme) for scheme in url_schemes)


def read_lazy_general(path_or_url_or_adata: str | Path | ad.AnnData):
    """Read an AnnData object lazily from a file path, URL, or in-memory AnnData object.

    This function automatically detects the input type and handles it appropriately:
    - **URLs**: Uses AccessTrackingStore.from_url() to create a zarr store, then reads lazily
    - **File paths**: Uses read_lazy directly (works with regular paths and RAM filesystem paths like /dev/shm/...)
    - **AnnData objects**: Converts to an in-memory zarr MemoryStore, then reads lazily (no filesystem writes)

    Parameters
    ----------
    path_or_url_or_adata : str | Path | AnnData
        Either:
        - A file system path (e.g., "data/test.h5ad", "data/test.zarr", or "/dev/shm/data.zarr")
        - A URL (e.g., "https://example.com/data.zarr/")
        - An in-memory AnnData object (will be converted to MemoryStore, no disk writes)

    Returns
    -------
    AnnData
        A lazily-loaded AnnData object

    Examples
    --------
    >>> from anndata_mcp.tools.utils import read_lazy_general
    >>> import anndata as ad
    >>>
    >>> # From file path
    >>> adata = read_lazy_general("data.h5ad")
    >>>
    >>> # From URL
    >>> adata = read_lazy_general("https://example.com/data.zarr/")
    >>>
    >>> # From RAM filesystem path
    >>> adata = read_lazy_general("/dev/shm/data.zarr")
    >>>
    >>> # From in-memory AnnData object (no disk writes!)
    >>> adata_in_memory = ad.read_h5ad("data.h5ad")
    >>> adata_lazy = read_lazy_general(adata_in_memory)
    """
    # Check if input is an AnnData object
    if isinstance(path_or_url_or_adata, ad.AnnData):
        # Convert in-memory AnnData to MemoryStore and read lazily
        store = anndata_to_memory_store(path_or_url_or_adata)
        return read_lazy(store)

    # Convert Path objects to strings
    path_str = str(path_or_url_or_adata)

    if _is_url(path_str):
        # For URLs, use AccessTrackingStore.from_url() then read_lazy
        store = AccessTrackingStore.from_url(path_str, read_only=True)
        return read_lazy(store)
    else:
        # For file paths (including RAM filesystem paths like /dev/shm/...),
        # use read_lazy directly
        return read_lazy(path_str)


def anndata_to_memory_store(adata: ad.AnnData) -> MemoryStore:
    """Convert an in-memory AnnData object to an in-memory zarr store.

    This allows you to use `read_lazy` on an AnnData object that's already in memory
    **without any filesystem writes**. The data is stored in a zarr MemoryStore, which
    is entirely in-memory (no disk I/O, no temporary files).

    **Important**: This is the only method that truly avoids any filesystem operations.
    For cross-process sharing, see `anndata_to_shared_memory_store()` with `use_mmap=True`,
    which uses a RAM-based filesystem (/dev/shm) but still creates files.

    Parameters
    ----------
    adata : AnnData
        An in-memory AnnData object

    Returns
    -------
    MemoryStore
        An in-memory zarr store containing the AnnData data (no filesystem writes)

    Examples
    --------
    >>> import anndata as ad
    >>> from anndata_mcp.tools.utils import anndata_to_memory_store, read_lazy
    >>>
    >>> # Create or load an AnnData object
    >>> adata = ad.read_h5ad("data.h5ad")
    >>>
    >>> # Convert to in-memory zarr store (no disk writes!)
    >>> store = anndata_to_memory_store(adata)
    >>>
    >>> # Use read_lazy on the in-memory store
    >>> adata_lazy = read_lazy(store)
    """
    # Create an in-memory zarr store
    # This is truly in-memory - no filesystem operations at all
    store = MemoryStore()

    # Write the AnnData object to the zarr store
    # Note: This writes to the in-memory store, NOT to disk
    adata.write_zarr(store)

    return store


def read_lazy_from_memory(adata: ad.AnnData):
    """Read an in-memory AnnData object lazily using an in-memory zarr store.

    This is a convenience function that combines `anndata_to_memory_store` and `read_lazy`.

    Parameters
    ----------
    adata : AnnData
        An in-memory AnnData object

    Returns
    -------
    AnnData
        A lazily-loaded AnnData object backed by an in-memory zarr store

    Examples
    --------
    >>> import anndata as ad
    >>> from anndata_mcp.tools.utils import read_lazy_from_memory
    >>>
    >>> # Create or load an AnnData object
    >>> adata = ad.read_h5ad("data.h5ad")
    >>>
    >>> # Read it lazily from memory
    >>> adata_lazy = read_lazy_from_memory(adata)
    """
    store = anndata_to_memory_store(adata)
    return read_lazy(store)


def anndata_to_shared_memory_store(
    adata: ad.AnnData, name: str | None = None, use_mmap: bool = False, mmap_dir: str | None = None
) -> tuple[MemoryStore | LocalStore, str]:
    """Convert an AnnData object to a store that can be shared between processes.

    This function provides two options for sharing AnnData data between processes:

    1. **Memory-mapped file** (use_mmap=True): Creates a file in a RAM-based filesystem
       (e.g., /dev/shm on Linux, which is a tmpfs mounted in RAM). While this creates
       "files", they exist entirely in RAM with no disk I/O. The files persist until
       explicitly deleted and can be accessed by multiple processes.

       **Note**: This DOES create files (even if in RAM), so it's not truly "no filesystem
       writes". For truly in-memory operation without any files, use `anndata_to_memory_store()`.

    2. **In-memory store** (use_mmap=False): Uses zarr's MemoryStore, which is truly
       in-memory with no filesystem operations. However, MemoryStore doesn't support
       cross-process sharing natively - this option is for single-process use only.

    Parameters
    ----------
    adata : AnnData
        An in-memory AnnData object
    name : str, optional
        A name identifier for the shared store. If None, a unique name is generated.
    use_mmap : bool, default False
        If True, use a RAM-based filesystem (/dev/shm on Linux) for cross-process sharing.
        Files are created but stored in RAM (tmpfs), not on disk.
        If False, use an in-memory store (single process only, no filesystem writes).
    mmap_dir : str, optional
        Directory for memory-mapped files. Defaults to /dev/shm on Linux (RAM-based),
        or system temp directory if /dev/shm is not available.

    Returns
    -------
    tuple[MemoryStore | LocalStore, str]
        A tuple containing:
        - The zarr store (MemoryStore or LocalStore)
        - A path/name that can be used to access the store from another process

    Examples
    --------
    >>> import anndata as ad
    >>> from anndata_mcp.tools.utils import anndata_to_shared_memory_store, read_lazy
    >>>
    >>> # Create or load an AnnData object
    >>> adata = ad.read_h5ad("data.h5ad")
    >>>
    >>> # Create a RAM-based store for cross-process sharing
    >>> # (creates files in /dev/shm, but they're in RAM, not on disk)
    >>> store, store_path = anndata_to_shared_memory_store(adata, use_mmap=True)
    >>>
    >>> # In another process, you can access it via:
    >>> # from zarr.storage import LocalStore
    >>> # store = LocalStore(store_path)
    >>> # adata_lazy = read_lazy(store)
    """
    if name is None:
        name = f"anndata_{uuid.uuid4().hex[:8]}"

    if use_mmap:
        # Use memory-mapped file for cross-process sharing
        if mmap_dir is None:
            # Try /dev/shm first (Linux shared memory)
            if os.path.exists("/dev/shm"):
                mmap_dir = "/dev/shm"
            else:
                # Fall back to system temp directory
                mmap_dir = tempfile.gettempdir()

        store_path = os.path.join(mmap_dir, f"{name}.zarr")
        store = LocalStore(store_path)

        # Write to the memory-mapped location
        adata.write_zarr(store)

        return store, store_path
    else:
        # Use in-memory store (single process only)
        store = MemoryStore()
        adata.write_zarr(store)
        return store, name


def truncate_string(string: str, max_output_len: int | None = None) -> str:
    """Truncate a string to the maximum length."""
    max_output_len = max_output_len or int(os.getenv("MCP_MAX_OUTPUT_LEN", "1000"))
    if len(string) > max_output_len:
        return string[:max_output_len] + "..."
    return string


def get_shape_str(obj: Any) -> str:
    """Get the shape of an object as a string."""
    try:
        return str(obj.shape)
    except AttributeError:
        return "NA"


def class_string_to_type(class_string: str) -> str:
    """Convert a class string to a type."""
    return class_string.split("'")[1]


def raw_type_to_string(raw_type: type, full_name: bool = False) -> str:
    """Convert a raw type to a string."""
    if full_name:
        return class_string_to_type(str(raw_type))
    else:
        return raw_type.__name__


def extract_original_type(obj: Any) -> type:
    """Extract the original type of an object."""
    if isinstance(obj, dask.array.core.Array):
        return type(obj._meta)
    elif isinstance(obj, Dataset2D):
        return pd.DataFrame
    else:
        return type(obj)


def extract_original_type_string(obj: Any, full_name: bool = False) -> str:
    """Extract the original type of an object and convert it to a string."""
    return raw_type_to_string(extract_original_type(obj), full_name=full_name)


def parse_slice(slice_str: str | None) -> slice:
    """Parse a slice string like '0:10' or ':100' into a slice object.

    Parameters
    ----------
    slice_str : str, optional
        Slice string

    Returns
    -------
    slice
        Parsed slice object
    """
    if slice_str is None:
        return slice(None)

    if ":" not in slice_str:
        raise ValueError("Slice string must contain ':'")

    parts = slice_str.split(":")
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None

    return slice(start, stop, step)


def extract_slice_from_dask_array(array: dask.array.core.Array, row_slice: slice, col_slice: slice) -> np.ndarray:
    """Extract a slice from a dask array."""
    return array[row_slice, col_slice].compute()


def extract_indices_from_dask_array(
    array: dask.array.core.Array, row_slice: slice, col_indices: list[int]
) -> np.ndarray:
    """Extract data from a dask array using column indices."""
    return array[row_slice, col_indices].compute()


def array_to_csv(array: np.ndarray) -> str:
    """Convert a numpy array to a CSV string."""
    return truncate_string("\n".join(pd.DataFrame(array).to_csv(index=False).split("\n")[1::]))


def extract_data_from_dask_array(
    array: dask.array.core.Array, row_slice: slice, col_slice: slice, return_shape: bool = False
) -> tuple[str, str] | str:
    """Extract data from a dask array."""
    data = extract_slice_from_dask_array(array, row_slice, col_slice)
    if return_shape:
        return truncate_string(array_to_csv(data)), str(data.shape)
    else:
        return truncate_string(array_to_csv(data))


def extract_data_from_dask_array_with_indices(
    array: dask.array.core.Array, row_slice: slice, col_indices: list[int], return_shape: bool = False
) -> tuple[str, str] | str:
    """Extract data from a dask array using column indices."""
    data = extract_indices_from_dask_array(array, row_slice, col_indices)
    if return_shape:
        return truncate_string(array_to_csv(data)), str(data.shape)
    else:
        return truncate_string(array_to_csv(data))


def extract_data_from_dataset2d(
    dataset2d: Dataset2D,
    columns: list[str],
    row_slice: slice | None = None,
    index: bool = True,
    return_shape: bool = False,
) -> tuple[str, str] | str:
    """Extract data from a dataset2d."""
    if row_slice is not None:
        data = dataset2d.iloc[row_slice][columns].to_memory()
    else:
        data = dataset2d[columns].to_memory()
    if return_shape:
        return truncate_string(data.to_csv(index=index)), str(data.shape)
    else:
        return truncate_string(data.to_csv(index=index))


def select_by_glob(items: list[str] | pd.Index, pattern: str):
    """Select items from a list or index matching a glob pattern."""
    return fnmatch.filter(items, pattern)


def match_patterns(items: list[str] | pd.Index, pattern_list: list[str]) -> tuple[list[str], str | None]:
    """Match items to patterns and return the matched items and a message listing any patterns that were not found."""
    result = []
    errors = []
    for pattern in pattern_list:
        selected = select_by_glob(items, pattern)
        if len(selected) == 0:
            errors.append(pattern)
            continue
        result.extend(selected)
    # Remove duplicates while preserving order
    result = list(dict.fromkeys(result))
    return result, f"No matches found for: {', '.join(errors)}" if len(errors) > 0 else None


def get_nested_key(obj: Any, keys: list[str]) -> Any:
    """Retrieve a nested value from an object using a list of keys.

    This function traverses through nested structures (dicts, objects with attributes, etc.)
    using the provided list of keys. It uses `get()` for dict-like objects and `hasattr()`/`getattr()`
    for objects with attributes where possible.

    Parameters
    ----------
    obj : Any
        The object to traverse
    keys : list[str]
        List of keys to traverse through the nested structure

    Returns
    -------
    Any
        The value at the nested key path

    Raises
    ------
    KeyError
        If any key in the path is not found
    AttributeError
        If any attribute in the path is not found
    """
    current = obj
    path_traversed = []

    for key in keys:
        path_traversed.append(key)

        # Try dict-like access first (supports dict, Mapping, etc.)
        if hasattr(current, "get"):
            get_method = current.get
            if callable(get_method):
                if key in current:
                    current = current[key]
                    continue

        # Try attribute access
        if hasattr(current, key):
            current = getattr(current, key)
            continue

        # Try direct indexing (for list-like or other indexable objects)
        try:
            current = current[key]
            continue
        except (KeyError, TypeError, IndexError):
            pass

        # If we get here, the key was not found
        path_str = " -> ".join(path_traversed)
        raise KeyError(f"Key path '{path_str}' not found in object")

    return current
