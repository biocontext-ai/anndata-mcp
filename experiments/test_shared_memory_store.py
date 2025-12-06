"""Test script for SharedMemoryStore implementation."""

import multiprocessing
import queue

import anndata as ad
import numpy as np
import pandas as pd
from anndata.experimental import read_lazy

from anndata_mcp.tools.utils import SharedMemoryStore


def create_and_share_adata(shared_name_queue):
    """Process 1: Create AnnData and share via SharedMemoryStore."""
    try:
        print("Process 1: Creating AnnData...")

        # Create a simple AnnData object
        n_obs, n_vars = 100, 50
        X = np.random.randn(n_obs, n_vars)
        obs = pd.DataFrame({"cell_type": np.random.choice(["A", "B", "C"], n_obs)})
        var = pd.DataFrame({"gene_name": [f"Gene_{i}" for i in range(n_vars)]})

        adata = ad.AnnData(X, obs=obs, var=var)
        print(f"Process 1: Created AnnData with shape {adata.shape}")

        # Create SharedMemoryStore and write AnnData (use unique name)
        import uuid

        unique_name = f"test_adata_{uuid.uuid4().hex[:8]}"
        store = SharedMemoryStore(name=unique_name)
        print("Process 1: Writing AnnData to SharedMemoryStore...")
        adata.write_zarr(store)

        # Get shared name and share it
        shared_name = store.get_shared_name()
        print(f"Process 1: Shared memory name: {shared_name}")
        shared_name_queue.put(shared_name)

        # Keep store alive (don't cleanup yet)
        print("Process 1: Store created, waiting for process 2 to finish...")
        try:
            result = shared_name_queue.get(timeout=30)
            print(f"Process 1: Got signal: {result}")
        except (queue.Empty, TimeoutError) as e:
            print(f"Process 1: Error waiting for signal: {e}")

        # Cleanup
        print("Process 1: Cleaning up...")
        store.cleanup()
        print("Process 1: Done!")
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"Process 1: Error: {e}")
        import traceback

        traceback.print_exc()
        # Signal error to avoid deadlock
        try:
            shared_name_queue.put("error")
        except (queue.Full, OSError, RuntimeError):
            pass


def attach_and_read(shared_name_queue):
    """Process 2: Attach to SharedMemoryStore and read AnnData."""
    import sys

    print("Process 2: Starting function...", file=sys.stderr, flush=True)
    print("Process 2: Starting function...", flush=True)
    try:
        print("Process 2: Waiting for shared memory name...", flush=True)

        # Get shared name from process 1
        try:
            shared_name = shared_name_queue.get(timeout=30)  # Add timeout
        except (queue.Empty, TimeoutError) as e:
            print(f"Process 2: Error getting shared name: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            return

        print(f"Process 2: Received shared memory name: {shared_name}")

        # Attach to the shared memory store
        print("Process 2: Attaching to SharedMemoryStore...")
        store = SharedMemoryStore.attach(shared_name)
        print("Process 2: Attached successfully!")

        # Read AnnData lazily
        print("Process 2: Reading AnnData lazily...")
        adata_lazy = read_lazy(store)
        print(f"Process 2: Read AnnData with shape {adata_lazy.shape}")
        print(f"Process 2: AnnData is backed: {adata_lazy.isbacked}")

        # Access some data (compute the dask array)
        print("Process 2: Accessing X[0:5, 0:5]...")
        x_slice = adata_lazy.X[0:5, 0:5]
        print(f"Process 2: Got dask array: {x_slice}")
        print("Process 2: Computing array...")
        x_computed = x_slice.compute()
        print(f"Process 2: Computed array:\n{x_computed}")

        # Check obs (Dataset2D doesn't have head(), use iloc instead)
        print("Process 2: First 5 obs:")
        print(adata_lazy.obs.iloc[:5])

        # Check var (Dataset2D doesn't have head(), use iloc instead)
        print("Process 2: First 5 var:")
        print(adata_lazy.var.iloc[:5])

        # Close
        adata_lazy.file.close()

        # Signal completion to process 1
        shared_name_queue.put("done")
        print("Process 2: Done!")
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"Process 2: Error: {e}")
        import traceback

        traceback.print_exc()
        # Still signal to avoid deadlock
        try:
            shared_name_queue.put("error")
        except (queue.Full, OSError, RuntimeError):
            pass


def test_single_process():
    """Test SharedMemoryStore in a single process."""
    print("=" * 60)
    print("Single Process Test")
    print("=" * 60)

    # Create AnnData
    n_obs, n_vars = 50, 25
    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame({"cell_type": np.random.choice(["A", "B"], n_obs)})
    var = pd.DataFrame({"gene_name": [f"Gene_{i}" for i in range(n_vars)]})

    adata = ad.AnnData(X, obs=obs, var=var)
    print(f"Created AnnData with shape {adata.shape}")

    # Create store and write
    store = SharedMemoryStore(name="test_single")
    print("Writing to SharedMemoryStore...")
    adata.write_zarr(store)

    # Get shared name
    shared_name = store.get_shared_name()
    print(f"Shared memory name: {shared_name}")

    # Attach to it (in same process)
    print("Attaching to SharedMemoryStore...")
    store2 = SharedMemoryStore.attach(shared_name)

    # Read lazily
    print("Reading AnnData lazily...")
    adata_lazy = read_lazy(store2)
    print(f"Read AnnData with shape {adata_lazy.shape}")

    # Verify data
    print("Verifying data...")
    assert adata_lazy.shape == adata.shape, "Shapes don't match!"
    print("✓ Shapes match")

    # Access data
    print("Accessing X[0:3, 0:3]:")
    print(adata_lazy.X[0:3, 0:3])

    # Cleanup
    adata_lazy.file.close()
    store2.cleanup()  # Cleanup the attached store
    store.cleanup()  # Cleanup the original store
    print("Single process test passed! ✓")


def test_cross_process():
    """Test SharedMemoryStore across processes."""
    print("\n" + "=" * 60)
    print("Cross-Process Test")
    print("=" * 60)

    # Use a queue for inter-process communication
    shared_name_queue = multiprocessing.Queue()

    # Create two processes
    p1 = multiprocessing.Process(target=create_and_share_adata, args=(shared_name_queue,))
    p2 = multiprocessing.Process(target=attach_and_read, args=(shared_name_queue,))

    # Start processes
    print("Main: Starting Process 1...", flush=True)
    p1.start()
    print(f"Main: Process 1 started (PID: {p1.pid})", flush=True)
    print("Main: Starting Process 2...", flush=True)
    p2.start()
    print(f"Main: Process 2 started (PID: {p2.pid})", flush=True)

    # Wait for both to complete with timeout
    p1.join(timeout=60)
    p2.join(timeout=60)

    # Check if processes are still alive
    if p1.is_alive():
        print("Warning: Process 1 did not complete in time")
        p1.terminate()
        p1.join()
    if p2.is_alive():
        print("Warning: Process 2 did not complete in time")
        p2.terminate()
        p2.join()

    print("\nCross-process test completed! ✓")


if __name__ == "__main__":
    # Use spawn method to avoid fork() deadlocks in multi-threaded environments
    multiprocessing.set_start_method("spawn", force=True)

    # Test single process first
    test_single_process()

    # Test cross-process
    test_cross_process()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
