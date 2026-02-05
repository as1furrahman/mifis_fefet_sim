"""
MIFIS FeFET Simulation - Parallel Execution Framework
=====================================================
Provides parallel execution for parameter sweeps using multiprocessing.

Key Features:
- Multi-process execution (DEVSIM-safe)
- Progress tracking
- Error handling and graceful degradation
- Configurable worker pool size

Author: Thesis Project
Date: February 2026
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
import pandas as pd


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    n_workers: Optional[int] = None  # None = auto-detect (n_cpus - 1)
    show_progress: bool = True
    timeout_per_task: Optional[float] = None  # seconds, None = no timeout

    def __post_init__(self):
        """Set default number of workers if not specified."""
        if self.n_workers is None:
            # Use n_cpus - 1 to leave one core free
            self.n_workers = max(1, mp.cpu_count() - 1)
        else:
            # Ensure at least 1 worker
            self.n_workers = max(1, self.n_workers)


class ParallelSweepRunner:
    """
    Execute parameter sweeps in parallel using multiprocessing.

    This is safe to use with DEVSIM because each process gets its own
    DEVSIM instance. Thread-based parallelism would NOT be safe.

    Example usage:
        runner = ParallelSweepRunner(n_workers=4)
        results = runner.run_sweep(
            sweep_func=simulate_single_point,
            parameter_list=[10, 12, 14, 16],
            config=device_config,
            fast_mode=True
        )
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel sweep runner.

        Args:
            config: Parallel execution configuration
        """
        self.config = config or ParallelConfig()
        print(f"[ParallelSweepRunner] Configured with {self.config.n_workers} workers")

    def run_sweep(
        self,
        sweep_func: Callable,
        parameter_list: List[Any],
        **sweep_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run a parameter sweep in parallel.

        Args:
            sweep_func: Function to call for each parameter value.
                        Signature: func(param_value, **kwargs) -> Dict
            parameter_list: List of parameter values to sweep over
            **sweep_kwargs: Additional keyword arguments passed to sweep_func

        Returns:
            List of result dictionaries in the same order as parameter_list
        """
        n_params = len(parameter_list)

        if self.config.show_progress:
            print(f"  Running {n_params} simulations in parallel "
                  f"({self.config.n_workers} workers)...")

        start_time = time.time()
        results = [None] * n_params  # Pre-allocate to maintain order

        # Create process pool and submit all tasks
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Submit all tasks and track their futures
            future_to_idx = {}
            for idx, param_value in enumerate(parameter_list):
                future = executor.submit(
                    _run_single_simulation,
                    sweep_func,
                    param_value,
                    sweep_kwargs
                )
                future_to_idx[future] = idx

            # Collect results as they complete
            n_completed = 0
            n_failed = 0

            for future in as_completed(future_to_idx.keys()):
                idx = future_to_idx[future]
                param_value = parameter_list[idx]

                try:
                    result = future.result(timeout=self.config.timeout_per_task)
                    results[idx] = result
                    n_completed += 1

                    if self.config.show_progress:
                        elapsed = time.time() - start_time
                        print(f"    [{n_completed}/{n_params}] "
                              f"Param={param_value} completed "
                              f"(elapsed: {elapsed:.1f}s)")

                except Exception as e:
                    n_failed += 1
                    print(f"    ERROR: Param={param_value} failed: {e}")
                    # Leave results[idx] as None to indicate failure

        elapsed_total = time.time() - start_time

        if self.config.show_progress:
            print(f"  Parallel sweep completed in {elapsed_total:.1f}s")
            print(f"    Success: {n_completed}/{n_params}, Failed: {n_failed}/{n_params}")
            if n_completed > 0:
                avg_time = elapsed_total / n_completed
                speedup_estimate = avg_time * n_params / elapsed_total
                print(f"    Est. speedup: {speedup_estimate:.1f}x vs sequential")

        # Filter out None results from failures
        return [r for r in results if r is not None]

    def run_sweep_batched(
        self,
        sweep_func: Callable,
        parameter_list: List[Any],
        batch_size: Optional[int] = None,
        **sweep_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run sweep in batches (useful for very large parameter spaces).

        Args:
            sweep_func: Function to call for each parameter value
            parameter_list: List of parameter values to sweep over
            batch_size: Number of tasks per batch (default: n_workers * 2)
            **sweep_kwargs: Additional keyword arguments passed to sweep_func

        Returns:
            List of result dictionaries
        """
        if batch_size is None:
            batch_size = self.config.n_workers * 2

        all_results = []
        n_total = len(parameter_list)

        for i in range(0, n_total, batch_size):
            batch = parameter_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            n_batches = (n_total + batch_size - 1) // batch_size

            print(f"\n[Batch {batch_num}/{n_batches}] Processing {len(batch)} parameters...")

            batch_results = self.run_sweep(sweep_func, batch, **sweep_kwargs)
            all_results.extend(batch_results)

        return all_results


def _run_single_simulation(
    sweep_func: Callable,
    param_value: Any,
    sweep_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Worker function to run a single simulation.

    This function is executed in a separate process, so it has its own
    DEVSIM instance and can't interfere with other processes.

    Args:
        sweep_func: The simulation function to call
        param_value: The parameter value for this simulation
        sweep_kwargs: Additional keyword arguments

    Returns:
        Dictionary with simulation results
    """
    try:
        # Suppress DEVSIM output in worker processes (optional)
        # This keeps the console output clean
        import sys
        import io

        # Uncomment the following to suppress all worker output:
        # old_stdout = sys.stdout
        # sys.stdout = io.StringIO()

        # Run the simulation
        result = sweep_func(param_value, **sweep_kwargs)

        # Restore stdout if it was suppressed
        # sys.stdout = old_stdout

        return result

    except Exception as e:
        # Return error information instead of raising
        # This allows the main process to handle it gracefully
        return {
            "error": str(e),
            "param_value": param_value,
            "success": False
        }


# =============================================================================
# HELPER FUNCTIONS FOR COMMON PATTERNS
# =============================================================================

def get_optimal_worker_count() -> int:
    """
    Get recommended number of workers based on system resources.

    Returns:
        Optimal number of workers (typically n_cpus - 1)
    """
    n_cpus = mp.cpu_count()

    # Leave one core free for system/main thread
    optimal = max(1, n_cpus - 1)

    # For small systems (1-2 cores), use all cores
    if n_cpus <= 2:
        optimal = n_cpus

    return optimal


def estimate_parallel_speedup(
    n_tasks: int,
    n_workers: int,
    overhead_fraction: float = 0.1
) -> float:
    """
    Estimate speedup from parallelization.

    Uses Amdahl's law with overhead estimation.

    Args:
        n_tasks: Number of tasks to run
        n_workers: Number of parallel workers
        overhead_fraction: Fraction of time spent on overhead (0-1)

    Returns:
        Expected speedup factor
    """
    if n_workers <= 0 or n_tasks <= 0:
        return 1.0

    # Parallel fraction (what can be parallelized)
    parallel_fraction = 1.0 - overhead_fraction

    # Amdahl's law
    speedup = 1.0 / (
        overhead_fraction + parallel_fraction / min(n_workers, n_tasks)
    )

    return speedup


def benchmark_sweep(
    sweep_func: Callable,
    parameter_list: List[Any],
    n_workers_list: Optional[List[int]] = None,
    **sweep_kwargs
) -> pd.DataFrame:
    """
    Benchmark a sweep with different numbers of workers.

    Useful for finding optimal parallelization settings.

    Args:
        sweep_func: Simulation function to benchmark
        parameter_list: Parameters to sweep
        n_workers_list: List of worker counts to test (None = [1, 2, 4, 8])
        **sweep_kwargs: Additional arguments for sweep_func

    Returns:
        DataFrame with benchmark results
    """
    if n_workers_list is None:
        max_workers = get_optimal_worker_count()
        n_workers_list = [1, 2, 4, 8]
        n_workers_list = [n for n in n_workers_list if n <= max_workers]
        if max_workers not in n_workers_list:
            n_workers_list.append(max_workers)

    benchmark_results = []

    for n_workers in n_workers_list:
        print(f"\n[Benchmark] Testing with {n_workers} workers...")

        config = ParallelConfig(n_workers=n_workers, show_progress=False)
        runner = ParallelSweepRunner(config=config)

        start_time = time.time()
        results = runner.run_sweep(sweep_func, parameter_list, **sweep_kwargs)
        elapsed = time.time() - start_time

        n_success = len([r for r in results if r.get("success", True)])

        benchmark_results.append({
            "n_workers": n_workers,
            "elapsed_time": elapsed,
            "n_tasks": len(parameter_list),
            "n_success": n_success,
            "tasks_per_second": n_success / elapsed if elapsed > 0 else 0,
            "speedup_vs_sequential": None  # Will fill in below
        })

        print(f"  Completed in {elapsed:.1f}s ({n_success} successful)")

    # Calculate speedup relative to sequential (1 worker)
    df = pd.DataFrame(benchmark_results)
    if len(df) > 0:
        sequential_time = df[df["n_workers"] == 1]["elapsed_time"].iloc[0]
        df["speedup_vs_sequential"] = sequential_time / df["elapsed_time"]

    return df
