"""
MIFIS FeFET Simulation - Parameter Optimization Sweeps
======================================================
Run parameter sweeps to optimize FE and IL thicknesses.

Now supports PARALLEL EXECUTION for significant speedup!

Author: Thesis Project
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from core.solver import MIFISSolver
from core.config import get_baseline_config, get_fast_simulation_config, DeviceGeometry
from core.postprocess import extract_memory_window, extract_vth
from core.parallel import ParallelSweepRunner, ParallelConfig, get_optimal_worker_count


# =============================================================================
# SINGLE-PARAMETER SIMULATION FUNCTIONS (for parallel execution)
# =============================================================================

def _simulate_single_fe_thickness(t_fe: float, fast_mode: bool = True) -> Dict[str, Any]:
    """
    Simulate a single FE thickness value.

    This function is called by parallel workers.

    Args:
        t_fe: Ferroelectric thickness in nm
        fast_mode: If True, uses faster settings

    Returns:
        Dictionary with simulation results
    """
    try:
        sim_config = get_fast_simulation_config() if fast_mode else None
        if sim_config is None:
            from core.config import SimulationConfig
            sim_config = SimulationConfig()

        # Create configuration
        config = get_baseline_config()
        config.geometry.t_fe = t_fe

        # Run simulation
        solver = MIFISSolver(f"fe_sweep_{t_fe:.1f}nm")
        solver.create_1d_mesh(config, n_points=200 if fast_mode else 500)
        solver.setup_physics(config)
        results = solver.run_voltage_sweep(config, sim_config)

        # Extract metrics
        mw = extract_memory_window(results)
        vth_fwd, vth_rev = extract_vth(results)
        Id_on = results["Id"].max()

        return {
            "t_fe_nm": t_fe,
            "memory_window": mw,
            "vth_forward": vth_fwd,
            "vth_reverse": vth_rev,
            "max_current": Id_on,
            "success": True
        }
    except Exception as e:
        print(f"  ERROR simulating t_fe={t_fe}: {e}")
        return {
            "t_fe_nm": t_fe,
            "memory_window": 0.0,
            "vth_forward": 0.0,
            "vth_reverse": 0.0,
            "max_current": 0.0,
            "success": False,
            "error": str(e)
        }


def _simulate_single_il_thickness(t_il: float, fast_mode: bool = True) -> Dict[str, Any]:
    """
    Simulate a single IL thickness value.

    Args:
        t_il: Top interlayer thickness in nm
        fast_mode: If True, uses faster settings

    Returns:
        Dictionary with simulation results
    """
    try:
        sim_config = get_fast_simulation_config() if fast_mode else None
        if sim_config is None:
            from core.config import SimulationConfig
            sim_config = SimulationConfig()

        config = get_baseline_config()
        config.geometry.t_top_il = t_il

        solver = MIFISSolver(f"il_sweep_{t_il:.1f}nm")
        solver.create_1d_mesh(config, n_points=200 if fast_mode else 500)
        solver.setup_physics(config)
        results = solver.run_voltage_sweep(config, sim_config)

        mw = extract_memory_window(results)
        vth_fwd, vth_rev = extract_vth(results)

        return {
            "t_top_il_nm": t_il,
            "memory_window": mw,
            "vth_forward": vth_fwd,
            "vth_reverse": vth_rev,
            "success": True
        }
    except Exception as e:
        print(f"  ERROR simulating t_il={t_il}: {e}")
        return {
            "t_top_il_nm": t_il,
            "memory_window": 0.0,
            "vth_forward": 0.0,
            "vth_reverse": 0.0,
            "success": False,
            "error": str(e)
        }


def _simulate_single_il_material(material_info: tuple, fast_mode: bool = True) -> Dict[str, Any]:
    """
    Simulate a single IL material.

    Args:
        material_info: Tuple of (mat_name, mat_props_dict)
        fast_mode: If True, uses faster settings

    Returns:
        Dictionary with simulation results
    """
    try:
        mat_name, mat_props = material_info

        sim_config = get_fast_simulation_config() if fast_mode else None
        if sim_config is None:
            from core.config import SimulationConfig
            sim_config = SimulationConfig()

        config = get_baseline_config()
        # Modify IL permittivity
        config.materials.eps_top_il = mat_props["eps_r"]

        solver = MIFISSolver(f"il_material_{mat_name}")
        solver.create_1d_mesh(config, n_points=200 if fast_mode else 500)
        solver.setup_physics(config)
        results = solver.run_voltage_sweep(config, sim_config)

        mw = extract_memory_window(results)
        vth_fwd, vth_rev = extract_vth(results)

        return {
            "material": mat_name,
            "material_name": mat_props["name"],
            "eps_r": mat_props["eps_r"],
            "memory_window": mw,
            "vth_forward": vth_fwd,
            "vth_reverse": vth_rev,
            "success": True
        }
    except Exception as e:
        mat_name, mat_props = material_info
        print(f"  ERROR simulating material={mat_name}: {e}")
        return {
            "material": mat_name,
            "material_name": mat_props.get("name", "Unknown"),
            "eps_r": mat_props.get("eps_r", 0.0),
            "memory_window": 0.0,
            "vth_forward": 0.0,
            "vth_reverse": 0.0,
            "success": False,
            "error": str(e)
        }


def _simulate_single_gate_length(L_g: float, fast_mode: bool = True) -> Dict[str, Any]:
    """
    Simulate a single gate length value.

    Args:
        L_g: Gate length in nm
        fast_mode: If True, uses faster settings

    Returns:
        Dictionary with simulation results
    """
    try:
        sim_config = get_fast_simulation_config() if fast_mode else None
        if sim_config is None:
            from core.config import SimulationConfig
            sim_config = SimulationConfig()

        config = get_baseline_config()
        config.geometry.L_gate = L_g

        solver = MIFISSolver(f"gate_length_{L_g}nm")
        solver.create_1d_mesh(config, n_points=200 if fast_mode else 500)
        solver.setup_physics(config)
        results = solver.run_voltage_sweep(config, sim_config)

        mw = extract_memory_window(results)
        vth_fwd, vth_rev = extract_vth(results)
        Id_on = results["Id"].max()

        return {
            "L_gate_nm": L_g,
            "memory_window": mw,
            "vth_forward": vth_fwd,
            "vth_reverse": vth_rev,
            "max_current": Id_on,
            "success": True
        }
    except Exception as e:
        print(f"  ERROR simulating L_g={L_g}: {e}")
        return {
            "L_gate_nm": L_g,
            "memory_window": 0.0,
            "vth_forward": 0.0,
            "vth_reverse": 0.0,
            "max_current": 0.0,
            "success": False,
            "error": str(e)
        }


# =============================================================================
# MAIN SWEEP FUNCTIONS (with parallel support)
# =============================================================================

def run_fe_thickness_sweep(fast_mode: bool = True, parallel: bool = True, n_workers: int = None):
    """
    Sweep ferroelectric thickness to find optimal value.

    Args:
        fast_mode: If True, uses faster settings
        parallel: If True, runs simulations in parallel (default: True)
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        DataFrame with sweep results
    """
    print("\n[FE Thickness Sweep]")
    print("="*50)

    # Thickness range (nm)
    t_fe_values = [8.0, 10.0, 12.0, 13.8, 16.0, 18.0, 20.0]

    print(f"  Testing FE thicknesses: {t_fe_values} nm")
    print(f"  Fast mode: {fast_mode}")
    print(f"  Parallel: {parallel}" + (f" ({n_workers or get_optimal_worker_count()} workers)" if parallel else ""))

    if parallel:
        # PARALLEL EXECUTION
        config = ParallelConfig(n_workers=n_workers, show_progress=True)
        runner = ParallelSweepRunner(config=config)
        sweep_results = runner.run_sweep(
            sweep_func=_simulate_single_fe_thickness,
            parameter_list=t_fe_values,
            fast_mode=fast_mode
        )
    else:
        # SEQUENTIAL EXECUTION (original behavior)
        sweep_results = []
        for t_fe in t_fe_values:
            print(f"\n  Simulating t_FE = {t_fe} nm...")
            result = _simulate_single_fe_thickness(t_fe, fast_mode=fast_mode)
            print(f"    MW: {result['memory_window']:.3f} V, Vth_fwd: {result['vth_forward']:.3f} V")
            sweep_results.append(result)

    # Create DataFrame
    sweep_df = pd.DataFrame(sweep_results)

    # Find optimal
    optimal_idx = sweep_df["memory_window"].idxmax()
    optimal_t_fe = sweep_df.loc[optimal_idx, "t_fe_nm"]
    optimal_mw = sweep_df.loc[optimal_idx, "memory_window"]

    print(f"\n  Optimal t_FE: {optimal_t_fe} nm (MW = {optimal_mw:.3f} V)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"fe_thickness_sweep_{timestamp}.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    # Plot
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sweep_df["t_fe_nm"], sweep_df["memory_window"],
            'o-', linewidth=2, markersize=8)
    ax.axvline(optimal_t_fe, color='r', linestyle='--',
               label=f'Optimal: {optimal_t_fe} nm')
    ax.set_xlabel("FE Thickness (nm)")
    ax.set_ylabel("Memory Window (V)")
    ax.set_title("Memory Window vs FE Thickness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = plot_dir / f"fe_thickness_sweep_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")
    plt.close()

    return sweep_df


def run_il_thickness_sweep(fast_mode: bool = True, parallel: bool = True, n_workers: int = None):
    """
    Sweep top interlayer thickness.

    Args:
        fast_mode: If True, uses faster settings
        parallel: If True, runs simulations in parallel (default: True)
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        DataFrame with sweep results
    """
    print("\n[Top IL Thickness Sweep]")
    print("="*50)

    # IL thickness range (nm)
    t_il_values = [2.0, 3.0, 4.0, 5.0, 6.0]

    print(f"  Testing top IL thicknesses: {t_il_values} nm")
    print(f"  Fast mode: {fast_mode}")
    print(f"  Parallel: {parallel}" + (f" ({n_workers or get_optimal_worker_count()} workers)" if parallel else ""))

    if parallel:
        # PARALLEL EXECUTION
        config = ParallelConfig(n_workers=n_workers, show_progress=True)
        runner = ParallelSweepRunner(config=config)
        sweep_results = runner.run_sweep(
            sweep_func=_simulate_single_il_thickness,
            parameter_list=t_il_values,
            fast_mode=fast_mode
        )
    else:
        # SEQUENTIAL EXECUTION
        sweep_results = []
        for t_il in t_il_values:
            print(f"\n  Simulating t_top_IL = {t_il} nm...")
            result = _simulate_single_il_thickness(t_il, fast_mode=fast_mode)
            print(f"    MW: {result['memory_window']:.3f} V")
            sweep_results.append(result)

    sweep_df = pd.DataFrame(sweep_results)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"il_thickness_sweep_{timestamp}.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    return sweep_df


def run_il_material_comparison(fast_mode: bool = True, parallel: bool = True, n_workers: int = None):
    """
    Compare different interlayer materials (SiO2, Al2O3, HfO2).

    Args:
        fast_mode: If True, uses faster settings
        parallel: If True, runs simulations in parallel (default: True)
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        DataFrame with material comparison results
    """
    print("\n[IL Material Comparison]")
    print("="*50)

    # Material comparison (using dielectric constant as proxy)
    materials = {
        "SiO2": {"eps_r": 3.9, "name": "Silicon Dioxide"},
        "Al2O3": {"eps_r": 9.0, "name": "Aluminum Oxide"},
        "HfO2": {"eps_r": 25.0, "name": "Hafnium Oxide"},
    }

    print(f"  Testing materials: {list(materials.keys())}")
    print(f"  Fast mode: {fast_mode}")
    print(f"  Parallel: {parallel}" + (f" ({n_workers or get_optimal_worker_count()} workers)" if parallel else ""))

    if parallel:
        # PARALLEL EXECUTION
        # Convert materials dict to list of tuples for parallel processing
        material_list = list(materials.items())

        config = ParallelConfig(n_workers=n_workers, show_progress=True)
        runner = ParallelSweepRunner(config=config)
        sweep_results = runner.run_sweep(
            sweep_func=_simulate_single_il_material,
            parameter_list=material_list,
            fast_mode=fast_mode
        )
    else:
        # SEQUENTIAL EXECUTION
        sweep_results = []
        for mat_name, mat_props in materials.items():
            print(f"\n  Simulating IL material: {mat_props['name']}...")
            result = _simulate_single_il_material((mat_name, mat_props), fast_mode=fast_mode)
            print(f"    MW: {result['memory_window']:.3f} V")
            sweep_results.append(result)

    sweep_df = pd.DataFrame(sweep_results)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"il_material_comparison_{timestamp}.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    return sweep_df


def run_gate_length_scaling(fast_mode: bool = True, parallel: bool = True, n_workers: int = None):
    """
    Study gate length scaling effects.

    Args:
        fast_mode: If True, uses faster settings
        parallel: If True, runs simulations in parallel (default: True)
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        DataFrame with gate length scaling results
    """
    print("\n[Gate Length Scaling Study]")
    print("="*50)

    # Gate lengths to test (nm)
    gate_lengths = [20, 30, 50, 100, 150, 200]

    print(f"  Testing gate lengths: {gate_lengths} nm")
    print(f"  Fast mode: {fast_mode}")
    print(f"  Parallel: {parallel}" + (f" ({n_workers or get_optimal_worker_count()} workers)" if parallel else ""))

    if parallel:
        # PARALLEL EXECUTION
        config = ParallelConfig(n_workers=n_workers, show_progress=True)
        runner = ParallelSweepRunner(config=config)
        sweep_results = runner.run_sweep(
            sweep_func=_simulate_single_gate_length,
            parameter_list=gate_lengths,
            fast_mode=fast_mode
        )
    else:
        # SEQUENTIAL EXECUTION
        sweep_results = []
        for L_g in gate_lengths:
            print(f"\n  Simulating L_g = {L_g} nm...")
            result = _simulate_single_gate_length(L_g, fast_mode=fast_mode)
            print(f"    MW: {result['memory_window']:.3f} V, I_on: {result['max_current']:.2e} A")
            sweep_results.append(result)

    sweep_df = pd.DataFrame(sweep_results)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"gate_length_scaling_{timestamp}.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"  Data saved: {csv_path}")

    return sweep_df


def run_all_sweeps(fast_mode: bool = True, parallel: bool = True, n_workers: int = None):
    """
    Run all parameter sweeps.

    Args:
        fast_mode: If True, uses faster settings
        parallel: If True, runs simulations in parallel (default: True)
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        Dictionary with all sweep results
    """
    import time

    print("\n[Parameter Optimization Sweeps]")
    print("="*60)

    if parallel:
        print(f"  PARALLEL MODE: Using {n_workers or get_optimal_worker_count()} workers")
    else:
        print("  SEQUENTIAL MODE")

    overall_start = time.time()
    results = {}

    # FE thickness sweep
    results["fe_thickness"] = run_fe_thickness_sweep(
        fast_mode=fast_mode, parallel=parallel, n_workers=n_workers
    )

    # IL thickness sweep
    results["il_thickness"] = run_il_thickness_sweep(
        fast_mode=fast_mode, parallel=parallel, n_workers=n_workers
    )

    # IL material comparison
    results["il_material"] = run_il_material_comparison(
        fast_mode=fast_mode, parallel=parallel, n_workers=n_workers
    )

    # Gate length scaling
    results["gate_length"] = run_gate_length_scaling(
        fast_mode=fast_mode, parallel=parallel, n_workers=n_workers
    )

    overall_time = time.time() - overall_start

    print("\n" + "="*60)
    print(f"  All parameter sweeps completed in {overall_time:.1f}s!")
    print("="*60)

    return results


if __name__ == "__main__":
    # Default: run in parallel mode
    run_all_sweeps(parallel=True)
