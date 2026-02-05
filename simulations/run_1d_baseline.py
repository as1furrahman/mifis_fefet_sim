"""
MIFIS FeFET Simulation - 1D Baseline
====================================
Run 1D baseline characterization of the MIFIS stack.
Outputs match fefet_simulation_base format exactly.

Author: Thesis Project
Date: February 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from core.solver import MIFISSolver
from core.config import (
    get_baseline_config,
    get_fast_simulation_config,
    get_ultra_fast_config,
    get_instant_config,
    get_demo_config,
    get_balanced_config,
    get_accurate_config,
    SimulationConfig
)
from core.postprocess import extract_memory_window, extract_vth
from core.results_generator import generate_all_mifis_outputs, generate_mifis_results_plot


def get_speed_config():
    """Get simulation config based on global speed mode."""
    import builtins
    speed_mode = getattr(builtins, 'SIMULATION_SPEED_MODE', 'balanced')

    config_map = {
        'instant': get_instant_config,
        'demo': get_demo_config,
        'fast': get_fast_simulation_config,
        'ultrafast': get_ultra_fast_config,
        'balanced': get_balanced_config,
        'accurate': get_accurate_config,
    }

    config_func = config_map.get(speed_mode, get_balanced_config)
    return config_func(), speed_mode


def run_1d_baseline(fast_mode: bool = True):
    """
    Run 1D baseline MIFIS FeFET simulation.
    
    Outputs in fefet_simulation_base format:
    - 6-panel plot (P-V, E-V, P-E, evolution, structure, summary)
    - CSV summary with MW, geometry, materials
    - Pickle file with full results

    Args:
        fast_mode: If True, uses faster (less accurate) settings

    Returns:
        DataFrame with simulation results
    """
    print("\n[1D Baseline Simulation]")
    print("="*50)

    # Get configuration based on speed mode
    device_config = get_baseline_config()
    sim_config, speed_mode = get_speed_config()

    # Override parameters from device_params.json if available
    import json
    config_path = Path(__file__).parent.parent / "config" / "device_params.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            params = json.load(f)
            # Override simulation parameters
            if 'simulation' in params:
                sim_config.Vg_start = params['simulation'].get('Vg_start', sim_config.Vg_start)
                sim_config.Vg_end = params['simulation'].get('Vg_end', sim_config.Vg_end)
                sim_config.Vg_step = params['simulation'].get('Vg_step', sim_config.Vg_step)
            # Override geometry parameters
            if 'device' in params and 'geometry' in params['device']:
                geom = params['device']['geometry']
                device_config.geometry.t_top_il = geom.get('t_top_il', device_config.geometry.t_top_il)
                device_config.geometry.t_fe = geom.get('t_fe', device_config.geometry.t_fe)
                device_config.geometry.t_bottom_il = geom.get('t_bottom_il', device_config.geometry.t_bottom_il)
            # Override material parameters
            if 'materials' in params and 'ferroelectric' in params['materials']:
                fe = params['materials']['ferroelectric']
                device_config.fe_material.Pr = fe.get('Pr', device_config.fe_material.Pr)
                device_config.fe_material.Ps = fe.get('Ps', device_config.fe_material.Ps)
                device_config.fe_material.Ec = fe.get('Ec', device_config.fe_material.Ec)
                device_config.fe_material.epsilon_r = fe.get('epsilon_r', device_config.fe_material.epsilon_r)

    # Calculate mesh points based on speed
    mesh_points = {
        'instant': 50,
        'demo': 100,
        'fast': 150,
        'ultrafast': 150,
        'balanced': 300,
        'accurate': 500,
    }
    n_points = mesh_points.get(speed_mode, 300)

    print(f"  Device: MIFIS FeFET")
    print(f"  Stack: TiN({device_config.geometry.t_gate}nm) / "
          f"SiO2({device_config.geometry.t_top_il}nm) / "
          f"HZO({device_config.geometry.t_fe}nm) / "
          f"SiO2({device_config.geometry.t_bottom_il}nm) / "
          f"Si({device_config.geometry.t_channel}nm)")
    print(f"  Voltage sweep: {sim_config.Vg_start}V to {sim_config.Vg_end}V, "
          f"step={sim_config.Vg_step}V")
    print(f"  Speed mode: {speed_mode.upper()} (mesh: {n_points} pts)")

    # Create solver
    print("\n  Creating 1D mesh...")
    solver = MIFISSolver("baseline_1d")
    solver.create_1d_mesh(device_config, n_points=n_points)

    # Setup physics
    print("  Setting up physics equations...")
    solver.setup_physics(device_config)

    # Run voltage sweep
    print("  Running voltage sweep...")
    results = solver.run_voltage_sweep(device_config, sim_config)

    # Extract metrics
    print("\n  Extracting metrics...")
    mw = extract_memory_window(results)
    vth_fwd, vth_rev = extract_vth(results)

    print(f"    Memory Window: {mw:.3f} V")
    print(f"    Vth (forward):  {vth_fwd:.3f} V")
    print(f"    Vth (reverse):  {vth_rev:.3f} V")

    # Calculate on/off ratio
    Id_on = results["Id"].max()
    Id_off = results["Id"].min()
    on_off_ratio = Id_on / Id_off if Id_off > 0 else np.inf
    print(f"    On/Off Ratio: {on_off_ratio:.2e}")

    # Prepare data for results generator
    geometry = {
        't_gate': device_config.geometry.t_gate,
        't_top_il': device_config.geometry.t_top_il,
        't_fe': device_config.geometry.t_fe,
        't_bottom_il': device_config.geometry.t_bottom_il,
        't_channel': device_config.geometry.t_channel,
    }
    
    # Extract V, P, E arrays from DEVSIM results
    # Column names from DEVSIM: Vg, Id, phi_surface, Polarization, E_fe, direction, converged
    V_gate = results['Vg'].values if 'Vg' in results.columns else np.linspace(-3, 3, len(results))

    # Polarization in µC/cm² - use 'Polarization' column from DEVSIM
    if 'Polarization' in results.columns:
        P_fe = results['Polarization'].values * 1e-6  # Convert µC/cm² to C/cm²
    elif 'P_fe' in results.columns:
        P_fe = results['P_fe'].values
    else:
        P_fe = np.zeros(len(results))

    # FIXED: Use E-field from simulation results (with voltage division) or calculate correctly
    if 'E_fe' in results.columns:
        E_fe = results['E_fe'].values  # Already calculated correctly in solver
    else:
        # Fallback: Calculate E-field with proper voltage division
        t_top_il = device_config.geometry.t_top_il * 1e-9  # nm to m
        t_fe = device_config.geometry.t_fe * 1e-9  # nm to m
        t_bottom_il = device_config.geometry.t_bottom_il * 1e-9  # nm to m

        eps_top_il = device_config.top_il.epsilon_r
        eps_fe = device_config.fe_material.epsilon_r
        eps_bottom_il = device_config.top_il.epsilon_r

        # Voltage division: V_i = V_total * (t_i/ε_i) / Σ(t_j/ε_j)
        sum_t_over_eps = (t_top_il / eps_top_il +
                         t_fe / eps_fe +
                         t_bottom_il / eps_bottom_il)
        V_fe = V_gate * (t_fe / eps_fe) / sum_t_over_eps
        E_fe = V_fe / t_fe  # V/m
    
    simulation_results = {
        'V_gate': V_gate,
        'P_fe': P_fe,
        'E_fe': E_fe,
        'memory_window': mw,
    }

    # Generate outputs in fefet_simulation_base format
    # Individual plots go to plots/1D/, results go to results/
    print("\n  Generating outputs (fefet_simulation_base format)...")
    
    outputs = generate_all_mifis_outputs(
        simulation_results=simulation_results,
        geometry=geometry,
        output_dir=".",  # Project root - creates plots/1D/ and results/
        phase="1d",
        show_plot=False
    )
    
    # Also save raw data (fixed name - replaces previous)
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old 1D raw data files
    for old_file in raw_dir.glob("baseline_1d*.csv"):
        try:
            old_file.unlink()
        except:
            pass
    
    csv_path = raw_dir / "baseline_1d.csv"
    results.to_csv(csv_path, index=False)
    print(f"  Raw data saved: {csv_path}")

    print("\n  ✓ All outputs generated successfully!")
    print(f"    Plot: {outputs.get('plot', 'N/A')}")
    print(f"    CSV:  {outputs.get('csv', 'N/A')}")
    print(f"    PKL:  {outputs.get('pkl', 'N/A')}")

    return results


if __name__ == "__main__":
    run_1d_baseline()
