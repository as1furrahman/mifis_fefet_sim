"""
MIFIS FeFET Simulation - GAA Architecture Comparison
====================================================
Compare GAA architectures with different wrap angles.

Author: Thesis Project
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from core.solver import MIFIS3DSolver
from core.config import get_gaa_config, get_fast_simulation_config
from core.postprocess import extract_memory_window


def run_gaa_comparison(fast_mode: bool = True):
    """
    Compare GAA architectures with different wrap angles.

    Tests wrap angles: 0° (planar), 90°, 180° (FinFET), 270°, 360° (full GAA)

    Args:
        fast_mode: If True, uses faster settings

    Returns:
        DataFrame with comparison results
    """
    print("\n[GAA Architecture Comparison]")
    print("="*50)

    # Wrap angles to test
    wrap_angles = [0, 90, 180, 270, 360]

    sim_config = get_fast_simulation_config() if fast_mode else None
    if sim_config is None:
        from core.config import SimulationConfig
        sim_config = SimulationConfig()

    print(f"  Testing wrap angles: {wrap_angles}°")
    print(f"  Fast mode: {fast_mode}")

    comparison_results = []

    for angle in wrap_angles:
        print(f"\n  Simulating wrap angle = {angle}°...")

        # Get configuration
        device_config = get_gaa_config(wrap_angle=angle)

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

        # Create solver
        solver = MIFIS3DSolver(f"gaa_{angle}deg")
        solver.create_3d_mesh(device_config, output_dir="data/meshes")
        solver.setup_physics(device_config)

        # Run simulation
        results = solver.run_voltage_sweep(device_config, sim_config)

        # Extract metrics
        mw = extract_memory_window(results)
        Id_max = results["Id"].max()

        # Enhancement factor
        enhancement = solver.architecture_factor

        print(f"    MW: {mw:.3f} V, Max Id: {Id_max:.2e} A, Enhancement: {enhancement:.3f}x")

        comparison_results.append({
            "wrap_angle": angle,
            "memory_window": mw,
            "max_current": Id_max,
            "enhancement_factor": enhancement,
            "architecture": "Planar" if angle == 0 else "FinFET" if angle == 180 else "GAA"
        })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)

    # Calculate improvement over planar
    planar_mw = comparison_df[comparison_df["wrap_angle"] == 0]["memory_window"].values[0]
    comparison_df["mw_improvement_pct"] = (
        (comparison_df["memory_window"] - planar_mw) / planar_mw * 100
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"gaa_comparison_{timestamp}.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n  Data saved: {csv_path}")

    # Plot comparison
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Memory window vs wrap angle
    ax1.plot(comparison_df["wrap_angle"], comparison_df["memory_window"],
             'o-', linewidth=2, markersize=8)
    ax1.set_xlabel("Wrap Angle (degrees)")
    ax1.set_ylabel("Memory Window (V)")
    ax1.set_title("Memory Window vs Wrap Angle")
    ax1.grid(True, alpha=0.3)

    # Enhancement factor
    ax2.bar(comparison_df["wrap_angle"], comparison_df["enhancement_factor"],
            color=['blue', 'green', 'orange', 'red', 'purple'])
    ax2.set_xlabel("Wrap Angle (degrees)")
    ax2.set_ylabel("Enhancement Factor")
    ax2.set_title("Architecture Enhancement Factor")
    ax2.axhline(y=1.0, color='k', linestyle='--', label='Baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = plot_dir / f"gaa_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")
    plt.close()

    # Print summary
    print("\n  Summary:")
    print(comparison_df.to_string(index=False))

    return comparison_df


if __name__ == "__main__":
    run_gaa_comparison()
