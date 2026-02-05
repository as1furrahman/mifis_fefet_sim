"""
MIFIS FeFET Simulation - Dual-Gate Multi-Level Memory
=====================================================
Simulate dual-gate MIFIS FeFET for 4-level (2-bit) memory.

Author: Thesis Project
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from core.solver import MIFIS3DSolver
from core.config import get_dualgate_config, get_fast_simulation_config, Architecture


def run_dualgate_simulation(fast_mode: bool = True):
    """
    Simulate dual-gate MIFIS FeFET with 4 polarization states.

    States:
        "00": Both gates in down state (low current)
        "01": Top down, Bottom up
        "10": Top up, Bottom down
        "11": Both gates in up state (high current)

    Args:
        fast_mode: If True, uses faster settings

    Returns:
        Dictionary with results for each state
    """
    print("\n[Dual-Gate Multi-Level Memory Simulation]")
    print("="*50)

    states = ["00", "01", "10", "11"]

    sim_config = get_fast_simulation_config() if fast_mode else None
    if sim_config is None:
        from core.config import SimulationConfig
        sim_config = SimulationConfig()

    print(f"  Testing 4 polarization states: {states}")
    print(f"  Fast mode: {fast_mode}")

    state_results = {}
    state_currents = []

    for state in states:
        print(f"\n  Simulating state {state}...")

        # Get configuration
        device_config = get_dualgate_config(state=state)

        # Create solver
        solver = MIFIS3DSolver(f"dualgate_{state}")
        solver.create_3d_mesh(device_config, output_dir="data/meshes")
        solver.setup_physics(device_config)

        # Run simulation
        results = solver.run_voltage_sweep(device_config, sim_config)

        # Extract read current at Vg=0V
        try:
            read_current = results[np.abs(results["Vg"]) < 0.1]["Id"].mean()
        except:
            read_current = results["Id"].mean()

        state_results[state] = results

        print(f"    State {state}: Read current = {read_current:.2e} A")

        state_currents.append({
            "state": state,
            "read_current": read_current,
            "level": int(state, 2),  # Binary to decimal
        })

    # Create summary DataFrame
    summary_df = pd.DataFrame(state_currents)
    summary_df = summary_df.sort_values("level")

    # Calculate separation ratios
    print("\n  State Separation Analysis:")
    currents_sorted = summary_df["read_current"].values
    for i in range(len(currents_sorted) - 1):
        ratio = currents_sorted[i+1] / currents_sorted[i]
        print(f"    Level {i} to {i+1}: {ratio:.2f}x")

    # Total on/off ratio
    on_off = currents_sorted[-1] / currents_sorted[0]
    print(f"    Total On/Off: {on_off:.2e}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"dualgate_{timestamp}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Data saved: {csv_path}")

    # Plot
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of read currents
    ax1.bar(summary_df["state"], summary_df["read_current"],
            color=['blue', 'green', 'orange', 'red'])
    ax1.set_xlabel("Polarization State")
    ax1.set_ylabel("Read Current (A)")
    ax1.set_title("4-Level Memory State Currents")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Transfer characteristics for all states
    for state in states:
        results = state_results[state]
        fwd = results[results["direction"] == "forward"]
        ax2.semilogy(fwd["Vg"], np.abs(fwd["Id"]),
                     label=f"State {state}", linewidth=2)

    ax2.set_xlabel("Gate Voltage (V)")
    ax2.set_ylabel("|Id| (A)")
    ax2.set_title("Transfer Characteristics - All States")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = plot_dir / f"dualgate_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")
    plt.close()

    return state_results


if __name__ == "__main__":
    run_dualgate_simulation()
