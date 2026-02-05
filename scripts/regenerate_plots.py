#!/usr/bin/env python3
"""
Unified Plot Regeneration Script
=================================
Regenerates all 1D, 2D, and 3D plots from simulation data.

Usage:
    python scripts/regenerate_plots.py --all              # Generate all plots
    python scripts/regenerate_plots.py --1d               # Generate 1D plots only
    python scripts/regenerate_plots.py --2d               # Generate 2D plots only
    python scripts/regenerate_plots.py --3d               # Generate 3D plots only
    python scripts/regenerate_plots.py --1d --2d          # Generate 1D and 2D plots
    python scripts/regenerate_plots.py --dpi 600          # High-resolution plots

Author: MIFIS FeFET Simulation Framework
Date: February 2026
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_config(config_path="config/device_params.json"):
    """Load device configuration from JSON."""
    config_file = project_root / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def regenerate_1d_plots(config, output_dir, dpi=300):
    """
    Regenerate 1D baseline plots.

    Generates:
    - P-E hysteresis loop (ferroelectric characterization)
    - P-V hysteresis (memory window)
    - E-field distribution across MIFIS stack
    """
    print("\n=== Regenerating 1D Plots ===")

    # Import 1D plotting functions
    from core.results_generator import (
        plot_pe_loop,
        plot_pv_hysteresis,
        plot_efield_distribution
    )

    output_path = Path(output_dir) / "1D"
    output_path.mkdir(parents=True, exist_ok=True)

    print("  → P-E Loop...")
    # Generate P-E loop from config parameters
    try:
        plot_pe_loop(
            config=config,
            save_path=output_path / "pe_loop_1d.png",
            dpi=dpi
        )
        print("    ✓ pe_loop_1d.png")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    print("  → P-V Hysteresis...")
    try:
        plot_pv_hysteresis(
            config=config,
            save_path=output_path / "pv_hysteresis_1d.png",
            dpi=dpi
        )
        print("    ✓ pv_hysteresis_1d.png")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    print("  → E-field Distribution...")
    try:
        plot_efield_distribution(
            config=config,
            save_path=output_path / "efield_voltage_1d.png",
            dpi=dpi
        )
        print("    ✓ efield_voltage_1d.png")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    print(f"✓ 1D plots saved to {output_path}")


def regenerate_2d_plots(config, output_dir, dpi=300):
    """
    Regenerate 2D transistor plots.

    Generates:
    - Id-Vg transfer characteristics (dual polarization states)
    - Id-Vd output characteristics
    - 2D potential distribution maps
    """
    print("\n=== Regenerating 2D Plots ===")

    # Import 2D plotting functions
    from core.results_generator import (
        plot_id_vg_dual_state,
        plot_id_vd_family,
        plot_2d_potential_comparison
    )

    output_path = Path(output_dir) / "2D"
    output_path.mkdir(parents=True, exist_ok=True)

    print("  → Id-Vg Transfer Characteristics...")
    try:
        plot_id_vg_dual_state(
            config=config,
            save_path=output_path / "id_vg_2d.png",
            dpi=dpi
        )
        print("    ✓ id_vg_2d.png")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    print("  → Id-Vd Output Characteristics...")
    try:
        plot_id_vd_family(
            config=config,
            save_path=output_path / "id_vd_2d.png",
            dpi=dpi
        )
        print("    ✓ id_vd_2d.png")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    print("  → 2D Potential Maps...")
    try:
        plot_2d_potential_comparison(
            config=config,
            save_path=output_path / "potential_2d.png",
            dpi=dpi
        )
        print("    ✓ potential_2d.png")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    print(f"✓ 2D plots saved to {output_path}")


def regenerate_3d_plots(config, output_dir, dpi=300):
    """
    Regenerate 3D architecture comparison plots.

    Generates:
    - 3D potential distribution comparison (Planar vs FinFET vs GAA)
    - Memory window vs architecture bar chart
    - GAA enhancement analysis
    """
    print("\n=== Regenerating 3D Plots ===")

    # Import 3D plotting functions
    from core.results_generator import (
        plot_3d_potential_comparison,
        plot_mw_vs_architecture
    )

    output_path = Path(output_dir) / "3D"
    output_path.mkdir(parents=True, exist_ok=True)

    print("  → 3D Potential Comparison...")
    try:
        plot_3d_potential_comparison(
            config=config,
            save_path=output_path / "potential_3d.png",
            dpi=dpi
        )
        print("    ✓ potential_3d.png")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    print("  → Memory Window vs Architecture...")
    try:
        plot_mw_vs_architecture(
            config=config,
            save_path=output_path / "mw_architecture.png",
            dpi=dpi
        )
        print("    ✓ mw_architecture.png")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    print(f"✓ 3D plots saved to {output_path}")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Unified plot regeneration for MIFIS FeFET simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/regenerate_plots.py --all              # All plots
  python scripts/regenerate_plots.py --1d --2d          # 1D and 2D only
  python scripts/regenerate_plots.py --3d --dpi 600     # 3D high-res
        """
    )

    # Plot selection arguments
    parser.add_argument('--all', action='store_true',
                        help='Generate all plots (1D, 2D, 3D)')
    parser.add_argument('--1d', action='store_true',
                        help='Generate 1D plots only')
    parser.add_argument('--2d', action='store_true',
                        help='Generate 2D plots only')
    parser.add_argument('--3d', action='store_true',
                        help='Generate 3D plots only')

    # Output configuration
    parser.add_argument('--output', type=str, default='plots',
                        help='Output directory (default: plots/)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Plot resolution in DPI (default: 300)')
    parser.add_argument('--config', type=str, default='config/device_params.json',
                        help='Device configuration file (default: config/device_params.json)')

    args = parser.parse_args()

    # If no specific dimension selected, default to --all
    if not (args.all or args.__dict__.get('1d') or args.__dict__.get('2d') or args.__dict__.get('3d')):
        args.all = True

    print("=" * 60)
    print("MIFIS FeFET Unified Plot Regeneration")
    print("=" * 60)

    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    try:
        config = load_config(args.config)
        print(f"✓ Configuration loaded")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return 1

    # Generate requested plots
    try:
        if args.all or args.__dict__.get('1d'):
            regenerate_1d_plots(config, args.output, args.dpi)

        if args.all or args.__dict__.get('2d'):
            regenerate_2d_plots(config, args.output, args.dpi)

        if args.all or args.__dict__.get('3d'):
            regenerate_3d_plots(config, args.output, args.dpi)

        print("\n" + "=" * 60)
        print("✓ All requested plots generated successfully")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Error during plot generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
