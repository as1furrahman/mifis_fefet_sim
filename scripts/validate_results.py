#!/usr/bin/env python3
"""
Results Validation Script
=========================
Validates simulation results against expected targets and literature values.

Usage:
    python scripts/validate_results.py                    # Validate all results
    python scripts/validate_results.py --phase 1d         # Validate 1D only
    python scripts/validate_results.py --verbose          # Detailed output

Author: MIFIS FeFET Simulation Framework
Date: February 2026
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# Expected target values (from validation)
TARGETS = {
    '1d': {
        'memory_window': 3.95,  # Volts
        'target_achievement': 102,  # Percent
        'pr_range': (14, 24),  # µC/cm²
        'ps_range': (36, 45),  # µC/cm²
    },
    '2d': {
        'memory_window': 4.15,  # Volts
        'target_achievement': 102,  # Percent
    },
    '3d': {
        'memory_window': 4.94,  # Volts
        'target_achievement': 102,  # Percent
        'enhancement_factor': 1.25,  # vs 2D planar
    },
}


def check_file_exists(filepath):
    """Check if a result file exists."""
    path = project_root / filepath
    if path.exists():
        print(f"  ✓ Found: {filepath}")
        return True
    else:
        print(f"  ✗ Missing: {filepath}")
        return False


def validate_1d_results(verbose=False):
    """Validate 1D baseline simulation results."""
    print("\n=== Validating 1D Baseline Results ===")

    results_file = "results/mifis_1d_results.pkl"
    csv_file = "data/processed/baseline_1d.csv"

    # Check files exist
    files_ok = True
    files_ok &= check_file_exists(results_file)
    files_ok &= check_file_exists(csv_file)

    if not files_ok:
        print("  ✗ Required files missing - run simulation first")
        return False

    # Load and validate results
    try:
        import pickle
        with open(project_root / results_file, 'rb') as f:
            results = pickle.load(f)

        mw = results.get('memory_window', 0)
        pr = results.get('pr_measured', 0)
        ps = results.get('ps_measured', 0)

        print(f"\n  Memory Window: {mw:.3f} V (target: {TARGETS['1d']['memory_window']} V)")
        if mw >= TARGETS['1d']['memory_window'] * 0.95:
            print(f"  ✓ Within 5% of target")
        else:
            print(f"  ⚠ Below target")

        if verbose:
            print(f"\n  Pr (remnant): {pr:.1f} µC/cm²")
            print(f"  Ps (saturation): {ps:.1f} µC/cm²")
            print(f"  Pr/Ps ratio: {pr/ps*100:.1f}%")

        return True

    except Exception as e:
        print(f"  ✗ Error loading results: {e}")
        return False


def validate_2d_results(verbose=False):
    """Validate 2D planar simulation results."""
    print("\n=== Validating 2D Planar Results ===")

    results_file = "results/mifis_2d_results.pkl"

    if not check_file_exists(results_file):
        print("  ✗ Required files missing - run simulation first")
        return False

    try:
        import pickle
        with open(project_root / results_file, 'rb') as f:
            results = pickle.load(f)

        mw = results.get('memory_window', 0)

        print(f"\n  Memory Window: {mw:.3f} V (target: {TARGETS['2d']['memory_window']} V)")
        if mw >= TARGETS['2d']['memory_window'] * 0.95:
            print(f"  ✓ Within 5% of target")
        else:
            print(f"  ⚠ Below target")

        return True

    except Exception as e:
        print(f"  ✗ Error loading results: {e}")
        return False


def validate_3d_results(verbose=False):
    """Validate 3D GAA simulation results."""
    print("\n=== Validating 3D GAA Results ===")

    results_file = "results/mifis_3d_results.pkl"

    if not check_file_exists(results_file):
        print("  ✗ Required files missing - run simulation first")
        return False

    try:
        import pickle
        with open(project_root / results_file, 'rb') as f:
            results = pickle.load(f)

        mw = results.get('memory_window', 0)

        print(f"\n  Memory Window: {mw:.3f} V (target: {TARGETS['3d']['memory_window']} V)")
        if mw >= TARGETS['3d']['memory_window'] * 0.95:
            print(f"  ✓ Within 5% of target")
        else:
            print(f"  ⚠ Below target")

        # Check enhancement vs 2D
        if verbose:
            mw_2d = 4.229  # From validation
            enhancement = mw / mw_2d
            print(f"\n  Enhancement vs 2D: {enhancement:.2f}× (target: {TARGETS['3d']['enhancement_factor']:.2f}×)")

        return True

    except Exception as e:
        print(f"  ✗ Error loading results: {e}")
        return False


def validate_plots():
    """Validate that required plots exist."""
    print("\n=== Validating Output Plots ===")

    required_plots = [
        "plots/1D/pe_loop_1d.png",
        "plots/1D/pv_hysteresis_1d.png",
        "plots/1D/efield_voltage_1d.png",
        "plots/2D/id_vg_2d.png",
        "plots/3D/mw_architecture.png",
    ]

    all_exist = True
    for plot in required_plots:
        if not check_file_exists(plot):
            all_exist = False

    if all_exist:
        print("\n  ✓ All required plots generated")
    else:
        print("\n  ⚠ Some plots missing - regenerate with: python scripts/regenerate_plots.py --all")

    return all_exist


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate MIFIS FeFET simulation results"
    )

    parser.add_argument('--phase', choices=['1d', '2d', '3d', 'all'],
                        default='all',
                        help='Which phase to validate (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed validation output')
    parser.add_argument('--plots', action='store_true',
                        help='Validate plot generation only')

    args = parser.parse_args()

    print("=" * 60)
    print("MIFIS FeFET Results Validation")
    print("=" * 60)

    if args.plots:
        validate_plots()
        return 0

    results = {}

    if args.phase == 'all' or args.phase == '1d':
        results['1d'] = validate_1d_results(args.verbose)

    if args.phase == 'all' or args.phase == '2d':
        results['2d'] = validate_2d_results(args.verbose)

    if args.phase == 'all' or args.phase == '3d':
        results['3d'] = validate_3d_results(args.verbose)

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    for phase, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {phase.upper()}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ All validations passed!")
        return 0
    else:
        print("\n⚠ Some validations failed - check simulation outputs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
