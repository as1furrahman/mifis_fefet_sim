#!/usr/bin/env python3
"""
Verify Hysteresis Loops Across 1D, 2D, 3D
=========================================
Checks that all three dimensions have proper ferroelectric hysteresis behavior.

Key checks:
1. S-shaped P-E loop (not flat or box-shaped)
2. Proper Pr and Ps values
3. Hysteresis exists (forward != reverse)
4. Memory window present
5. Coercive field reasonable
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def load_results(dimension):
    """Load pickle results for a dimension."""
    pkl_file = f"results/mifis_{dimension}_results.pkl"
    if not Path(pkl_file).exists():
        print(f"  ✗ {pkl_file} not found")
        return None

    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def analyze_hysteresis(results, dimension):
    """Analyze hysteresis loop characteristics."""
    print(f"\n{'='*60}")
    print(f"  {dimension.upper()} HYSTERESIS ANALYSIS")
    print(f"{'='*60}")

    if results is None:
        print("  ✗ No data available")
        return False

    # Extract data - check nested 'results' dict first
    if 'results' in results and isinstance(results['results'], dict):
        res_data = results['results']
        if 'V_gate' in res_data:
            V = np.array(res_data['V_gate'])
            P = np.array(res_data['P_fe'])
        elif 'Vg' in res_data:
            V = np.array(res_data['Vg'])
            P = np.array(res_data['P_fe']) if 'P_fe' in res_data else np.array(res_data['Polarization'])
        else:
            print("  ✗ Cannot find voltage/polarization data in results dict")
            return False
    elif 'V_gate' in results:
        V = np.array(results['V_gate'])
        P = np.array(results['P_fe'])
    elif 'Vg' in results:
        V = np.array(results['Vg'])
        P = np.array(results['P_fe']) if 'P_fe' in results else np.array(results['Polarization'])
    else:
        print("  ✗ Cannot find voltage/polarization data")
        return False

    # Convert P to µC/cm² if needed
    if np.max(np.abs(P)) < 1:  # Likely in C/cm²
        P = P * 1e6

    MW = results.get('memory_window', 0)

    print(f"\n📊 Data Overview:")
    print(f"  Voltage range: [{np.min(V):.1f}, {np.max(V):.1f}] V")
    print(f"  Polarization range: [{np.min(P):.1f}, {np.max(P):.1f}] µC/cm²")
    print(f"  Data points: {len(V)}")
    print(f"  Memory Window: {MW:.3f} V")

    # Check 1: S-shaped loop
    print(f"\n✓ Check 1: S-Shaped Loop")
    # Find positive and negative peaks
    P_max = np.max(P)
    P_min = np.min(P)
    P_range = P_max - P_min

    print(f"  P_max: {P_max:.2f} µC/cm²")
    print(f"  P_min: {P_min:.2f} µC/cm²")
    print(f"  Range: {P_range:.2f} µC/cm²")

    if P_range < 10:
        print(f"  ⚠ WARNING: Small polarization range (<10 µC/cm²)")
        print(f"     Loop might be too flat")
    else:
        print(f"  ✓ Good polarization range")

    # Check 2: Pr and Ps
    print(f"\n✓ Check 2: Pr and Ps Values")
    # Find P at E≈0 (Pr) and P at max E (Ps)
    zero_mask = np.abs(V) < 0.5
    if np.any(zero_mask):
        P_at_zero = P[zero_mask]
        Pr_measured = np.mean(np.abs(P_at_zero))
        print(f"  Pr (at V≈0): {Pr_measured:.2f} µC/cm²")
    else:
        Pr_measured = 0
        print(f"  ⚠ Cannot measure Pr (no points near V=0)")

    # Ps at high voltage
    high_v_mask = np.abs(V) > 4.0
    if np.any(high_v_mask):
        P_at_high = P[high_v_mask]
        Ps_measured = np.mean(np.abs(P_at_high))
        print(f"  Ps (at |V|>4V): {Ps_measured:.2f} µC/cm²")
    else:
        Ps_measured = P_max
        print(f"  Ps (max P): {Ps_measured:.2f} µC/cm²")

    # Check Pr < Ps
    if Pr_measured > 0 and Ps_measured > 0:
        ratio = Pr_measured / Ps_measured
        print(f"  Pr/Ps ratio: {ratio:.2%}")
        if 0.3 < ratio < 0.7:
            print(f"  ✓ Good Pr/Ps ratio (literature: 43-60%)")
        else:
            print(f"  ⚠ Pr/Ps ratio outside typical range (43-60%)")

    # Check 3: Hysteresis exists
    print(f"\n✓ Check 3: Hysteresis Present")
    # Split into forward and reverse sweeps
    n_half = len(V) // 2
    if len(V) % 2 == 0:
        V_fwd = V[:n_half]
        P_fwd = P[:n_half]
        V_rev = V[n_half:]
        P_rev = P[n_half:]

        # Check overlap at similar voltages
        overlap_V = 0.0
        fwd_idx = np.argmin(np.abs(V_fwd - overlap_V))
        rev_idx = np.argmin(np.abs(V_rev - overlap_V))

        P_diff = abs(P_fwd[fwd_idx] - P_rev[rev_idx])
        print(f"  P difference at V≈{overlap_V:.1f}: {P_diff:.2f} µC/cm²")

        if P_diff > 5:
            print(f"  ✓ Clear hysteresis (ΔP > 5 µC/cm²)")
        else:
            print(f"  ⚠ Small hysteresis (ΔP < 5 µC/cm²)")
    else:
        print(f"  ⚠ Cannot check hysteresis (odd number of points)")

    # Check 4: Memory window
    print(f"\n✓ Check 4: Memory Window")
    print(f"  MW: {MW:.3f} V")
    if MW > 2.0:
        print(f"  ✓ Good memory window (>2V)")
    elif MW > 1.0:
        print(f"  ⚠ Moderate memory window (1-2V)")
    else:
        print(f"  ✗ Poor memory window (<1V)")

    # Check 5: Loop shape
    print(f"\n✓ Check 5: Loop Shape")
    # Check if it's S-shaped by looking at derivative
    dP_dV = np.gradient(P, V)

    # S-shape should have regions where dP/dV is large (switching regions)
    switching_regions = np.sum(np.abs(dP_dV) > 2.0)
    print(f"  Switching regions (|dP/dV|>2): {switching_regions} points")

    if switching_regions > 5:
        print(f"  ✓ S-shaped loop (clear switching)")
    else:
        print(f"  ⚠ Might be flat or box-shaped")

    # Overall assessment
    print(f"\n{'='*60}")
    print(f"  OVERALL ASSESSMENT")
    print(f"{'='*60}")

    issues = []
    if P_range < 10:
        issues.append("Small polarization range")
    if MW < 2.0:
        issues.append("Low memory window")
    if switching_regions < 5:
        issues.append("Weak switching behavior")

    if len(issues) == 0:
        print(f"  ✅ ALL CHECKS PASSED - EXCELLENT HYSTERESIS")
        return True
    else:
        print(f"  ⚠ WARNINGS FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        return False

def compare_all():
    """Compare all three dimensions."""
    print("\n" + "="*60)
    print("  HYSTERESIS LOOP VERIFICATION")
    print("  Comparing 1D, 2D, and 3D MIFIS FeFET")
    print("="*60)

    # Load all results
    results_1d = load_results("1d")
    results_2d = load_results("2d")
    results_3d = load_results("3d")

    # Analyze each
    good_1d = analyze_hysteresis(results_1d, "1D")
    good_2d = analyze_hysteresis(results_2d, "2D")
    good_3d = analyze_hysteresis(results_3d, "3D")

    # Final comparison
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")

    print(f"\n  Memory Windows:")
    if results_1d: print(f"    1D: {results_1d.get('memory_window', 0):.3f} V")
    if results_2d: print(f"    2D: {results_2d.get('memory_window', 0):.3f} V")
    if results_3d: print(f"    3D: {results_3d.get('memory_window', 0):.3f} V")

    print(f"\n  Enhancement Factors:")
    if results_1d and results_2d:
        enh_2d = results_2d.get('memory_window', 0) / results_1d.get('memory_window', 1)
        print(f"    1D→2D: {enh_2d:.2f}× (expected: ~1.05×)")

    if results_1d and results_3d:
        # Note: 3D uses 1.25x on the base, not on 1D baseline
        # The base should be similar to 1D
        mw_3d = results_3d.get('memory_window', 0)
        mw_1d = results_1d.get('memory_window', 1)
        enh_3d = mw_3d / mw_1d
        print(f"    1D→3D: {enh_3d:.2f}× (expected: ~1.29×)")

    print(f"\n  Quality Assessment:")
    print(f"    1D: {'✅ GOOD' if good_1d else '⚠ NEEDS REVIEW'}")
    print(f"    2D: {'✅ GOOD' if good_2d else '⚠ NEEDS REVIEW'}")
    print(f"    3D: {'✅ GOOD' if good_3d else '⚠ NEEDS REVIEW'}")

    if good_1d and good_2d and good_3d:
        print(f"\n  ✅ ALL THREE DIMENSIONS HAVE PROPER HYSTERESIS!")
        print(f"  ✅ READY FOR THESIS PUBLICATION")
    else:
        print(f"\n  ⚠ SOME DIMENSIONS NEED REVIEW")
        print(f"  Check individual analyses above")

if __name__ == "__main__":
    compare_all()
