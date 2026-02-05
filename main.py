#!/usr/bin/env python3
"""
MIFIS FeFET Simulation Framework
================================
Main Thesis Simulation Driver (v2.1)

REQUIRES DEVSIM - No mock mode available.

Features:
- Auto-cleans previous outputs before new runs
- 6 simulation phases
- Fast/accurate mode selection

Author: Thesis Project
Date: February 2026
"""

import sys
import os
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Setup path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def clean_outputs(keep_latest: bool = False):
    """
    Clean ALL previous simulation outputs.
    Use clean_phase_outputs() for phase-specific cleaning.
    
    Args:
        keep_latest: If True, keeps the most recent file in each directory
    """
    output_dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "meshes",
        PROJECT_ROOT / "plots",
        PROJECT_ROOT / "logs",
    ]
    
    print("\n[Cleaning ALL previous outputs...]")
    
    for dir_path in output_dirs:
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            if files:
                if keep_latest:
                    # Keep most recent file
                    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                    files_to_remove = files[1:]  # Keep first (newest)
                else:
                    files_to_remove = files
                
                for f in files_to_remove:
                    if f.is_file():
                        f.unlink()
                        print(f"  Removed: {f.name}")
                    elif f.is_dir():
                        shutil.rmtree(f)
                        print(f"  Removed dir: {f.name}")
    
    print("  Cleanup complete.\n")


def clean_phase_outputs(phase: str):
    """
    Clean ONLY outputs from a specific simulation phase.
    
    Phase patterns:
    - '1d': Files containing 'baseline', '1d', 'baseline_1d'
    - '2d': Files containing 'planar', '2d', 'planar_2d'
    - '3d': Files containing '3d', 'mifis_3d', 'gaa'
    - 'gaa': Files containing 'gaa', 'architecture'
    - 'dualgate': Files containing 'dualgate', 'multilevel'
    - 'sweeps': Files containing 'sweep', 'optimization'
    
    Args:
        phase: Phase identifier ('1d', '2d', '3d', 'gaa', 'dualgate', 'sweeps')
    """
    # Define file patterns for each phase
    phase_patterns = {
        '1d': ['baseline', '1d_', '_1d', 'baseline_1d'],
        '2d': ['planar', '2d_', '_2d', 'planar_2d'],
        '3d': ['3d_', '_3d', 'mifis_3d'],
        'gaa': ['gaa', 'architecture', 'nanowire'],
        'dualgate': ['dualgate', 'dual_gate', 'multilevel'],
        'sweeps': ['sweep', 'optimization', 'param_'],
    }
    
    patterns = phase_patterns.get(phase.lower(), [])
    if not patterns:
        print(f"  Unknown phase: {phase}. No files removed.")
        return
    
    output_dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "meshes",
        PROJECT_ROOT / "plots",
    ]
    
    print(f"\n[Cleaning {phase.upper()} phase outputs only...]")
    removed_count = 0
    
    for dir_path in output_dirs:
        if dir_path.exists():
            for f in dir_path.iterdir():
                f_name_lower = f.name.lower()
                # Check if file matches any pattern for this phase
                if any(pattern in f_name_lower for pattern in patterns):
                    if f.is_file():
                        f.unlink()
                        print(f"  Removed: {f.name}")
                        removed_count += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        print(f"  Removed dir: {f.name}")
                        removed_count += 1
    
    if removed_count == 0:
        print(f"  No {phase.upper()} files found to clean.")
    else:
        print(f"  Removed {removed_count} {phase.upper()} file(s).\n")


def ensure_directories():
    """Create all required directories."""
    dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "meshes",
        PROJECT_ROOT / "plots",
        PROJECT_ROOT / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def check_devsim():
    """Verify DEVSIM is installed and working."""
    try:
        import devsim
        version = devsim.__version__ if hasattr(devsim, '__version__') else "unknown"
        print(f"  DEVSIM: ✓ Version {version}")
        return True
    except ImportError:
        print("  DEVSIM: ✗ NOT INSTALLED")
        print("\n" + "="*60)
        print("  ERROR: DEVSIM is REQUIRED")
        print("="*60)
        print("\n  Install DEVSIM with:")
        print("    pip install devsim")
        print("="*60 + "\n")
        return False


def check_gmsh():
    """Check if Gmsh is available for 3D meshing."""
    try:
        import gmsh
        print(f"  Gmsh:   ✓ Available")
        return True
    except ImportError:
        print("  Gmsh:   ✗ Not installed (optional, needed for 3D)")
        return False


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     MIFIS FeFET 3D TCAD SIMULATION FRAMEWORK  v2.1              ║
║     ═══════════════════════════════════════════                  ║
║                                                                  ║
║     Device:  Metal-Insulator-Ferroelectric-Insulator-Si         ║
║     Stack:   TiN(50nm)/SiO2(4nm)/HZO(13.8nm)/SiO2(0.7nm)/Si     ║
║                                                                  ║
║     Solver:  DEVSIM (REQUIRED)                                  ║
║                                                                  ║
║     Thesis Project - February 2026                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_phase(phase_num: int, name: str, runner_func, skip: bool = False) -> Any:
    """Run a simulation phase with timing."""
    print(f"\n{'='*60}")
    print(f"  PHASE {phase_num}: {name.upper()}")
    print(f"{'='*60}")
    
    if skip:
        print("  [SKIPPED]")
        return None
    
    start = time.time()
    try:
        result = runner_func()
        elapsed = time.time() - start
        print(f"\n  Phase {phase_num} completed in {elapsed:.1f} seconds")
        return result
    except Exception as e:
        print(f"\n  [ERROR] Phase {phase_num} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(phases: Dict[str, bool] = None, clean: bool = True):
    """
    Main driver function.
    
    Args:
        phases: Dictionary of phases to run. Default runs all.
                Keys: '1d', '2d', '3d', 'gaa', 'dualgate', 'sweeps'
        clean: If True, removes previous outputs for ONLY the phases being run
               (other phase outputs are preserved)
    """
    print_banner()
    
    start_time = time.time()
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    print("\nChecking Dependencies:")
    if not check_devsim():
        print("Cannot proceed without DEVSIM. Exiting.")
        sys.exit(1)
    
    gmsh_ok = check_gmsh()
    
    # Ensure directories exist
    ensure_directories()
    
    # Default: run all phases
    if phases is None:
        phases = {
            '1d': True,
            '2d': True,
            '3d': gmsh_ok,
            'gaa': True,
            'dualgate': True,
            'sweeps': True,
        }
    
    # Phase-specific cleaning: only clean outputs for phases being run
    if clean:
        for phase_name, should_run in phases.items():
            if should_run:
                clean_phase_outputs(phase_name)
    
    # Import runners
    from simulations.run_1d_baseline import run_1d_baseline
    from simulations.run_2d_planar import run_2d_planar
    from simulations.run_3d_mifis import run_3d_mifis
    from simulations.run_gaa import run_gaa_comparison
    from simulations.run_dualgate import run_dualgate_simulation
    from simulations.run_sweeps import run_all_sweeps
    
    # Execute phases
    results = {}
    
    if phases.get('1d', False):
        results['1d'] = run_phase(1, "1D Baseline Characterization", run_1d_baseline)
    
    if phases.get('2d', False):
        results['2d'] = run_phase(2, "2D Planar FeFET Simulation", run_2d_planar)
    
    if phases.get('3d', False):
        if gmsh_ok:
            results['3d'] = run_phase(3, "3D MIFIS FeFET Simulation", run_3d_mifis)
        else:
            print("\n  Phase 3 skipped: Gmsh not available")
    
    if phases.get('gaa', False):
        results['gaa'] = run_phase(4, "GAA Architecture Comparison", run_gaa_comparison)
    
    if phases.get('dualgate', False):
        results['dualgate'] = run_phase(5, "Dual-Gate Multi-Level Study", run_dualgate_simulation)
    
    if phases.get('sweeps', False):
        # Get parallel config from simulation settings
        from core.config import get_fast_simulation_config
        sim_config = get_fast_simulation_config()
        sweep_func = lambda: run_all_sweeps(
            fast_mode=True,
            parallel=sim_config.enable_parallel,
            n_workers=sim_config.n_workers
        )
        results['sweeps'] = run_phase(6, "Parameter Optimization Sweeps", sweep_func)
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("  ALL SIMULATIONS COMPLETE")
    print("="*60)
    print(f"\n  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print(f"\n  Phases executed:")
    for phase, result in results.items():
        status = "✓ SUCCESS" if result is not None else "✗ FAILED/SKIPPED"
        print(f"    - {phase}: {status}")
    
    print(f"\n  Output directories:")
    print(f"    - Data:   {PROJECT_ROOT / 'data'}")
    print(f"    - Plots:  {PROJECT_ROOT / 'plots'}")
    print()
    
    # Generate log
    log_path = PROJECT_ROOT / "logs" / f"simulation_log.txt"
    with open(log_path, 'w') as f:
        f.write("MIFIS FeFET Simulation Log v2.1\n")
        f.write(f"{'='*40}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Total Time: {total_time:.1f}s\n\n")
        f.write(f"Phases Completed:\n")
        for phase, result in results.items():
            status = "SUCCESS" if result is not None else "FAILED/SKIPPED"
            f.write(f"  - {phase}: {status}\n")
    
    print(f"  Log saved: {log_path}")
    
    return results


def run_quick():
    """Run quick validation (1D only)."""
    return main(phases={'1d': True}, clean=True)


def run_1d_2d():
    """Run 1D and 2D phases."""
    return main(phases={'1d': True, '2d': True}, clean=True)


def run_3d_study():
    """Run 3D phases only."""
    return main(phases={'3d': True, 'gaa': True}, clean=True)


def run_all():
    """Run all phases."""
    return main(clean=True)


def run_no_clean():
    """Run all phases WITHOUT cleaning previous outputs."""
    return main(clean=False)


if __name__ == "__main__":
    # Speed mode selection
    SPEED_MODES = {
        'instant': 'get_instant_config',   # ~10-30 seconds
        'demo': 'get_demo_config',          # ~30-60 seconds
        'fast': 'get_fast_simulation_config',  # ~1-2 minutes
        'ultrafast': 'get_ultra_fast_config',  # ~1-2 minutes
        'balanced': 'get_balanced_config',  # ~8-12 minutes (RECOMMENDED)
        'accurate': 'get_accurate_config',  # ~30+ minutes (high precision)
    }
    
    # Check for speed argument
    speed_mode = 'balanced'  # DEFAULT TO BALANCED MODE (10min, good accuracy)
    clean_mode = True
    run_mode = 'all'
    
    for arg in sys.argv[1:]:
        arg_lower = arg.lower()
        if arg_lower in SPEED_MODES:
            speed_mode = arg_lower
        elif arg_lower in ['quick', '1d', '1d2d', '2d', '3d', 'all', 'gaa', 'sweeps']:
            run_mode = arg_lower
        elif arg_lower == 'noclean':
            clean_mode = False
        elif arg_lower == 'clean':
            clean_outputs()
            print("Outputs cleaned. No simulation run.")
            sys.exit(0)
        elif arg_lower == 'help':
            print("""
MIFIS FeFET Simulation Framework
================================

Usage: python main.py [speed] [phase] [options]

Speed Modes (choose one):
  instant    - Minimal points, ~10-30 seconds (for testing only)
  demo       - Few points, ~30-60 seconds (for quick demos)
  fast       - Medium points, ~1-2 minutes (for rapid testing)
  ultrafast  - Same as fast
  balanced   - Good accuracy, ~8-12 minutes (RECOMMENDED - DEFAULT)
  accurate   - High precision, ~30+ minutes (for publication results)

Phase Selection (choose one):
  quick / 1d - 1D only (cleans ONLY 1D outputs)
  1d2d / 2d  - 1D + 2D (cleans ONLY 1D and 2D outputs)
  3d         - 3D + GAA (cleans ONLY 3D and GAA outputs)
  gaa        - GAA only (cleans ONLY GAA outputs)
  sweeps     - Parameter sweeps only (cleans ONLY sweep outputs)
  all        - All phases (DEFAULT)

Options:
  noclean    - Don't clean previous outputs (keep all old files)
  clean      - Just clean ALL outputs, no simulation
  help       - Show this help

NOTE: Each phase only cleans its OWN previous outputs!
      Running '1d' does NOT remove 2D or 3D results.

Examples:
  python main.py 1d               # Run 1D only, clean only 1D outputs
  python main.py 2d noclean       # Run 2D, keep previous 2D outputs
  python main.py 3d               # Run 3D+GAA, clean only 3D/GAA outputs
  python main.py all              # Run all, clean all phase outputs
""")
            sys.exit(0)
    
    # Set global speed config (runners will check this)
    import builtins
    builtins.SIMULATION_SPEED_MODE = speed_mode
    
    print(f"\n  Speed Mode: {speed_mode.upper()} ({SPEED_MODES.get(speed_mode, 'default')})")
    print(f"  Run Mode: {run_mode}")
    print(f"  Clean: {clean_mode}")
    
    # Execute based on run_mode
    if run_mode == 'quick' or run_mode == '1d':
        main(phases={'1d': True}, clean=clean_mode)
    elif run_mode == '1d2d' or run_mode == '2d':
        main(phases={'1d': True, '2d': True}, clean=clean_mode)
    elif run_mode == '3d':
        main(phases={'3d': True, 'gaa': True}, clean=clean_mode)
    elif run_mode == 'sweeps':
        main(phases={'sweeps': True}, clean=clean_mode)
    elif run_mode == 'gaa':
        main(phases={'gaa': True}, clean=clean_mode)
    else:  # all
        main(clean=clean_mode)

