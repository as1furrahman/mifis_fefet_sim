"""
MIFIS FeFET Simulation - Simulations Package
=============================================

Complete runners for all simulation phases:
- Phase 1: 1D Baseline
- Phase 2: 2D Planar
- Phase 3: 3D Full Device, GAA, Dual-Gate
- Phase 4: Parameter Sweeps (FE, IL, Materials, Gate Length)
"""

from .run_1d_baseline import run_1d_baseline
from .run_2d_planar import run_2d_planar
from .run_3d_mifis import run_3d_mifis
from .run_3d_gaa import run_3d_gaa
from .run_gaa import run_gaa_comparison
from .run_dualgate import run_dualgate_simulation
from .run_sweeps import (
    run_all_sweeps,
    run_fe_thickness_sweep,
    run_il_thickness_sweep,
    run_il_material_comparison,
    run_gate_length_scaling,
)

__all__ = [
    "run_1d_baseline",
    "run_2d_planar",
    "run_3d_mifis",
    "run_3d_gaa",
    "run_gaa_comparison",
    "run_dualgate_simulation",
    "run_all_sweeps",
    "run_fe_thickness_sweep",
    "run_il_thickness_sweep",
    "run_il_material_comparison",
    "run_gate_length_scaling",
]
