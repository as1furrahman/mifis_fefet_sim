"""
MIFIS FeFET Simulation Framework - Core Module
================================================

Complete simulation framework for Metal-Insulator-Ferroelectric-Insulator-Si
Field-Effect Transistors supporting 1D, 2D, and 3D device simulations.

Version: 2.0.0 (with 2D/3D support)
Author: Thesis Project
Date: February 2026
"""

# Configuration
from .config import (
    DeviceGeometry,
    DeviceConfig,
    SimulationConfig,
    FerroelectricMaterial,
    DielectricMaterial,
    SemiconductorMaterial,
    Architecture,
    SweepDirection,
    get_baseline_config,
    get_gaa_config,
    get_dualgate_config,
    get_fast_simulation_config,
    get_ultra_fast_config,
    get_instant_config,
    get_demo_config,
)


# Materials Database
from .materials import (
    FERROELECTRIC_DB,
    DIELECTRIC_DB,
    SEMICONDUCTOR_DB,
    METAL_DB,
    get_fe_material,
    get_dielectric,
    get_semiconductor,
    list_all_materials,
    LITERATURE_BENCHMARKS,
)

# Geometry & Mesh Generation
from .geometry import (
    MeshParameters,
    DEFAULT_MESH_PARAMS,
    FAST_MESH_PARAMS,
    ACCURATE_MESH_PARAMS,
    create_1d_mesh,
    create_2d_mesh,
    create_3d_mesh_gmsh,
    create_gaa_mesh_gmsh,
    GMSH_AVAILABLE,
)

# Physics Models
from .physics import (
    LandauKhalatnikovModel,
    FerroelectricModelWithMemory,
    HysteresisModel,
    coupled_displacement_field,
    bound_charge_density,
    calculate_memory_window,
    estimate_memory_window,
    EPS0,
    Q_E,
    K_B,
)

# Pure Python Solvers (work without DEVSIM)
from .pure_python_solver import (
    MIFIS1DSolver as PurePython1DSolver,
    MIFIS2DPlanarSolver as PurePython2DSolver,
    MIFIS3DGAASolver as PurePython3DSolver,
    MIFISGeometry,
    HZOProperties,
    generate_voltage_sweep,
    run_all_architectures,
)

# Solvers (1D, 2D, 3D)
from .solver import (
    MIFISSolver,
    MIFIS2DSolver,
    MIFIS3DSolver,
    SimulationDimension,
    run_baseline_simulation,
    run_2d_simulation,
    run_3d_simulation,
    DEVSIM_AVAILABLE,
)

# Post-processing
from .postprocess import (
    FeFETMetrics,
    extract_metrics,
    extract_threshold_voltage,
    extract_subthreshold_swing,
    extract_memory_window,
    extract_vth,
    compare_with_target,
    generate_summary_report,
)

# Visualization
from .visualization import (
    setup_thesis_style,
    plot_id_vg,
    plot_multiple_id_vg,
    plot_polarization_loop,
    plot_memory_window_vs_parameter,
    plot_architecture_comparison,
    plot_mifis_stack,
    plot_1d_stack_with_results,
    plot_2d_structure_with_results,
    plot_3d_gaa_structure,
    plot_architecture_summary,
    save_all_formats,
    # Paper-style plots (from literature analysis)
    plot_cv_hysteresis,
    plot_id_vg_linear_with_gm,
    plot_subthreshold_swing,
    plot_band_diagram,
    plot_retention,
    plot_endurance,
    plot_mw_vs_il_thickness,
    plot_dual_sweep_comparison,
)


__version__ = "2.0.0"
__author__ = "Thesis Project"

__all__ = [
    # Config
    "DeviceGeometry",
    "DeviceConfig", 
    "SimulationConfig",
    "FerroelectricMaterial",
    "DielectricMaterial",
    "SemiconductorMaterial",
    "Architecture",
    "SweepDirection",
    "get_baseline_config",
    "get_gaa_config",
    "get_dualgate_config",
    "get_fast_simulation_config",
    "get_ultra_fast_config",
    "get_instant_config",
    "get_demo_config",
    
    # Materials
    "FERROELECTRIC_DB",
    "DIELECTRIC_DB",
    "SEMICONDUCTOR_DB",
    "METAL_DB",
    "get_fe_material",
    "get_dielectric",
    "get_semiconductor",
    "list_all_materials",
    "LITERATURE_BENCHMARKS",
    
    # Geometry
    "MeshParameters",
    "DEFAULT_MESH_PARAMS",
    "FAST_MESH_PARAMS",
    "ACCURATE_MESH_PARAMS",
    "create_1d_mesh",
    "create_2d_mesh",
    "create_3d_mesh_gmsh",
    "create_gaa_mesh_gmsh",
    "GMSH_AVAILABLE",
    
    # Physics
    "LandauKhalatnikovModel",
    "HysteresisModel",
    "coupled_displacement_field",
    "bound_charge_density",
    "estimate_memory_window",
    "EPS0",
    "Q_E",
    "K_B",
    
    # Solvers
    "MIFISSolver",
    "MIFIS2DSolver",
    "MIFIS3DSolver",
    "SimulationDimension",
    "run_baseline_simulation",
    "run_2d_simulation",
    "run_3d_simulation",
    "DEVSIM_AVAILABLE",
    
    # Postprocess
    "FeFETMetrics",
    "extract_metrics",
    "extract_threshold_voltage",
    "extract_subthreshold_swing",
    "extract_memory_window",
    "extract_vth",
    "compare_with_target",
    "generate_summary_report",
    
    # Visualization - Core
    "setup_thesis_style",
    "plot_id_vg",
    "plot_multiple_id_vg",
    "plot_polarization_loop",
    "plot_memory_window_vs_parameter",
    "plot_architecture_comparison",
    "plot_mifis_stack",
    "save_all_formats",
    # Visualization - Structure plots
    "plot_1d_stack_with_results",
    "plot_2d_structure_with_results",
    "plot_3d_gaa_structure",
    "plot_architecture_summary",
    # Visualization - Paper-style plots
    "plot_cv_hysteresis",
    "plot_id_vg_linear_with_gm",
    "plot_subthreshold_swing",
    "plot_band_diagram",
    "plot_retention",
    "plot_endurance",
    "plot_mw_vs_il_thickness",
    "plot_dual_sweep_comparison",
]
