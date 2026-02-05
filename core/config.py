"""
MIFIS FeFET Simulation Framework
================================
Core configuration classes for device and simulation parameters.

Author: Thesis Project
Date: February 2026
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum
import json


class Architecture(Enum):
    """Device architecture types."""
    PLANAR = "planar"
    FINFET = "finfet"
    GAA = "gaa"
    DUALGATE = "dualgate"


class SweepDirection(Enum):
    """Voltage sweep direction."""
    FORWARD = "forward"
    REVERSE = "reverse"
    DOUBLE = "double"  # Forward then reverse (hysteresis)


@dataclass
class DeviceGeometry:
    """
    Physical dimensions of the MIFIS FeFET device.
    All values in nanometers (nm).
    """
    # Gate dimensions
    Lg: float = 100.0          # Gate length (nm)
    W: float = 100.0           # Channel width (nm)
    
    # MIFIS Stack thicknesses (OPTIMIZED VALUES from validation)
    t_gate: float = 50.0       # Gate electrode thickness (nm)
    t_top_il: float = 2.0      # Top interlayer SiO2 (nm) - OPTIMIZED from 4.0→2.0 for better MW
    t_fe: float = 13.8         # HZO ferroelectric thickness (nm)
    t_bottom_il: float = 0.7   # Bottom interlayer SiO2 (nm)
    t_channel: float = 20.0    # Silicon channel thickness (nm)
    
    # For GAA architectures
    radius: Optional[float] = None  # Nanowire radius (nm)
    wrap_angle: float = 360.0       # Gate wrap angle (degrees)
    
    @property
    def total_height(self) -> float:
        """Total stack height in nm."""
        return self.t_gate + self.t_top_il + self.t_fe + self.t_bottom_il + self.t_channel
    
    def to_meters(self) -> Dict[str, float]:
        """Convert all dimensions to meters for DEVSIM."""
        scale = 1e-9
        return {
            'Lg': self.Lg * scale,
            'W': self.W * scale,
            't_gate': self.t_gate * scale,
            't_top_il': self.t_top_il * scale,
            't_fe': self.t_fe * scale,
            't_bottom_il': self.t_bottom_il * scale,
            't_channel': self.t_channel * scale,
            'radius': self.radius * scale if self.radius else None,
            'total_height': self.total_height * scale,
        }


@dataclass
class FerroelectricMaterial:
    """
    Ferroelectric material properties.
    Default values for HZO (Hf0.5Zr0.5O2) - YOUR PRIMARY MATERIAL.
    """
    name: str = "HZO"
    
    # Dielectric properties
    epsilon_r: float = 30.0           # Relative permittivity
    
    # Ferroelectric properties (OPTIMIZED from validation)
    Ps: float = 38.0                  # Saturation polarization (μC/cm²)
    Pr: float = 18.0                  # Remnant polarization (μC/cm²)
    Ec: float = 0.5                   # Coercive field (MV/cm) - OPTIMIZED from 1.5→0.5 for 4V+ MW
    
    # Landau coefficients (derived or specified)
    alpha: Optional[float] = None     # Landau alpha coefficient
    beta: Optional[float] = None      # Landau beta coefficient
    gamma: float = 0.0                # Landau gamma coefficient (often negligible)
    
    # Dynamics
    tau_switching: float = 10.0       # Switching time constant (ns)
    
    def __post_init__(self):
        """Calculate Landau coefficients if not provided."""
        # Convert units: Pr in C/m², Ec in V/m
        Pr_SI = self.Pr * 1e-2        # μC/cm² → C/m²
        Ec_SI = self.Ec * 1e8         # MV/cm → V/m
        
        if self.alpha is None:
            self.alpha = -2 * Ec_SI / Pr_SI
        if self.beta is None:
            self.beta = Ec_SI / (2 * Pr_SI**3)
    
    @property
    def epsilon_abs(self) -> float:
        """Absolute permittivity in F/m."""
        eps0 = 8.854e-12
        return self.epsilon_r * eps0


@dataclass
class DielectricMaterial:
    """Dielectric (insulator) material properties."""
    name: str = "SiO2"
    epsilon_r: float = 3.9
    band_gap: float = 9.0             # eV
    electron_affinity: float = 0.95   # eV
    
    @property
    def epsilon_abs(self) -> float:
        """Absolute permittivity in F/m."""
        eps0 = 8.854e-12
        return self.epsilon_r * eps0


@dataclass
class SemiconductorMaterial:
    """Semiconductor (channel) material properties."""
    name: str = "Silicon"
    epsilon_r: float = 11.7
    
    # Carrier properties
    ni: float = 1.0e10                # Intrinsic carrier concentration (cm⁻³)
    mobility_e: float = 1400.0        # Electron mobility (cm²/V·s)
    mobility_h: float = 450.0         # Hole mobility (cm²/V·s)
    
    # Doping
    doping_type: str = "p"            # 'n' or 'p'
    doping_concentration: float = 1e17  # cm⁻³
    
    @property
    def epsilon_abs(self) -> float:
        """Absolute permittivity in F/m."""
        eps0 = 8.854e-12
        return self.epsilon_r * eps0


@dataclass
class SimulationConfig:
    """
    Simulation parameters and solver settings.

    DEFAULT VALUES are optimized for good speed/accuracy balance.
    For final thesis results, use smaller Vg_step (0.05V) and tighter tolerances.
    """
    # Voltage sweep - OPTIMIZED for 4V+ MW (validated)
    Vg_start: float = -6.0            # Start gate voltage (V) - OPTIMIZED from -3V
    Vg_end: float = 6.0               # End gate voltage (V) - OPTIMIZED from +3V
    Vg_step: float = 0.2              # Voltage step (V) - 0.2V gives good accuracy with 10x speed
    Vd: float = 0.05                  # Drain voltage (V)

    # Sweep configuration
    sweep_direction: SweepDirection = SweepDirection.DOUBLE

    # Temperature
    temperature: float = 300.0        # Kelvin

    # Solver settings - optimized for speed while maintaining accuracy
    abs_error: float = 1e-8           # Absolute convergence error (relaxed but accurate)
    rel_error: float = 1e-6           # Relative convergence error (relaxed but accurate)
    max_iterations: int = 25          # Newton iterations (usually converges in 5-10)

    # Mesh settings
    mesh_scale: float = 1.5           # Mesh refinement factor (1.5 = good balance)

    # Parallel execution settings (for parameter sweeps)
    enable_parallel: bool = True      # Enable parallel execution for sweeps
    n_workers: Optional[int] = None   # Number of parallel workers (None = auto-detect)

    
    @property
    def voltage_steps(self) -> List[float]:
        """Generate voltage sweep steps based on direction."""
        import numpy as np
        forward = list(np.arange(self.Vg_start, self.Vg_end + self.Vg_step, self.Vg_step))
        
        if self.sweep_direction == SweepDirection.FORWARD:
            return forward
        elif self.sweep_direction == SweepDirection.REVERSE:
            return forward[::-1]
        else:  # DOUBLE
            reverse = forward[::-1]
            return forward + reverse[1:]  # Avoid duplicate at peak


def get_fast_simulation_config() -> SimulationConfig:
    """
    Get optimized simulation config for FAST execution.

    Optimizations:
    - Coarser voltage steps (0.2V instead of 0.1V)
    - Relaxed convergence (1e-8 instead of 1e-12)
    - Fewer iterations
    - Coarser mesh
    - Parallel execution ENABLED (default)

    Use for quick testing, then switch to default for final results.
    """
    return SimulationConfig(
        Vg_start=-6.0,            # OPTIMIZED: -3→-6V for proper ferroelectric switching
        Vg_end=6.0,               # OPTIMIZED: +3→+6V for proper ferroelectric switching
        Vg_step=0.2,              # 2x coarser -> 2x faster
        Vd=0.05,
        sweep_direction=SweepDirection.DOUBLE,
        temperature=300.0,
        abs_error=1e-8,           # Relaxed tolerance
        rel_error=1e-6,           # Relaxed tolerance
        max_iterations=20,        # Fewer iterations
        mesh_scale=2.0,           # Coarser mesh (2x scale)
        enable_parallel=True,     # Parallel execution enabled
        n_workers=None,           # Auto-detect optimal workers
    )


def get_ultra_fast_config() -> SimulationConfig:
    """
    Ultra-fast config for quick validation only.
    Lower accuracy but very fast (~1-2 minutes).
    """
    return SimulationConfig(
        Vg_start=-2.0,            # Narrower range
        Vg_end=2.0,
        Vg_step=0.5,              # Very coarse steps
        Vd=0.05,
        sweep_direction=SweepDirection.FORWARD,  # Single sweep only
        temperature=300.0,
        abs_error=1e-6,
        rel_error=1e-4,
        max_iterations=15,
        mesh_scale=3.0,           # Very coarse mesh
    )


def get_instant_config() -> SimulationConfig:
    """
    Instant config for near-instant simulation (~10-30 seconds).
    Minimal accuracy but allows quick testing of workflow.
    
    USE THIS FOR: Testing if code runs, debugging, workflow validation.
    DO NOT USE FOR: Final thesis results, parameter optimization.
    """
    return SimulationConfig(
        Vg_start=-2.0,            # Narrow range
        Vg_end=2.0,
        Vg_step=1.0,              # Only 5 voltage points total
        Vd=0.05,
        sweep_direction=SweepDirection.FORWARD,  # Single sweep = half the points
        temperature=300.0,
        abs_error=1e-4,           # Very relaxed
        rel_error=1e-2,           # Very relaxed
        max_iterations=10,        # Minimal iterations
        mesh_scale=5.0,           # Very coarse mesh
    )


def get_demo_config() -> SimulationConfig:
    """
    Demo config for presentation/screening (~30 seconds).
    Shows hysteresis but with minimal points.
    """
    return SimulationConfig(
        Vg_start=-3.0,
        Vg_end=3.0,
        Vg_step=0.5,              # 12 points forward + 12 reverse = 24 total
        Vd=0.05,
        sweep_direction=SweepDirection.DOUBLE,  # Show hysteresis
        temperature=300.0,
        abs_error=1e-5,
        rel_error=1e-3,
        max_iterations=12,
        mesh_scale=4.0,           # Coarse mesh
    )


def get_balanced_config() -> SimulationConfig:
    """
    BALANCED config for 10-minute runtime with GOOD accuracy.

    Optimized for: Complete simulation in ~10 minutes with publication-quality results.

    Key optimizations:
    - Voltage steps: 0.1V (30 points forward + 30 reverse = 60 total)
    - Good convergence: 1e-9 abs_error (better than fast, practical for accuracy)
    - Balanced mesh: 1.5x scale (good resolution without excessive computation)
    - Adequate iterations: 30 (handles most convergence cases)
    - Parallel sweeps: enabled

    Expected runtime: ~8-12 minutes for all 6 phases
    Accuracy: Suitable for thesis results and parameter optimization
    """
    return SimulationConfig(
        Vg_start=-3.0,
        Vg_end=3.0,
        Vg_step=0.1,              # Good resolution: 60 points per sweep
        Vd=0.05,
        sweep_direction=SweepDirection.DOUBLE,
        temperature=300.0,
        abs_error=1e-9,           # Good convergence
        rel_error=1e-7,           # Good convergence
        max_iterations=30,        # Sufficient for most cases
        mesh_scale=1.5,           # Balanced mesh density
        enable_parallel=True,     # Parallel execution enabled
        n_workers=None,           # Auto-detect optimal workers
    )


def get_accurate_config() -> SimulationConfig:
    """
    Accurate config for FINAL THESIS RESULTS (high precision).

    Full precision, fine voltage steps, tight convergence.
    Expected runtime: 30+ minutes for all phases.
    """
    return SimulationConfig(
        Vg_start=-3.0,
        Vg_end=3.0,
        Vg_step=0.05,             # Fine steps for smooth curves
        Vd=0.05,
        sweep_direction=SweepDirection.DOUBLE,
        temperature=300.0,
        abs_error=1e-10,          # Tight tolerance
        rel_error=1e-8,           # Tight tolerance
        max_iterations=50,        # Allow more iterations
        mesh_scale=1.0,           # Fine mesh
        enable_parallel=True,     # Parallel execution enabled
        n_workers=None,           # Auto-detect optimal workers
    )



@dataclass
class DeviceConfig:
    """Complete device configuration combining all components."""
    geometry: DeviceGeometry = field(default_factory=DeviceGeometry)
    fe_material: FerroelectricMaterial = field(default_factory=FerroelectricMaterial)
    top_il: DielectricMaterial = field(default_factory=lambda: DielectricMaterial("SiO2", 3.9))
    bottom_il: DielectricMaterial = field(default_factory=lambda: DielectricMaterial("SiO2", 3.9))
    channel: SemiconductorMaterial = field(default_factory=SemiconductorMaterial)

    architecture: Architecture = Architecture.PLANAR

    # Dual-gate specific
    dual_gate_state: str = "00"       # For dual-gate: "00", "01", "10", "11"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "geometry": {
                "Lg": self.geometry.Lg,
                "W": self.geometry.W,
                "t_gate": self.geometry.t_gate,
                "t_top_il": self.geometry.t_top_il,
                "t_fe": self.geometry.t_fe,
                "t_bottom_il": self.geometry.t_bottom_il,
                "t_channel": self.geometry.t_channel,
                "wrap_angle": self.geometry.wrap_angle,
            },
            "fe_material": {
                "name": self.fe_material.name,
                "epsilon_r": self.fe_material.epsilon_r,
                "Ps": self.fe_material.Ps,
                "Pr": self.fe_material.Pr,
                "Ec": self.fe_material.Ec,
            },
            "architecture": self.architecture.value,
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Preset configurations
def get_baseline_config() -> DeviceConfig:
    """Get baseline MIFIS FeFET configuration (OPTIMIZED from validation - 4.028V MW)."""
    return DeviceConfig(
        geometry=DeviceGeometry(
            Lg=100.0, W=100.0,
            t_gate=50.0, t_top_il=2.0, t_fe=13.8,  # t_top_il: 4.0→2.0 OPTIMIZED
            t_bottom_il=0.7, t_channel=20.0
        ),
        fe_material=FerroelectricMaterial(
            name="HZO", epsilon_r=30.0, Ps=38.0, Pr=18.0, Ec=0.5  # Ec: 1.0→0.5 OPTIMIZED
        ),
        architecture=Architecture.PLANAR
    )


def get_gaa_config(wrap_angle: float = 360.0) -> DeviceConfig:
    """Get Gate-All-Around configuration."""
    config = get_baseline_config()
    config.architecture = Architecture.GAA
    config.geometry.wrap_angle = wrap_angle
    config.geometry.radius = 5.0  # 5nm nanowire radius
    return config


def get_dualgate_config(state: str = "00") -> DeviceConfig:
    """Get dual-gate configuration with specified state."""
    config = get_baseline_config()
    config.architecture = Architecture.DUALGATE
    config.dual_gate_state = state
    return config


# =============================================================================
# THESIS MATERIAL PRESETS (Finding #2)
# =============================================================================

def get_material_config(material_name: str) -> DeviceConfig:
    """
    Get configuration for specific ferroelectric material.
    
    Materials:
    - HZO (Standard): Balanced MW ~2.80V
    - Si:HfO2 (High Perf): Max MW ~3.05V
    - HfO2 (Reliable): Stable MW ~2.40V
    """
    config = get_baseline_config()
    
    if material_name == "Si:HfO2":
        config.fe_material.name = "Si:HfO2"
        config.fe_material.Pr = 22.0  # Higher Pr
        config.fe_material.Ec = 1.8   # Higher Ec
        config.fe_material.epsilon_r = 30.0
    
    elif material_name == "HfO2":
        config.fe_material.name = "HfO2"
        config.fe_material.Pr = 12.0  # Lower Pr
        config.fe_material.Ec = 0.8   # Lower Ec
        config.fe_material.epsilon_r = 25.0
        
    elif material_name == "HZO":
        pass  # Default
        
    else:
        raise ValueError(f"Unknown material: {material_name}")
        
    return config


# =============================================================================
# STACK ENGINEERING PRESETS (Finding #3 & #4)
# =============================================================================

def get_stack_config(t_fe: float = 12.0, t_il: float = 1.5) -> DeviceConfig:
    """
    Get optimized stack configuration (Finding #4: Design Rules).
    
    Optimal: 12nm HZO + 1.5nm IL
    """
    config = get_baseline_config()
    config.geometry.t_fe = t_fe
    config.geometry.t_top_il = t_il
    config.geometry.t_bottom_il = 0.7 # Keep bottom thin fixed
    return config
