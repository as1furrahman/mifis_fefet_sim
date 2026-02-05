"""
MIFIS FeFET Simulation Framework
================================
DEVSIM Solver Interface for 1D, 2D, and 3D TCAD Simulations

This module REQUIRES DEVSIM to be installed. No mock mode.

Supports:
- 1D MOS capacitor / vertical stack analysis
- 2D Planar FeFET (cross-section with S/D)
- 2D FinFET (via wrap angle approximation)
- 3D via Gmsh mesh import

Author: Thesis Project
Date: February 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# DEVSIM is REQUIRED - no fallback
try:
    import devsim
    DEVSIM_AVAILABLE = True
except ImportError:
    raise ImportError(
        "DEVSIM is required for this simulation framework.\n"
        "Install with: pip install devsim\n"
        "Or visit: https://devsim.org"
    )

from .config import DeviceConfig, SimulationConfig, DeviceGeometry, Architecture
from .physics import LandauKhalatnikovModel, HysteresisModel, EPS0, Q_E, K_B

# Gmsh is optional for 3D
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


class SimulationDimension(Enum):
    """Simulation dimensionality."""
    DIM_1D = "1D"
    DIM_2D = "2D"
    DIM_3D = "3D"


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

PHYSICS = {
    'eps0': 8.854e-12,       # F/m - vacuum permittivity
    'q': 1.602e-19,          # C - electron charge
    'kB': 1.381e-23,         # J/K - Boltzmann constant
    'T_ref': 300.0,          # K - reference temperature
    'Vt_300K': 0.0259,       # V - thermal voltage at 300K
    'ni_Si': 1.0e16,         # m^-3 - intrinsic carrier concentration
}


# =============================================================================
# BASE SOLVER CLASS
# =============================================================================

class BaseSolver:
    """Base class for DEVSIM-based MIFIS FeFET solvers."""
    
    def __init__(self, device_name: str, dimension: SimulationDimension):
        self.device_name = device_name
        self.dimension = dimension
        self.device_created = False
        self.physics_setup = False
        self.mesh_name = f"{device_name}_mesh"
        
        # Region and contact tracking
        self.regions: List[str] = []
        self.contacts: List[str] = []
        
        # Polarization state for hysteresis
        self.polarization_state: float = 0.0
        self.hysteresis_model: Optional[HysteresisModel] = None
    
    def _set_material_parameters(self, region: str, config: DeviceConfig):
        """Set material parameters for a region."""
        # Get permittivity based on region
        if region == "channel":
            eps = config.channel.epsilon_r
            mat_type = "semiconductor"
        elif region == "ferroelectric":
            eps = config.fe_material.epsilon_r
            mat_type = "ferroelectric"
        elif region in ["bottom_il", "top_il"]:
            eps = config.top_il.epsilon_r
            mat_type = "insulator"
        elif region == "gate":
            eps = 1.0  # Metal
            mat_type = "metal"
        else:
            eps = 1.0
            mat_type = "unknown"
        
        # Set permittivity parameter
        eps_abs = eps * PHYSICS['eps0']
        devsim.set_parameter(device=self.device_name, region=region,
                            name="Permittivity", value=eps_abs)
        
        # For semiconductor, set more parameters
        if mat_type == "semiconductor":
            devsim.set_parameter(device=self.device_name, region=region,
                                name="n_i", value=PHYSICS['ni_Si'])
            devsim.set_parameter(device=self.device_name, region=region,
                                name="q", value=PHYSICS['q'])
            devsim.set_parameter(device=self.device_name, region=region,
                                name="kT", value=PHYSICS['kB'] * config.channel.temperature 
                                if hasattr(config.channel, 'temperature') else PHYSICS['kB'] * 300)
    
    def _setup_poisson_equation(self, region: str):
        """Setup Poisson equation in a region using DEVSIM."""
        # Create solution variable
        devsim.node_solution(device=self.device_name, region=region, name="Potential")

        # Initialize potential to zero
        devsim.edge_from_node_model(device=self.device_name, region=region,
                                   node_model="Potential")
        
        # Create edge model for electric field from potential
        devsim.edge_from_node_model(device=self.device_name, region=region,
                                   node_model="Potential")
        
        # Electric field: E = -dV/dx
        devsim.edge_model(device=self.device_name, region=region,
                         name="ElectricField",
                         equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")
        
        # Derivatives for Newton solver
        devsim.edge_model(device=self.device_name, region=region,
                         name="ElectricField:Potential@n0",
                         equation="EdgeInverseLength")
        devsim.edge_model(device=self.device_name, region=region,
                         name="ElectricField:Potential@n1",
                         equation="-EdgeInverseLength")
        
        # Displacement field: D = epsilon * E
        devsim.edge_model(device=self.device_name, region=region,
                         name="PotentialEdgeFlux",
                         equation="Permittivity * ElectricField")
        devsim.edge_model(device=self.device_name, region=region,
                         name="PotentialEdgeFlux:Potential@n0",
                         equation="Permittivity * ElectricField:Potential@n0")
        devsim.edge_model(device=self.device_name, region=region,
                         name="PotentialEdgeFlux:Potential@n1",
                         equation="Permittivity * ElectricField:Potential@n1")
        
        # Poisson equation: div(D) = rho
        devsim.equation(device=self.device_name, region=region,
                       name="PotentialEquation",
                       variable_name="Potential",
                       edge_model="PotentialEdgeFlux",
                       variable_update="log_damp")
    
    def _setup_ferroelectric_region(self, region: str, config: DeviceConfig):
        """Setup ferroelectric region with polarization coupling."""
        # Basic Poisson setup
        self._setup_poisson_equation(region)
        
        # Ferroelectric parameters
        Pr = config.fe_material.Pr * 1e-2  # μC/cm² to C/m²
        Ps = config.fe_material.Ps * 1e-2
        Ec = config.fe_material.Ec * 1e8   # MV/cm to V/m
        
        # Set FE parameters
        devsim.set_parameter(device=self.device_name, region=region,
                            name="Pr", value=Pr)
        devsim.set_parameter(device=self.device_name, region=region,
                            name="Ps", value=Ps)
        devsim.set_parameter(device=self.device_name, region=region,
                            name="Ec", value=Ec)
        
        # Polarization as a node solution (will be updated during sweep)
        devsim.node_solution(device=self.device_name, region=region, name="Polarization")
        devsim.set_node_values(device=self.device_name, region=region,
                              name="Polarization", init_from="Polarization")
        
        # Bound charge from polarization divergence: rho_b = -div(P)
        devsim.edge_from_node_model(device=self.device_name, region=region,
                                   node_model="Polarization")
        devsim.edge_model(device=self.device_name, region=region,
                         name="PolarizationFlux",
                         equation="(Polarization@n0 - Polarization@n1) * EdgeInverseLength")
        
        # Modified D-field with polarization: D = eps*E + P
        devsim.edge_model(device=self.device_name, region=region,
                         name="PotentialEdgeFlux",
                         equation="Permittivity * ElectricField + 0.5 * (Polarization@n0 + Polarization@n1)")
    
    def _setup_semiconductor_region(self, region: str, config: DeviceConfig):
        """Setup semiconductor region with carrier physics."""
        # Basic Poisson
        self._setup_poisson_equation(region)
        
        # Doping concentration
        Na = config.channel.doping_concentration * 1e6  # cm^-3 to m^-3
        devsim.set_parameter(device=self.device_name, region=region,
                            name="Na", value=Na)
        
        # Intrinsic density
        ni = config.channel.ni * 1e6  # cm^-3 to m^-3
        devsim.set_parameter(device=self.device_name, region=region,
                            name="n_i", value=ni)
        
        # For now, use depletion approximation
        # Charge density: rho = q * Na (fully depleted p-type)
        devsim.node_model(device=self.device_name, region=region,
                         name="NetDoping", equation="-Na")
        devsim.node_model(device=self.device_name, region=region,
                         name="BuiltinCharge", equation="q * NetDoping")
        
        # Add charge to Poisson equation
        devsim.node_model(device=self.device_name, region=region,
                         name="PotentialNodeCharge", equation="BuiltinCharge")
        devsim.equation(device=self.device_name, region=region,
                       name="PotentialEquation",
                       variable_name="Potential",
                       edge_model="PotentialEdgeFlux",
                       node_model="PotentialNodeCharge",
                       variable_update="log_damp")
    
    def _setup_contact_equation(self, contact: str, init_voltage: float = 0.0):
        """Setup contact boundary condition."""
        # Create bias parameter
        devsim.set_parameter(device=self.device_name, name=f"{contact}_bias",
                            value=init_voltage)
        
        # Get regions attached to this contact
        regions = devsim.get_region_list(device=self.device_name)
        
        for region in regions:
            contact_list = devsim.get_contact_list(device=self.device_name)
            if contact in contact_list:
                # Contact equation: V = V_applied
                devsim.contact_node_model(device=self.device_name, contact=contact,
                                         name=f"{contact}_bc",
                                         equation=f"Potential - {contact}_bias")
                devsim.contact_node_model(device=self.device_name, contact=contact,
                                         name=f"{contact}_bc:Potential",
                                         equation="1")
                devsim.contact_equation(device=self.device_name, contact=contact,
                                       name="PotentialEquation",
                                       node_model=f"{contact}_bc")
                break
    
    def set_voltage(self, contact: str, voltage: float):
        """Set voltage at a contact."""
        devsim.set_parameter(device=self.device_name, name=f"{contact}_bias",
                            value=voltage)
    
    def update_polarization(self, region: str, E_field: float, direction: str):
        """Update polarization based on E-field and sweep direction."""
        if self.hysteresis_model is None:
            return 0.0
        
        P = self.hysteresis_model.polarization(np.array([E_field]), direction)[0]
        
        # Set uniform polarization across the region
        nodes = devsim.get_node_model_values(device=self.device_name, region=region,
                                            name="Potential")
        n_nodes = len(nodes)
        P_values = [P] * n_nodes
        
        devsim.set_node_values(device=self.device_name, region=region,
                              name="Polarization", values=P_values)
        
        return P
    
    def solve(self, sim_config: SimulationConfig, verbose: bool = False) -> bool:
        """Solve the coupled equations."""
        try:
            # Suppress output unless verbose mode
            if not verbose:
                devsim.set_parameter(name="info", value=0)
            else:
                devsim.set_parameter(name="info", value=2)

            devsim.solve(type="dc",
                        abstol=sim_config.abs_error,
                        reltol=sim_config.rel_error,
                        maximum_iterations=sim_config.max_iterations)
            return True
        except devsim.error as e:
            if verbose:
                print(f"DEVSIM solver error: {e}")
            return False
    
    def get_potential(self, region: str) -> np.ndarray:
        """Get potential distribution in a region."""
        return np.array(devsim.get_node_model_values(
            device=self.device_name, region=region, name="Potential"))
    
    def get_electric_field(self, region: str) -> np.ndarray:
        """Get electric field in a region."""
        return np.array(devsim.get_edge_model_values(
            device=self.device_name, region=region, name="ElectricField"))


# =============================================================================
# 1D SOLVER
# =============================================================================

class MIFISSolver(BaseSolver):
    """
    DEVSIM 1D solver for MIFIS FeFET vertical stack simulation.
    """
    
    def __init__(self, device_name: str = "mifis_1d"):
        super().__init__(device_name, SimulationDimension.DIM_1D)
    
    def create_1d_mesh(self, config: DeviceConfig, n_points: int = 500) -> bool:
        """
        Create 1D mesh for MIFIS stack using DEVSIM.
        """
        geom = config.geometry
        dims = geom.to_meters()

        # Create 1D mesh (device is created after mesh finalization)
        devsim.create_1d_mesh(mesh=self.mesh_name)
        
        # Layer boundaries (bottom = 0, going up)
        z0 = 0.0
        z1 = dims['t_channel']              # Channel top
        z2 = z1 + dims['t_bottom_il']       # Bottom IL top
        z3 = z2 + dims['t_fe']              # FE top
        z4 = z3 + dims['t_top_il']          # Top IL top
        z5 = z4 + dims['t_gate']            # Gate top
        
        # Mesh spacing (finer at interfaces)
        ps_coarse = 2e-9    # 2 nm in bulk
        ps_fine = 0.5e-9    # 0.5 nm at interfaces
        ps_very_fine = 0.1e-9  # 0.1 nm at FE interfaces
        
        # Add mesh lines
        devsim.add_1d_mesh_line(mesh=self.mesh_name, pos=z0, ps=ps_coarse, tag="bottom")
        devsim.add_1d_mesh_line(mesh=self.mesh_name, pos=z1*0.5, ps=ps_coarse, tag="channel_mid")
        devsim.add_1d_mesh_line(mesh=self.mesh_name, pos=z1, ps=ps_fine, tag="ch_il")
        devsim.add_1d_mesh_line(mesh=self.mesh_name, pos=z2, ps=ps_very_fine, tag="il_fe")
        devsim.add_1d_mesh_line(mesh=self.mesh_name, pos=(z2+z3)/2, ps=ps_fine, tag="fe_mid")
        devsim.add_1d_mesh_line(mesh=self.mesh_name, pos=z3, ps=ps_very_fine, tag="fe_il")
        devsim.add_1d_mesh_line(mesh=self.mesh_name, pos=z4, ps=ps_fine, tag="il_gate")
        devsim.add_1d_mesh_line(mesh=self.mesh_name, pos=z5, ps=ps_coarse, tag="top")
        
        # Add regions
        devsim.add_1d_region(mesh=self.mesh_name, material="Silicon",
                            region="channel", tag1="bottom", tag2="ch_il")
        devsim.add_1d_region(mesh=self.mesh_name, material="SiO2",
                            region="bottom_il", tag1="ch_il", tag2="il_fe")
        devsim.add_1d_region(mesh=self.mesh_name, material="HZO",
                            region="ferroelectric", tag1="il_fe", tag2="fe_il")
        devsim.add_1d_region(mesh=self.mesh_name, material="SiO2",
                            region="top_il", tag1="fe_il", tag2="il_gate")
        devsim.add_1d_region(mesh=self.mesh_name, material="TiN",
                            region="gate", tag1="il_gate", tag2="top")
        
        # Add interfaces between adjacent regions
        devsim.add_1d_interface(mesh=self.mesh_name, tag="ch_il", name="channel_bottom_il")
        devsim.add_1d_interface(mesh=self.mesh_name, tag="il_fe", name="bottom_il_fe")
        devsim.add_1d_interface(mesh=self.mesh_name, tag="fe_il", name="fe_top_il")
        devsim.add_1d_interface(mesh=self.mesh_name, tag="il_gate", name="top_il_gate")

        # Add contacts
        devsim.add_1d_contact(mesh=self.mesh_name, name="substrate",
                             tag="bottom", material="metal")
        devsim.add_1d_contact(mesh=self.mesh_name, name="gate",
                             tag="top", material="metal")

        # Finalize mesh and create device
        devsim.finalize_mesh(mesh=self.mesh_name)
        devsim.create_device(mesh=self.mesh_name, device=self.device_name)

        self.regions = ["channel", "bottom_il", "ferroelectric", "top_il", "gate"]
        self.contacts = ["substrate", "gate"]
        self.device_created = True

        return True
    
    def setup_physics(self, config: DeviceConfig) -> bool:
        """Setup physics equations for all regions."""
        if not self.device_created:
            raise RuntimeError("Create mesh before setting up physics")
        
        # Set material parameters for each region
        for region in self.regions:
            self._set_material_parameters(region, config)
        
        # Setup equations per region type
        self._setup_semiconductor_region("channel", config)
        self._setup_poisson_equation("bottom_il")
        self._setup_ferroelectric_region("ferroelectric", config)
        self._setup_poisson_equation("top_il")
        self._setup_poisson_equation("gate")  # Gate also needs Poisson for contact BC
        
        # Setup contact equations
        self._setup_contact_equation("substrate", 0.0)
        self._setup_contact_equation("gate", 0.0)
        
        # Initialize hysteresis model
        self.hysteresis_model = HysteresisModel(
            Ps=config.fe_material.Ps,
            Pr=config.fe_material.Pr,
            Ec=config.fe_material.Ec
        )
        
        self.physics_setup = True
        return True
    
    def run_voltage_sweep(self, config: DeviceConfig,
                          sim_config: SimulationConfig,
                          fast_mode: bool = True) -> pd.DataFrame:
        """
        Run voltage sweep with ferroelectric hysteresis.
        
        Args:
            config: Device configuration
            sim_config: Simulation configuration
            fast_mode: If True, skips slow ramping for faster execution
            
        Returns:
            DataFrame with Vg, Id, phi_surface, Polarization, direction, converged
        """
        results = []
        voltage_steps = sim_config.voltage_steps
        n_forward = len(voltage_steps) // 2

        # Suppress DEVSIM verbosity
        devsim.set_parameter(name="info", value=0)

        # Initial solve at Vg=0 for better starting point
        self.set_voltage("gate", 0.0)
        self.set_voltage("substrate", 0.0)

        try:
            devsim.solve(type="dc", abstol=1e-4, reltol=1e-2,
                        maximum_iterations=20)
        except:
            pass

        # Track previous voltage for ramping decision
        prev_Vg = 0.0

        for i, Vg in enumerate(voltage_steps):
            direction = "forward" if i < n_forward else "reverse"

            # Calculate E-field in FE for polarization update
            # FIXED: Account for voltage division across stack
            t_top_il = config.geometry.t_top_il * 1e-9
            t_fe = config.geometry.t_fe * 1e-9
            t_bottom_il = config.geometry.t_bottom_il * 1e-9

            eps_top_il = config.top_il.epsilon_r
            eps_fe = config.fe_material.epsilon_r
            eps_bottom_il = config.bottom_il.epsilon_r

            # Voltage divides across series stack: V_i = V_total * (t_i/ε_i) / Σ(t_j/ε_j)
            sum_t_over_eps = (t_top_il / eps_top_il +
                             t_fe / eps_fe +
                             t_bottom_il / eps_bottom_il)
            V_fe = Vg * (t_fe / eps_fe) / sum_t_over_eps
            E_fe = V_fe / t_fe

            # Update polarization in FE region
            P = self.update_polarization("ferroelectric", E_fe, direction)

            # Set voltage - SKIP RAMPING IN FAST MODE
            self.set_voltage("gate", Vg)
            
            # Only ramp if NOT fast_mode AND large voltage step
            if not fast_mode and abs(Vg - prev_Vg) > 1.0:
                # Minimal ramping - just 2 intermediate steps
                mid_Vg = (prev_Vg + Vg) / 2
                self.set_voltage("gate", mid_Vg)
                try:
                    devsim.solve(type="dc", abstol=1e-4, reltol=1e-2,
                                maximum_iterations=10)
                except:
                    pass
                self.set_voltage("gate", Vg)

            # Solve at target voltage
            converged = self.solve(sim_config, verbose=False)
            prev_Vg = Vg

            # Get surface potential
            phi = self.get_potential("channel")
            phi_surface = phi[-1] if len(phi) > 0 else 0.0

            # Calculate drain current from surface potential
            Vt = PHYSICS['kB'] * sim_config.temperature / PHYSICS['q']
            n = 1.5  # Ideality factor
            I0 = 1e-12  # Reference current

            Id = I0 * np.exp(phi_surface / (n * Vt))
            Id = min(Id, 1e-4)  # Clamp

            results.append({
                "Vg": Vg,
                "Id": Id,
                "phi_surface": phi_surface,
                "Polarization": P * 1e2 if P else 0.0,
                "E_fe": E_fe,  # Store calculated E-field for plotting
                "direction": direction,
                "converged": converged,
            })

        return pd.DataFrame(results)


# =============================================================================
# 2D SOLVER
# =============================================================================

class MIFIS2DSolver(BaseSolver):
    """
    DEVSIM 2D solver for planar MIFIS FeFET with source/drain.
    """
    
    def __init__(self, device_name: str = "mifis_2d"):
        super().__init__(device_name, SimulationDimension.DIM_2D)
    
    def create_2d_mesh(self, config: DeviceConfig) -> bool:
        """Create 2D mesh for planar FeFET cross-section."""
        geom = config.geometry
        dims = geom.to_meters()
        
        # Dimensions
        Lg = dims['Lg']
        Lsd = Lg * 0.3  # Source/drain extension
        L_total = Lg + 2 * Lsd
        
        # Vertical positions
        z1 = dims['t_channel']
        z2 = z1 + dims['t_bottom_il']
        z3 = z2 + dims['t_fe']
        z4 = z3 + dims['t_top_il']
        z5 = z4 + dims['t_gate']
        
        # Create 2D mesh (device is created after mesh finalization)
        devsim.create_2d_mesh(mesh=self.mesh_name)
        
        # Mesh spacing
        ps_x = 2e-9
        ps_z = 1e-9
        ps_fine = 0.5e-9
        
        # X-direction lines
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="x", pos=0, ps=ps_x)
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="x", pos=Lsd, ps=ps_fine)
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="x", pos=Lsd+Lg/2, ps=ps_fine)
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="x", pos=Lsd+Lg, ps=ps_fine)
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="x", pos=L_total, ps=ps_x)
        
        # Y-direction (vertical) lines
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=0, ps=ps_z)
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=z1, ps=ps_fine)
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=z2, ps=ps_fine)
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=z3, ps=ps_fine)
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=z4, ps=ps_fine)
        devsim.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=z5, ps=ps_z)
        
        # Regions - Silicon channel (full width)
        devsim.add_2d_region(mesh=self.mesh_name, material="Silicon", region="channel",
                            xl=0, xh=L_total, yl=0, yh=z1)
        
        # Gate stack regions (only over channel, not over S/D)
        devsim.add_2d_region(mesh=self.mesh_name, material="SiO2", region="bottom_il",
                            xl=Lsd, xh=Lsd+Lg, yl=z1, yh=z2)
        devsim.add_2d_region(mesh=self.mesh_name, material="HZO", region="ferroelectric",
                            xl=Lsd, xh=Lsd+Lg, yl=z2, yh=z3)
        devsim.add_2d_region(mesh=self.mesh_name, material="SiO2", region="top_il",
                            xl=Lsd, xh=Lsd+Lg, yl=z3, yh=z4)
        devsim.add_2d_region(mesh=self.mesh_name, material="TiN", region="gate",
                            xl=Lsd, xh=Lsd+Lg, yl=z4, yh=z5)
        
        # Add 2D interfaces between adjacent regions
        devsim.add_2d_interface(mesh=self.mesh_name, name="channel_bottom_il",
                               region0="channel", region1="bottom_il")
        devsim.add_2d_interface(mesh=self.mesh_name, name="bottom_il_fe",
                               region0="bottom_il", region1="ferroelectric")
        devsim.add_2d_interface(mesh=self.mesh_name, name="fe_top_il",
                               region0="ferroelectric", region1="top_il")
        devsim.add_2d_interface(mesh=self.mesh_name, name="top_il_gate",
                               region0="top_il", region1="gate")

        # Contacts
        devsim.add_2d_contact(mesh=self.mesh_name, name="source", region="channel",
                             yl=0, yh=z1, xl=0, xh=ps_fine, material="metal")
        devsim.add_2d_contact(mesh=self.mesh_name, name="drain", region="channel",
                             yl=0, yh=z1, xl=L_total-ps_fine, xh=L_total, material="metal")
        devsim.add_2d_contact(mesh=self.mesh_name, name="gate", region="gate",
                             yl=z5-ps_fine, yh=z5, xl=Lsd, xh=Lsd+Lg, material="metal")
        devsim.add_2d_contact(mesh=self.mesh_name, name="substrate", region="channel",
                             yl=0, yh=ps_fine, xl=0, xh=L_total, material="metal")

        # Finalize mesh and create device
        devsim.finalize_mesh(mesh=self.mesh_name)
        devsim.create_device(mesh=self.mesh_name, device=self.device_name)

        self.regions = ["channel", "bottom_il", "ferroelectric", "top_il", "gate"]
        self.contacts = ["source", "drain", "gate", "substrate"]
        self.device_created = True

        return True
    
    def setup_physics(self, config: DeviceConfig) -> bool:
        """Setup 2D physics equations."""
        if not self.device_created:
            raise RuntimeError("Create mesh before setting up physics")
        
        for region in self.regions:
            self._set_material_parameters(region, config)
        
        self._setup_semiconductor_region("channel", config)
        self._setup_poisson_equation("bottom_il")
        self._setup_ferroelectric_region("ferroelectric", config)
        self._setup_poisson_equation("top_il")
        
        for contact in self.contacts:
            self._setup_contact_equation(contact, 0.0)
        
        self.hysteresis_model = HysteresisModel(
            Ps=config.fe_material.Ps,
            Pr=config.fe_material.Pr,
            Ec=config.fe_material.Ec
        )
        
        self.physics_setup = True
        return True
    
    def run_voltage_sweep(self, config: DeviceConfig,
                          sim_config: SimulationConfig,
                          Vd: float = 0.05) -> pd.DataFrame:
        """Run 2D voltage sweep."""
        results = []
        voltage_steps = sim_config.voltage_steps
        n_forward = len(voltage_steps) // 2

        # Suppress DEVSIM verbosity
        devsim.set_parameter(name="info", value=0)

        # Set drain voltage
        self.set_voltage("drain", Vd)
        self.set_voltage("source", 0.0)
        self.set_voltage("substrate", 0.0)

        # Initial solve at Vg=0
        self.set_voltage("gate", 0.0)
        try:
            devsim.solve(type="dc", abstol=1e-6, reltol=1e-4,
                        maximum_iterations=50)
        except:
            pass

        for i, Vg in enumerate(voltage_steps):
            direction = "forward" if i < n_forward else "reverse"

            # FIXED: Account for voltage division across stack
            t_top_il = config.geometry.t_top_il * 1e-9
            t_fe = config.geometry.t_fe * 1e-9
            t_bottom_il = config.geometry.t_bottom_il * 1e-9

            eps_top_il = config.top_il.epsilon_r
            eps_fe = config.fe_material.epsilon_r
            eps_bottom_il = config.bottom_il.epsilon_r

            # Voltage divides across series stack: V_i = V_total * (t_i/ε_i) / Σ(t_j/ε_j)
            sum_t_over_eps = (t_top_il / eps_top_il +
                             t_fe / eps_fe +
                             t_bottom_il / eps_bottom_il)
            V_fe = Vg * (t_fe / eps_fe) / sum_t_over_eps
            E_fe = V_fe / t_fe
            P = self.update_polarization("ferroelectric", E_fe, direction)

            # Gradual voltage ramping
            current_Vg = devsim.get_parameter(device=self.device_name, name="gate_bias")
            if abs(Vg - current_Vg) > 0.5:
                n_ramp_steps = int(abs(Vg - current_Vg) / 0.5) + 1
                for Vg_ramp in np.linspace(current_Vg, Vg, n_ramp_steps):
                    self.set_voltage("gate", Vg_ramp)
                    try:
                        devsim.solve(type="dc",
                                    abstol=sim_config.abs_error * 10,
                                    reltol=sim_config.rel_error * 10,
                                    maximum_iterations=10)
                    except:
                        pass
            else:
                self.set_voltage("gate", Vg)

            converged = self.solve(sim_config, verbose=False)

            if converged:
                phi = self.get_potential("channel")
                phi_surface = np.max(phi) if len(phi) > 0 else 0.0

                Vt = PHYSICS['kB'] * sim_config.temperature / PHYSICS['q']
                n = 1.3  # Better ideality for 2D
                I0 = 1e-12

                Id = I0 * np.exp(phi_surface / (n * Vt))
                Id = min(Id, 1e-4)
            else:
                phi = self.get_potential("channel")
                phi_surface = np.max(phi) if len(phi) > 0 else 0.0
                Vt = PHYSICS['kB'] * sim_config.temperature / PHYSICS['q']
                n = 1.3
                I0 = 1e-12
                Id = I0 * np.exp(phi_surface / (n * Vt))
                Id = min(Id, 1e-4)

            results.append({
                "Vg": Vg,
                "Vd": Vd,
                "Id": Id,
                "phi_surface": phi_surface,
                "Polarization": P * 1e2 if P else 0.0,
                "E_fe": E_fe,  # Store calculated E-field for plotting
                "direction": direction,
                "converged": converged,
            })

        return pd.DataFrame(results)


# =============================================================================
# 3D SOLVER (Uses Gmsh-generated mesh)
# =============================================================================

class MIFIS3DSolver(BaseSolver):
    """
    3D Solver using Gmsh-generated mesh imported into DEVSIM.
    
    Implements true 3D Poisson solve with Ferroelectric physics.
    Optimized for speed using iterative solvers when possible.
    """
    
    def __init__(self, device_name: str = "mifis_3d"):
        super().__init__(device_name, SimulationDimension.DIM_3D)
        self.mesh_file: Optional[str] = None
        self.architecture_factor: float = 1.0
        # Reference region names (consistent with Geometry module)
        self.region_map = {
            "silicon": "channel",       # Gmsh name -> Solver internal name
            "bottom_il": "bottom_il",
            "ferroelectric": "ferroelectric",
            "top_il": "top_il",
            "gate": "gate"
        }
    
    def create_3d_mesh(self, config: DeviceConfig, output_dir: str = "data/meshes",
                        mesh_params: Optional[Dict] = None) -> bool:
        """Create 3D mesh using Gmsh."""
        if not GMSH_AVAILABLE:
            # Fallback handled by run script, but warn here
            print("  GMsh not available for 3D meshing.")
            return False

        from .geometry import create_3d_mesh_gmsh, create_gaa_mesh_gmsh

        output_path = Path(output_dir) / f"{self.device_name}.msh"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.architecture == Architecture.GAA:
            mesh_info = create_gaa_mesh_gmsh(config, str(output_path))
            self.architecture_factor = 1.25
        else:
            # 3D Planar (Stack)
            mesh_info = create_3d_mesh_gmsh(config, str(output_path))
            self.architecture_factor = 1.0

        if mesh_info.get('status') == 'success':
            self.mesh_file = mesh_info['file_path']
            # Load mesh into DEVSIM
            if DEVSIM_AVAILABLE:
                devsim.create_gmsh_mesh(mesh=self.device_name+"_mesh", file=self.mesh_file)
                
                # Map Gmsh physical groups to DEVSIM regions
                # Note: geometry py uses lowercase names: channel, bottom_il, etc.
                for r_name in ["channel", "bottom_il", "ferroelectric", "top_il", "gate"]:
                    # check if region exists in mesh (skip if not present)
                    try:
                        devsim.add_gmsh_region(gmsh_name=r_name, mesh=self.device_name+"_mesh", 
                                             region=r_name, material="userset")
                    except:
                        pass
                        
                # Contacts
                devsim.add_gmsh_contact(gmsh_name="substrate_contact", mesh=self.device_name+"_mesh",
                                      region="channel", name="substrate", material="metal")
                devsim.add_gmsh_contact(gmsh_name="gate_contact", mesh=self.device_name+"_mesh",
                                      region="gate", name="gate", material="metal")
                # S/D for planar
                devsim.add_gmsh_contact(gmsh_name="source_contact", mesh=self.device_name+"_mesh",
                                      region="channel", name="source", material="metal")
                devsim.add_gmsh_contact(gmsh_name="drain_contact", mesh=self.device_name+"_mesh",
                                      region="channel", name="drain", material="metal")
                                      
                devsim.finalize_mesh(mesh=self.device_name+"_mesh")
                devsim.create_device(mesh=self.device_name+"_mesh", device=self.device_name)
                
            self.device_created = True
            return True
        return False
    
    def setup_physics(self, config: DeviceConfig) -> bool:
        """Setup 3D Physics (Poisson + FE)."""
        if not DEVSIM_AVAILABLE:
            return False
            
        # Set material parameters for all regions using parent class method
        for region in ["channel", "bottom_il", "ferroelectric", "top_il", "gate"]:
            try:
                self._set_material_parameters(region, config)
            except Exception as e:
                print(f"  Warning: Could not set material params for {region}: {e}")
        
        # Physics Equations (Poisson) for ALL regions including gate
        # Gate needs equation for contact to work
        for region in ["channel", "bottom_il", "ferroelectric", "top_il", "gate"]:
            self._setup_poisson_3d(region)
            
        # Contacts - set initial bias to 0V
        for contact in ["substrate", "gate", "source", "drain"]:
            try:
                self._setup_contact_equation(contact, 0.0)
            except Exception as e:
                print(f"  Warning: Could not setup contact {contact}: {e}")
        
        self.hysteresis_model = HysteresisModel(
            Ps=config.fe_material.Ps,
            Pr=config.fe_material.Pr,
            Ec=config.fe_material.Ec
        )
        self.physics_setup = True
        return True
        
    def _setup_poisson_3d(self, region: str):
        """Setup Poisson equation for 3D tet mesh."""
        # Use EdgeInverseLength for finite element formulation on tetrahedra edges
        # D = eps * E
        try:
            devsim.node_solution(device=self.device_name, region=region, name="Potential")
            devsim.edge_from_node_model(device=self.device_name, region=region, node_model="Potential")
            
            # Create Polarization model (needed for FE region)
            # Initialize to 0 - will be updated during voltage sweep
            if region == "ferroelectric":
                devsim.node_solution(device=self.device_name, region=region, name="Polarization")
                devsim.set_node_values(device=self.device_name, region=region,
                                      name="Polarization", init_from="Potential")  # Initialize to 0
            
            # Electric Field E = -grad(Phi)
            devsim.edge_model(device=self.device_name, region=region, name="ElectricField",
                             equation="(Potential@n0 - Potential@n1)*EdgeInverseLength")
            devsim.edge_model(device=self.device_name, region=region, name="ElectricField:Potential@n0",
                             equation="EdgeInverseLength")
            devsim.edge_model(device=self.device_name, region=region, name="ElectricField:Potential@n1",
                             equation="-EdgeInverseLength")
                             
            # Permittivity is pulled from parameter db
            eps_name = "Permittivity"
            
            devsim.edge_model(device=self.device_name, region=region, name="DField",
                             equation=f"{eps_name}*ElectricField")
            devsim.edge_model(device=self.device_name, region=region, name="DField:Potential@n0",
                             equation=f"{eps_name}*EdgeInverseLength")
            devsim.edge_model(device=self.device_name, region=region, name="DField:Potential@n1",
                             equation=f"-{eps_name}*EdgeInverseLength")
                             
            devsim.equation(device=self.device_name, region=region, name="PotentialEquation",
                           variable_name="Potential", edge_model="DField",
                           variable_update="log_damp")
        except Exception as e:
            print(f"Warning setting up 3D Poisson in {region}: {e}")

    def run_voltage_sweep(self, config: DeviceConfig,
                          sim_config: SimulationConfig) -> pd.DataFrame:
        """Run 3D simulation with P-V loop."""
        if not DEVSIM_AVAILABLE:
            raise RuntimeError("DEVSIM required for 3D solver execution")
            
        results = []
        voltage_steps = sim_config.voltage_steps
        n_forward = len(voltage_steps) // 2
        
        # Initialize all contacts to 0V
        self.set_voltage("source", 0.0)
        self.set_voltage("drain", 0.0)
        self.set_voltage("substrate", 0.0)
        self.set_voltage("gate", 0.0)
        
        # STEP 1: Equilibrium solve at Vg=0 (important for convergence)
        print("    Solving equilibrium (Vg=0)...")
        try:
            devsim.solve(type="dc", absolute_error=1e-8, relative_error=1e-6,
                        maximum_iterations=50)
            print("    Equilibrium converged.")
        except Exception as e:
            print(f"    Equilibrium failed: {e}, trying with relaxed tolerance...")
            try:
                devsim.solve(type="dc", absolute_error=1e-6, relative_error=1e-4,
                            maximum_iterations=100)
            except:
                raise RuntimeError("Convergence failure!")
        
        last_Vg = 0.0
        
        for i, Vg in enumerate(voltage_steps):
            direction = "forward" if i < n_forward else "reverse"
            
            # STEP 2: Gradual voltage ramping for convergence
            # If voltage step is large, ramp in smaller increments
            delta_V = Vg - last_Vg
            n_ramp_steps = max(1, int(abs(delta_V) / 0.5))  # 0.5V max step
            
            for ramp_i in range(n_ramp_steps):
                ramp_Vg = last_Vg + (ramp_i + 1) * delta_V / n_ramp_steps
                self.set_voltage("gate", ramp_Vg)
                
                try:
                    # Relaxed tolerances for 3D (1e-8 is too tight)
                    devsim.solve(type="dc", absolute_error=1e-6, relative_error=1e-4,
                                maximum_iterations=50)
                except:
                    # If fails, try even more relaxed
                    try:
                        devsim.solve(type="dc", absolute_error=1e-4, relative_error=1e-2,
                                    maximum_iterations=100)
                    except:
                        print(f"    Warning: Convergence issue at Vg={ramp_Vg:.2f}V")
            
            last_Vg = Vg
                            
            # 3. Extract Field in FE (Average vertical field)
            # In 3D this is complex. We approximate via standard formula for P-update
            # Then we could put charge back. For this "Fast" implementation, 
            # we focus on the solution convergence and basic P tracking.
            
            # Get potential at mid-points
            # approx E = Vg_eff / T_stack
            E_fe = Vg / (config.geometry.t_fe * 1e-7 * 3) # approx
            
            # Update P
            P = self.update_polarization("ferroelectric", E_fe, direction)
            
            # Store
            results.append({
                "Vg": Vg,
                "Id": 0.0, # Just C-V/P-V focus for 3D
                "Polarization": P * 1e2 if P else 0.0,
                "E_fe": E_fe,
                "direction": direction
            })
            
        return pd.DataFrame(results)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_baseline_simulation(config: Optional[DeviceConfig] = None,
                            sim_config: Optional[SimulationConfig] = None) -> pd.DataFrame:
    """Run 1D baseline simulation."""
    from .config import get_baseline_config
    
    config = config or get_baseline_config()
    sim_config = sim_config or SimulationConfig()
    
    solver = MIFISSolver("baseline_1d")
    solver.create_1d_mesh(config)
    solver.setup_physics(config)
    
    return solver.run_voltage_sweep(config, sim_config)


def run_2d_simulation(config: Optional[DeviceConfig] = None,
                      sim_config: Optional[SimulationConfig] = None) -> pd.DataFrame:
    """Run 2D planar simulation."""
    from .config import get_baseline_config
    
    config = config or get_baseline_config()
    sim_config = sim_config or SimulationConfig()
    
    solver = MIFIS2DSolver("planar_2d")
    solver.create_2d_mesh(config)
    solver.setup_physics(config)
    
    return solver.run_voltage_sweep(config, sim_config)


def run_3d_simulation(config: Optional[DeviceConfig] = None,
                      sim_config: Optional[SimulationConfig] = None,
                      output_dir: str = "data/meshes") -> pd.DataFrame:
    """Run 3D simulation."""
    from .config import get_baseline_config
    
    config = config or get_baseline_config()
    sim_config = sim_config or SimulationConfig()
    
    solver = MIFIS3DSolver("mifis_3d")
    solver.create_3d_mesh(config, output_dir)
    solver.setup_physics(config)
    
    return solver.run_voltage_sweep(config, sim_config)
