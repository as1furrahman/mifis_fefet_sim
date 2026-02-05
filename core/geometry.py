"""
MIFIS FeFET Simulation Framework
================================
Geometry and Mesh Generation Module

Provides mesh generation for 1D, 2D, and 3D device structures.
Uses DEVSIM built-in meshing for 1D/2D and Gmsh for 3D.

Author: Thesis Project
Date: February 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from .config import DeviceConfig, DeviceGeometry, Architecture

# DEVSIM is REQUIRED
try:
    import devsim
    DEVSIM_AVAILABLE = True
except ImportError:
    raise ImportError(
        "DEVSIM is required for mesh generation.\n"
        "Install with: pip install devsim"
    )

# Gmsh is optional for 3D
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


# =============================================================================
# MESH PARAMETERS - SYNCHRONIZED WITH ROADMAP
# =============================================================================

@dataclass
class MeshParameters:
    """Mesh generation parameters aligned with TCAD_Simulation_Parameters.csv"""
    
    # Resolution settings
    mesh_scale: float = 1.0          # Global refinement factor
    
    # 1D mesh
    n_points_1d: int = 500           # Points for 1D mesh
    
    # 2D mesh
    n_x_2d: int = 100                # Points in x-direction (gate length)
    n_z_2d: int = 150                # Points in z-direction (vertical)
    
    # 3D mesh
    n_x_3d: int = 50                 # Points in x (gate length)
    n_y_3d: int = 50                 # Points in y (width)
    n_z_3d: int = 100                # Points in z (vertical)
    
    # Interface refinement (critical for FE/IL interfaces)
    interface_refinement: float = 0.1  # nm - fine mesh at interfaces
    bulk_spacing: float = 1.0          # nm - coarse mesh in bulk
    
    # Quality targets
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0


# Default mesh parameters - OPTIMIZED for balance of speed/accuracy
DEFAULT_MESH_PARAMS = MeshParameters(
    mesh_scale=1.0,
    n_points_1d=200,              # Reduced from 500
    n_x_2d=50,                    # Reduced from 100
    n_z_2d=75,                    # Reduced from 150
    interface_refinement=0.2,     # For 1D/2D (nm)
    bulk_spacing=2.0              # For 1D/2D (nm)
)

# Fast mesh for quick testing
FAST_MESH_PARAMS = MeshParameters(
    mesh_scale=2.0,
    n_points_1d=100,              # Very fast 1D
    n_x_2d=25,                    # Coarse 2D
    n_z_2d=40,
    interface_refinement=0.5,     # For 1D/2D
    bulk_spacing=5.0              # For 1D/2D
)

# High accuracy mesh (slow but precise)
ACCURATE_MESH_PARAMS = MeshParameters(
    mesh_scale=0.5,
    n_points_1d=500,
    n_x_2d=100,
    n_z_2d=150,
    interface_refinement=0.1,
    bulk_spacing=1.0
)

# 3D-SPECIFIC mesh parameters (MEMORY OPTIMIZED)
# For 3D we need larger elements than 1D/2D to fit in RAM
# Reduced for denser mesh (was 5/15, now 3/8)
MESH_3D_FINE = 3.0     # nm - Fine regions (interfaces)
MESH_3D_COARSE = 8.0   # nm - Bulk regions


# =============================================================================
# 1D MESH GENERATION
# =============================================================================

def create_1d_mesh(device_name: str, config: DeviceConfig, 
                   mesh_params: Optional[MeshParameters] = None) -> Dict:
    """
    Create 1D mesh for MIFIS stack (vertical slice).
    
    Mesh structure (bottom to top):
    - Si Channel
    - Bottom IL (SiO2)
    - Ferroelectric (HZO)
    - Top IL (SiO2)
    - Gate (TiN)
    
    Args:
        device_name: Name for the DEVSIM device
        config: Device configuration
        mesh_params: Mesh generation parameters
        
    Returns:
        Dictionary with mesh info and region boundaries
    """
    if mesh_params is None:
        mesh_params = DEFAULT_MESH_PARAMS
    
    geom = config.geometry
    dims = geom.to_meters()
    
    # Calculate layer boundaries (bottom to top, in meters)
    boundaries = {
        'z_bottom': 0.0,
        'z_channel_top': dims['t_channel'],
        'z_bottom_il_top': dims['t_channel'] + dims['t_bottom_il'],
        'z_fe_top': dims['t_channel'] + dims['t_bottom_il'] + dims['t_fe'],
        'z_top_il_top': dims['t_channel'] + dims['t_bottom_il'] + dims['t_fe'] + dims['t_top_il'],
        'z_top': dims['total_height'],
    }
    
    if DEVSIM_AVAILABLE:
        mesh_name = f"{device_name}_mesh"
        
        # Create device
        devsim.create_device(device=device_name)
        devsim.create_1d_mesh(mesh=mesh_name)
        
        # Calculate spacing based on mesh refinement
        fine_spacing = mesh_params.interface_refinement * 1e-9 * mesh_params.mesh_scale
        coarse_spacing = mesh_params.bulk_spacing * 1e-9 * mesh_params.mesh_scale
        
        # Add mesh lines with appropriate spacing
        # Channel bulk (coarser)
        devsim.add_1d_mesh_line(mesh=mesh_name, pos=0.0, 
                                ps=coarse_spacing, tag="bottom")
        
        # Channel/IL interface (fine)
        devsim.add_1d_mesh_line(mesh=mesh_name, pos=boundaries['z_channel_top'],
                                ps=fine_spacing, tag="channel_top")
        
        # Bottom IL/FE interface (very fine - critical)
        devsim.add_1d_mesh_line(mesh=mesh_name, pos=boundaries['z_bottom_il_top'],
                                ps=fine_spacing * 0.5, tag="bottom_il_top")
        
        # FE bulk (medium)
        fe_mid = (boundaries['z_bottom_il_top'] + boundaries['z_fe_top']) / 2
        devsim.add_1d_mesh_line(mesh=mesh_name, pos=fe_mid,
                                ps=fine_spacing * 2, tag="fe_mid")
        
        # FE/Top IL interface (very fine - critical)
        devsim.add_1d_mesh_line(mesh=mesh_name, pos=boundaries['z_fe_top'],
                                ps=fine_spacing * 0.5, tag="fe_top")
        
        # Top IL/Gate interface
        devsim.add_1d_mesh_line(mesh=mesh_name, pos=boundaries['z_top_il_top'],
                                ps=fine_spacing, tag="top_il_top")
        
        # Gate top
        devsim.add_1d_mesh_line(mesh=mesh_name, pos=boundaries['z_top'],
                                ps=coarse_spacing, tag="top")
        
        # Add regions
        devsim.add_1d_region(mesh=mesh_name, material="Silicon",
                            region="channel", tag1="bottom", tag2="channel_top")
        devsim.add_1d_region(mesh=mesh_name, material="SiO2",
                            region="bottom_il", tag1="channel_top", tag2="bottom_il_top")
        devsim.add_1d_region(mesh=mesh_name, material="HZO",
                            region="ferroelectric", tag1="bottom_il_top", tag2="fe_top")
        devsim.add_1d_region(mesh=mesh_name, material="SiO2",
                            region="top_il", tag1="fe_top", tag2="top_il_top")
        devsim.add_1d_region(mesh=mesh_name, material="TiN",
                            region="gate", tag1="top_il_top", tag2="top")
        
        # Add contacts
        devsim.add_1d_contact(mesh=mesh_name, name="substrate", 
                             tag="bottom", material="Metal")
        devsim.add_1d_contact(mesh=mesh_name, name="gate", 
                             tag="top", material="Metal")
        
        # Finalize
        devsim.finalize_mesh(mesh=mesh_name)
        devsim.create_device_from_mesh(mesh=mesh_name, device=device_name)
    
    return {
        'device_name': device_name,
        'mesh_type': '1D',
        'boundaries': boundaries,
        'regions': ['channel', 'bottom_il', 'ferroelectric', 'top_il', 'gate'],
        'contacts': ['substrate', 'gate'],
        'n_points': mesh_params.n_points_1d,
    }


# =============================================================================
# 2D MESH GENERATION
# =============================================================================

def create_2d_mesh(device_name: str, config: DeviceConfig,
                   mesh_params: Optional[MeshParameters] = None) -> Dict:
    """
    Create 2D mesh for planar MIFIS FeFET (x-z cross section).
    
    Includes source/drain regions for transistor operation.
    
    Structure:
        x-direction: Source | Channel | Drain
        z-direction: Substrate | IL | FE | IL | Gate
        
    Args:
        device_name: Name for the DEVSIM device
        config: Device configuration
        mesh_params: Mesh generation parameters
        
    Returns:
        Dictionary with mesh info
    """
    if mesh_params is None:
        mesh_params = DEFAULT_MESH_PARAMS
    
    geom = config.geometry
    dims = geom.to_meters()
    
    # Horizontal dimensions
    Lg = dims['Lg']                    # Gate length
    Lsd = Lg * 0.3                     # Source/Drain extension (30% of Lg)
    L_total = Lg + 2 * Lsd             # Total device length
    
    # Vertical boundaries
    z_sub_bottom = 0.0
    z_channel_top = dims['t_channel']
    z_bottom_il = z_channel_top + dims['t_bottom_il']
    z_fe = z_bottom_il + dims['t_fe']
    z_top_il = z_fe + dims['t_top_il']
    z_gate = z_top_il + dims['t_gate']
    
    mesh_info = {
        'device_name': device_name,
        'mesh_type': '2D',
        'dimensions': {
            'L_total': L_total,
            'Lg': Lg,
            'Lsd': Lsd,
            'height': z_gate,
        },
        'n_points_x': mesh_params.n_x_2d,
        'n_points_z': mesh_params.n_z_2d,
        'total_elements': mesh_params.n_x_2d * mesh_params.n_z_2d,
    }
    
    if DEVSIM_AVAILABLE:
        mesh_name = f"{device_name}_mesh"
        
        devsim.create_device(device=device_name)
        devsim.create_2d_mesh(mesh=mesh_name)
        
        # X-direction mesh lines
        fine_x = mesh_params.interface_refinement * 1e-9
        coarse_x = mesh_params.bulk_spacing * 1e-9
        
        # Source region
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=0.0, ps=coarse_x)
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=Lsd, ps=fine_x)
        
        # Channel region under gate
        n_channel = int(Lg / (fine_x * 2))
        for i in range(n_channel):
            x_pos = Lsd + (i / n_channel) * Lg
            devsim.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=x_pos, ps=fine_x)
        
        # Drain region
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=Lsd + Lg, ps=fine_x)
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=L_total, ps=coarse_x)
        
        # Z-direction mesh lines
        fine_z = mesh_params.interface_refinement * 1e-9 * 0.5
        coarse_z = mesh_params.bulk_spacing * 1e-9
        
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=0.0, ps=coarse_z)
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=z_channel_top, ps=fine_z)
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=z_bottom_il, ps=fine_z)
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=z_fe, ps=fine_z)
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=z_top_il, ps=fine_z)
        devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=z_gate, ps=coarse_z)
        
        # Define regions (simplified - channel + MIFIS stack)
        devsim.add_2d_region(mesh=mesh_name, material="Silicon", region="silicon",
                            xl=0, xh=L_total, yl=0, yh=z_channel_top)
        devsim.add_2d_region(mesh=mesh_name, material="SiO2", region="bottom_il",
                            xl=Lsd, xh=Lsd+Lg, yl=z_channel_top, yh=z_bottom_il)
        devsim.add_2d_region(mesh=mesh_name, material="HZO", region="ferroelectric",
                            xl=Lsd, xh=Lsd+Lg, yl=z_bottom_il, yh=z_fe)
        devsim.add_2d_region(mesh=mesh_name, material="SiO2", region="top_il",
                            xl=Lsd, xh=Lsd+Lg, yl=z_fe, yh=z_top_il)
        devsim.add_2d_region(mesh=mesh_name, material="TiN", region="gate",
                            xl=Lsd, xh=Lsd+Lg, yl=z_top_il, yh=z_gate)
        
        # Contacts
        devsim.add_2d_contact(mesh=mesh_name, name="source", region="silicon",
                             xl=0, xh=fine_x, yl=0, yh=z_channel_top, material="Metal")
        devsim.add_2d_contact(mesh=mesh_name, name="drain", region="silicon",
                             xl=L_total-fine_x, xh=L_total, yl=0, yh=z_channel_top, material="Metal")
        devsim.add_2d_contact(mesh=mesh_name, name="gate", region="gate",
                             xl=Lsd, xh=Lsd+Lg, yl=z_gate-fine_z, yh=z_gate, material="Metal")
        devsim.add_2d_contact(mesh=mesh_name, name="substrate", region="silicon",
                             xl=0, xh=L_total, yl=0, yh=fine_z, material="Metal")
        
        devsim.finalize_mesh(mesh=mesh_name)
        devsim.create_device_from_mesh(mesh=mesh_name, device=device_name)
        
        mesh_info['contacts'] = ['source', 'drain', 'gate', 'substrate']
        mesh_info['regions'] = ['silicon', 'bottom_il', 'ferroelectric', 'top_il', 'gate']
    
    return mesh_info


# =============================================================================
# 3D MESH GENERATION (GMSH)
# =============================================================================

def create_3d_mesh_gmsh(config: DeviceConfig, output_path: str,
                        mesh_params: Optional[MeshParameters] = None) -> Dict:
    """
    Create 3D mesh for MIFIS FeFET using Gmsh.
    
    Creates a complete 3D device with:
    - TiN Gate electrode
    - Top SiO2 interlayer (4 nm)
    - HZO ferroelectric (13.8 nm)
    - Bottom SiO2 interlayer (0.7 nm)
    - Si channel
    - Source/Drain contacts
    
    Args:
        config: Device configuration
        output_path: Path to save .msh file
        mesh_params: Mesh generation parameters
        
    Returns:
        Dictionary with mesh info and file path
    """
    if not GMSH_AVAILABLE:
        return {
            'status': 'error',
            'message': 'Gmsh not available. Install with: pip install gmsh',
            'mesh_type': '3D',
        }
    
    if mesh_params is None:
        mesh_params = DEFAULT_MESH_PARAMS
    
    geom = config.geometry
    
    # Dimensions in nm (Gmsh uses arbitrary units, we use nm)
    Lg = geom.Lg           # Gate length
    W = geom.W             # Width
    Lsd = Lg * 0.3         # Source/Drain length
    L_total = Lg + 2*Lsd
    
    # Layer thicknesses
    t_channel = geom.t_channel
    t_bottom_il = geom.t_bottom_il
    t_fe = geom.t_fe
    t_top_il = geom.t_top_il
    t_gate = geom.t_gate
    
    # Z positions (bottom to top)
    z0 = 0
    z1 = t_channel
    z2 = z1 + t_bottom_il
    z3 = z2 + t_fe
    z4 = z3 + t_top_il
    z5 = z4 + t_gate
    
    gmsh.initialize()
    gmsh.model.add("mifis_3d")
    
    # Mesh size - USE 3D-SPECIFIC SIZES (MEMORY OPTIMIZED)
    # The 1D/2D parameters (0.2-2nm) create 20M+ elements in 3D
    # We need 5-15nm spacing for practical 3D meshing (~100k-500k elements)
    lc_fine = MESH_3D_FINE      # 5nm for interfaces
    lc_coarse = MESH_3D_COARSE  # 15nm for bulk
    
    # Create boxes for each layer
    # Channel (full width)
    channel = gmsh.model.occ.addBox(0, 0, z0, L_total, W, t_channel)
    
    # Bottom IL (under gate only)
    bottom_il = gmsh.model.occ.addBox(Lsd, 0, z1, Lg, W, t_bottom_il)
    
    # Ferroelectric
    fe = gmsh.model.occ.addBox(Lsd, 0, z2, Lg, W, t_fe)
    
    # Top IL
    top_il = gmsh.model.occ.addBox(Lsd, 0, z3, Lg, W, t_top_il)
    
    # Gate
    gate = gmsh.model.occ.addBox(Lsd, 0, z4, Lg, W, t_gate)
    
    gmsh.model.occ.synchronize()
    
    # Physical groups for regions (Volumes)
    gmsh.model.addPhysicalGroup(3, [channel], tag=1, name="channel")
    gmsh.model.addPhysicalGroup(3, [bottom_il], tag=2, name="bottom_il")
    gmsh.model.addPhysicalGroup(3, [fe], tag=3, name="ferroelectric")
    gmsh.model.addPhysicalGroup(3, [top_il], tag=4, name="top_il")
    gmsh.model.addPhysicalGroup(3, [gate], tag=5, name="gate")
    
    # Physical groups for contacts (Surfaces) - REQUIRED BY DEVSIM
    # Get boundary surfaces of each volume
    channel_surfs = gmsh.model.getBoundary([(3, channel)], combined=False, oriented=False)
    gate_surfs = gmsh.model.getBoundary([(3, gate)], combined=False, oriented=False)
    
    # Identify contact surfaces by bounding box
    eps = 0.01  # tolerance for coordinate matching
    
    source_surfaces = []
    drain_surfaces = []
    substrate_surfaces = []
    gate_contact_surfaces = []
    
    for dim, tag in channel_surfs:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        
        # Source contact: x=0 face
        if abs(xmin) < eps and abs(xmax) < eps:
            source_surfaces.append(tag)
        # Drain contact: x=L_total face
        elif abs(xmin - L_total) < eps and abs(xmax - L_total) < eps:
            drain_surfaces.append(tag)
        # Substrate contact: z=0 face (bottom)
        elif abs(zmin) < eps and abs(zmax) < eps:
            substrate_surfaces.append(tag)
            
    for dim, tag in gate_surfs:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        # Gate contact: z=z5 face (top of gate)
        if abs(zmax - z5) < eps and abs(zmin - z5) < eps:
            gate_contact_surfaces.append(tag)
    
    # Add physical surface groups for contacts
    if source_surfaces:
        gmsh.model.addPhysicalGroup(2, source_surfaces, tag=101, name="source_contact")
    if drain_surfaces:
        gmsh.model.addPhysicalGroup(2, drain_surfaces, tag=102, name="drain_contact")
    if substrate_surfaces:
        gmsh.model.addPhysicalGroup(2, substrate_surfaces, tag=103, name="substrate_contact")
    if gate_contact_surfaces:
        gmsh.model.addPhysicalGroup(2, gate_contact_surfaces, tag=104, name="gate_contact")
    
    # Set mesh sizes
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_coarse)
    
    # Refine at FE interfaces (Vertical)
    # Get surfaces at z2 (bottom_il/FE) and z3 (FE/top_il)
    fe_surfaces = gmsh.model.getBoundary([(3, fe)], combined=False)
    for surf in fe_surfaces:
        gmsh.model.mesh.setSize(gmsh.model.getBoundary([surf], combined=False), lc_fine)
        
    # Refine at Source/Channel and Channel/Drain junctions (Lateral)
    # Critical for short-channel effects and junction fields
    # We refine the points at x=Lsd and x=Lsd+Lg across the stack
    
    # Defines points of interest for lateral refinement
    refinement_points = []
    
    # Find points at junction locations (approximate using bounding box search)
    # x = Lsd (Source junction) and x = Lsd + Lg (Drain junction)
    eps = 0.1 # tolerance
    
    # Get all lines/curves in the model
    entities = gmsh.model.getEntities(1) 
    for dim, tag in entities:
        # Check bounding box of curve
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        
        # If curve is at junction X-location
        if (abs(xmin - Lsd) < eps and abs(xmax - Lsd) < eps) or \
           (abs(xmin - (Lsd+Lg)) < eps and abs(xmax - (Lsd+Lg)) < eps):
            gmsh.model.mesh.setSize([(dim, tag)], lc_fine)
            
    # Also refine the channel volume itself slightly more than bulk
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_coarse) # Base
    gmsh.model.mesh.setSize(gmsh.model.getBoundary([(3, channel)], recursive=True), lc_fine * 2.0) # Channel surfaces
    
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    
    # Get mesh statistics
    nodes = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements()
    n_nodes = len(nodes[0])
    n_elements = sum(len(e) for e in elements[1])
    
    # Save mesh in MSH 2.2 ASCII format (DEVSIM requirement)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Force MSH 2.2 format for DEVSIM compatibility
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(output_file))
    
    gmsh.finalize()
    
    return {
        'mesh_type': '3D',
        'file_path': str(output_file),
        'n_nodes': n_nodes,
        'n_elements': n_elements,
        'dimensions': {
            'L_total': L_total,
            'W': W,
            'height': z5,
        },
        'regions': ['channel', 'bottom_il', 'ferroelectric', 'top_il', 'gate'],
        'status': 'success',
    }


# =============================================================================
# GAA MESH GENERATION
# =============================================================================

def create_gaa_mesh_gmsh(config: DeviceConfig, output_path: str,
                         mesh_params: Optional[MeshParameters] = None) -> Dict:
    """
    Create 3D GAA (Gate-All-Around) mesh using Gmsh.
    
    Creates cylindrical nanowire structure with wrapped gate stack.
    
    Args:
        config: Device configuration (must have radius set)
        output_path: Path to save .msh file
        mesh_params: Mesh parameters
        
    Returns:
        Dictionary with mesh info
    """
    if not GMSH_AVAILABLE:
        return {'status': 'error', 'message': 'Gmsh not available'}
    
    if mesh_params is None:
        mesh_params = DEFAULT_MESH_PARAMS
    
    geom = config.geometry
    
    # Get dimensions
    Lg = geom.Lg
    r_channel = geom.radius if geom.radius else 5.0  # Default 5nm radius
    Lsd = Lg * 0.3
    L_total = Lg + 2*Lsd
    
    # Layer thicknesses (radial)
    t_bottom_il = geom.t_bottom_il
    t_fe = geom.t_fe
    t_top_il = geom.t_top_il
    t_gate = geom.t_gate
    
    # Radii
    r1 = r_channel
    r2 = r1 + t_bottom_il
    r3 = r2 + t_fe
    r4 = r3 + t_top_il
    r5 = r4 + t_gate
    
    gmsh.initialize()
    gmsh.model.add("gaa_3d")
    
    # Use 3D-optimized mesh size (not 1D/2D parameters)
    lc = MESH_3D_FINE  # 5nm for GAA (more detail needed for curved surfaces)
    
    # Create cylinders (centered at origin, along x-axis)
    # We create the gate region only around the channel, not S/D
    
    # Full length channel cylinder
    channel = gmsh.model.occ.addCylinder(0, 0, 0, L_total, 0, 0, r_channel)
    
    # Gate stack only around channel (Lsd to Lsd+Lg)
    bottom_il = gmsh.model.occ.addCylinder(Lsd, 0, 0, Lg, 0, 0, r2)
    bottom_il_inner = gmsh.model.occ.addCylinder(Lsd, 0, 0, Lg, 0, 0, r1)
    
    fe = gmsh.model.occ.addCylinder(Lsd, 0, 0, Lg, 0, 0, r3)
    fe_inner = gmsh.model.occ.addCylinder(Lsd, 0, 0, Lg, 0, 0, r2)
    
    top_il = gmsh.model.occ.addCylinder(Lsd, 0, 0, Lg, 0, 0, r4)
    top_il_inner = gmsh.model.occ.addCylinder(Lsd, 0, 0, Lg, 0, 0, r3)
    
    gate = gmsh.model.occ.addCylinder(Lsd, 0, 0, Lg, 0, 0, r5)
    gate_inner = gmsh.model.occ.addCylinder(Lsd, 0, 0, Lg, 0, 0, r4)
    
    # Boolean operations to create hollow shells
    gmsh.model.occ.cut([(3, bottom_il)], [(3, bottom_il_inner)])
    gmsh.model.occ.cut([(3, fe)], [(3, fe_inner)])
    gmsh.model.occ.cut([(3, top_il)], [(3, top_il_inner)])
    gmsh.model.occ.cut([(3, gate)], [(3, gate_inner)])
    
    gmsh.model.occ.synchronize()
    
    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    
    # Generate mesh
    gmsh.model.mesh.generate(3)
    
    # Statistics
    nodes = gmsh.model.mesh.getNodes()
    n_nodes = len(nodes[0])
    
    # Save in MSH 2.2 format (DEVSIM requirement)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_path)
    gmsh.finalize()
    
    return {
        'mesh_type': '3D_GAA',
        'file_path': output_path,
        'n_nodes': n_nodes,
        'radii': {'r_channel': r_channel, 'r_outer': r5},
        'wrap_angle': config.geometry.wrap_angle,
        'status': 'success',
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_mesh(mesh_info: Dict) -> Dict:
    """Validate mesh quality."""
    validation = {
        'valid': True,
        'warnings': [],
        'errors': [],
    }
    
    if mesh_info.get('status') == 'error':
        validation['valid'] = False
        validation['errors'].append(mesh_info.get('message', 'Unknown error'))
        return validation
    
    # Check node count
    n_nodes = mesh_info.get('n_nodes', 0) or mesh_info.get('n_points', 0)
    if n_nodes < 100:
        validation['warnings'].append(f"Low node count: {n_nodes}")
    if n_nodes > 1000000:
        validation['warnings'].append(f"Very high node count: {n_nodes}")
    
    return validation


def get_mesh_statistics(device_name: str) -> Dict:
    """Get mesh statistics from DEVSIM device."""
    if not DEVSIM_AVAILABLE:
        return {'available': False}
    
    try:
        regions = devsim.get_region_list(device=device_name)
        stats = {}
        for region in regions:
            nodes = devsim.get_node_model_list(device=device_name, region=region)
            stats[region] = {'models': len(nodes)}
        return {'available': True, 'regions': stats}
    except:
        return {'available': False}
