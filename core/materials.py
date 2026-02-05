"""
MIFIS FeFET Simulation Framework
================================
Material property database with literature-validated parameters.

Author: Thesis Project
Date: February 2026
"""

from dataclasses import dataclass
from typing import Dict, Optional
from .config import FerroelectricMaterial, DielectricMaterial, SemiconductorMaterial


# =============================================================================
# FERROELECTRIC MATERIALS DATABASE
# =============================================================================

FERROELECTRIC_DB: Dict[str, FerroelectricMaterial] = {
    # HZO - YOUR PRIMARY MATERIAL (Kuk 2024, Hu 2024)
    "HZO": FerroelectricMaterial(
        name="HZO",
        epsilon_r=30.0,
        Ps=38.0,      # μC/cm²
        Pr=18.0,      # μC/cm²
        Ec=1.0,       # MV/cm
        tau_switching=10.0
    ),
    
    # Standard HfO2 (Hafnium oxide-based FeFET paper)
    "HfO2": FerroelectricMaterial(
        name="HfO2",
        epsilon_r=25.0,
        Ps=35.0,
        Pr=12.0,
        Ec=1.5,
        tau_switching=15.0
    ),
    
    # Silicon-doped HfO2 (highest MW)
    "Si-HfO2": FerroelectricMaterial(
        name="Si-HfO2",
        epsilon_r=29.0,
        Ps=40.0,
        Pr=20.0,
        Ec=1.3,
        tau_switching=12.0
    ),
    
    # Aluminum-doped HfO2
    "Al-HfO2": FerroelectricMaterial(
        name="Al-HfO2",
        epsilon_r=22.0,
        Ps=30.0,
        Pr=8.0,
        Ec=1.8,
        tau_switching=20.0
    ),
    
    # PZT (classical ferroelectric, scaling limited)
    "PZT": FerroelectricMaterial(
        name="PZT",
        epsilon_r=500.0,
        Ps=45.0,
        Pr=20.0,
        Ec=1.0,
        tau_switching=5.0
    ),
}


# =============================================================================
# DIELECTRIC MATERIALS DATABASE
# =============================================================================

DIELECTRIC_DB: Dict[str, DielectricMaterial] = {
    # Silicon dioxide - standard IL
    "SiO2": DielectricMaterial(
        name="SiO2",
        epsilon_r=3.9,
        band_gap=9.0,
        electron_affinity=0.95
    ),
    
    # Aluminum oxide - higher-k alternative
    "Al2O3": DielectricMaterial(
        name="Al2O3",
        epsilon_r=9.0,
        band_gap=8.8,
        electron_affinity=1.0
    ),
    
    # Silicon oxynitride - lower noise
    "SiON": DielectricMaterial(
        name="SiON",
        epsilon_r=5.0,
        band_gap=7.5,
        electron_affinity=1.2
    ),
    
    # High-k hafnium oxide (as IL)
    "HfO2_IL": DielectricMaterial(
        name="HfO2_IL",
        epsilon_r=25.0,
        band_gap=5.8,
        electron_affinity=2.0
    ),
}


# =============================================================================
# SEMICONDUCTOR MATERIALS DATABASE
# =============================================================================

SEMICONDUCTOR_DB: Dict[str, SemiconductorMaterial] = {
    # Silicon p-type (YOUR DESIGN)
    "Si_p": SemiconductorMaterial(
        name="Silicon",
        epsilon_r=11.7,
        ni=1.0e10,
        mobility_e=1400.0,
        mobility_h=450.0,
        doping_type="p",
        doping_concentration=1e17
    ),
    
    # Silicon n-type
    "Si_n": SemiconductorMaterial(
        name="Silicon",
        epsilon_r=11.7,
        ni=1.0e10,
        mobility_e=1400.0,
        mobility_h=450.0,
        doping_type="n",
        doping_concentration=1e17
    ),
    
    # Germanium (high mobility alternative)
    "Ge": SemiconductorMaterial(
        name="Germanium",
        epsilon_r=16.0,
        ni=2.4e13,
        mobility_e=3900.0,
        mobility_h=1900.0,
        doping_type="p",
        doping_concentration=1e17
    ),
}


# =============================================================================
# METAL GATE DATABASE
# =============================================================================

@dataclass
class MetalGate:
    """Metal gate electrode properties."""
    name: str
    work_function: float  # eV
    resistivity: float    # Ω·cm


METAL_DB: Dict[str, MetalGate] = {
    "TiN": MetalGate("TiN", work_function=4.7, resistivity=25e-6),
    "W": MetalGate("W", work_function=4.55, resistivity=5.3e-6),
    "Ir": MetalGate("Ir", work_function=5.3, resistivity=4.7e-6),
    "Pt": MetalGate("Pt", work_function=5.65, resistivity=10.6e-6),
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_fe_material(name: str) -> FerroelectricMaterial:
    """Get ferroelectric material by name."""
    if name not in FERROELECTRIC_DB:
        raise ValueError(f"Unknown FE material: {name}. Available: {list(FERROELECTRIC_DB.keys())}")
    return FERROELECTRIC_DB[name]


def get_dielectric(name: str) -> DielectricMaterial:
    """Get dielectric material by name."""
    if name not in DIELECTRIC_DB:
        raise ValueError(f"Unknown dielectric: {name}. Available: {list(DIELECTRIC_DB.keys())}")
    return DIELECTRIC_DB[name]


def get_semiconductor(name: str) -> SemiconductorMaterial:
    """Get semiconductor material by name."""
    if name not in SEMICONDUCTOR_DB:
        raise ValueError(f"Unknown semiconductor: {name}. Available: {list(SEMICONDUCTOR_DB.keys())}")
    return SEMICONDUCTOR_DB[name]


def list_all_materials() -> Dict[str, list]:
    """List all available materials in the database."""
    return {
        "ferroelectrics": list(FERROELECTRIC_DB.keys()),
        "dielectrics": list(DIELECTRIC_DB.keys()),
        "semiconductors": list(SEMICONDUCTOR_DB.keys()),
        "metals": list(METAL_DB.keys()),
    }


# =============================================================================
# LITERATURE BENCHMARK VALUES
# =============================================================================

LITERATURE_BENCHMARKS = {
    "Kuk_2024": {
        "description": "Record MIFIS memory window",
        "memory_window_V": 12.2,
        "fe_material": "HZO",
        "structure": "MIFIS",
    },
    "Hu_2024": {
        "description": "Optimized top IL study",
        "memory_window_V": 6.3,
        "fe_material": "HZO",
        "structure": "MIFIS",
    },
    "Your_Target": {
        "description": "Thesis baseline target",
        "memory_window_V": 2.5,  # Conservative lower bound
        "fe_material": "HZO",
        "structure": "MIFIS",
    },
}
