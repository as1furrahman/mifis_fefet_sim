"""
MIFIS FeFET Simulation Framework
================================
Post-processing and metric extraction utilities.

Author: Thesis Project
Date: February 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeFETMetrics:
    """Container for extracted FeFET performance metrics."""
    # Threshold voltages
    Vth_forward: float
    Vth_reverse: float
    
    # Memory window
    memory_window: float
    
    # Current metrics
    Ion: float
    Ioff: float
    Ion_Ioff_ratio: float
    
    # Subthreshold swing
    SS_forward: float
    SS_reverse: float
    SS_average: float
    
    # Coercive voltages
    Vc_positive: Optional[float] = None
    Vc_negative: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "Vth_forward": self.Vth_forward,
            "Vth_reverse": self.Vth_reverse,
            "Memory_Window_V": self.memory_window,
            "Ion_A": self.Ion,
            "Ioff_A": self.Ioff,
            "Ion_Ioff_ratio": self.Ion_Ioff_ratio,
            "SS_forward_mV_dec": self.SS_forward,
            "SS_reverse_mV_dec": self.SS_reverse,
            "SS_average_mV_dec": self.SS_average,
        }


def extract_threshold_voltage(df: pd.DataFrame, 
                               Vg_col: str = "Vg",
                               Id_col: str = "Id",
                               I_threshold: float = 1e-7,
                               direction: str = "forward") -> float:
    """
    Extract threshold voltage using constant current method.
    
    Args:
        df: DataFrame with Vg and Id columns
        Vg_col: Name of gate voltage column
        Id_col: Name of drain current column
        I_threshold: Target current for Vth (A)
        direction: "forward" or "reverse"
        
    Returns:
        Threshold voltage (V)
    """
    # Filter by direction if column exists
    if "direction" in df.columns:
        subset = df[df["direction"] == direction].copy()
    else:
        subset = df.copy()
    
    if len(subset) == 0:
        return np.nan
    
    # Sort by Vg
    subset = subset.sort_values(by=Vg_col)
    
    Vg = subset[Vg_col].values
    Id = np.abs(subset[Id_col].values)
    
    # Find crossing point using log interpolation
    try:
        log_Id = np.log10(Id + 1e-20)  # Avoid log(0)
        log_target = np.log10(I_threshold)
        
        # Interpolate
        Vth = np.interp(log_target, log_Id, Vg)
        return float(Vth)
    except:
        return np.nan


def extract_subthreshold_swing(df: pd.DataFrame,
                                Vg_col: str = "Vg",
                                Id_col: str = "Id",
                                direction: str = "forward") -> float:
    """
    Extract subthreshold swing (SS).
    
    SS = dVg / d(log10(Id))
    
    Args:
        df: DataFrame with simulation results
        Vg_col: Gate voltage column name
        Id_col: Drain current column name
        direction: Sweep direction
        
    Returns:
        Subthreshold swing (mV/decade)
    """
    # Filter by direction
    if "direction" in df.columns:
        subset = df[df["direction"] == direction].copy()
    else:
        subset = df.copy()
    
    if len(subset) < 3:
        return np.nan
    
    subset = subset.sort_values(by=Vg_col)
    
    Vg = subset[Vg_col].values
    Id = np.abs(subset[Id_col].values)
    
    # Calculate SS in subthreshold region (low current)
    # Find subthreshold region (Id < 1e-8 to 1e-6)
    mask = (Id > 1e-12) & (Id < 1e-6)
    
    if mask.sum() < 2:
        return np.nan
    
    Vg_sub = Vg[mask]
    log_Id = np.log10(Id[mask] + 1e-20)
    
    # Linear fit: Vg = SS * log10(Id) + b
    try:
        slope, _ = np.polyfit(log_Id, Vg_sub, 1)
        SS_mV = abs(slope) * 1000  # V/dec to mV/dec
        return SS_mV
    except:
        return np.nan


def extract_metrics(df: pd.DataFrame,
                    Vg_col: str = "Vg",
                    Id_col: str = "Id") -> FeFETMetrics:
    """
    Extract all FeFET performance metrics from simulation results.
    
    Args:
        df: DataFrame with Vg, Id, and optionally direction columns
        
    Returns:
        FeFETMetrics object with all extracted values
    """
    # Threshold voltages
    Vth_fwd = extract_threshold_voltage(df, Vg_col, Id_col, direction="forward")
    Vth_rev = extract_threshold_voltage(df, Vg_col, Id_col, direction="reverse")
    
    # Memory window
    MW = abs(Vth_fwd - Vth_rev) if not (np.isnan(Vth_fwd) or np.isnan(Vth_rev)) else 0.0
    
    # Ion / Ioff
    Id = np.abs(df[Id_col].values)
    Ion = float(np.max(Id))
    
    # Ioff at Vg ≈ 0
    Vg = df[Vg_col].values
    idx_zero = np.argmin(np.abs(Vg))
    Ioff = float(Id[idx_zero])
    
    if Ioff > 0:
        ratio = Ion / Ioff
    else:
        ratio = np.inf
    
    # Subthreshold swing
    SS_fwd = extract_subthreshold_swing(df, direction="forward")
    SS_rev = extract_subthreshold_swing(df, direction="reverse")
    SS_avg = np.nanmean([SS_fwd, SS_rev])
    
    return FeFETMetrics(
        Vth_forward=Vth_fwd,
        Vth_reverse=Vth_rev,
        memory_window=MW,
        Ion=Ion,
        Ioff=Ioff,
        Ion_Ioff_ratio=ratio,
        SS_forward=SS_fwd if not np.isnan(SS_fwd) else 0,
        SS_reverse=SS_rev if not np.isnan(SS_rev) else 0,
        SS_average=SS_avg if not np.isnan(SS_avg) else 0
    )


def calculate_polarization_from_cv(C: np.ndarray, V: np.ndarray, 
                                   area: float) -> np.ndarray:
    """
    Calculate polarization from C-V data by integration.
    
    P = ∫ C dV / A
    
    Args:
        C: Capacitance array (F)
        V: Voltage array (V)
        area: Device area (m²)
        
    Returns:
        Polarization array (C/m²)
    """
    # Integrate C-V
    Q = np.cumsum(C * np.gradient(V))  # Charge
    P = Q / area
    return P


def compare_with_target(metrics: FeFETMetrics, 
                        target_MW: float = 2.5) -> Dict[str, float]:
    """
    Compare extracted metrics with target values.
    
    Args:
        metrics: Extracted FeFET metrics
        target_MW: Target memory window (V)
        
    Returns:
        Dictionary with comparison results
    """
    return {
        "MW_achieved": metrics.memory_window,
        "MW_target": target_MW,
        "MW_ratio": metrics.memory_window / target_MW if target_MW > 0 else 0,
        "meets_target": metrics.memory_window >= target_MW,
        "Ion_Ioff_log": np.log10(metrics.Ion_Ioff_ratio) if metrics.Ion_Ioff_ratio > 0 else 0,
        "SS_meets_ideal": metrics.SS_average < 100,  # < 100 mV/dec is good
    }


def generate_summary_report(df: pd.DataFrame, metrics: FeFETMetrics) -> str:
    """
    Generate text summary of simulation results.

    Args:
        df: Simulation results DataFrame
        metrics: Extracted metrics

    Returns:
        Formatted summary string
    """
    report = f"""
================================================================================
                    MIFIS FeFET SIMULATION SUMMARY
================================================================================

THRESHOLD VOLTAGES:
  Forward sweep Vth:    {metrics.Vth_forward:+.3f} V
  Reverse sweep Vth:    {metrics.Vth_reverse:+.3f} V

MEMORY PERFORMANCE:
  Memory Window (MW):   {metrics.memory_window:.3f} V

CURRENT CHARACTERISTICS:
  Ion:                  {metrics.Ion:.2e} A
  Ioff:                 {metrics.Ioff:.2e} A
  Ion/Ioff Ratio:       {metrics.Ion_Ioff_ratio:.2e}

SUBTHRESHOLD SWING:
  SS (forward):         {metrics.SS_forward:.1f} mV/dec
  SS (reverse):         {metrics.SS_reverse:.1f} mV/dec
  SS (average):         {metrics.SS_average:.1f} mV/dec

DATA POINTS:
  Total samples:        {len(df)}
  Voltage range:        {df['Vg'].min():.1f} V to {df['Vg'].max():.1f} V

================================================================================
"""
    return report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_memory_window(df: pd.DataFrame,
                          Vg_col: str = "Vg",
                          Id_col: str = "Id",
                          I_threshold: float = 1e-7,
                          use_polarization: bool = True) -> float:
    """
    Extract memory window from simulation results.
    
    Uses polarization-based calculation (matching fefet_simulation_base) 
    if Polarization column exists, otherwise falls back to current threshold method.

    Args:
        df: DataFrame with Vg, Id, and optionally Polarization columns
        Vg_col: Gate voltage column name
        Id_col: Drain current column name
        I_threshold: Current threshold for Vth extraction
        use_polarization: If True, use polarization-based MW calculation

    Returns:
        Memory window (V)
    """
    # Try polarization-based calculation first (preferred, matches fefet_simulation_base)
    if use_polarization and 'Polarization' in df.columns:
        return extract_memory_window_from_polarization(df, Vg_col)
    
    # Fallback to threshold voltage method
    Vth_fwd = extract_threshold_voltage(df, Vg_col, Id_col, I_threshold, "forward")
    Vth_rev = extract_threshold_voltage(df, Vg_col, Id_col, I_threshold, "reverse")

    if np.isnan(Vth_fwd) or np.isnan(Vth_rev):
        # If Vth extraction fails, try polarization anyway
        if 'Polarization' in df.columns:
            return extract_memory_window_from_polarization(df, Vg_col)
        return 0.0

    return abs(Vth_fwd - Vth_rev)


def extract_memory_window_from_polarization(df: pd.DataFrame,
                                             Vg_col: str = "Vg",
                                             t_fe: float = 13.8e-9,
                                             eps_r: float = 30.0) -> float:
    """
    Calculate memory window from polarization hysteresis data.
    
    Uses the fefet_simulation_base formula:
    MW = ΔP * t_FE / (ε₀ * ε_FE) / 10
    
    Args:
        df: DataFrame with Vg and Polarization columns
        Vg_col: Gate voltage column name
        t_fe: Ferroelectric layer thickness (m)
        eps_r: Relative permittivity of HZO
    
    Returns:
        Memory window (V)
    """
    if 'Polarization' not in df.columns:
        return 0.0
    
    P_fe = df['Polarization'].values  # In µC/cm²
    V_gate = df[Vg_col].values if Vg_col in df.columns else np.arange(len(P_fe))
    
    n_points = len(V_gate)
    if n_points < 4:
        return 0.0
    
    quarter = n_points // 4
    
    # Find P+ state: After positive saturation, when V crosses 0 (around quarter point)
    try:
        idx_p_plus = quarter + np.argmin(np.abs(V_gate[quarter:2*quarter]))
        P_positive = P_fe[idx_p_plus]
    except:
        P_positive = np.max(P_fe)
    
    # Find P- state: After negative saturation, when V crosses 0 (around 3*quarter point)
    try:
        idx_p_minus = np.argmin(np.abs(V_gate[3*quarter:]))
        P_negative = P_fe[3*quarter + idx_p_minus]
    except:
        P_negative = np.min(P_fe)
    
    # If signs are wrong, just use max/min
    if P_positive < 0 or P_negative > 0:
        P_positive = np.max(P_fe)
        P_negative = np.min(P_fe)
    
    # Calculate MW using fefet_simulation_base formula
    # P is in µC/cm² = 1e-6 C/cm² = 1e-6 * 1e4 C/m² = 1e-2 C/m²
    Delta_P_uC_cm2 = abs(P_positive - P_negative)  # µC/cm²
    Delta_P_si = Delta_P_uC_cm2 * 1e-2  # Convert µC/cm² to C/m²
    
    eps_0 = 8.854e-12  # F/m
    
    # Basic formula: MW = ΔP * t / (ε₀ * ε_r)
    MW_basic = Delta_P_si * t_fe / (eps_0 * eps_r)
    
    # Correction factor for series capacitance (empirical, from fefet_simulation_base)
    MW = MW_basic / 10.0
    
    return MW


def extract_vth(df: pd.DataFrame,
                Vg_col: str = "Vg",
                Id_col: str = "Id",
                I_threshold: float = 1e-7) -> Tuple[float, float]:
    """
    Extract threshold voltages for both forward and reverse sweeps.

    Args:
        df: DataFrame with simulation results
        Vg_col: Gate voltage column name
        Id_col: Drain current column name
        I_threshold: Current threshold

    Returns:
        Tuple of (Vth_forward, Vth_reverse)
    """
    Vth_fwd = extract_threshold_voltage(df, Vg_col, Id_col, I_threshold, "forward")
    Vth_rev = extract_threshold_voltage(df, Vg_col, Id_col, I_threshold, "reverse")

    return (Vth_fwd, Vth_rev)
