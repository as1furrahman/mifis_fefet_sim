"""
MIFIS FeFET Simulation Framework
================================
Visualization and plotting utilities.

Author: Thesis Project
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path


# =============================================================================
# PLOT STYLE CONFIGURATION
# =============================================================================

def setup_thesis_style():
    """Configure matplotlib for thesis-quality figures."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# =============================================================================
# TRANSFER CHARACTERISTIC PLOTS
# =============================================================================

def plot_id_vg(df: pd.DataFrame,
               label: str = "MIFIS FeFET",
               title: str = "Transfer Characteristic",
               save_path: Optional[str] = None,
               show_hysteresis: bool = True) -> plt.Figure:
    """
    Plot Id-Vg transfer characteristic with hysteresis.
    
    Args:
        df: DataFrame with Vg and Id columns
        label: Legend label
        title: Plot title
        save_path: Path to save figure (if provided)
        show_hysteresis: Whether to show forward/reverse separately
        
    Returns:
        matplotlib Figure object
    """
    setup_thesis_style()
    fig, ax = plt.subplots()
    
    if show_hysteresis and "direction" in df.columns:
        # Separate forward and reverse
        fwd = df[df["direction"] == "forward"]
        rev = df[df["direction"] == "reverse"]
        
        ax.semilogy(fwd["Vg"], np.abs(fwd["Id"]), 'b-', 
                    label=f"{label} (forward)", linewidth=2)
        ax.semilogy(rev["Vg"], np.abs(rev["Id"]), 'r--', 
                    label=f"{label} (reverse)", linewidth=2)
    else:
        ax.semilogy(df["Vg"], np.abs(df["Id"]), 'b-', label=label, linewidth=2)
    
    ax.set_xlabel("Gate Voltage $V_{gs}$ (V)")
    ax.set_ylabel("Drain Current $|I_d|$ (A)")
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Add memory window annotation if hysteresis visible
    if show_hysteresis and "direction" in df.columns:
        # Find Vth difference
        I_th = 1e-7
        try:
            Vth_fwd = np.interp(np.log10(I_th), 
                               np.log10(np.abs(fwd["Id"].values) + 1e-20), 
                               fwd["Vg"].values)
            Vth_rev = np.interp(np.log10(I_th),
                               np.log10(np.abs(rev["Id"].values) + 1e-20),
                               rev["Vg"].values)
            MW = abs(Vth_fwd - Vth_rev)
            ax.annotate(f"MW = {MW:.2f} V", xy=(0.05, 0.95), 
                       xycoords='axes fraction', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            pass
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_multiple_id_vg(dfs: List[pd.DataFrame],
                        labels: List[str],
                        title: str = "Transfer Characteristic Comparison",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot multiple Id-Vg curves for comparison.
    
    Args:
        dfs: List of DataFrames
        labels: List of labels for each curve
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots()
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(dfs)))
    
    for df, label, color in zip(dfs, labels, colors):
        ax.semilogy(df["Vg"], np.abs(df["Id"]), color=color, 
                    label=label, linewidth=2)
    
    ax.set_xlabel("Gate Voltage $V_{gs}$ (V)")
    ax.set_ylabel("Drain Current $|I_d|$ (A)")
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# HYSTERESIS LOOP PLOTS
# =============================================================================

def plot_polarization_loop(E: np.ndarray, P: np.ndarray,
                           title: str = "Ferroelectric P-E Hysteresis Loop (HZO)",
                           save_path: Optional[str] = None,
                           Ec_target: float = None,
                           Pr_target: float = None) -> plt.Figure:
    """
    Plot P-E ferroelectric hysteresis loop with Pr and Ec annotations.

    Args:
        E: Electric field array (MV/cm)
        P: Polarization array (μC/cm²)
        title: Plot title
        save_path: Path to save figure
        Ec_target: Expected coercive field (MV/cm) for comparison
        Pr_target: Expected remnant polarization (μC/cm²) for comparison

    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Main hysteresis loop
    ax.plot(E, P, 'b-', linewidth=2.5, label='P-E Loop')

    # Axes through origin
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8)

    # Extract Pr and Ec from the data
    if len(P) > 1 and len(E) > 1:
        # Find Pr: Polarization values where E is closest to 0
        # For a proper loop, there should be two crossings (upper and lower branch)
        E_near_zero_mask = np.abs(E) < 0.1 * np.max(np.abs(E))
        if np.any(E_near_zero_mask):
            P_at_zero = P[E_near_zero_mask]
            Pr_pos = np.max(P_at_zero)  # Upper branch
            Pr_neg = np.min(P_at_zero)  # Lower branch
        else:
            # Fallback: find P at minimum |E|
            idx_zero = np.argmin(np.abs(E))
            Pr_pos = abs(P[idx_zero])
            Pr_neg = -Pr_pos

        # Find Ec: E values where P crosses zero
        # Look for sign changes in P
        sign_changes = np.where(np.diff(np.sign(P)))[0]
        Ec_values = []
        for idx in sign_changes:
            if idx < len(E) - 1:
                # Linear interpolation to find exact E where P=0
                E1, E2 = E[idx], E[idx+1]
                P1, P2 = P[idx], P[idx+1]
                if P2 != P1:
                    E_cross = E1 - P1 * (E2 - E1) / (P2 - P1)
                    Ec_values.append(E_cross)

        if len(Ec_values) >= 2:
            Ec_pos = max(Ec_values)  # Positive coercive field
            Ec_neg = min(Ec_values)  # Negative coercive field
        elif len(Ec_values) == 1:
            Ec_pos = abs(Ec_values[0])
            Ec_neg = -Ec_pos
        else:
            # Fallback estimate
            Ec_pos = 0.1  # Default HZO value
            Ec_neg = -0.1

        # Draw Pr markers (horizontal dashed lines at E=0)
        ax.axhline(y=Pr_pos, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=Pr_neg, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

        # Draw Ec markers (vertical dashed lines at P=0)
        ax.axvline(x=Ec_pos, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=Ec_neg, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Mark points on the loop
        ax.plot(0, Pr_pos, 'go', markersize=10, markeredgecolor='darkgreen',
                markeredgewidth=2, label=f'+$P_r$ = {Pr_pos:.1f} μC/cm²')
        ax.plot(0, Pr_neg, 'go', markersize=10, markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(Ec_pos, 0, 'r^', markersize=10, markeredgecolor='darkred',
                markeredgewidth=2, label=f'+$E_c$ = {Ec_pos:.2f} MV/cm')
        ax.plot(Ec_neg, 0, 'r^', markersize=10, markeredgecolor='darkred', markeredgewidth=2)

        # Annotations with arrows
        ax.annotate(f'$+P_r$ = {Pr_pos:.1f}', xy=(0, Pr_pos), xytext=(0.15*max(E), Pr_pos*1.1),
                   fontsize=11, color='darkgreen', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        ax.annotate(f'$-P_r$ = {Pr_neg:.1f}', xy=(0, Pr_neg), xytext=(0.15*max(E), Pr_neg*1.1),
                   fontsize=11, color='darkgreen', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        ax.annotate(f'$+E_c$ = {Ec_pos:.2f}', xy=(Ec_pos, 0), xytext=(Ec_pos*1.3, 0.2*max(P)),
                   fontsize=11, color='darkred', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        ax.annotate(f'$-E_c$ = {Ec_neg:.2f}', xy=(Ec_neg, 0), xytext=(Ec_neg*1.3, -0.2*max(P)),
                   fontsize=11, color='darkred', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # Add summary box
        summary_text = (f"Remnant: $P_r$ = ±{abs(Pr_pos):.1f} μC/cm²\n"
                       f"Coercive: $E_c$ = ±{abs(Ec_pos):.2f} MV/cm")
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

    ax.set_xlabel("Electric Field $E$ (MV/cm)", fontsize=12)
    ax.set_ylabel("Polarization $P$ (μC/cm²)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig


# =============================================================================
# PAPER-STYLE FEFET PLOTS (Based on literature analysis)
# =============================================================================

def plot_cv_hysteresis(Vg: np.ndarray, C_fwd: np.ndarray, C_rev: np.ndarray,
                        title: str = "C-V Hysteresis",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Capacitance-Voltage hysteresis (standard FeFET paper plot).
    
    Based on: Impact_of_Top_SiO2_Interlayer_Thickness papers.
    
    Args:
        Vg: Gate voltage array
        C_fwd: Capacitance (forward sweep), normalized to Cox or absolute
        C_rev: Capacitance (reverse sweep)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots()
    
    # Plot forward and reverse sweeps
    ax.plot(Vg, C_fwd, 'b-', linewidth=2, label='Forward')
    ax.plot(Vg, C_rev, 'r--', linewidth=2, label='Reverse')
    
    # Add arrows to show sweep direction
    mid_idx = len(Vg) // 2
    ax.annotate('', xy=(Vg[mid_idx+5], C_fwd[mid_idx+5]),
               xytext=(Vg[mid_idx], C_fwd[mid_idx]),
               arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.annotate('', xy=(Vg[mid_idx-5], C_rev[mid_idx-5]),
               xytext=(Vg[mid_idx], C_rev[mid_idx]),
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # Find flatband voltage shifts (memory window)
    try:
        C_mid = (np.max(C_fwd) + np.min(C_fwd)) / 2
        Vfb_fwd = np.interp(C_mid, C_fwd, Vg)
        Vfb_rev = np.interp(C_mid, C_rev[::-1], Vg[::-1])
        MW = abs(Vfb_rev - Vfb_fwd)
        
        # Annotate memory window
        ax.axhline(y=C_mid, color='gray', linestyle=':', alpha=0.7)
        ax.annotate(f'MW = {MW:.2f} V', xy=(0.6, 0.15),
                   xycoords='axes fraction', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    except:
        pass
    
    ax.set_xlabel("Gate Voltage $V_g$ (V)")
    ax.set_ylabel("Capacitance (F/μm²)")
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_id_vg_linear_with_gm(df: pd.DataFrame,
                               title: str = "Linear Transfer Characteristic",
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot linear Id-Vg with transconductance (gm) - standard paper figure.
    
    Shows: Linear Id vs Vg + gm curve for Vth extraction.
    
    Args:
        df: DataFrame with Vg and Id columns
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax1 = plt.subplots()
    
    Vg = df["Vg"].values
    Id = df["Id"].values
    
    # Calculate transconductance gm = dId/dVg
    gm = np.gradient(Id, Vg)
    
    # Plot Id (linear scale)
    color1 = 'blue'
    ax1.plot(Vg, Id * 1e6, color=color1, linewidth=2, label='$I_d$')  # Convert to μA
    ax1.set_xlabel("Gate Voltage $V_g$ (V)")
    ax1.set_ylabel("Drain Current $I_d$ (μA)", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot gm on secondary axis
    ax2 = ax1.twinx()
    color2 = 'red'
    ax2.plot(Vg, gm * 1e6, color=color2, linewidth=2, linestyle='--', label='$g_m$')
    ax2.set_ylabel("Transconductance $g_m$ (μS)", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Find Vth from gm peak
    gm_max_idx = np.argmax(gm)
    Vth_approx = Vg[gm_max_idx]
    
    # Mark Vth
    ax1.axvline(x=Vth_approx, color='green', linestyle=':', alpha=0.7)
    ax1.annotate(f'$V_{{th}}$ ≈ {Vth_approx:.2f} V', 
                xy=(Vth_approx, Id[gm_max_idx] * 1e6),
                xytext=(Vth_approx + 0.5, Id[gm_max_idx] * 1e6 * 1.2),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))
    
    ax1.set_title(title)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_subthreshold_swing(df: pd.DataFrame,
                             title: str = "Subthreshold Swing Extraction",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Id-Vg with SS extraction (common in FeFET papers).
    
    Based on: Scaling of the Ferroelectric FET paper analysis.
    
    Args:
        df: DataFrame with Vg and Id columns
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure with SS annotation
    """
    setup_thesis_style()
    fig, ax = plt.subplots()
    
    Vg = df["Vg"].values
    Id = np.abs(df["Id"].values)
    
    # Log scale plot
    ax.semilogy(Vg, Id, 'b-', linewidth=2)
    
    # Calculate SS in subthreshold region
    log_Id = np.log10(Id + 1e-20)
    dlogId_dVg = np.gradient(log_Id, Vg)
    
    # Find subthreshold region (steepest slope)
    # Filter out noise near off-state
    valid_mask = Id > 1e-12
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        
        # Find minimum SS (steepest slope = maximum dlogId/dVg)
        ss_region = dlogId_dVg[valid_indices]
        max_slope_idx = valid_indices[np.argmax(ss_region)]
        
        # SS = 1 / (dlog10(Id)/dVg) in mV/decade
        if dlogId_dVg[max_slope_idx] > 0:
            SS = 1000.0 / dlogId_dVg[max_slope_idx]  # mV/decade
        else:
            SS = np.inf
        
        # Draw SS extraction line
        Vg_ss_start = Vg[max_slope_idx] - 0.3
        Vg_ss_end = Vg[max_slope_idx] + 0.3
        
        # Draw tangent line
        slope = dlogId_dVg[max_slope_idx]
        intercept = log_Id[max_slope_idx] - slope * Vg[max_slope_idx]
        Id_line = 10 ** (slope * np.array([Vg_ss_start, Vg_ss_end]) + intercept)
        ax.plot([Vg_ss_start, Vg_ss_end], Id_line, 'r-', linewidth=3, 
                label=f'SS = {SS:.0f} mV/dec')
        
        # Annotate SS value
        ax.annotate(f'SS = {SS:.0f} mV/dec', 
                   xy=(Vg[max_slope_idx], Id[max_slope_idx]),
                   xytext=(Vg[max_slope_idx] + 0.5, Id[max_slope_idx] * 10),
                   fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel("Gate Voltage $V_g$ (V)")
    ax.set_ylabel("Drain Current $|I_d|$ (A)")
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_band_diagram(geometry: dict,
                       polarization_state: str = "positive",
                       title: str = "MIFIS Band Diagram",
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot energy band diagram through MIFIS stack.
    
    Based on: FeFET_review_arxiv and Hafnium oxide-based FeFET papers.
    
    Args:
        geometry: Device layer thicknesses
        polarization_state: "positive", "negative", or "zero"
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure showing band structure
    """
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Layer thicknesses (nm)
    t_gate = geometry.get('t_gate', 50)
    t_top_il = geometry.get('t_top_il', 4.0)
    t_fe = geometry.get('t_fe', 13.8)
    t_bottom_il = geometry.get('t_bottom_il', 0.7)
    t_channel = geometry.get('t_channel', 20)
    
    # Work functions and band offsets (approximate values)
    # TiN work function: ~4.5 eV
    # SiO2 conduction band offset from Si: ~3.1 eV
    # HZO conduction band offset from Si: ~1.5 eV
    # Si bandgap: 1.12 eV
    
    E_vacuum = 0
    phi_TiN = 4.5  # TiN work function
    chi_Si = 4.05  # Si electron affinity
    Eg_Si = 1.12   # Si bandgap
    dEc_SiO2 = 3.1  # SiO2 conduction band offset
    dEc_HZO = 1.5   # HZO conduction band offset
    
    # X positions for each layer
    x_gate = [0, t_gate]
    x_top_il = [t_gate, t_gate + t_top_il]
    x_fe = [t_gate + t_top_il, t_gate + t_top_il + t_fe]
    x_bottom_il = [t_gate + t_top_il + t_fe, t_gate + t_top_il + t_fe + t_bottom_il]
    x_channel = [t_gate + t_top_il + t_fe + t_bottom_il, 
                 t_gate + t_top_il + t_fe + t_bottom_il + t_channel]
    
    # Polarization effect on band bending
    if polarization_state == "positive":
        P_shift = 0.5  # Band bend downward in channel (accumulation)
    elif polarization_state == "negative":
        P_shift = -0.5  # Band bend upward in channel (depletion)
    else:
        P_shift = 0
    
    # Draw bands
    # Metal (flat)
    ax.fill_between([x_gate[0], x_gate[1]], -phi_TiN, -phi_TiN - 0.5,
                    color='#4169E1', alpha=0.3, label='TiN Gate')
    ax.axhline(y=-phi_TiN, xmin=x_gate[0]/x_channel[1], xmax=x_gate[1]/x_channel[1],
               color='#4169E1', linewidth=3)
    
    # Top IL (SiO2) - conduction band
    x_il1 = np.linspace(x_top_il[0], x_top_il[1], 50)
    Ec_il1 = -chi_Si + dEc_SiO2
    ax.fill_between(x_il1, Ec_il1, Ec_il1 + 2, color='#FFD700', alpha=0.3)
    ax.plot(x_il1, np.ones_like(x_il1) * Ec_il1, 'k-', linewidth=2)
    
    # Ferroelectric (HZO)
    x_fe_arr = np.linspace(x_fe[0], x_fe[1], 50)
    Ec_fe = -chi_Si + dEc_HZO
    ax.fill_between(x_fe_arr, Ec_fe, Ec_fe + 1.5, color='#FF4444', alpha=0.3, label='HZO')
    ax.plot(x_fe_arr, np.ones_like(x_fe_arr) * Ec_fe, 'k-', linewidth=2)
    
    # Bottom IL (SiO2)
    x_il2 = np.linspace(x_bottom_il[0], x_bottom_il[1], 50)
    ax.fill_between(x_il2, Ec_il1, Ec_il1 + 2, color='#FFD700', alpha=0.3)
    ax.plot(x_il2, np.ones_like(x_il2) * Ec_il1, 'k-', linewidth=2)
    
    # Si channel - with band bending
    x_si = np.linspace(x_channel[0], x_channel[1], 100)
    # Simple exponential band bending
    decay_length = 5  # nm
    bend = P_shift * np.exp(-(x_si - x_channel[0]) / decay_length)
    
    Ec_Si = -chi_Si + bend
    Ev_Si = Ec_Si - Eg_Si
    Ei_Si = (Ec_Si + Ev_Si) / 2  # Intrinsic level
    
    ax.fill_between(x_si, Ec_Si, Ev_Si, color='#90EE90', alpha=0.3, label='Si Channel')
    ax.plot(x_si, Ec_Si, 'b-', linewidth=2, label='$E_c$')
    ax.plot(x_si, Ev_Si, 'r-', linewidth=2, label='$E_v$')
    ax.plot(x_si, Ei_Si, 'k--', linewidth=1, label='$E_i$')
    
    # Fermi level (dashed line through stack)
    Ef = -phi_TiN
    ax.axhline(y=Ef, color='green', linestyle=':', linewidth=2, label='$E_F$')
    
    # Polarization arrow
    arrow_x = (x_fe[0] + x_fe[1]) / 2
    if polarization_state == "positive":
        ax.annotate('', xy=(arrow_x, Ec_fe - 0.5), xytext=(arrow_x, Ec_fe + 0.5),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=3))
        ax.text(arrow_x + 2, Ec_fe, '+P', fontsize=14, color='purple', fontweight='bold')
    elif polarization_state == "negative":
        ax.annotate('', xy=(arrow_x, Ec_fe + 0.5), xytext=(arrow_x, Ec_fe - 0.5),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=3))
        ax.text(arrow_x + 2, Ec_fe, '-P', fontsize=14, color='purple', fontweight='bold')
    
    # Labels
    ax.set_xlabel("Position (nm)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, x_channel[1])
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add layer labels at bottom
    layer_labels = [
        ('TiN', (x_gate[0] + x_gate[1])/2),
        ('SiO₂', (x_top_il[0] + x_top_il[1])/2),
        ('HZO', (x_fe[0] + x_fe[1])/2),
        ('SiO₂', (x_bottom_il[0] + x_bottom_il[1])/2),
        ('Si', (x_channel[0] + x_channel[1])/2),
    ]
    for label, x_pos in layer_labels:
        ax.text(x_pos, ax.get_ylim()[0] + 0.3, label, ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_retention(time_hours: np.ndarray, MW_values: np.ndarray,
                    MW_initial: float = None,
                    title: str = "Retention Characteristics",
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot retention characteristics (MW vs time).
    
    Standard plot in FeFET reliability papers.
    
    Args:
        time_hours: Time array in hours
        MW_values: Memory window at each time point
        MW_initial: Initial memory window for normalization
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots()
    
    if MW_initial is None:
        MW_initial = MW_values[0]
    
    # Normalize to initial MW
    MW_normalized = MW_values / MW_initial * 100
    
    # Log scale for time
    ax.semilogx(time_hours, MW_normalized, 'bo-', markersize=8, linewidth=2)
    
    # 10 year target line
    ten_years_hours = 10 * 365 * 24
    ax.axvline(x=ten_years_hours, color='red', linestyle='--', alpha=0.7)
    ax.text(ten_years_hours * 1.1, 50, '10 years', fontsize=10, color='red')
    
    # Retention limit (typically 50% or 70%)
    ax.axhline(y=70, color='green', linestyle=':', alpha=0.7)
    ax.text(time_hours[0] * 1.5, 72, '70% retention limit', fontsize=9, color='green')
    
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Normalized Memory Window (%)")
    ax.set_title(title)
    ax.set_ylim(0, 110)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Add annotation for projected 10-year retention
    if time_hours[-1] < ten_years_hours:
        # Extrapolate (simple log fit)
        log_time = np.log10(time_hours)
        coeffs = np.polyfit(log_time, MW_normalized, 1)
        MW_10yr = coeffs[0] * np.log10(ten_years_hours) + coeffs[1]
        ax.annotate(f'Projected 10yr: {MW_10yr:.0f}%', 
                   xy=(ten_years_hours, MW_10yr),
                   xytext=(ten_years_hours/10, MW_10yr - 10),
                   fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_endurance(cycles: np.ndarray, MW_values: np.ndarray,
                    target_cycles: float = 1e9,
                    title: str = "Endurance Characteristics",
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot endurance characteristics (MW vs programming cycles).
    
    Standard plot in FeFET reliability papers.
    
    Args:
        cycles: Number of program/erase cycles
        MW_values: Memory window at each cycle count
        target_cycles: Target endurance (default 10^9)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots()
    
    # Log scale for cycles
    ax.semilogx(cycles, MW_values, 'bs-', markersize=8, linewidth=2, label='MIFIS FeFET')
    
    # Initial MW for reference
    MW_initial = MW_values[0]
    ax.axhline(y=MW_initial, color='gray', linestyle='--', alpha=0.5)
    
    # Target endurance line
    ax.axvline(x=target_cycles, color='green', linestyle='--', alpha=0.7)
    ax.text(target_cycles * 0.5, MW_initial * 0.3, f'{target_cycles:.0e}\ncycles', 
            fontsize=10, color='green', ha='center')
    
    # Failure threshold (typically 50% of initial)
    failure_threshold = MW_initial * 0.5
    ax.axhline(y=failure_threshold, color='red', linestyle=':', alpha=0.7)
    ax.text(cycles[0] * 2, failure_threshold * 0.9, '50% failure threshold', 
            fontsize=9, color='red')
    
    ax.set_xlabel("Number of Cycles")
    ax.set_ylabel("Memory Window (V)")
    ax.set_title(title)
    ax.set_ylim(0, MW_initial * 1.2)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_mw_vs_il_thickness(il_thickness: np.ndarray, MW_values: np.ndarray,
                             il_material: str = "SiO₂",
                             optimal_thickness: float = None,
                             title: str = "Memory Window vs IL Thickness",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Memory Window vs Interlayer thickness (key paper figure).
    
    Based on: Impact_of_Top_SiO2_Interlayer_Thickness papers.
    
    Args:
        il_thickness: IL thickness array in nm
        MW_values: Memory window values
        il_material: IL material name
        optimal_thickness: Mark optimal thickness point
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots()
    
    ax.plot(il_thickness, MW_values, 'bo-', markersize=10, linewidth=2,
            markerfacecolor='white', markeredgewidth=2)
    
    # Find and mark optimal point
    if optimal_thickness is None:
        optimal_idx = np.argmax(MW_values)
        optimal_thickness = il_thickness[optimal_idx]
        optimal_MW = MW_values[optimal_idx]
    else:
        optimal_MW = np.interp(optimal_thickness, il_thickness, MW_values)
    
    ax.scatter([optimal_thickness], [optimal_MW], s=200, c='red', marker='*', 
               zorder=5, label=f'Optimal: {optimal_thickness:.1f} nm')
    
    # Add regions annotation
    ax.axvspan(0, 1.5, alpha=0.1, color='red')
    ax.axvspan(1.5, 3.0, alpha=0.1, color='green')
    ax.axvspan(3.0, max(il_thickness), alpha=0.1, color='orange')
    
    ax.text(0.75, max(MW_values) * 0.3, 'High\nLeakage', ha='center', fontsize=9, color='red')
    ax.text(2.25, max(MW_values) * 0.3, 'Optimal', ha='center', fontsize=9, color='green')
    ax.text(max(il_thickness) * 0.8, max(MW_values) * 0.3, 'Reduced\nCoupling', 
            ha='center', fontsize=9, color='orange')
    
    ax.set_xlabel(f"{il_material} Thickness (nm)")
    ax.set_ylabel("Memory Window (V)")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dual_sweep_comparison(df: pd.DataFrame,
                                title: str = "Program/Erase State Comparison",
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot programmed and erased states on same axes (standard memory paper plot).
    
    Args:
        df: DataFrame with Vg, Id, and direction columns
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots()
    
    # Separate forward (LVT/erased) and reverse (HVT/programmed) states
    if "direction" in df.columns:
        fwd = df[df["direction"] == "forward"]
        rev = df[df["direction"] == "reverse"]
    else:
        # Split in half
        mid = len(df) // 2
        fwd = df.iloc[:mid]
        rev = df.iloc[mid:]
    
    # Log scale plot
    ax.semilogy(fwd["Vg"], np.abs(fwd["Id"]), 'b-', linewidth=2.5, label='LVT (Erased)')
    ax.semilogy(rev["Vg"], np.abs(rev["Id"]), 'r-', linewidth=2.5, label='HVT (Programmed)')
    
    # Find Vth for each state
    I_read = 1e-7  # Read current threshold
    try:
        Vth_LVT = np.interp(np.log10(I_read), np.log10(np.abs(fwd["Id"].values) + 1e-20), 
                           fwd["Vg"].values)
        Vth_HVT = np.interp(np.log10(I_read), np.log10(np.abs(rev["Id"].values[::-1]) + 1e-20), 
                           rev["Vg"].values[::-1])
        MW = abs(Vth_HVT - Vth_LVT)
        
        # Mark Vth points
        ax.axvline(x=Vth_LVT, color='blue', linestyle=':', alpha=0.7)
        ax.axvline(x=Vth_HVT, color='red', linestyle=':', alpha=0.7)
        
        # Memory window annotation with arrow
        ax.annotate('', xy=(Vth_HVT, I_read), xytext=(Vth_LVT, I_read),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text((Vth_LVT + Vth_HVT)/2, I_read * 3, f'MW = {MW:.2f} V',
               ha='center', fontsize=12, fontweight='bold', color='green')
        
        # State labels
        ax.text(Vth_LVT - 0.3, 1e-6, '$V_{th}^{LVT}$', fontsize=11, color='blue')
        ax.text(Vth_HVT + 0.1, 1e-6, '$V_{th}^{HVT}$', fontsize=11, color='red')
    except:
        pass
    
    # Read voltage line
    ax.axhline(y=I_read, color='gray', linestyle='--', alpha=0.5)
    ax.text(ax.get_xlim()[0] + 0.2, I_read * 2, '$I_{read}$', fontsize=10, color='gray')
    
    ax.set_xlabel("Gate Voltage $V_g$ (V)")
    ax.set_ylabel("Drain Current $|I_d|$ (A)")
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# MEMORY WINDOW TREND PLOTS
# =============================================================================

def plot_memory_window_vs_parameter(param_values: np.ndarray,
                                     MW_values: np.ndarray,
                                     param_name: str = "Parameter",
                                     param_unit: str = "",
                                     title: str = "Memory Window Trend",
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot memory window as function of a parameter (thickness, etc.).
    
    Args:
        param_values: Array of parameter values
        MW_values: Array of memory window values
        param_name: Name of the parameter
        param_unit: Unit of the parameter
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots()
    
    ax.plot(param_values, MW_values, 'bo-', markersize=10, linewidth=2,
            markerfacecolor='white', markeredgewidth=2)
    
    unit_str = f" ({param_unit})" if param_unit else ""
    ax.set_xlabel(f"{param_name}{unit_str}")
    ax.set_ylabel("Memory Window (V)")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Highlight optimal point
    max_idx = np.argmax(MW_values)
    ax.scatter([param_values[max_idx]], [MW_values[max_idx]], 
               s=200, c='red', marker='*', zorder=5,
               label=f"Optimal: {param_values[max_idx]:.1f}")
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# ARCHITECTURE COMPARISON
# =============================================================================

def plot_architecture_comparison(architectures: List[str],
                                  MW_values: List[float],
                                  title: str = "Architecture Comparison",
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Bar chart comparing memory window across architectures.
    
    Args:
        architectures: List of architecture names
        MW_values: Corresponding memory window values
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(architectures))
    bars = ax.bar(x, MW_values, color='steelblue', edgecolor='navy', linewidth=2)
    
    # Color best performer differently
    best_idx = np.argmax(MW_values)
    bars[best_idx].set_color('darkgreen')
    
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Memory Window (V)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars, MW_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f} V', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylim(0, max(MW_values) * 1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# DEVICE STACK VISUALIZATION - 1D, 2D, 3D Structures
# =============================================================================

def plot_mifis_stack(geometry: dict,
                     title: str = "MIFIS Stack Structure",
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the MIFIS layer stack (basic 1D view).
    
    Args:
        geometry: Dictionary with layer thicknesses
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Layer definitions (bottom to top)
    layers = [
        ("Si Channel", geometry.get('t_channel', 20), '#90EE90'),
        ("Bottom IL (SiO₂)", geometry.get('t_bottom_il', 0.7), '#FFD700'),
        ("HZO Ferroelectric", geometry.get('t_fe', 13.8), '#FF6B6B'),
        ("Top IL (SiO₂)", geometry.get('t_top_il', 4.0), '#FFD700'),
        ("TiN Gate", geometry.get('t_gate', 50), '#4169E1'),
    ]
    
    y_pos = 0
    for name, thickness, color in layers:
        rect = plt.Rectangle((0.1, y_pos), 0.8, thickness, 
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add label
        ax.text(0.5, y_pos + thickness/2, f"{name}\n({thickness:.1f} nm)",
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        y_pos += thickness
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_pos * 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add total height
    ax.text(0.95, y_pos/2, f"Total:\n{y_pos:.1f} nm",
            ha='left', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_1d_stack_with_results(geometry: dict,
                                results: pd.DataFrame = None,
                                title: str = "1D MIFIS Structure & Output",
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 1D MIFIS stack with simulation results (Id-Vg and potential profile).
    
    Args:
        geometry: Dictionary with layer thicknesses
        results: DataFrame with simulation results (Vg, Id columns)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    
    if results is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        axes = [ax, None, None]
    
    ax_stack = axes[0]
    
    # Layer definitions (bottom to top) with proper colors
    layers = [
        ("Si p-type\n(Channel)", geometry.get('t_channel', 20), '#90EE90', 'Semiconductor'),
        ("SiO₂\n(Bottom IL)", geometry.get('t_bottom_il', 0.7), '#FFA500', 'Dielectric'),
        ("HZO\n(Ferroelectric)", geometry.get('t_fe', 13.8), '#FF4444', 'Ferroelectric'),
        ("SiO₂\n(Top IL)", geometry.get('t_top_il', 4.0), '#FFA500', 'Dielectric'),
        ("TiN\n(Gate)", geometry.get('t_gate', 50), '#4169E1', 'Metal'),
    ]
    
    y_pos = 0
    for name, thickness, color, mat_type in layers:
        # Scale for visibility (thin layers are hard to see)
        display_thickness = max(thickness, 5)  # Minimum display thickness
        
        rect = plt.Rectangle((0.15, y_pos), 0.7, display_thickness, 
                              facecolor=color, edgecolor='black', linewidth=2,
                              alpha=0.8)
        ax_stack.add_patch(rect)
        
        # Add label
        label_text = f"{name}\n{thickness:.1f} nm"
        ax_stack.text(0.5, y_pos + display_thickness/2, label_text,
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        y_pos += display_thickness
    
    # Add contacts
    ax_stack.annotate('Gate Contact\n(V_g)', xy=(0.5, y_pos), xytext=(0.5, y_pos + 10),
                     ha='center', fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='black'))
    ax_stack.annotate('Substrate\n(Ground)', xy=(0.5, 0), xytext=(0.5, -15),
                     ha='center', fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='black'))
    
    ax_stack.set_xlim(0, 1)
    ax_stack.set_ylim(-20, y_pos + 20)
    ax_stack.axis('off')
    ax_stack.set_title("1D MIFIS Stack", fontsize=12, fontweight='bold')
    
    # Plot Id-Vg if results provided
    if results is not None and len(axes) > 1:
        ax_idvg = axes[1]
        
        if "direction" in results.columns:
            fwd = results[results["direction"] == "forward"]
            rev = results[results["direction"] == "reverse"]
            ax_idvg.semilogy(fwd["Vg"], np.abs(fwd["Id"]), 'b-', 
                            label="Forward", linewidth=2)
            ax_idvg.semilogy(rev["Vg"], np.abs(rev["Id"]), 'r--', 
                            label="Reverse", linewidth=2)
        else:
            ax_idvg.semilogy(results["Vg"], np.abs(results["Id"]), 'b-', linewidth=2)
        
        ax_idvg.set_xlabel("Gate Voltage (V)")
        ax_idvg.set_ylabel("|Id| (A)")
        ax_idvg.set_title("Transfer Characteristic", fontweight='bold')
        ax_idvg.legend()
        ax_idvg.grid(True, alpha=0.3)
        
        # Add metrics box
        ax_metrics = axes[2]
        ax_metrics.axis('off')
        
        # Calculate metrics
        Id_on = results["Id"].max()
        Id_off = results["Id"].min()
        on_off = Id_on / Id_off if Id_off > 0 else np.inf
        
        metrics_text = f"""
Simulation Output
─────────────────
Stack: MIFIS (1D)
FE: HZO {geometry.get('t_fe', 13.8):.1f}nm
IL: SiO₂ {geometry.get('t_top_il', 4.0):.1f}nm

Results:
  Ion:  {Id_on:.2e} A
  Ioff: {Id_off:.2e} A
  On/Off: {on_off:.2e}
  
Voltage Range:
  Vg: {results['Vg'].min():.1f}V to {results['Vg'].max():.1f}V
  Points: {len(results)}
"""
        ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax_metrics.set_title("Metrics", fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_2d_structure_with_results(geometry: dict,
                                    results: pd.DataFrame = None,
                                    title: str = "2D Planar FeFET Structure",
                                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2D planar FeFET structure with source/drain contacts.
    
    Args:
        geometry: Dictionary with device dimensions
        results: DataFrame with simulation results
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    
    if results is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        axes = [ax, None]
    
    ax_struct = axes[0]
    
    Lg = geometry.get('Lg', 100)  # Gate length in nm
    
    # Layer thicknesses
    t_channel = geometry.get('t_channel', 20)
    t_bottom_il = geometry.get('t_bottom_il', 0.7)
    t_fe = geometry.get('t_fe', 13.8)
    t_top_il = geometry.get('t_top_il', 4.0)
    t_gate = geometry.get('t_gate', 50)
    
    # Scale factor for display
    scale = 1.0
    
    # Draw substrate (full width)
    substrate_width = Lg * 2
    ax_struct.add_patch(plt.Rectangle((-substrate_width/2, -50*scale), substrate_width, 50*scale,
                                       facecolor='#808080', edgecolor='black', linewidth=1))
    ax_struct.text(0, -25*scale, 'Si Substrate\n(p-type)', ha='center', va='center', fontsize=9)
    
    # Draw channel under gate
    channel_y = 0
    ax_struct.add_patch(plt.Rectangle((-Lg/2, channel_y), Lg, t_channel*scale,
                                       facecolor='#90EE90', edgecolor='black', linewidth=2))
    ax_struct.text(0, channel_y + t_channel*scale/2, 'Channel', ha='center', va='center', fontsize=8)
    
    # Source n+ region
    source_width = Lg * 0.4
    ax_struct.add_patch(plt.Rectangle((-Lg/2 - source_width, channel_y), source_width, t_channel*scale,
                                       facecolor='#FF9999', edgecolor='black', linewidth=2))
    ax_struct.text(-Lg/2 - source_width/2, channel_y + t_channel*scale/2, 'n+ Source', 
                  ha='center', va='center', fontsize=8)
    
    # Drain n+ region
    ax_struct.add_patch(plt.Rectangle((Lg/2, channel_y), source_width, t_channel*scale,
                                       facecolor='#FF9999', edgecolor='black', linewidth=2))
    ax_struct.text(Lg/2 + source_width/2, channel_y + t_channel*scale/2, 'n+ Drain',
                  ha='center', va='center', fontsize=8)
    
    # Gate stack layers
    stack_y = t_channel * scale
    
    # Bottom IL
    ax_struct.add_patch(plt.Rectangle((-Lg/2, stack_y), Lg, max(t_bottom_il*scale, 3),
                                       facecolor='#FFD700', edgecolor='black', linewidth=1))
    stack_y += max(t_bottom_il*scale, 3)
    
    # Ferroelectric
    ax_struct.add_patch(plt.Rectangle((-Lg/2, stack_y), Lg, t_fe*scale,
                                       facecolor='#FF4444', edgecolor='black', linewidth=2))
    ax_struct.text(0, stack_y + t_fe*scale/2, f'HZO\n{t_fe}nm', 
                  ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    stack_y += t_fe*scale
    
    # Top IL
    ax_struct.add_patch(plt.Rectangle((-Lg/2, stack_y), Lg, max(t_top_il*scale, 3),
                                       facecolor='#FFD700', edgecolor='black', linewidth=1))
    stack_y += max(t_top_il*scale, 3)
    
    # Gate metal
    ax_struct.add_patch(plt.Rectangle((-Lg/2, stack_y), Lg, t_gate*scale,
                                       facecolor='#4169E1', edgecolor='black', linewidth=2))
    ax_struct.text(0, stack_y + t_gate*scale/2, f'TiN Gate\nLg={Lg}nm', 
                  ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Add dimension arrows
    ax_struct.annotate('', xy=(Lg/2, stack_y + t_gate*scale + 10), 
                      xytext=(-Lg/2, stack_y + t_gate*scale + 10),
                      arrowprops=dict(arrowstyle='<->', color='black'))
    ax_struct.text(0, stack_y + t_gate*scale + 15, f'Lg = {Lg} nm', ha='center', fontsize=10)
    
    # Contact labels
    ax_struct.text(-Lg/2 - source_width/2, -60*scale, 'S', ha='center', fontsize=12, fontweight='bold')
    ax_struct.text(Lg/2 + source_width/2, -60*scale, 'D', ha='center', fontsize=12, fontweight='bold')
    ax_struct.text(0, stack_y + t_gate*scale + 30, 'G', ha='center', fontsize=12, fontweight='bold')
    
    ax_struct.set_xlim(-substrate_width/2 - 20, substrate_width/2 + 20)
    ax_struct.set_ylim(-70*scale, stack_y + t_gate*scale + 50)
    ax_struct.set_aspect('equal')
    ax_struct.axis('off')
    ax_struct.set_title("2D Planar FeFET Cross-Section", fontsize=12, fontweight='bold')
    
    # Plot results if provided
    if results is not None and axes[1] is not None:
        ax_idvg = axes[1]
        
        if "direction" in results.columns:
            fwd = results[results["direction"] == "forward"]
            rev = results[results["direction"] == "reverse"]
            ax_idvg.semilogy(fwd["Vg"], np.abs(fwd["Id"]), 'b-', label="Forward", linewidth=2)
            ax_idvg.semilogy(rev["Vg"], np.abs(rev["Id"]), 'r--', label="Reverse", linewidth=2)
        else:
            ax_idvg.semilogy(results["Vg"], np.abs(results["Id"]), 'b-', linewidth=2)
        
        ax_idvg.set_xlabel("Gate Voltage (V)")
        ax_idvg.set_ylabel("|Id| (A)")
        ax_idvg.set_title("2D Transfer Characteristic", fontweight='bold')
        ax_idvg.legend()
        ax_idvg.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_3d_gaa_structure(geometry: dict,
                           wrap_angle: float = 360.0,
                           results: pd.DataFrame = None,
                           title: str = "3D GAA FeFET Structure",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3D GAA (Gate-All-Around) structure as cross-section and perspective.
    
    Args:
        geometry: Dictionary with device dimensions
        wrap_angle: Gate wrap angle in degrees (0=planar, 360=full GAA)
        results: DataFrame with simulation results
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    
    if results is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax_cross = axes[0]
    
    radius = geometry.get('radius', 5)  # nm
    t_fe = geometry.get('t_fe', 13.8)
    t_il = geometry.get('t_top_il', 4.0)
    
    # Draw cross-section view (circular)
    # Si core (nanowire)
    circle_si = plt.Circle((0, 0), radius, facecolor='#90EE90', edgecolor='black', linewidth=2)
    ax_cross.add_patch(circle_si)
    
    # IL layer
    circle_il = plt.Circle((0, 0), radius + t_il, facecolor='#FFD700', 
                           edgecolor='black', linewidth=1, fill=False)
    ax_cross.add_patch(circle_il)
    wedge_il = plt.matplotlib.patches.Wedge((0, 0), radius + t_il, 0, wrap_angle,
                                            width=t_il, facecolor='#FFD700', 
                                            edgecolor='black', linewidth=1, alpha=0.8)
    ax_cross.add_patch(wedge_il)
    
    # FE layer
    wedge_fe = plt.matplotlib.patches.Wedge((0, 0), radius + t_il + t_fe, 0, wrap_angle,
                                            width=t_fe, facecolor='#FF4444',
                                            edgecolor='black', linewidth=2, alpha=0.9)
    ax_cross.add_patch(wedge_fe)
    
    # Gate metal (outer)
    gate_thickness = 10
    wedge_gate = plt.matplotlib.patches.Wedge((0, 0), radius + t_il + t_fe + gate_thickness, 
                                               0, wrap_angle, width=gate_thickness,
                                               facecolor='#4169E1', edgecolor='black', 
                                               linewidth=2, alpha=0.9)
    ax_cross.add_patch(wedge_gate)
    
    # Labels
    ax_cross.text(0, 0, 'Si', ha='center', va='center', fontsize=10, fontweight='bold')
    
    outer_r = radius + t_il + t_fe + gate_thickness + 5
    ax_cross.set_xlim(-outer_r, outer_r)
    ax_cross.set_ylim(-outer_r, outer_r)
    ax_cross.set_aspect('equal')
    ax_cross.axis('off')
    
    # Determine architecture name
    if wrap_angle == 0:
        arch_name = "Planar"
    elif wrap_angle <= 180:
        arch_name = "FinFET"
    else:
        arch_name = f"GAA-{int(wrap_angle)}°"
    
    ax_cross.set_title(f"Cross-Section: {arch_name}\n(wrap angle = {wrap_angle}°)", 
                       fontsize=12, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', edgecolor='black', label=f'Si Core (r={radius}nm)'),
        Patch(facecolor='#FFD700', edgecolor='black', label=f'IL ({t_il}nm)'),
        Patch(facecolor='#FF4444', edgecolor='black', label=f'HZO ({t_fe}nm)'),
        Patch(facecolor='#4169E1', edgecolor='black', label='TiN Gate'),
    ]
    ax_cross.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Side view (perspective)
    ax_side = axes[1]
    
    Lg = geometry.get('Lg', 100)
    
    # Draw as a 3D-like side view
    # Channel (cylinder representation)
    from matplotlib.patches import FancyBboxPatch
    
    # Source contact
    ax_side.add_patch(FancyBboxPatch((-80, -radius), 30, 2*radius,
                                     boxstyle="round,pad=0.02", facecolor='#CCCCCC',
                                     edgecolor='black', linewidth=2))
    ax_side.text(-65, 0, 'Source', ha='center', va='center', fontsize=9)
    
    # Channel region
    ax_side.add_patch(FancyBboxPatch((-50, -radius), Lg, 2*radius,
                                     boxstyle="round,pad=0.01", facecolor='#90EE90',
                                     edgecolor='black', linewidth=2))
    ax_side.text(-50 + Lg/2, 0, 'Si\nNanowire', ha='center', va='center', fontsize=8)
    
    # Gate stack (surrounding the channel)
    gate_r = radius + t_il + t_fe + gate_thickness
    ax_side.add_patch(FancyBboxPatch((-50, -gate_r), Lg, 2*gate_r,
                                     boxstyle="round,pad=0.05", facecolor='none',
                                     edgecolor='#4169E1', linewidth=3, linestyle='--'))
    ax_side.text(-50 + Lg/2, gate_r + 10, f'Gate\n(Lg={Lg}nm)', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Drain contact
    ax_side.add_patch(FancyBboxPatch((Lg - 50 + 10, -radius), 30, 2*radius,
                                     boxstyle="round,pad=0.02", facecolor='#CCCCCC',
                                     edgecolor='black', linewidth=2))
    ax_side.text(Lg - 50 + 25, 0, 'Drain', ha='center', va='center', fontsize=9)
    
    ax_side.set_xlim(-100, Lg + 10)
    ax_side.set_ylim(-gate_r - 20, gate_r + 30)
    ax_side.set_aspect('equal')
    ax_side.axis('off')
    ax_side.set_title("Side View (Along Channel)", fontsize=12, fontweight='bold')
    
    # Plot results if provided
    if results is not None and len(axes) > 2:
        ax_idvg = axes[2]
        
        if "direction" in results.columns:
            fwd = results[results["direction"] == "forward"]
            rev = results[results["direction"] == "reverse"]
            ax_idvg.semilogy(fwd["Vg"], np.abs(fwd["Id"]), 'b-', label="Forward", linewidth=2)
            ax_idvg.semilogy(rev["Vg"], np.abs(rev["Id"]), 'r--', label="Reverse", linewidth=2)
        else:
            ax_idvg.semilogy(results["Vg"], np.abs(results["Id"]), 'b-', linewidth=2)
        
        ax_idvg.set_xlabel("Gate Voltage (V)")
        ax_idvg.set_ylabel("|Id| (A)")
        ax_idvg.set_title(f"3D {arch_name} Transfer Characteristic", fontweight='bold')
        ax_idvg.legend()
        ax_idvg.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_architecture_summary(results_dict: dict,
                               geometry: dict,
                               title: str = "Architecture Comparison Summary",
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive summary comparing all architectures.
    
    Args:
        results_dict: Dictionary with architecture names as keys, DataFrames as values
        geometry: Device geometry
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    setup_thesis_style()
    
    fig = plt.figure(figsize=(16, 10))
    
    # Grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1D stack schematic
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, "1D: Vertical Stack\n(Capacitor Model)", 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.axis('off')
    ax1.set_title("1D MIFIS", fontweight='bold')
    
    # 2D planar schematic
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.5, "2D: Planar FeFET\n(S/D + Gate)", 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax2.axis('off')
    ax2.set_title("2D Planar", fontweight='bold')
    
    # 3D GAA schematic  
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.5, "3D: GAA Nanowire\n(360° Gate Wrap)", 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.8))
    ax3.axis('off')
    ax3.set_title("3D GAA", fontweight='bold')
    
    # Combined Id-Vg plot
    ax4 = fig.add_subplot(gs[1, :2])
    
    colors = {'1d': 'blue', '2d': 'green', '3d': 'red', 'gaa': 'purple'}
    
    for arch_name, df in results_dict.items():
        color = colors.get(arch_name.lower(), 'gray')
        if "direction" in df.columns:
            fwd = df[df["direction"] == "forward"]
            ax4.semilogy(fwd["Vg"], np.abs(fwd["Id"]), color=color, 
                        label=arch_name, linewidth=2)
        else:
            ax4.semilogy(df["Vg"], np.abs(df["Id"]), color=color,
                        label=arch_name, linewidth=2)
    
    ax4.set_xlabel("Gate Voltage (V)")
    ax4.set_ylabel("|Id| (A)")
    ax4.set_title("Transfer Characteristics Comparison", fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Metrics table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Create metrics table
    table_data = []
    for arch_name, df in results_dict.items():
        Id_on = df["Id"].max()
        Id_off = df["Id"].min()
        on_off = Id_on / Id_off if Id_off > 0 else np.inf
        table_data.append([arch_name, f"{Id_on:.2e}", f"{on_off:.2e}"])
    
    table = ax5.table(cellText=table_data,
                      colLabels=["Architecture", "Ion (A)", "On/Off"],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax5.set_title("Performance Metrics", fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_all_formats(fig: plt.Figure, base_path: str):
    """Save figure in multiple formats (PNG, PDF, SVG)."""
    base = Path(base_path).with_suffix('')
    fig.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{base}.pdf", bbox_inches='tight')
    fig.savefig(f"{base}.svg", bbox_inches='tight')
    print(f"Saved: {base}.png, .pdf, .svg")

