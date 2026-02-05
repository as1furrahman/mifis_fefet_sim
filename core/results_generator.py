"""
MIFIS FeFET Results Generator
==============================
Generates plots and results in EXACT format matching fefet_simulation_base.

Output Structure:
- plots/1D/  - Individual 1D plots (P-V, E-V, P-E, evolution, structure, summary)
- plots/2D/  - Individual 2D plots
- plots/3D/  - Individual 3D plots
- results/   - CSV summaries and pickle files

Author: Thesis Project
Date: February 2026
"""

import numpy as np
import os
import csv
from datetime import datetime
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    import pickle
    PICKLE_OK = True
except ImportError:
    PICKLE_OK = False


# =============================================================================
# INDIVIDUAL PLOT GENERATORS
# =============================================================================

def plot_pv_hysteresis(V_gate: np.ndarray, P_uC: np.ndarray, MW: float,
                        save_path: str = None, show: bool = False):
    """Generate P-V Hysteresis Loop plot."""
    if not MATPLOTLIB_OK:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot(V_gate, P_uC, 'b-', linewidth=2)
    
    # Add arrows to indicate direction (CCW)
    if len(V_gate) > 10:
        # Arrow on top branch (Reverse sweep: Leftward)
        # Find point near V=0 on reverse sweep (last half of data)
        mid_idx = len(V_gate) // 2
        rev_idx = mid_idx + len(V_gate)//4
        if rev_idx < len(V_gate) - 1:
            ax.annotate('', xy=(V_gate[rev_idx-1], P_uC[rev_idx-1]), 
                        xytext=(V_gate[rev_idx], P_uC[rev_idx]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Arrow on bottom branch (Forward sweep: Rightward)
        # Find point near V=0 on forward sweep (first half)
        fwd_idx = len(V_gate) // 4
        if fwd_idx < mid_idx:
            ax.annotate('', xy=(V_gate[fwd_idx+1], P_uC[fwd_idx+1]), 
                        xytext=(V_gate[fwd_idx], P_uC[fwd_idx]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Gate Voltage (V)', fontsize=12)
    ax.set_ylabel('Polarization (µC/cm²)', fontsize=12)
    ax.set_title('P-V Hysteresis Loop (HZO)', fontsize=14, fontweight='bold')
    ax.text(0.05, 0.95, f'MW = {MW:.2f} V', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
    return fig


def plot_ev_field(V_gate: np.ndarray, E_MV: np.ndarray,
                   save_path: str = None, show: bool = False):
    """Generate E-field vs Voltage plot."""
    if not MATPLOTLIB_OK:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(V_gate, E_MV, 'r-', linewidth=2, label='HZO Layer')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Gate Voltage (V)', fontsize=12)
    ax.set_ylabel('Electric Field (MV/cm)', fontsize=12)
    ax.set_title('E-field in HZO vs Voltage', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
    return fig


def plot_pe_loop(E_MV: np.ndarray, P_uC: np.ndarray,
                  save_path: str = None, show: bool = False):
    """Generate P-E Loop plot with Pr and Ec annotations."""
    if not MATPLOTLIB_OK:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Main hysteresis loop
    ax.plot(E_MV, P_uC, 'b-', linewidth=2.5, label='HZO P-E Loop')

    # Axes through origin
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8)

    # Extract Pr and Ec from the data
    if len(P_uC) > 1 and len(E_MV) > 1:
        E_max = np.max(np.abs(E_MV))
        P_max = np.max(np.abs(P_uC))

        # Find Pr: Polarization values where E is closest to 0
        E_near_zero_mask = np.abs(E_MV) < 0.1 * E_max
        if np.any(E_near_zero_mask):
            P_at_zero = P_uC[E_near_zero_mask]
            Pr_pos = np.max(P_at_zero)
            Pr_neg = np.min(P_at_zero)
        else:
            idx_zero = np.argmin(np.abs(E_MV))
            Pr_pos = abs(P_uC[idx_zero])
            Pr_neg = -Pr_pos

        # Find Ec: E values where P crosses zero
        sign_changes = np.where(np.diff(np.sign(P_uC)))[0]
        Ec_values = []
        for idx in sign_changes:
            if idx < len(E_MV) - 1:
                E1, E2 = E_MV[idx], E_MV[idx+1]
                P1, P2 = P_uC[idx], P_uC[idx+1]
                if P2 != P1:
                    E_cross = E1 - P1 * (E2 - E1) / (P2 - P1)
                    Ec_values.append(E_cross)

        if len(Ec_values) >= 2:
            Ec_pos = max(Ec_values)
            Ec_neg = min(Ec_values)
        elif len(Ec_values) == 1:
            Ec_pos = abs(Ec_values[0])
            Ec_neg = -Ec_pos
        else:
            Ec_pos = 0.1
            Ec_neg = -0.1

        # Draw Pr markers (horizontal dashed lines)
        ax.axhline(y=Pr_pos, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=Pr_neg, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

        # Draw Ec markers (vertical dashed lines)
        ax.axvline(x=Ec_pos, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=Ec_neg, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Mark points
        ax.plot(0, Pr_pos, 'go', markersize=10, markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(0, Pr_neg, 'go', markersize=10, markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(Ec_pos, 0, 'r^', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
        ax.plot(Ec_neg, 0, 'r^', markersize=10, markeredgecolor='darkred', markeredgewidth=2)

        # Annotations
        ax.annotate(f'+$P_r$={Pr_pos:.1f}', xy=(0, Pr_pos),
                   xytext=(0.15*E_max, Pr_pos + 0.1*P_max),
                   fontsize=10, color='darkgreen', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))
        ax.annotate(f'-$P_r$={Pr_neg:.1f}', xy=(0, Pr_neg),
                   xytext=(0.15*E_max, Pr_neg - 0.1*P_max),
                   fontsize=10, color='darkgreen', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))
        ax.annotate(f'+$E_c$={Ec_pos:.2f}', xy=(Ec_pos, 0),
                   xytext=(Ec_pos + 0.1*E_max, 0.25*P_max),
                   fontsize=10, color='darkred', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1))
        ax.annotate(f'-$E_c$={Ec_neg:.2f}', xy=(Ec_neg, 0),
                   xytext=(Ec_neg - 0.1*E_max, -0.25*P_max),
                   fontsize=10, color='darkred', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1))

        # Summary box
        summary_text = (f"$P_r$ = ±{abs(Pr_pos):.1f} μC/cm²\n"
                       f"$E_c$ = ±{abs(Ec_pos):.2f} MV/cm")
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Electric Field (MV/cm)', fontsize=12)
    ax.set_ylabel('Polarization (µC/cm²)', fontsize=12)
    ax.set_title('P-E Hysteresis Loop (HZO Ferroelectric)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
    return fig


def plot_polarization_evolution(P_uC: np.ndarray,
                                 save_path: str = None, show: bool = False):
    """Generate Polarization Evolution plot."""
    if not MATPLOTLIB_OK:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(P_uC, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Voltage Step', fontsize=12)
    ax.set_ylabel('Polarization (µC/cm²)', fontsize=12)
    ax.set_title('Polarization Evolution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
    return fig


def plot_id_vg_dual_state(Vg_fwd: np.ndarray, Id_fwd: np.ndarray,
                          Vg_rev: np.ndarray, Id_rev: np.ndarray,
                          Vd: float = 0.1, MW: float = None,
                          save_path: str = None, show: bool = False):
    """
    Plot Id-Vg transfer characteristics for both polarization states.

    Shows programmed (HVT) and erased (LVT) states on same plot with
    memory window extraction and annotation.

    Args:
        Vg_fwd: Gate voltage array for forward (erase) sweep
        Id_fwd: Drain current array for forward sweep (A)
        Vg_rev: Gate voltage array for reverse (program) sweep
        Id_rev: Drain current array for reverse sweep (A)
        Vd: Drain voltage used (V)
        MW: Pre-calculated memory window (optional)
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_OK:
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot both states (log scale)
    ax.semilogy(Vg_fwd, np.abs(Id_fwd), 'b-', linewidth=2.5, label='LVT (Erased State)')
    ax.semilogy(Vg_rev, np.abs(Id_rev), 'r-', linewidth=2.5, label='HVT (Programmed State)')

    # Extract Vth for each state using constant current method
    I_th = 1e-7  # 100 nA threshold current
    Vth_LVT = None
    Vth_HVT = None

    try:
        # LVT (forward/erased) - find where current crosses threshold
        Id_fwd_abs = np.abs(Id_fwd)
        idx_cross_fwd = np.where(Id_fwd_abs >= I_th)[0]
        if len(idx_cross_fwd) > 0:
            idx = idx_cross_fwd[0]
            if idx > 0:
                # Linear interpolation
                Vth_LVT = np.interp(np.log10(I_th),
                                   np.log10(Id_fwd_abs[idx-1:idx+1] + 1e-20),
                                   Vg_fwd[idx-1:idx+1])
            else:
                Vth_LVT = Vg_fwd[idx]

        # HVT (reverse/programmed)
        Id_rev_abs = np.abs(Id_rev)
        idx_cross_rev = np.where(Id_rev_abs >= I_th)[0]
        if len(idx_cross_rev) > 0:
            idx = idx_cross_rev[0]
            if idx > 0:
                Vth_HVT = np.interp(np.log10(I_th),
                                   np.log10(Id_rev_abs[idx-1:idx+1] + 1e-20),
                                   Vg_rev[idx-1:idx+1])
            else:
                Vth_HVT = Vg_rev[idx]
    except Exception:
        pass

    # Calculate MW if not provided
    if MW is None and Vth_LVT is not None and Vth_HVT is not None:
        MW = abs(Vth_HVT - Vth_LVT)

    # Draw Vth markers
    if Vth_LVT is not None:
        ax.axvline(x=Vth_LVT, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.plot(Vth_LVT, I_th, 'bo', markersize=10)
        ax.text(Vth_LVT - 0.2, I_th * 0.1, f'$V_{{th}}^{{LVT}}$={Vth_LVT:.2f}V',
                fontsize=10, color='blue', fontweight='bold')

    if Vth_HVT is not None:
        ax.axvline(x=Vth_HVT, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.plot(Vth_HVT, I_th, 'ro', markersize=10)
        ax.text(Vth_HVT + 0.1, I_th * 0.1, f'$V_{{th}}^{{HVT}}$={Vth_HVT:.2f}V',
                fontsize=10, color='red', fontweight='bold')

    # Memory window annotation
    if Vth_LVT is not None and Vth_HVT is not None and MW is not None:
        ax.annotate('', xy=(Vth_HVT, I_th), xytext=(Vth_LVT, I_th),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
        ax.text((Vth_LVT + Vth_HVT) / 2, I_th * 5, f'MW = {MW:.2f} V',
               ha='center', fontsize=14, fontweight='bold', color='green',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Read current reference line
    ax.axhline(y=I_th, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(ax.get_xlim()[0] + 0.1, I_th * 2, f'$I_{{th}}$ = {I_th:.0e} A',
            fontsize=9, color='gray')

    ax.set_xlabel('Gate Voltage $V_g$ (V)', fontsize=12)
    ax.set_ylabel('Drain Current $|I_d|$ (A)', fontsize=12)
    ax.set_title(f'Transfer Characteristics (Id-Vg) at $V_d$ = {Vd} V', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_ylim(1e-14, 1e-3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()

    return fig


def plot_id_vd_family(Vd_arrays: list, Id_arrays: list, Vg_values: list,
                      polarization_state: str = "Programmed",
                      save_path: str = None, show: bool = False):
    """
    Plot Id-Vd output characteristics (family of curves at different Vg).

    Args:
        Vd_arrays: List of Vd arrays for each Vg
        Id_arrays: List of Id arrays for each Vg
        Vg_values: List of Vg values (V)
        polarization_state: "Programmed" or "Erased"
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_OK:
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(Vg_values)))

    for i, (Vd, Id, Vg) in enumerate(zip(Vd_arrays, Id_arrays, Vg_values)):
        ax.plot(Vd, np.abs(Id) * 1e6, '-', color=colors[i], linewidth=2,
                label=f'$V_g$ = {Vg:.1f} V')

    ax.set_xlabel('Drain Voltage $V_d$ (V)', fontsize=12)
    ax.set_ylabel('Drain Current $|I_d|$ (μA)', fontsize=12)
    ax.set_title(f'Output Characteristics (Id-Vd) - {polarization_state} State',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, title='Gate Voltage')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    # Add saturation region annotation
    ax.text(0.7, 0.9, 'Saturation', transform=ax.transAxes,
            fontsize=11, style='italic', color='gray')
    ax.text(0.2, 0.9, 'Linear', transform=ax.transAxes,
            fontsize=11, style='italic', color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()

    return fig


def plot_2d_potential_map(x: np.ndarray, y: np.ndarray, potential: np.ndarray,
                          geometry: dict, polarization_state: str = "Programmed",
                          Vg: float = 0.0, Vd: float = 0.1,
                          save_path: str = None, show: bool = False):
    """
    Plot 2D electrostatic potential distribution in device cross-section.

    Args:
        x: X-coordinates (nm) - lateral direction
        y: Y-coordinates (nm) - vertical direction (depth)
        potential: 2D potential array (V)
        geometry: Device geometry dict
        polarization_state: "Programmed" or "Erased"
        Vg: Gate voltage (V)
        Vd: Drain voltage (V)
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_OK:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create meshgrid if needed
    if potential.ndim == 1:
        # Assume 1D data, create simple 2D representation
        X, Y = np.meshgrid(x, y)
        # Expand 1D potential to 2D (uniform in x)
        potential = np.tile(potential.reshape(-1, 1), (1, len(x)))
    else:
        X, Y = np.meshgrid(x, y)

    # Plot potential contour
    levels = 50
    contour = ax.contourf(X, Y, potential, levels=levels, cmap='RdYlBu_r')
    cbar = fig.colorbar(contour, ax=ax, label='Potential φ (V)')

    # Add contour lines
    ax.contour(X, Y, potential, levels=15, colors='k', linewidths=0.5, alpha=0.3)

    # Extract geometry
    t_top_il = geometry.get('t_top_il', 4.0)
    t_fe = geometry.get('t_fe', 13.8)
    t_bot_il = geometry.get('t_bottom_il', 0.7)
    t_channel = geometry.get('t_channel', 20)
    Lg = geometry.get('Lg', 100)

    # Mark layer boundaries with horizontal lines
    y_max = np.max(y)
    y_channel = t_channel
    y_bot_il = y_channel + t_bot_il
    y_fe = y_bot_il + t_fe
    y_top_il = y_fe + t_top_il

    for y_bound, label in [(y_channel, 'Si/Bot-IL'),
                           (y_bot_il, 'Bot-IL/HZO'),
                           (y_fe, 'HZO/Top-IL')]:
        if y_bound < y_max:
            ax.axhline(y=y_bound, color='white', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add layer labels
    ax.text(np.max(x) * 0.85, t_channel / 2, 'Si Channel', fontsize=10,
            color='white', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    ax.set_xlabel('Lateral Position (nm)', fontsize=12)
    ax.set_ylabel('Depth from Bottom (nm)', fontsize=12)
    ax.set_title(f'2D Potential Distribution - {polarization_state} State\n'
                f'$V_g$ = {Vg:.1f} V, $V_d$ = {Vd:.2f} V',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()

    return fig


def plot_2d_potential_comparison(geometry: dict, Lg: float = 100,
                                  save_path: str = None, show: bool = False):
    """
    Generate 2D potential comparison for Programmed vs Erased states.

    Creates a synthetic potential distribution to show channel modulation
    by ferroelectric polarization state.

    Args:
        geometry: Device geometry dict
        Lg: Gate length (nm)
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_OK:
        return None

    # Device dimensions
    t_top_il = geometry.get('t_top_il', 4.0)
    t_fe = geometry.get('t_fe', 13.8)
    t_bot_il = geometry.get('t_bottom_il', 0.7)
    t_channel = geometry.get('t_channel', 20)

    total_height = t_channel + t_bot_il + t_fe + t_top_il
    Lsd = Lg * 0.3  # Source/drain extension
    L_total = Lg + 2 * Lsd

    # Create mesh
    nx, ny = 100, 80
    x = np.linspace(0, L_total, nx)
    y = np.linspace(0, total_height, ny)
    X, Y = np.meshgrid(x, y)

    # Layer boundaries
    y_channel = t_channel
    y_bot_il = y_channel + t_bot_il
    y_fe = y_bot_il + t_fe

    # Gate region in x
    x_gate_start = Lsd
    x_gate_end = Lsd + Lg

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (state, Vth_shift) in enumerate([('Erased (LVT)', -0.3), ('Programmed (HVT)', 0.5)]):
        ax = axes[idx]

        # Create potential distribution
        potential = np.zeros((ny, nx))

        # Base potential (grounded substrate)
        for j in range(ny):
            y_pos = y[j]

            if y_pos < y_channel:
                # Silicon channel
                # Potential varies based on gate proximity and polarization
                for i in range(nx):
                    x_pos = x[i]
                    # Under gate region
                    if x_gate_start <= x_pos <= x_gate_end:
                        # Channel potential affected by Vth shift
                        surface_pot = 0.3 + Vth_shift
                        depth_factor = np.exp(-(y_channel - y_pos) / 5)
                        potential[j, i] = surface_pot * depth_factor
                    else:
                        # Source/drain regions
                        potential[j, i] = 0.0

            elif y_pos < y_bot_il:
                # Bottom IL - linear potential drop
                for i in range(nx):
                    if x_gate_start <= x[i] <= x_gate_end:
                        frac = (y_pos - y_channel) / t_bot_il
                        potential[j, i] = (0.3 + Vth_shift) + frac * 0.5

            elif y_pos < y_fe:
                # HZO ferroelectric - major potential drop
                for i in range(nx):
                    if x_gate_start <= x[i] <= x_gate_end:
                        frac = (y_pos - y_bot_il) / t_fe
                        potential[j, i] = (0.8 + Vth_shift) + frac * 1.5

            else:
                # Top IL
                for i in range(nx):
                    if x_gate_start <= x[i] <= x_gate_end:
                        frac = (y_pos - y_fe) / t_top_il
                        potential[j, i] = (2.3 + Vth_shift) + frac * 0.7

        # Plot
        levels = 30
        contour = ax.contourf(X, Y, potential, levels=levels, cmap='RdYlBu_r')
        fig.colorbar(contour, ax=ax, label='φ (V)')
        ax.contour(X, Y, potential, levels=10, colors='k', linewidths=0.3, alpha=0.5)

        # Layer boundaries
        for y_bound in [y_channel, y_bot_il, y_fe]:
            ax.axhline(y=y_bound, color='white', linestyle='--', linewidth=1, alpha=0.7)

        # Gate region marker
        ax.axvline(x=x_gate_start, color='white', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=x_gate_end, color='white', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_xlabel('Lateral Position (nm)', fontsize=11)
        ax.set_ylabel('Depth (nm)', fontsize=11)
        ax.set_title(f'{state}', fontsize=12, fontweight='bold')

        # Add labels
        ax.text(L_total/2, t_channel/2, 'Channel', ha='center', color='white',
                fontsize=9, fontweight='bold')
        ax.text(L_total/2, y_bot_il + t_fe/2, 'HZO', ha='center', color='white',
                fontsize=9, fontweight='bold')

    plt.suptitle('2D Potential Distribution: Channel Modulation by FE State\n'
                 '($V_g$ = 1.0 V, $V_d$ = 0.1 V)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()

    return fig


def plot_electric_field_distribution(geometry: dict, V_biases: list = None,
                                      save_path: str = None, show: bool = False):
    """
    Generate Electric Field Distribution E(x) vs depth through MIFIS stack.

    Shows field distribution at different gate biases to demonstrate:
    - Field concentration in HZO ferroelectric layer
    - Voltage division between IL and FE layers
    - Fields remain within safe operating limits

    Args:
        geometry: Dict with layer thicknesses (t_gate, t_top_il, t_fe, t_bottom_il, t_channel)
        V_biases: List of gate voltages to plot [V]. Default: [0, +3, -3]
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        matplotlib Figure
    """
    if not MATPLOTLIB_OK:
        return None

    # Default biases if not provided
    if V_biases is None:
        V_biases = [0.0, 3.0, -3.0]  # Vg = 0, +Vprog, -Verase

    # Extract geometry
    t_gate = geometry.get('t_gate', 50)      # nm
    t_top_il = geometry.get('t_top_il', 4.0)  # nm
    t_fe = geometry.get('t_fe', 13.8)         # nm
    t_bottom_il = geometry.get('t_bottom_il', 0.7)  # nm
    t_channel = geometry.get('t_channel', 20)  # nm

    # Material permittivities
    eps_il = 3.9      # SiO2
    eps_fe = 30.0     # HZO (high-k)
    eps_si = 11.7     # Silicon

    # HZO ferroelectric parameters
    Pr = 18.0  # μC/cm² remnant polarization
    Ec = 1.0   # MV/cm coercive field (100 kV/cm = 0.1 MV/cm, but use 1.0 for visibility)

    # Position array (depth from gate surface)
    # Layer boundaries (cumulative from top)
    x_gate_end = t_gate
    x_top_il_end = x_gate_end + t_top_il
    x_fe_end = x_top_il_end + t_fe
    x_bot_il_end = x_fe_end + t_bottom_il
    x_total = x_bot_il_end + t_channel

    # Create position array for each layer
    n_pts = 100
    x_top_il = np.linspace(x_gate_end, x_top_il_end, n_pts//4)
    x_fe = np.linspace(x_top_il_end, x_fe_end, n_pts//2)
    x_bot_il = np.linspace(x_fe_end, x_bot_il_end, n_pts//8)
    x_channel = np.linspace(x_bot_il_end, x_total, n_pts//4)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['blue', 'red', 'green', 'purple', 'orange']
    linestyles = ['-', '--', '-.', ':', '-']

    for i, Vg in enumerate(V_biases):
        # Calculate voltage division using capacitor model
        # C_total = 1 / (1/C_top_il + 1/C_fe + 1/C_bot_il)
        # E = V / t for each layer, weighted by capacitance

        # Effective capacitance per unit area (proportional to eps/t)
        C_top_il = eps_il / t_top_il
        C_fe = eps_fe / t_fe
        C_bot_il = eps_il / t_bottom_il

        # Total series capacitance
        C_total_inv = 1/C_top_il + 1/C_fe + 1/C_bot_il
        C_total = 1 / C_total_inv

        # Voltage drops across each layer (V = Q/C, Q same in series)
        V_top_il = Vg * (C_total / C_top_il)
        V_fe = Vg * (C_total / C_fe)
        V_bot_il = Vg * (C_total / C_bot_il)

        # Electric fields (MV/cm) = V / t(nm) * 0.1 = V / t * 10^-1
        # E (V/m) = V / (t * 1e-9), convert to MV/cm: * 1e-6 / 1e-2 = * 1e-4
        E_top_il = V_top_il / (t_top_il * 1e-9) * 1e-8  # MV/cm
        E_fe = V_fe / (t_fe * 1e-9) * 1e-8              # MV/cm
        E_bot_il = V_bot_il / (t_bottom_il * 1e-9) * 1e-8  # MV/cm

        # For channel, field decays exponentially (depletion approximation)
        E_channel_surface = E_bot_il * eps_il / eps_si
        decay_length = 10  # nm characteristic decay

        # Build E(x) arrays for each layer
        E_top_il_arr = np.ones_like(x_top_il) * E_top_il
        E_fe_arr = np.ones_like(x_fe) * E_fe
        E_bot_il_arr = np.ones_like(x_bot_il) * E_bot_il
        E_channel_arr = E_channel_surface * np.exp(-(x_channel - x_bot_il_end) / decay_length)

        # Combine arrays
        x_full = np.concatenate([x_top_il, x_fe, x_bot_il, x_channel])
        E_full = np.concatenate([E_top_il_arr, E_fe_arr, E_bot_il_arr, E_channel_arr])

        # Label for legend
        if Vg == 0:
            label = f'$V_g$ = 0 V'
        elif Vg > 0:
            label = f'$V_g$ = +{Vg:.1f} V (Program)'
        else:
            label = f'$V_g$ = {Vg:.1f} V (Erase)'

        ax.plot(x_full, E_full, color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2.5, label=label)

    # Add layer region shading
    ax.axvspan(x_gate_end, x_top_il_end, alpha=0.15, color='gold', label='Top IL (SiO₂)')
    ax.axvspan(x_top_il_end, x_fe_end, alpha=0.2, color='orange', label='HZO (FE)')
    ax.axvspan(x_fe_end, x_bot_il_end, alpha=0.15, color='gold')
    ax.axvspan(x_bot_il_end, x_total, alpha=0.1, color='green', label='Si Channel')

    # Add breakdown field reference line for SiO2 (~10 MV/cm)
    E_breakdown_sio2 = 10.0
    ax.axhline(y=E_breakdown_sio2, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(y=-E_breakdown_sio2, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(x_total * 0.7, E_breakdown_sio2 * 1.05, 'SiO₂ Breakdown (~10 MV/cm)',
            fontsize=9, color='red', alpha=0.8)

    # Layer boundary labels
    layer_centers = [
        (x_gate_end + x_top_il_end) / 2,
        (x_top_il_end + x_fe_end) / 2,
        (x_fe_end + x_bot_il_end) / 2,
        (x_bot_il_end + x_total) / 2
    ]
    layer_names = ['Top IL', 'HZO', 'Bot IL', 'Si']

    y_max = ax.get_ylim()[1]
    for xc, name in zip(layer_centers, layer_names):
        ax.text(xc, y_max * 0.9, name, ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Depth from Gate Surface (nm)', fontsize=12)
    ax.set_ylabel('Electric Field (MV/cm)', fontsize=12)
    ax.set_title('Electric Field Distribution E(x) Through MIFIS Stack', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)

    # Add geometry info box
    info_text = (f"Stack: SiO₂({t_top_il}nm) / HZO({t_fe}nm) / SiO₂({t_bottom_il}nm)\n"
                f"ε(IL)={eps_il:.1f}, ε(HZO)={eps_fe:.0f}")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()

    return fig


def plot_device_structure(geometry: dict, phase: str = "1D",
                           save_path: str = None, show: bool = False):
    """Generate dimension-specific MIFIS Device Structure sketch."""
    if not MATPLOTLIB_OK:
        return None
        
    phase_clean = phase.upper()
    
    try:
        if "1D" in phase_clean:
            return plot_device_structure_1d(geometry, save_path, show)
        elif "2D" in phase_clean:
            return plot_device_structure_2d(geometry, save_path, show)
        elif "3D" in phase_clean:
            return plot_device_structure_3d(geometry, save_path, show)
        else:
            return plot_device_structure_1d(geometry, save_path, show)
    except Exception as e:
        print(f"ERROR generating {phase} structure plot: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# AESTHETIC CONFIGURATION
# =============================================================================

STYLE_CONFIG = {
    'colors': {
        'silicon': '#F5F5F5',      # White Smoke (Base)
        'source_drain': '#E0E0E0', # Light Grey
        'sio2': '#FFFFFF',         # White (hatch will define it)
        'hzo': '#FFF3E0',          # Very pale orange/cream
        'gate': '#B0BEC5',         # Blue Grey Light
        'contact': '#78909C',      # Blue Grey Medium
        'text': '#212121',         # Almost Black
        'line': '#000000',         # Black
        'channel': '#FF5252'       # Soft Red (still needs to stand out slightly)
    },
    'font': {
        'family': 'serif',
        'weight': 'normal',
        'size_title': 14,
        'size_label': 10,
        'size_annot': 9
    }
}

def plot_device_structure_1d(geometry: dict, save_path: str = None, show: bool = False, ax=None):
    """1D Vertical Stack Sketch - Classic Aesthetic."""
    t_gate = geometry.get('t_gate', 50)
    t_top_il = geometry.get('t_top_il', 4.0)
    t_fe = geometry.get('t_fe', 13.8)
    t_bottom_il = geometry.get('t_bottom_il', 0.7)
    t_channel = geometry.get('t_channel', 20)
    
    c = STYLE_CONFIG['colors']
    
    # Layers: (Name, Physical Thickness, Color, Display Height)
    layers = [
        ('Gate Contact (W/TiN)', t_gate, c['gate'], 15),
        ('Top Interlayer (SiO₂)', t_top_il, c['sio2'], 4),
        ('Ferroelectric (HZO)', t_fe, c['hzo'], 10),
        ('Bottom Interlayer (SiO₂)', t_bottom_il, c['sio2'], 4),
        ('p-Si Substrate (Body)', t_channel, c['silicon'], 15),
    ]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 9))
    else:
        fig = ax.figure
    ax.axis('off')
    
    # Center setup
    x_center = 0.4
    width = 0.5
    y_curr = 0
    
    # Draw layers
    for name, real_th, color, h_disp in reversed(layers):
        # Main Block
        rect = patches.Rectangle((x_center-width/2, y_curr), width, h_disp, 
                               linewidth=1.2, edgecolor=c['line'], facecolor=color)
        ax.add_patch(rect)
        
        # Annotation (Right side, strictly aligned)
        label_x = x_center + width/2 + 0.1
        label_y = y_curr + h_disp/2
        
        # Line connection
        ax.plot([x_center + width/2, label_x-0.02], [label_y, label_y], 
                color=c['line'], linewidth=0.8)
        
        # Text
        ax.text(label_x, label_y, f"{name}", 
                va='bottom', ha='left', fontsize=11, color=c['text'], 
                family=STYLE_CONFIG['font']['family'], fontweight='bold')
        ax.text(label_x, label_y, f"{real_th} nm", 
                va='top', ha='left', fontsize=10, color=c['text'], 
                family=STYLE_CONFIG['font']['family'], style='italic')
        
        y_curr += h_disp

    # Circuit Symbols
    # Ground at bottom (Symbol drawn first, label below it)
    symbol_x = x_center
    symbol_y_start = 0
    symbol_y_end = -2.5
    
    # Vertical line
    ax.plot([symbol_x, symbol_x], [symbol_y_start, symbol_y_end], color=c['line'], lw=1.5)
    # Horizontal bars (decreasing width)
    ax.plot([symbol_x-0.05, symbol_x+0.05], [symbol_y_end, symbol_y_end], color=c['line'], lw=1.5)
    ax.plot([symbol_x-0.03, symbol_x+0.03], [symbol_y_end-0.3, symbol_y_end-0.3], color=c['line'], lw=1.5)
    ax.plot([symbol_x-0.01, symbol_x+0.01], [symbol_y_end-0.6, symbol_y_end-0.6], color=c['line'], lw=1.5)
    
    # Text Label (Moved safely below symbol)
    ax.text(symbol_x, symbol_y_end - 1.2, "Body (GND)", ha='center', va='top', 
            fontsize=11, color=c['text'], family='serif', fontweight='bold')
    
    # Gate Vg at top
    ax.text(x_center, y_curr + 4, "Gate Voltage (Vg)", ha='center', va='bottom', 
            fontsize=11, color=c['text'], fontweight='bold', family='serif')
    ax.arrow(x_center, y_curr + 3, 0, -2.5, head_width=0.03, head_length=0.8, 
             fc=c['contact'], ec=c['contact'], lw=1.5)
    
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-8, y_curr + 6)
    ax.set_title("1D MIFIS Structure", fontsize=16, 
                color=c['text'], family='serif', fontweight='bold', pad=20)
    
    # Only save if we created the figure locally OR if specifically requested? Usually only if standalone.
    pass

    if save_path and ax is None: # Only save here if we own the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig

def plot_device_structure_2d(geometry: dict, save_path: str = None, show: bool = False, ax=None):
    """2D Planar FET Cross-Section - Minimalist Academic Style."""
    Lg = geometry.get('Lg', 100)
    t_fe = geometry.get('t_fe', 13.8)
    t_top_il = geometry.get('t_top_il', 4.0)
    t_bottom_il = geometry.get('t_bottom_il', 0.7)
    
    c = STYLE_CONFIG['colors']
    
    W_sub = 300
    H_sub = 70
    W_sd = 80
    H_sd = 25
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure
    ax.axis('off')
    
    # 1. Silicon Substrate (Base)
    ax.add_patch(patches.Rectangle((0, 0), W_sub, H_sub, 
                                 edgecolor='black', facecolor=c['silicon'], hatch='///', alpha=0.5))
    ax.text(W_sub/2, 10, r"$p$-Si (Substrate)", ha='center', color=c['text'], 
            family='serif', style='italic', fontsize=10)
    
    # 2. Source / Drain Regions
    ax.add_patch(patches.Rectangle((0, H_sub-H_sd), W_sd, H_sd, 
                                 edgecolor='black', facecolor=c['source_drain']))
    ax.add_patch(patches.Rectangle((W_sub-W_sd, H_sub-H_sd), W_sd, H_sd, 
                                 edgecolor='black', facecolor=c['source_drain']))
    
    # 3. Contacts
    # Source
    ax.add_patch(patches.Rectangle((10, H_sub), W_sd-20, 15, 
                                 edgecolor='black', facecolor=c['contact']))
    ax.text(W_sd/2, H_sub+7.5, r"Source ($n^{+}$)", ha='center', va='center', 
            color='white', fontsize=9, fontweight='bold', family='sans-serif')
    
    # Drain
    ax.add_patch(patches.Rectangle((W_sub-W_sd+10, H_sub), W_sd-20, 15, 
                                 edgecolor='black', facecolor=c['contact']))
    ax.text(W_sub-W_sd/2, H_sub+7.5, r"Drain ($n^{+}$)", ha='center', va='center', 
            color='white', fontsize=9, fontweight='bold', family='sans-serif')

    # Channel Highlight (Visible dashed box)
    channel_x = W_sd
    channel_w = W_sub - 2*W_sd
    channel_h = 3 
    
    # Dashed box for channel
    ax.add_patch(patches.Rectangle((channel_x, H_sub - channel_h - 1), channel_w, channel_h,
                                 edgecolor=c['channel'], facecolor='none', linestyle='--', linewidth=1.5))
    
    # 4. Gate Stack
    x_gate = (W_sub - Lg)/2
    y_curr = H_sub
    
    # Bottom IL
    h_bil = 4
    ax.add_patch(patches.Rectangle((x_gate, y_curr), Lg, h_bil, 
                                 edgecolor='black', facecolor=c['sio2']))
    y_curr += h_bil
    
    # HZO
    h_fe = 15
    ax.add_patch(patches.Rectangle((x_gate, y_curr), Lg, h_fe, 
                                 edgecolor='black', facecolor=c['hzo']))
    y_curr += h_fe
    
    # Top IL
    h_til = 4
    ax.add_patch(patches.Rectangle((x_gate, y_curr), Lg, h_til, 
                                 edgecolor='black', facecolor=c['sio2']))
    y_curr += h_til
    
    # Gate Metal
    h_gate = 20
    ax.add_patch(patches.Rectangle((x_gate, y_curr), Lg, h_gate, 
                                 edgecolor='black', facecolor=c['gate']))
    ax.text(x_gate+Lg/2, y_curr+h_gate/2, "Gate", ha='center', va='center',
            fontsize=10, family='serif', fontweight='bold', color='white')
            
    # Draw separate Vg contact point
    ax.arrow(x_gate+Lg/2, y_curr+h_gate, 0, 10, head_width=5, head_length=5, fc='black', ec='black')
    # Optimized gap for Vg text (closer but not overlapping: +18)
    ax.text(x_gate+Lg/2, y_curr+h_gate+18, r"$V_g$", ha='center', fontweight='bold', fontsize=12)


    # Annotations with MINIMAL POINTS (Straight lines, no arrowheads)
    
    # Annotations with ROUTED ARROWS (Clean & Separated)
    # Using 'angle' style for clear dog-leg connectors
    
    annot_x_start = W_sub + 40  # Push labels further right for separation
    
    def annotate_routed(text, y_target, x_target, y_text_pos):
        ax.annotate(text,
                    xy=(x_target, y_target), xycoords='data',
                    xytext=(annot_x_start, y_text_pos), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=5",
                                    color="black", lw=1.2),
                    fontsize=10, family='serif', va='center', ha='left',
                    bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.8)) # varying bg for readabilty

    # Optimize Label Positions for "Section Identification"
    # We spread them out vertically to distinct heights
    
    # Gate Contact (Topmost)
    annotate_routed("Gate Metal (W/TiN)", 
                   y_target = y_curr + h_gate/2, 
                   x_target = x_gate + Lg, 
                   y_text_pos = y_curr + 30)
                   
    # Multi-Layer Dielectric Stack (Grouped)
    # Single arrow pointing to the center of the stack (HZO)
    # Text lists all layers clearly
    y_stack_center = y_curr - h_til - h_fe/2
    
    stack_text = (f"Dielectric Stack:\n"
                  f"• Top IL ({t_top_il}nm)\n"
                  f"• HZO ({t_fe}nm)\n"
                  f"• Bot IL ({t_bottom_il}nm)")
                  
    annotate_routed(stack_text, 
                   y_target = y_stack_center, 
                   x_target = x_gate + Lg, 
                   y_text_pos = y_stack_center - 10)

    # Inversion Channel (Separate Section at Bottom)
    # Pointing to the red highlight box
    annotate_routed("Inversion Channel", 
                   y_target = H_sub - channel_h/2 - 1, 
                   x_target = channel_x + channel_w/2, 
                   y_text_pos = H_sub - 50)

    ax.set_xlim(-10, W_sub + 120)
    ax.set_ylim(0, H_sub + 80)
    ax.set_title("2D Planar FeFET Schematic", fontsize=16, family='serif', fontweight='bold')
    
    if save_path and ax is None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig

def plot_device_structure_3d(geometry: dict, save_path: str = None, show: bool = False, ax=None):
    """3D Device Structure - Planar Stack (Reference Implementation)."""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return plot_device_structure_1d(geometry, save_path, show)
        
    L = geometry.get('Lg', 100)
    W = geometry.get('W_channel', 100) # Default to square if not specified
    
    # Layer thicknesses
    T_si = geometry.get('t_channel', 20)
    T_bot_IL = geometry.get('t_bottom_il', 0.7)
    T_fe = geometry.get('t_fe', 13.8)
    T_top_IL = geometry.get('t_top_il', 4.0)
    T_gate_metal = geometry.get('t_gate', 50)
    
    c = STYLE_CONFIG['colors']
    
    # Use channel-centered coordinate system
    x_range = (-L / 2.0, L / 2.0)
    y_range = (-W / 2.0, W / 2.0)
    
    # EXPLOEDED VIEW PARAMETERS
    # We use "display limits" instead of physical limits to ensure thin layers (0.7nm) are visible
    # Ratio: 1 visual unit approx 1-2nm, but with minimums
    
    # Visual heights (tunable for aesthetics)
    h_si_vis = 15
    h_bil_vis = 4   # Exaggerated (real 0.7)
    h_fe_vis = 12
    h_til_vis = 4   # Preserved/Slightly Exaggerated
    h_gate_vis = 15
    
    gap = 8  # Vertical gap between exploded layers
    
    L_sd = L * 0.4 # Visual source/drain length extension
    
    # Define layers: (z_start_vis, height_vis, color, label_text, physical_th_text)
    vis_layers = []
    
    current_z = 0
    
    # 1. Silicon Body (Channel + S/D)
    # We will handle S/D geometry in the plotting loop specially
    vis_layers.append({
        'z': current_z, 'h': h_si_vis, 
        'c': c['silicon'], 
        'lbl': "p-Si Body", 
        'th': f"{T_si} nm",
        'type': 'channel_with_sd', # Special marker
        'alpha': 0.9
    })
    current_z += h_si_vis + gap
    
    # 2. Bottom IL
    vis_layers.append({
        'z': current_z, 'h': h_bil_vis, 
        'c': c['sio2'], 
        'lbl': "Bot IL", 
        'th': f"{T_bot_IL} nm",
        'alpha': 0.95
    })
    current_z += h_bil_vis + gap
    
    # 3. HZO
    vis_layers.append({
        'z': current_z, 'h': h_fe_vis, 
        'c': c['hzo'], 
        'lbl': "HZO", 
        'th': f"{T_fe} nm",
        'alpha': 0.9
    })
    current_z += h_fe_vis + gap
    
    # 4. Top IL
    vis_layers.append({
        'z': current_z, 'h': h_til_vis, 
        'c': c['sio2'], 
        'lbl': "Top IL", 
        'th': f"{T_top_IL} nm",
        'alpha': 0.95
    })
    current_z += h_til_vis + gap
    
    # 5. Gate
    vis_layers.append({
        'z': current_z, 'h': h_gate_vis, 
        'c': c['gate'], 
        'lbl': "Gate Metal", 
        'th': f"{T_gate_metal} nm",
        'alpha': 1.0
    })
    
    # Plotting
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    def _add_exploded_box(ax, xr, yr, z_start, height, col, alph, edge_c='black', line_w=0.5):
        x = [xr[0], xr[1], xr[1], xr[0]]
        y = [yr[0], yr[0], yr[1], yr[1]]
        z0 = z_start
        z1 = z_start + height
        
        verts = []
        # bottom and top
        verts.append([list(zip(x, y, [z0]*4))])
        verts.append([list(zip(x, y, [z1]*4))])
        # sides
        for i in range(4):
            x_s = [x[i], x[(i+1)%4], x[(i+1)%4], x[i]]
            y_s = [y[i], y[(i+1)%4], y[(i+1)%4], y[i]]
            z_s = [z0, z0, z1, z1]
            verts.append([list(zip(x_s, y_s, z_s))])
        
        for v in verts:
            poly = Poly3DCollection(v, facecolors=col, alpha=alph, 
                                  edgecolors=edge_c, linewidths=line_w)
            ax.add_collection3d(poly)

    # Draw layers and annotations
    for l in vis_layers:
        
        if l.get('type') == 'channel_with_sd':
            # Draw Central Channel
            _add_exploded_box(ax, x_range, y_range, l['z'], l['h'], l['c'], l['alpha'])
            
            # Draw Source (Left) - Shifted in X (Length direction)
            # Assuming Length is along X
            xs_range = (x_range[0] - L_sd, x_range[0])
            _add_exploded_box(ax, xs_range, y_range, l['z'], l['h'], 
                             c['source_drain'], l['alpha'])
            
            # Label Source
            # "In front view" -> Move negative Y (in front of block)
            # "More outside" -> Increase offset significantly (was 40)
            center_s = xs_range[0] + L_sd/2
            y_front = y_range[0] - 100 # Push out 100 units in front
            ax.text(center_s, y_front, l['z']+l['h']/2, "Source", 
                   ha='center', va='center', fontsize=10, fontweight='bold', color='darkred',
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

            # Draw Drain (Right)
            xd_range = (x_range[1], x_range[1] + L_sd)
            _add_exploded_box(ax, xd_range, y_range, l['z'], l['h'], 
                             c['source_drain'], l['alpha'])
            
            # Label Drain
            center_d = xd_range[0] + L_sd/2
            ax.text(center_d, y_front, l['z']+l['h']/2, "Drain", 
                   ha='center', va='center', fontsize=10, fontweight='bold', color='darkred',
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))
            
        else:
            # Standard Layer
            _add_exploded_box(ax, x_range, y_range, l['z'], l['h'], l['c'], l['alpha'])
        
        # Annotation: "Every section side"
        # Move to RIGHT side (Positive Y) to fix left cropping
        # "Middle box" -> Center of the face (x=0)
        xc = (x_range[0] + x_range[1]) / 2.0 # Center X
        yc = y_range[1] # Back edge Y
        zc = l['z'] + l['h']/2
        
        # Text position (pushed out FARTHER in Y to make line longer)
        y_text = yc + 100 # Increased from 40 to 100
        
        # Draw dotted leader line
        ax.plot([xc, xc], [yc, y_text], [zc, zc], 'k:', lw=1)
        
        # Label text
        label_full = f"{l['lbl']} ({l['th']})"
        ax.text(xc, y_text, zc, label_full, 
               ha='left', va='center', fontsize=10, fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_xlabel("Length (nm)")
    ax.set_ylabel("Width (nm)")
    ax.set_zlabel("Stack Order")
    ax.set_title("3D MIFIS Device Structure (Exploded View)", fontsize=16, fontweight='bold', pad=20)
    
    # Set limits - ZOOM OUT MORE
    # Drain text extends to the right (positive X), so we need more positive X buffer
    max_dim = max(L + 2*L_sd, W) 
    
    # Expand limits significantly
    ax.set_xlim(-max_dim*1.2, max_dim*1.5) # Extra space on right for Drain text
    ax.set_ylim(-max_dim*1.5, max_dim*1.0 + 150) # Increased Negative Y for Source/Drain labels
    ax.set_zlim(0, current_z + 50) # Extra vertical space for Z axis labels
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    
    # ZOOM OUT CAMERA to include Z-axis labels
    ax.dist = 16  # Increased from 13 to 16 to prevent cropping
    ax.set_zlabel("Stack Order", labelpad=20) # Increase padding to push label off axis
    
    ax.view_init(elev=25, azim=-45)
    
    # Use subplots_adjust instead of tight_layout default to guarantee margins
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    
    if save_path and ax is None:
        plt.savefig(save_path, dpi=300) # Removed bbox_inches='tight' which sometimes crops 3D labels
        plt.close(fig)
    return fig


def plot_performance_summary(MW: float, P_max: float, P_min: float, geometry: dict,
                              phase: str = "1D", save_path: str = None, show: bool = False):
    """Generate Performance Summary plot."""
    if not MATPLOTLIB_OK:
        return None
    
    t_top_il = geometry.get('t_top_il', 4.0)
    t_fe = geometry.get('t_fe', 13.8)
    t_bottom_il = geometry.get('t_bottom_il', 0.7)
    total_stack = t_bottom_il + t_fe + t_top_il
    delta_P = abs(P_max - P_min)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    metrics_text = f"""
    MIFIS {phase} PERFORMANCE SUMMARY
    {'='*40}
    
    Structure:
    • Metal / SiO₂({t_top_il}nm) / HZO({t_fe}nm) / SiO₂({t_bottom_il}nm) / Si
    • Total Stack: {total_stack:.1f} nm
    
    Key Metrics:
    • Memory Window:  {MW:.2f} V
    • Target Range:   2.5 - 3.0 V
    • P+ state:       {P_max:+.2f} µC/cm²
    • P- state:       {P_min:+.2f} µC/cm²
    • ΔP:             {delta_P:.2f} µC/cm²
    
    Status: {'✅ TARGET ACHIEVED' if MW >= 2.5 else '⚠️ Below Target'}
    """
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
    return fig


# =============================================================================
# COMBINED 6-PANEL PLOT (for backward compatibility)
# =============================================================================

def generate_mifis_results_plot(
    V_gate: np.ndarray,
    P_fe: np.ndarray,
    E_fe: np.ndarray,
    MW: float,
    geometry: dict,
    title: str = "MIFIS FeFET Optimization Results",
    save_path: str = None,
    show: bool = False
):
    """
    Generate 6-panel results plot matching fefet_simulation_base format.
    """
    if not MATPLOTLIB_OK:
        print("  WARNING: matplotlib not available, skipping plot")
        return None
    
    # Convert units based on magnitude
    P_max_abs = np.max(np.abs(P_fe)) if len(P_fe) > 0 else 0
    if P_max_abs < 0.001:
        P_uC = P_fe * 1e6
    elif P_max_abs > 100:
        P_uC = P_fe
    else:
        P_uC = P_fe
    
    E_max_abs = np.max(np.abs(E_fe)) if len(E_fe) > 0 else 0
    if E_max_abs > 1e4:
        E_MV = E_fe * 1e-6
    else:
        E_MV = E_fe
    
    fig = plt.figure(figsize=(16, 10))
    
    # Panel 1: P-V Hysteresis Loop
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(V_gate, P_uC, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Gate Voltage (V)', fontsize=12)
    ax1.set_ylabel('Polarization (µC/cm²)', fontsize=12)
    ax1.set_title('P-V Hysteresis Loop (HZO)', fontsize=13, fontweight='bold')
    ax1.text(0.05, 0.95, f'MW = {MW:.2f} V', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
             fontsize=11, fontweight='bold')
    
    # Panel 2: E-field vs Voltage
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(V_gate, E_MV, 'r-', linewidth=2, label='HZO Layer')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Gate Voltage (V)', fontsize=12)
    ax2.set_ylabel('Electric Field (MV/cm)', fontsize=12)
    ax2.set_title('E-field in HZO vs Voltage', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    
    # Panel 3: P-E Loop
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(E_MV, P_uC, 'purple', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Electric Field (MV/cm)', fontsize=12)
    ax3.set_ylabel('Polarization (µC/cm²)', fontsize=12)
    ax3.set_title('P-E Loop (Ferroelectric)', fontsize=13, fontweight='bold')
    
    # Panel 4: Polarization Evolution
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(P_uC, 'b-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Voltage Step', fontsize=12)
    ax4.set_ylabel('Polarization (µC/cm²)', fontsize=12)
    ax4.set_title('Polarization Evolution', fontsize=13, fontweight='bold')
    
    # Panel 5: MIFIS Structure
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    t_gate = geometry.get('t_gate', 50)
    t_top_il = geometry.get('t_top_il', 4.0)
    t_fe = geometry.get('t_fe', 13.8)
    t_bottom_il = geometry.get('t_bottom_il', 0.7)
    t_channel = geometry.get('t_channel', 20)
    
    layers = [
        ('Gate (W/TiN)', t_gate, 'silver'),
        ('Top IL (SiO₂)', t_top_il, 'lightblue'),
        ('HZO', t_fe, 'orange'),
        ('Bottom IL (SiO₂)', t_bottom_il, 'lightblue'),
        ('Silicon', t_channel, 'lightgreen'),
    ]
    
    y_base = 0
    for name, thickness, color in layers:
        ax5.barh(y_base, thickness, height=0.8, left=0, color=color,
                 edgecolor='black', linewidth=2)
        ax5.text(thickness/2, y_base, f'{name}\n{thickness:.1f} nm',
                 ha='center', va='center', fontsize=9, fontweight='bold')
        y_base += 1
    
    ax5.set_xlim(0, max([t for _, t, _ in layers]) * 1.2)
    ax5.set_ylim(-0.5, len(layers)-0.5)
    ax5.set_title('MIFIS Structure (1D Sketch)', fontsize=13, fontweight='bold')
    ax5.invert_yaxis()
    
    # Panel 6: Performance Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    P_max = np.max(P_uC)
    P_min = np.min(P_uC)
    delta_P = P_max - P_min
    total_stack = t_bottom_il + t_fe + t_top_il
    
    metrics_text = f"""
    MIFIS PERFORMANCE SUMMARY
    
    Structure:
    • Metal / SiO₂({t_top_il}nm) / HZO({t_fe}nm) / SiO₂({t_bottom_il}nm) / Si
    • Total Stack: {total_stack:.1f} nm
    
    Key Metrics:
    • Memory Window:  {MW:.2f} V
    • Target Range:    2.5 - 3.0 V
    • P+ state:        {P_max:+.2f} µC/cm²
    • P- state:        {P_min:+.2f} µC/cm²
    • ΔP:              {delta_P:.2f} µC/cm²
    """
    ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes,
             verticalalignment='top', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Combined plot saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# CSV AND PICKLE GENERATORS
# =============================================================================

def generate_mifis_summary_csv(MW: float, geometry: dict, material: dict = None,
                                save_path: str = None):
    """Generate CSV summary in fefet_simulation_base format."""
    t_fe = geometry.get('t_fe', 13.8)
    t_top_il = geometry.get('t_top_il', 4.0)
    t_bottom_il = geometry.get('t_bottom_il', 0.7)
    
    if material is None:
        material = {'Pr': 18.0, 'Ps': 38.0, 'Ec': 0.10}
    
    rows = [
        ['Metric', 'Value', 'Unit'],
        ['Memory Window', f'{MW:.3f}', 'V'],
        ['HZO Thickness', f'{t_fe:.1f}', 'nm'],
        ['Top IL Thickness', f'{t_top_il:.1f}', 'nm'],
        ['Bottom IL Thickness', f'{t_bottom_il:.2f}', 'nm'],
        ['Pr', f'{material["Pr"]:.1f}', 'µC/cm²'],
        ['Ps', f'{material["Ps"]:.1f}', 'µC/cm²'],
        ['Ec', f'{material["Ec"]:.2f}', 'MV/cm'],
    ]
    
    if save_path:
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
        print(f"  ✓ CSV saved: {save_path}")
    
    return rows


def generate_mifis_results_pkl(results: dict, MW: float, geometry: dict,
                                material: dict = None, save_path: str = None):
    """Save simulation data to pickle file."""
    if not PICKLE_OK:
        return None
    
    data = {
        'results': results,
        'memory_window': MW,
        'geometry': geometry,
        'material': material or {},
        'timestamp': datetime.now().isoformat()
    }
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ Pickle saved: {save_path}")
    
    return data


# =============================================================================
# MAIN OUTPUT GENERATOR (with individual plots)
# =============================================================================

def generate_all_mifis_outputs(
    simulation_results: dict,
    geometry: dict,
    output_dir: str = None,
    phase: str = "1d",
    show_plot: bool = False
):
    """
    Generate all outputs (individual plots, combined plot, CSV, pickle).
    
    Output Structure:
    - plots/{phase.upper()}/  - Individual plots
    - results/                - CSV and pickle files
    
    Args:
        simulation_results: Dict with V_gate, P_fe, E_fe, memory_window
        geometry: Dict with device thicknesses
        output_dir: Base output directory (default: project root)
        phase: Simulation phase (1d, 2d, 3d)
        show_plot: Whether to display plots
    
    Returns:
        Dict with paths to generated files
    """
    # Setup directories
    base_dir = Path(output_dir) if output_dir else Path(".")
    plots_dir = base_dir / "plots" / phase.upper()
    results_dir = base_dir / "results"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up old files for this phase (removes both timestamped and fixed-name files)
    import glob
    import os
    
    # Clean plots directory for this phase
    for old_file in plots_dir.glob(f"*_{phase}*.png"):
        try:
            old_file.unlink()
        except:
            pass
    for old_file in plots_dir.glob(f"*{phase}*.png"):
        try:
            old_file.unlink()
        except:
            pass
    
    # Clean results directory for this phase
    for old_file in results_dir.glob(f"mifis_{phase}*.csv"):
        try:
            old_file.unlink()
        except:
            pass
    for old_file in results_dir.glob(f"mifis_{phase}*.pkl"):
        try:
            old_file.unlink()
        except:
            pass
    
    # Extract data
    V_gate = np.array(simulation_results.get('V_gate', []))
    P_fe = np.array(simulation_results.get('P_fe', []))
    E_fe = np.array(simulation_results.get('E_fe', []))
    MW = simulation_results.get('memory_window', 0.0)
    
    # Convert units
    P_max_abs = np.max(np.abs(P_fe)) if len(P_fe) > 0 else 0
    if P_max_abs < 0.001:
        P_uC = P_fe * 1e6
    else:
        P_uC = P_fe
    
    E_max_abs = np.max(np.abs(E_fe)) if len(E_fe) > 0 else 0
    if E_max_abs > 1e4:
        E_MV = E_fe * 1e-6
    else:
        E_MV = E_fe
    
    # Use fixed filenames (no timestamps) - each run replaces previous results
    outputs = {'plots': {}, 'results': {}}
    
    print(f"\n  Generating {phase.upper()} outputs (replacing previous)...")
    
    # =========================================================================
    # INDIVIDUAL PLOTS (saved to plots/{phase}/)
    # =========================================================================
    
    # 1. P-V Hysteresis Loop
    pv_path = plots_dir / f"pv_hysteresis_{phase}.png"
    plot_pv_hysteresis(V_gate, P_uC, MW, save_path=str(pv_path))
    outputs['plots']['pv_hysteresis'] = str(pv_path)
    print(f"    ✓ P-V Hysteresis: {pv_path.name}")
    
    # 2. E-field vs Voltage
    ev_path = plots_dir / f"efield_voltage_{phase}.png"
    plot_ev_field(V_gate, E_MV, save_path=str(ev_path))
    outputs['plots']['efield_voltage'] = str(ev_path)
    print(f"    ✓ E-V Field: {ev_path.name}")
    
    # 3. P-E Loop
    pe_path = plots_dir / f"pe_loop_{phase}.png"
    plot_pe_loop(E_MV, P_uC, save_path=str(pe_path))
    outputs['plots']['pe_loop'] = str(pe_path)
    print(f"    ✓ P-E Loop: {pe_path.name}")
    
    # 4. Polarization Evolution
    evol_path = plots_dir / f"polarization_evolution_{phase}.png"
    plot_polarization_evolution(P_uC, save_path=str(evol_path))
    outputs['plots']['polarization_evolution'] = str(evol_path)
    print(f"    ✓ Polarization Evolution: {evol_path.name}")
    
    # 5. Device Structure
    struct_path = plots_dir / f"device_structure_{phase}.png"
    plot_device_structure(geometry, phase=phase.upper(), save_path=str(struct_path))
    outputs['plots']['device_structure'] = str(struct_path)
    print(f"    ✓ Device Structure: {struct_path.name}")
    
    # 6. Performance Summary
    P_max = np.max(P_uC) if len(P_uC) > 0 else 0
    P_min = np.min(P_uC) if len(P_uC) > 0 else 0
    summary_path = plots_dir / f"performance_summary_{phase}.png"
    plot_performance_summary(MW, P_max, P_min, geometry, phase=phase.upper(),
                              save_path=str(summary_path))
    outputs['plots']['performance_summary'] = str(summary_path)
    print(f"    ✓ Performance Summary: {summary_path.name}")

    # 7. Electric Field Distribution E(x) vs Depth (1D only - stack validation)
    if phase.lower() == "1d":
        field_dist_path = plots_dir / f"field_distribution_{phase}.png"
        plot_electric_field_distribution(geometry, V_biases=[0.0, 3.0, -3.0],
                                         save_path=str(field_dist_path))
        outputs['plots']['field_distribution'] = str(field_dist_path)
        print(f"    ✓ Field Distribution E(x): {field_dist_path.name}")

    # 8. Combined 6-panel plot
    combined_path = plots_dir / f"mifis_{phase}_combined.png"
    generate_mifis_results_plot(
        V_gate=V_gate, P_fe=P_fe, E_fe=E_fe, MW=MW, geometry=geometry,
        title=f"MIFIS FeFET {phase.upper()} Simulation Results",
        save_path=str(combined_path), show=show_plot
    )
    outputs['plots']['combined'] = str(combined_path)
    
    # =========================================================================
    # RESULTS (saved to results/)
    # =========================================================================
    
    # CSV Summary
    csv_path = results_dir / f"mifis_{phase}_summary.csv"
    generate_mifis_summary_csv(MW=MW, geometry=geometry, save_path=str(csv_path))
    outputs['results']['csv'] = str(csv_path)
    
    # Pickle file
    pkl_path = results_dir / f"mifis_{phase}_results.pkl"
    generate_mifis_results_pkl(
        results=simulation_results, MW=MW, geometry=geometry, save_path=str(pkl_path)
    )
    outputs['results']['pkl'] = str(pkl_path)
    
    print(f"\n  ✓ All {phase.upper()} outputs generated!")
    print(f"    Plots: {plots_dir}")
    print(f"    Results: {results_dir}")
    
    return outputs


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("MIFIS Results Generator Test")
    print("=" * 50)
    
    # Generate test data
    n_points = 200
    V_max = 3.0
    
    V_up = np.linspace(0, V_max, n_points//4)
    V_down = np.linspace(V_max, -V_max, n_points//2)
    V_return = np.linspace(-V_max, 0, n_points//4)
    V_gate = np.concatenate([V_up, V_down, V_return])
    
    P_fe = 38e-6 * np.tanh(2.0 * V_gate / 3.0)
    E_fe = V_gate / (13.8 * 1e-7)
    
    MW = 3.95
    
    geometry = {
        't_gate': 50,
        't_top_il': 4.0,
        't_fe': 13.8,
        't_bottom_il': 0.7,
        't_channel': 20,
    }
    
    results = {
        'V_gate': V_gate,
        'P_fe': P_fe,
        'E_fe': E_fe,
        'memory_window': MW,
    }
    
    outputs = generate_all_mifis_outputs(
        simulation_results=results,
        geometry=geometry,
        output_dir=".",
        phase="test",
        show_plot=False
    )
    
    print("\nGenerated files:")
    print("  Plots:", outputs['plots'])
    print("  Results:", outputs['results'])
