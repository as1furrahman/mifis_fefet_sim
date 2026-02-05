"""
MIFIS FeFET Simulation Framework
================================
Ferroelectric physics models including Landau-Khalatnikov dynamics.

Author: Thesis Project
Date: February 2026
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from .config import FerroelectricMaterial


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

EPS0 = 8.854e-12      # Vacuum permittivity (F/m)
Q_E = 1.602e-19       # Elementary charge (C)
K_B = 1.381e-23       # Boltzmann constant (J/K)


# =============================================================================
# LANDAU-KHALATNIKOV MODEL
# =============================================================================

@dataclass
class LandauKhalatnikovModel:
    """
    Landau-Khalatnikov (L-K) ferroelectric polarization model.
    
    Implements:
        dP/dt = -(1/τ) × ∂F/∂P
        
    Where free energy:
        F(P,E) = αP² + βP⁴ + γP⁶ - E·P
    """
    
    # Landau coefficients
    alpha: float          # Negative for ferroelectric
    beta: float           # Positive for stability
    gamma: float = 0.0    # Often negligible
    
    # Dynamics
    tau: float = 1e-8     # Switching time constant (s)
    
    # Saturation
    Ps: float = 0.38      # Saturation polarization (C/m²)
    
    @classmethod
    def from_material(cls, mat: FerroelectricMaterial) -> 'LandauKhalatnikovModel':
        """Create L-K model from material properties."""
        # Convert units
        Pr = mat.Pr * 1e-2   # μC/cm² → C/m²
        Ec = mat.Ec * 1e8    # MV/cm → V/m
        Ps = mat.Ps * 1e-2   # μC/cm² → C/m²
        
        # Derive coefficients
        alpha = -2 * Ec / Pr
        beta = Ec / (2 * Pr**3)
        tau = mat.tau_switching * 1e-9  # ns → s
        
        return cls(alpha=alpha, beta=beta, gamma=0.0, tau=tau, Ps=Ps)
    
    def free_energy(self, P: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        Calculate Landau free energy.
        
        Args:
            P: Polarization (C/m²)
            E: Electric field (V/m)
            
        Returns:
            Free energy density (J/m³)
        """
        return self.alpha * P**2 + self.beta * P**4 + self.gamma * P**6 - E * P
    
    def dF_dP(self, P: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        Derivative of free energy with respect to polarization.
        
        Args:
            P: Polarization (C/m²)
            E: Electric field (V/m)
            
        Returns:
            ∂F/∂P
        """
        return 2*self.alpha*P + 4*self.beta*P**3 + 6*self.gamma*P**5 - E
    
    def equilibrium_polarization(self, E: float) -> float:
        """
        Find equilibrium polarization for given electric field.
        Solves: ∂F/∂P = 0
        
        Args:
            E: Electric field (V/m)
            
        Returns:
            Equilibrium polarization (C/m²)
        """
        # Use tanh approximation for numerical stability
        # P_eq ≈ Ps * tanh((E - sign*Ec) / delta)
        Ec = -self.alpha / (2 * np.sqrt(self.beta * abs(self.alpha)))
        delta = Ec / 3  # Transition width

        # FIXED: Removed incorrect 1e8 scaling factor
        return self.Ps * np.tanh(E / delta)
    
    def dP_dt(self, P: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        Time derivative of polarization (L-K dynamics).
        
        Args:
            P: Current polarization (C/m²)
            E: Applied electric field (V/m)
            
        Returns:
            dP/dt (C/m²/s)
        """
        return -(1/self.tau) * self.dF_dP(P, E)
    
    def step(self, P: float, E: float, dt: float) -> float:
        """
        Single time step of L-K evolution (Euler method).
        
        Args:
            P: Current polarization
            E: Applied electric field
            dt: Time step
            
        Returns:
            Updated polarization
        """
        dP = self.dP_dt(np.array([P]), np.array([E]))[0]
        P_new = P + dP * dt
        
        # Clamp to ±Ps
        return np.clip(P_new, -self.Ps, self.Ps)
    
    def evolve(self, P0: float, E_history: np.ndarray, 
               dt: float) -> np.ndarray:
        """
        Evolve polarization through time with changing field.
        
        Args:
            P0: Initial polarization
            E_history: Array of electric field values over time
            dt: Time step
            
        Returns:
            Array of polarization values
        """
        n_steps = len(E_history)
        P = np.zeros(n_steps)
        P[0] = P0
        
        for i in range(1, n_steps):
            P[i] = self.step(P[i-1], E_history[i], dt)
        
        return P


# =============================================================================
# FERROELECTRIC MODEL WITH MEMORY (for target MW ~3.9V)
# =============================================================================

class FerroelectricModelWithMemory:
    """
    Enhanced ferroelectric model with hysteresis memory.
    Tracks polarization history for realistic P-V behavior.
    MATCHED to fefet_simulation_base for target MW ~3.9V.
    """
    
    def __init__(self, P_s: float, P_r: float, E_c: float, T: float = 300):
        """
        Initialize ferroelectric model with memory.
        
        Args:
            P_s: Spontaneous polarization (C/cm²)
            P_r: Remanent polarization (C/cm²)
            E_c: Coercive field (kV/cm) - use 100 for HZO
            T: Temperature (K)
        """
        self.P_s = P_s
        self.P_r = P_r
        self.E_c = E_c
        self.T = T
        
        # Hysteresis memory state
        self.P_previous = 0.0
        self.E_previous = 0.0
        self.switching_history = []
    
    def polarization_enhanced(self, E_field: float, alpha: float = 2.0) -> float:
        """Landau polarization with adjustable sharpness."""
        e_norm = E_field / self.E_c
        return self.P_s * np.tanh(alpha * e_norm)
    
    def polarization_with_memory(self, E_field: float, alpha: float = 2.0) -> float:
        """
        Polarization with hysteresis memory effect - CORRECTED FOR PROPER Pr BEHAVIOR.

        Implements proper ferroelectric hysteresis:
        - At |E| >> Ec: Saturates to ±Ps (saturation polarization)
        - At E ≈ 0: Relaxes to ±Pr (remnant polarization, NOT Ps!)
        - At intermediate E: Smooth transition between Pr and Ps
        - Maintains hysteresis memory for sweep direction

        Args:
            E_field: Electric field (kV/cm)
            alpha: Sharpness parameter

        Returns:
            Polarization with memory (C/cm²)
        """
        # Determine if we're in a new sweep or continuing
        E_diff = E_field - self.E_previous

        # Define field thresholds
        E_threshold_low = 0.15 * self.E_c   # Below this: remnant state
        E_threshold_high = 0.9 * self.E_c   # Above this: saturation state

        # Calculate ideal polarization for reference
        P_ideal = self.polarization_enhanced(E_field, alpha)

        # HIGH FIELD REGIME: |E| > Ec → Saturate to ±Ps
        if abs(E_field) > E_threshold_high:
            # Full switching to saturation
            P_new = P_ideal  # ≈ ±Ps * tanh(large) ≈ ±Ps

            # Track switching events
            if abs(P_new - self.P_previous) > 0.5 * self.P_r:
                self.switching_history.append({
                    'E_field': E_field, 'P_old': self.P_previous, 'P_new': P_new
                })

        # LOW FIELD REGIME: |E| << Ec → Relax to ±Pr (KEY FIX!)
        elif abs(E_field) < E_threshold_low:
            # Relaxation to remnant polarization
            # Sign determined by previous polarization state (memory)
            if self.P_previous != 0.0:
                P_sign = np.sign(self.P_previous)
            else:
                P_sign = np.sign(E_field) if E_field != 0 else 1.0

            # Relax to Pr, not Ps!
            P_new = self.P_r * P_sign

        # INTERMEDIATE REGIME: Transition between Pr and Ps
        else:
            # Smooth interpolation between Pr and Ps based on field strength
            field_fraction = (abs(E_field) - E_threshold_low) / (E_threshold_high - E_threshold_low)
            field_fraction = np.clip(field_fraction, 0.0, 1.0)

            # Determine sign from field direction or memory
            if abs(E_field) > 0.01 * self.E_c:
                P_sign = np.sign(E_field)
            else:
                P_sign = np.sign(self.P_previous) if self.P_previous != 0 else 1.0

            # Linear interpolation: P = Pr + (Ps - Pr) * field_fraction
            P_magnitude = self.P_r + (self.P_s - self.P_r) * field_fraction
            P_new = P_magnitude * P_sign

            # Add small memory effect for smoothness
            if abs(E_diff) < 0.05 * self.E_c:
                P_new = 0.7 * P_new + 0.3 * self.P_previous

        # Update memory state
        self.P_previous = P_new
        self.E_previous = E_field

        return P_new
    
    def reset_memory(self):
        """Reset hysteresis memory state."""
        self.P_previous = 0.0
        self.E_previous = 0.0
        self.switching_history = []


# =============================================================================
# HYSTERESIS MODEL (SIMPLIFIED)
# =============================================================================

class HysteresisModel:
    """
    Simplified hysteresis model using tanh switching function.
    Good for quasi-static simulations where full L-K dynamics not needed.
    """
    
    def __init__(self, Ps: float, Pr: float, Ec: float):
        """
        Initialize hysteresis model.
        
        Args:
            Ps: Saturation polarization (μC/cm²)
            Pr: Remnant polarization (μC/cm²)
            Ec: Coercive field (MV/cm)
        """
        self.Ps = Ps * 1e-2   # → C/m²
        self.Pr = Pr * 1e-2   # → C/m²
        self.Ec = Ec * 1e8    # → V/m
        
        # Transition parameters
        self.delta = self.Ec / 3  # Switching transition width
    
    def polarization(self, E: np.ndarray, direction: str = "forward") -> np.ndarray:
        """
        Calculate polarization with proper Pr and Ps behavior.

        Classical ferroelectric hysteresis model:
        - At |E| >> Ec: Saturates to ±Ps (saturation polarization)
        - At E ≈ 0: Remains at ±Pr (remnant polarization)
        - Smooth S-shaped switching with coercive field at ±Ec

        Model: P = Pr·tanh(E/δ₁) + (Ps-Pr)·tanh((E∓Ec)/δ₂)
        - First term: slow reversible component → Pr
        - Second term: fast irreversible switching → Ps
        - Hysteresis from ∓Ec shift in second term

        Args:
            E: Electric field (V/m)
            direction: "forward" (increasing E) or "reverse" (decreasing E)

        Returns:
            Polarization (C/m²)
        """
        # Two-component model for proper Pr/Ps behavior
        delta1 = 3.0 * self.Ec  # Slow component (reversible, gives Pr at E=0)
        delta2 = self.delta      # Fast component (irreversible switching)

        # Reversible component: contributes Pr at moderate fields
        P_reversible = self.Pr * np.tanh(E / delta1)

        # Irreversible component: switched hysteretic part
        if direction == "forward":
            # Forward: switch at +Ec (shift by -Ec)
            E_shift = E - self.Ec
        else:
            # Reverse: switch at -Ec (shift by +Ec)
            E_shift = E + self.Ec

        P_irreversible = (self.Ps - self.Pr) * np.tanh(E_shift / delta2)

        # Total polarization
        P = P_reversible + P_irreversible

        return P
    
    def hysteresis_loop(self, E_max: float, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete hysteresis loop.
        
        Args:
            E_max: Maximum electric field magnitude (V/m)
            n_points: Points per sweep direction
            
        Returns:
            (E_forward, P_forward, E_reverse, P_reverse)
        """
        # Forward sweep: -E_max to +E_max
        E_fwd = np.linspace(-E_max, E_max, n_points)
        P_fwd = self.polarization(E_fwd, "forward")
        
        # Reverse sweep: +E_max to -E_max
        E_rev = np.linspace(E_max, -E_max, n_points)
        P_rev = self.polarization(E_rev, "reverse")
        
        # Combine for complete loop
        E_loop = np.concatenate([E_fwd, E_rev])
        P_loop = np.concatenate([P_fwd, P_rev])
        
        return E_loop, P_loop


# =============================================================================
# POLARIZATION-FIELD COUPLING
# =============================================================================

def coupled_displacement_field(E: np.ndarray, P: np.ndarray, 
                                epsilon_r: float) -> np.ndarray:
    """
    Calculate total displacement field D = ε₀εᵣE + P
    
    Args:
        E: Electric field (V/m)
        P: Polarization (C/m²)
        epsilon_r: Background dielectric constant
        
    Returns:
        Displacement field D (C/m²)
    """
    return EPS0 * epsilon_r * E + P


def bound_charge_density(P: np.ndarray, dz: float) -> np.ndarray:
    """
    Calculate bound charge density from polarization gradient.
    ρ_bound = -∇·P
    
    Args:
        P: Polarization profile (C/m²)
        dz: Spatial step (m)
        
    Returns:
        Bound charge density (C/m³)
    """
    dP_dz = np.gradient(P, dz)
    return -dP_dz


# =============================================================================
# MEMORY WINDOW CALCULATION
# =============================================================================

def calculate_memory_window(Vth_forward: float, Vth_reverse: float) -> float:
    """
    Calculate memory window from threshold voltage shift.
    
    Args:
        Vth_forward: Threshold voltage on forward sweep (V)
        Vth_reverse: Threshold voltage on reverse sweep (V)
        
    Returns:
        Memory window (V)
    """
    return abs(Vth_forward - Vth_reverse)


def estimate_memory_window(Pr: float, t_fe: float, t_il: float, 
                           eps_fe: float, eps_il: float) -> float:
    """
    Analytical estimate of memory window.
    
    MW ≈ 2 × Pr × t_IL / (ε₀ × εᵣ_IL)
    
    Args:
        Pr: Remnant polarization (μC/cm²)
        t_fe: FE thickness (nm)
        t_il: IL thickness (nm)
        eps_fe: FE relative permittivity
        eps_il: IL relative permittivity
        
    Returns:
        Estimated memory window (V)
    """
    Pr_SI = Pr * 1e-2        # → C/m²
    t_il_SI = t_il * 1e-9    # → m
    
    MW = 2 * Pr_SI * t_il_SI / (EPS0 * eps_il)
    return MW
