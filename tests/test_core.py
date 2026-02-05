#!/usr/bin/env python3
"""
MIFIS FeFET Simulation Framework
================================
Unit Tests for Core Modules

Run with: pytest tests/test_core.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd


class TestConfig:
    """Tests for configuration module."""
    
    def test_device_geometry_defaults(self):
        from core import DeviceGeometry
        geom = DeviceGeometry()
        
        assert geom.Lg == 100.0
        assert geom.t_fe == 13.8
        assert geom.t_top_il == 4.0
        assert geom.total_height > 0
    
    def test_device_geometry_to_meters(self):
        from core import DeviceGeometry
        geom = DeviceGeometry(Lg=100)
        dims = geom.to_meters()
        
        assert dims['Lg'] == 100e-9
        assert 't_fe' in dims
    
    def test_fe_material_landau_coefficients(self):
        from core import FerroelectricMaterial
        mat = FerroelectricMaterial(Pr=18.0, Ec=1.0)
        
        assert mat.alpha is not None
        assert mat.alpha < 0  # Should be negative for ferroelectric
        assert mat.beta is not None
        assert mat.beta > 0  # Should be positive
    
    def test_get_baseline_config(self):
        from core import get_baseline_config
        config = get_baseline_config()
        
        assert config.fe_material.name == "HZO"
        assert config.geometry.t_fe == 13.8
        assert config.geometry.t_top_il == 4.0


class TestMaterials:
    """Tests for materials database."""
    
    def test_ferroelectric_db_has_hzo(self):
        from core import FERROELECTRIC_DB
        assert "HZO" in FERROELECTRIC_DB
    
    def test_get_fe_material(self):
        from core import get_fe_material
        hzo = get_fe_material("HZO")
        
        assert hzo.Pr == 18.0
        assert hzo.Ps == 38.0
        assert hzo.epsilon_r == 30.0
    
    def test_invalid_material_raises(self):
        from core import get_fe_material
        with pytest.raises(ValueError):
            get_fe_material("InvalidMaterial")
    
    def test_list_all_materials(self):
        from core import list_all_materials
        materials = list_all_materials()
        
        assert "ferroelectrics" in materials
        assert "dielectrics" in materials
        assert len(materials["ferroelectrics"]) > 0


class TestPhysics:
    """Tests for physics models."""
    
    def test_landau_khalatnikov_creation(self):
        from core import LandauKhalatnikovModel, FerroelectricMaterial
        
        mat = FerroelectricMaterial(Pr=18.0, Ps=38.0, Ec=1.0)
        lk = LandauKhalatnikovModel.from_material(mat)
        
        assert lk.alpha < 0
        assert lk.Ps > 0
    
    def test_hysteresis_model(self):
        from core.physics import HysteresisModel
        
        model = HysteresisModel(Ps=38.0, Pr=18.0, Ec=1.0)
        E_loop, P_loop = model.hysteresis_loop(E_max=2e8, n_points=50)
        
        assert len(E_loop) == 100  # 50 forward + 50 reverse
        assert len(P_loop) == 100
    
    def test_estimate_memory_window(self):
        from core import estimate_memory_window
        
        MW = estimate_memory_window(
            Pr=18.0,    # μC/cm²
            t_fe=13.8,  # nm
            t_il=4.0,   # nm
            eps_fe=30.0,
            eps_il=3.9
        )
        
        assert MW > 0
        assert MW < 20  # Reasonable range


class TestSolver:
    """Tests for solver interface."""
    
    def test_solver_creation(self):
        from core import MIFISSolver
        solver = MIFISSolver("test_device")
        
        assert solver.device_name == "test_device"
        assert not solver.device_created
    
    def test_solver_mesh_creation(self):
        from core import MIFISSolver, get_baseline_config
        
        solver = MIFISSolver("test_mesh")
        config = get_baseline_config()
        
        result = solver.create_1d_mesh(config)
        assert result == True
        assert solver.device_created
        assert "ferroelectric" in solver.regions
    
    def test_voltage_sweep(self):
        from core import MIFISSolver, get_baseline_config, SimulationConfig, SweepDirection
        
        solver = MIFISSolver("test_sweep")
        config = get_baseline_config()
        sim_config = SimulationConfig(
            Vg_start=-1.0,
            Vg_end=1.0,
            Vg_step=0.5,
            sweep_direction=SweepDirection.DOUBLE
        )
        
        solver.create_1d_mesh(config)
        solver.setup_physics(config)
        
        results = solver.run_voltage_sweep(config, sim_config)
        
        assert isinstance(results, pd.DataFrame)
        assert "Vg" in results.columns
        assert "Id" in results.columns
        assert len(results) > 0


class TestPostprocess:
    """Tests for post-processing."""
    
    def test_extract_metrics(self):
        from core import extract_metrics, FeFETMetrics
        
        # Create mock data
        df = pd.DataFrame({
            "Vg": np.linspace(-3, 3, 50),
            "Id": np.abs(np.exp(np.linspace(-3, 3, 50))),
            "direction": ["forward"] * 25 + ["reverse"] * 25
        })
        
        metrics = extract_metrics(df)
        
        assert isinstance(metrics, FeFETMetrics)
        assert metrics.Ion > metrics.Ioff
    
    def test_metrics_to_dict(self):
        from core.postprocess import FeFETMetrics
        
        metrics = FeFETMetrics(
            Vth_forward=-0.5,
            Vth_reverse=0.5,
            memory_window=1.0,
            Ion=1e-6,
            Ioff=1e-12,
            Ion_Ioff_ratio=1e6,
            SS_forward=100,
            SS_reverse=110,
            SS_average=105
        )
        
        d = metrics.to_dict()
        assert "Memory_Window_V" in d
        assert d["Memory_Window_V"] == 1.0


class TestVisualization:
    """Tests for visualization module."""
    
    def test_setup_style(self):
        from core import setup_thesis_style
        setup_thesis_style()
        # Should not raise
    
    def test_plot_id_vg(self):
        from core import plot_id_vg
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame({
            "Vg": np.linspace(-3, 3, 20),
            "Id": np.abs(np.exp(np.linspace(-15, -5, 20))),
            "direction": ["forward"] * 10 + ["reverse"] * 10
        })
        
        fig = plot_id_vg(df, label="Test")
        
        assert fig is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
