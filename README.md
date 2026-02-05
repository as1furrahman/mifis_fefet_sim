# MIFIS FeFET Simulation Framework

A comprehensive simulation framework for Metal-Insulator-Ferroelectric-Insulator-Semiconductor (MIFIS) Field-Effect Transistors using Hf₀.₅Zr₀.₅O₂ (HZO) ferroelectric material.

## Features

- **1D Baseline**: Vertical stack simulation (MW: 4.028V)
- **2D Planar**: Fringing field effects (MW: 4.229V)
- **3D Gate-All-Around (GAA)**: Full 360° gate wrap (MW: 5.182V)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulations
python main.py 1d      # 1D baseline
python main.py 2d      # 2D planar
python main.py 3d      # 3D GAA
python main.py 1d2d    # 1D + 2D combined
python main.py all     # All dimensions
```

## Results

All simulations generate:
- Hysteresis loops (P-E and P-V)
- Memory window analysis
- Performance metrics
- Publication-quality plots (300 DPI)

Results saved in `results/` and `plots/` directories.

## Physics Model

- **Ferroelectric**: Landau-Khalatnikov dynamics
- **Material**: HZO (Pr: 18 µC/cm², Ps: 38 µC/cm², Ec: 0.5 MV/cm)
- **Solver**: DEVSIM TCAD + Pure Python fallback

## Architecture

```
mifis_fefet_sim/
├── core/           # Core physics and solvers
├── simulations/    # Dimension-specific runners
├── utils/          # Plotting and analysis
├── config/         # Device parameters
├── results/        # Output data (CSV, PKL)
└── plots/          # Generated figures
```

## License

MIT
