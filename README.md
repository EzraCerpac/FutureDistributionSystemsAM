# FutureDistributionSystemsAM

Julia-based finite element method (FEM) package for electromagnetics simulations in power distribution transformers.

## Overview

Research project from TU Delft focused on modeling magnetic fields and thermal behavior in transformers for modern distribution grids.

## Quick Start

```bash
# Setup environment
julia --project=.
julia -e "using Pkg; Pkg.instantiate()"

# Run examples
julia --project=. examples/1D-Harmonic.jl
```

## Key Features

- Magnetostatics, time-harmonic magnetodynamics, transient analysis
- Linear and nonlinear material properties
- Coupled electromagnetic-thermal modeling
- VTK output for Paraview visualization

## Dependencies

- Julia 1.6+
- GMSH for mesh generation
- Paraview for visualization
