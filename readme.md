# Future Distribution Systems Advanced Modeling

## Introduction

This repository contains the work for "Future Distribution Grids: A Case Study of Modeling the Magnetic Field, the Thermal Field and the Electrical Circuit Parameters in a Power Transformer" - a project developed in collaboration with TU Delft and STEDIN Rotterdam.

The project focuses on modeling power distribution transformers to address the challenges of modern electrical distribution grids. With the increasing integration of renewable energy sources and electric vehicle charging, these grids face new operational challenges including higher frequency harmonics that affect transformer performance and lifespan.

## Project Overview

The operation of electrical distribution grids is rapidly evolving due to:
- Integration of renewable energy sources (e.g., solar panels)
- New loading patterns from battery charging (e.g., electric vehicles)
- Introduction of higher frequency harmonics in the system

This project aims to develop comprehensive models for analyzing power transformers, focusing on:
1. Electromagnetic field analysis
2. Core loss estimation
3. Copper loss calculation
4. Thermal behavior modeling
5. Transformer aging/lifetime prediction

Our models help chart the amplitude, frequency content, and location of electromagnetic losses, which is essential for monitoring the life-time expectancy of assets in the distribution grid.

## Repository Structure

The primary focus of this project is on modeling distribution transformers, with the assignment and starting code located in:

```
modeling_distribution_transformer/
```

## Getting Started & Development Setup

### Prerequisites
- Julia programming environment
- Python with Jupyter Notebook support
- GMSH for mesh generation
- Paraview for visualization

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/EzraCerpac/FutureDistributionSystemsAM.git
   cd FutureDistributionSystemsAM
   ```

2. Install required Julia packages using the Project.toml file:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

3. For mesh generation, install GMSH: https://gmsh.info/

## Key Components/Models

Our project consists of five interconnected sub-models:

1. **Magnetic Field-Circuit Coupled Model**: A 2D quasi-stationary electromagnetic field model coupled with an electrical circuit model for voltage-driven conductors.

2. **Core Loss Model**: Calculates ferromagnetic core losses due to eddy current effects.

3. **Copper Loss Model**: Computes losses in primary and secondary windings.

4. **Thermal Model**: A 3D temperature model that accounts for diffusion, convection, and energy conservation.

5. **Aging/Lifetime Model**: Predicts degradation of transformer components over time.

The implementation focuses on:
- Finite element modeling using Julia (Gridap, Ferrite)
- Parametric geometry generation with GMSH
- Time-harmonic and transient computations
- Linear and non-linear magnetic material properties

## Credits and References

This work builds upon previous academic research:

- Master thesis by Max van Dijk at TU Delft
- Previous implementations by Gijs Lagerweij
- Work by Philip Soliman and Auke Schaap on fast finite element assembly
- Extensions by Rahul Rane

### Project Supervision
- Domenico Lahaye (TU Delft)
- Jeroen Schuddebeurs (STEDIN Rotterdam)

### Key References
- [Max van Dijk's Master Thesis](https://repository.tudelft.nl/islandora/object/uuid%3A15b25b42-e04b-4ff2-a187-773bc170f061?collection=education)
