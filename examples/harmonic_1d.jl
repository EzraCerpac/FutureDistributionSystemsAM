# Transient 1D Magnetodynamics Example

# %% Setup
include(joinpath(dirname(@__DIR__), "config.jl"))
paths = get_project_paths(@__DIR__) # For OUT_DIR, GEO_DIR etc.
GEO_DIR = paths["GEO_DIR"]
OUT_DIR = paths["OUT_DIR"] # Ensure OUT_DIR is defined here
include("../src/MagnetostaticsFEM.jl")

using LinearAlgebra
using Plots
using LaTeXStrings
using Gridap
using .MagnetostaticsFEM
using Printf

println("--- Starting Transient 1D Magnetodynamics Example ---")

# %% Parameters
# --- Model Parameters ---
J0_amplitude = 2.2e4  # Source current density amplitude [A/m²]
μ0 = 4e-7 * pi     # Vacuum permeability [H/m]
μr_core = 50000.0    # Relative permeability of the core (linear assumption for transient)
σ_core = 1e7       # Conductivity of the core [S/m]
freq = 50.0          # Frequency of the source current [Hz]
ω_source = 2 * pi * freq # Angular frequency [rad/s]

# --- FEM Parameters ---
order_fem = 2
field_type = ComplexF64
dirichlet_tag = "D"
dirichlet_value = 0.0 + 0.0im # Dirichlet BC for A = u + iv

mesh_file = joinpath(GEO_DIR, "coil_geo.msh")
# --- Output Parameters ---
output_dir = joinpath(OUT_DIR, "harmonic_1d_results_jl")
if !isdir(output_dir)   
    mkdir(output_dir)
end

println("Mesh file: ", mesh_file)
println("Output PVD base: ", output_dir)

# %% Call the new preparation and solving function from TransientSolver
solution_harmonic, Az0_out, Ω_out, ν_func_map, σ_func_map, tags = 
    prepare_and_solve_harmonic_1d(
        mesh_file,
        order_fem, 
        field_type,
        dirichlet_tag,
        dirichlet_value,
        μ0,
        μr_core,
        σ_core,
        J0_amplitude,
        ω_source
    )

# %% Post-processing:
Az_mag, B_mag, J_eddy_mag, Az_re, Az_im, B_re, B_im, J_eddy_re, J_eddy_im, ν_field_linear = process_harmonic_solution(solution_harmonic, Ω_out, ν_func_map, σ_func_map, ω_source, tags)

save_results_vtk(Ω_out, output_dir, 
    Dict(
        "Az_re" => Az_re, "Az_im" => Az_im, "Az_mag" => Az_mag,
        "B_re" => B_re, "B_im" => B_im, "B_mag" => B_mag,
        "J_eddy_re" => J_eddy_re, "J_eddy_im" => J_eddy_im, "J_eddy_mag" => J_eddy_mag,
        "ν_linear" => ν_field_linear
    )
)


plot_harmonic_magnitude_1d(Az_mag, B_mag, J_eddy_mag, ν_field_linear; output_path=output_dir)

plot_harmonic_animation_1d(Az_re, Az_im, Az_mag, B_re, B_im, B_mag, J_eddy_re, J_eddy_im, J_eddy_mag, ν_field_linear, ω_source; output_path=output_dir)
