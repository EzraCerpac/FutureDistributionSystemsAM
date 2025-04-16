module MagnetostaticsFEM

using Gridap
using GridapGmsh
using LinearAlgebra
using WriteVTK

# Include sub-modules
include("mesh_handling.jl")
include("problem_definition.jl")
include("problems.jl") 
include("fem_solver.jl")
include("post_processing.jl")
include("nonlinear_solver.jl")

# Export functions
export load_mesh_and_tags, get_material_tags, get_tag_from_name
export define_reluctivity, define_conductivity, define_current_density
export define_nonlinear_reluctivity, update_reluctivity_from_field
export WeakFormProblem
export magnetostatics_1d_weak_form
export magnetodynamics_1d_harmonic_weak_form
export magnetodynamics_1d_harmonic_coupled_weak_form
export magnetodynamics_2d_harmonic_coupled_weak_form
export setup_fe_spaces, solve_fem_problem
export calculate_b_field, calculate_eddy_current, calculate_b_field_magnitude
export save_results_vtk
export solve_nonlinear_magnetodynamics

end # module MagnetostaticsFEM
