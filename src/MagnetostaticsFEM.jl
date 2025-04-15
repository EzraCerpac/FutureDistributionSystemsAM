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

# Export functions to be used by external scripts/notebooks
export load_mesh_and_tags, get_material_tags
export define_reluctivity, define_current_density, define_conductivity 
export WeakFormProblem, magnetostatics_1d_weak_form, magnetodynamics_1d_harmonic_coupled_weak_form 
export setup_fe_spaces, solve_fem_problem 
export calculate_b_field, calculate_eddy_current 
export save_results_vtk

end # module MagnetostaticsFEM
