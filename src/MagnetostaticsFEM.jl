module MagnetostaticsFEM

using Gridap
using GridapGmsh
using LinearAlgebra
using WriteVTK

# Include sub-modules
include("mesh_handling.jl")
include("problem_definition.jl")
include("problems.jl") # Include the new problems file
include("fem_solver.jl")
include("post_processing.jl")

# Export functions to be used by external scripts/notebooks
export load_mesh_and_tags, get_material_tags
export define_material_properties, define_reluctivity, define_current_density # Added problem_definition exports
export WeakFormProblem, magnetostatics_1d_weak_form # Added problems exports
export setup_fe_spaces, solve_fem_problem # Updated solver exports
export calculate_b_field
export save_results_vtk

end # module MagnetostaticsFEM
