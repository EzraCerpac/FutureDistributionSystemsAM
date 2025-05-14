module MagnetostaticsFEM

using Gridap
using Gridap.FESpaces # Explicit import for TestFESpace, TrialFESpace, etc.
using Gridap.MultiField # Explicit import for MultiFieldFESpace
using Gridap.CellData # Explicit import for CellField, Measure, Triangulation, Operation
using Gridap.ReferenceFEs # Explicit import for ReferenceFE, lagrangian
using Gridap.Geometry # Explicit import for DiscreteModel
using Gridap.Algebra # For LUSolver, solve, etc.

using WriteVTK # For post_processing usually
using LinearAlgebra
using Serialization # For save/load data

# Include source files
include("problem_definition.jl")
include("problems.jl")
include("fem_solver.jl") 
include("post_processing.jl")
include("mesh_handling.jl")
include("visualisation.jl")

# Export functions to be available for users of the module

# From problem_definition.jl
export define_reluctivity, define_conductivity, define_current_density, get_material_tags

# From problems.jl
export setup_fe_spaces, magnetostatics_weak_form, magnetodynamics_harmonic_coupled_weak_form

# From fem_solver.jl
export solve_fem_problem, get_fem_matrices_and_vector, save_data_serialized, load_data_serialized

# From post_processing.jl
export calculate_b_field, calculate_eddy_current, save_results_vtk

# From mesh_handling.jl
export load_mesh_and_tags

end
