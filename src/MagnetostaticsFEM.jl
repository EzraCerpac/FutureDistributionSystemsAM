module MagnetostaticsFEM

using Gridap
using Gridap.FESpaces # Explicit import for TestFESpace, TrialFESpace, etc.
using Gridap.MultiField # Explicit import for MultiFieldFESpace
using Gridap.CellData # Explicit import for CellField, Measure, Triangulation, Operation
using Gridap.ReferenceFEs # Explicit import for ReferenceFE, lagrangian
using Gridap.Geometry # Explicit import for DiscreteModel
using Gridap.Algebra # For LUSolver, solve, etc.
using Gridap.ODEs # Base ODEs module
# Specific imports for Gridap v0.17 structure
using Gridap.ODEs.ODETools: ThetaMethod, ODESolver # ODESolver for potential internal use or re-export if needed
using Gridap.ODEs.TransientFETools: TransientFEOperator, TransientTrialFESpace # And other necessary types from here

using WriteVTK # For post_processing usually (though Gridap.writevtk is preferred)
using LinearAlgebra
using Serialization # For save/load data
using FFTW # For signal processing
using Plots # For visualisation module, and potentially direct use
using LaTeXStrings # For plot labels

# Include source files
include("problem_definition.jl")
include("problems.jl")
include("fem_solver.jl") 
include("post_processing.jl")
include("mesh_handling.jl")
include("transient_solver.jl")   # New
include("nonlinear_solver.jl")  # Added nonlinear_solver.jl
include("signal_processing.jl")  # New
include("comparison_utils.jl")   # New
include("visualisation.jl")      # Expanded

# Export functions to be available for users of the module

# From problem_definition.jl
export define_reluctivity, define_conductivity, define_current_density, get_material_tags

# From problems.jl
export setup_fe_spaces, magnetostatics_weak_form, magnetodynamics_harmonic_coupled_weak_form
# For Gridap v0.17, TransientTrialFESpace is likely from Gridap.ODEs.TransientFETools
export TransientTrialFESpace # Now imported from Gridap.ODEs.TransientFETools

# From fem_solver.jl
export solve_fem_problem, get_fem_matrices_and_vector, save_data_serialized, load_data_serialized, prepare_and_solve_harmonic_1d

# From post_processing.jl
using .PostProcessing
export calculate_b_field, calculate_eddy_current, save_results_vtk, save_pvd_and_extract_signal, save_transient_pvd, process_harmonic_solution, calculate_b_magnitude_from_az_transient # Added calculate_b_magnitude_from_az_transient

# From mesh_handling.jl
export load_mesh_and_tags

# From transient_solver.jl
using .TransientSolver # This module uses qualified names internally like TransientFETools.TypeName
export setup_transient_operator, solve_transient_problem, prepare_and_solve_transient_1d

# For Gridap v0.17, ThetaMethod is likely from Gridap.ODEs.ODETools
export ThetaMethod # Now imported from Gridap.ODEs.ODETools

# From signal_processing.jl
using .SignalProcessing
export perform_fft, select_periodic_window, perform_fft_periodic

# From comparison_utils.jl
# using .ComparisonUtils (no exports yet from placeholder)

# From visualisation.jl
using .Visualisation
export plot_contour_2d, create_field_animation, plot_time_signal, plot_fft_spectrum, plot_line_1d, create_transient_animation, plot_harmonic_magnitude_1d, plot_harmonic_animation_1d

end
