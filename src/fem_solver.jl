using Gridap
using Gridap.FESpaces
using Gridap.MultiField
using LinearAlgebra
using WriteVTK
using Serialization
include("problems.jl")

"""
    setup_fe_spaces(model, order::Int, field_type::Type, dirichlet_tag::String, dirichlet_value)

Sets up the Test and Trial Finite Element spaces with Dirichlet boundary conditions.
Handles single-field (Real) or multi-field (for complex split) cases.
"""
function setup_fe_spaces(model, order::Int, field_type::Type, dirichlet_tag::String, dirichlet_value)
    if field_type == ComplexF64
        reffe = ReferenceFE(lagrangian, Float64, order)
        V_re = TestFESpace(model, reffe; dirichlet_tags=[dirichlet_tag])
        V_im = TestFESpace(model, reffe; dirichlet_tags=[dirichlet_tag])
        V = MultiFieldFESpace([V_re, V_im])
        u_D = real(field_type(dirichlet_value))
        v_D = imag(field_type(dirichlet_value))
        U_re = TrialFESpace(V_re, u_D)
        U_im = TrialFESpace(V_im, v_D)
        U = MultiFieldFESpace([U_re, U_im])
        return U, V
    elseif field_type <: Real
        reffe = ReferenceFE(lagrangian, field_type, order)
        V = TestFESpace(model, reffe; dirichlet_tags=[dirichlet_tag])
        U = TrialFESpace(V, field_type(dirichlet_value)) 
        return U, V
    else
        error("Unsupported field_type for setup_fe_spaces: $field_type")
    end
end


"""
    solve_fem_problem(
        problem::WeakFormProblem,
        U, # Can be SingleFieldFESpace or MultiFieldFESpace
        V  # Can be SingleFieldFESpace or MultiFieldFESpace
    )

Assembles and solves the Finite Element problem defined by the weak form.
Handles both single-field and multi-field problems.
"""
function solve_fem_problem(problem::WeakFormProblem, U, V)
    a = problem.a
    b = problem.b
    op = AffineFEOperator(a, b, U, V)
    ls = LUSolver()
    solver = LinearFESolver(ls)
    solution = solve(solver, op)
    return solution
end

"""
    extract_solution_values(solution)

Extracts values from a FE solution and returns them as matrix/matrices.
Handles both real and complex solutions.
"""
function extract_solution_values(solution)
    if isa(solution, MultiFieldFEFunction)
        u_field = solution[1]
        v_field = solution[2]
        u_cell = get_cell_dof_values(u_field)
        v_cell = get_cell_dof_values(v_field)

        u = convert_to_matrix(u_cell)
        v = convert_to_matrix(v_cell)
        return u + 1im * v
    else  # Real-valued solution case
        u_cell = get_cell_dof_values(solution)
        u = convert_to_matrix(u_cell)
        return u
    end
end

function convert_to_matrix(cell_values)
    values = vcat(collect.(cell_values)...)
"""
    convert_to_matrix(cell_values)

Helper function to convert cell values to a matrix format.
"""
    return values
end

# Function to solve a linear FE problem
# The following solve_fem_problem is the one intended to be used with FEOperators
function solve_fem_problem(problem::FEOperator, U, V) # U and V are FESpaces
    # Test if problem is an AffineFEOperator, which directly gives matrix and vector
    if isa(problem, AffineFEOperator)
        A = get_matrix(problem)
        b = get_vector(problem)
        # Gridap's default solve for matrix and vector
        # This returns a vector of free DoF values
        free_values = solve(A,b) 
        # Create a FEFunction from the free values and the trial FE space V
        # (Note: transient_1d.jl currently uses U as the TrialFESpace in FEFunction)
        # For MultiField, V is often the TestFESpace from which U (Trial) is derived,
        # but FEFunction needs the Trial space. The U passed here is the TrialFESpace.
        return FEFunction(U,free_values)
    elseif isa(problem,FEOperator) # General FEOperator
      solver = LUSolver() 
      @warn "Problem is a general FEOperator, not AffineFEOperator. Solving with LUSolver."
      # This path might need an initial guess if non-linear
      # x0 = zero_initial_guess(problem) 
      # return solve(solver,problem,x0)
      return solve(solver,problem)
    else
        error("Unsupported problem type: ", typeof(problem))
    end
end

function get_fem_matrices_and_vector(
    model::DiscreteModel,
    order::Int,
    dirichlet_tag::String,
    dirichlet_value_scalar::Float64,
    Ω::Triangulation,
    dΩ::Measure,
    ν_cellfield::CellField,
    σ_cellfield::CellField,
    Js_re_cellfield::CellField
)
    reffe_scalar = ReferenceFE(lagrangian, Float64, order)
    V_scalar_test = TestFESpace(model, reffe_scalar, conformity=:H1, dirichlet_tags=[dirichlet_tag])
    U_scalar_trial = TrialFESpace(V_scalar_test, dirichlet_value_scalar)
    a_S_form(u,v) = ∫( ν_cellfield * (∇(u) ⋅ ∇(v)) ) * dΩ
    S_matrix = assemble_matrix(a_S_form, U_scalar_trial, V_scalar_test)
    a_M_form(u,v) = ∫( σ_cellfield * u ⋅ v ) * dΩ
    M_matrix = assemble_matrix(a_M_form, U_scalar_trial, V_scalar_test)
    l_f_re_form(v) = ∫( Js_re_cellfield * v ) * dΩ
    f_re_vector = assemble_vector(l_f_re_form, V_scalar_test)
    return S_matrix, M_matrix, f_re_vector, U_scalar_trial
end

function save_data_serialized(filepath::String, data)
    mkpath(dirname(filepath))
    open(filepath, "w") do f
        serialize(f, data)
    end
end

function load_data_serialized(filepath::String)
    local data
    open(filepath, "r") do f
        data = deserialize(f)
    end
    return data
end

function prepare_and_solve_harmonic_1d(mesh_file::String, order::Int, field_type::Type, dirichlet_tag::String, dirichlet_value::ComplexF64, μ0::Float64, μr_core::Float64, σ_core::Float64, J0_amplitude::Float64, ω_source::Float64)
    model, labels, tags = load_mesh_and_tags(mesh_file)
    material_tags = get_material_tags(labels)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*order)
    ν_func_map = define_reluctivity(material_tags, μ0, μr_core)
    σ_func_map = define_conductivity(material_tags, σ_core)
    source_current_func = define_current_density(material_tags, J0_amplitude)
    U, V = setup_fe_spaces(model, order, field_type, dirichlet_tag, dirichlet_value)
    problem = magnetodynamics_harmonic_coupled_weak_form(Ω, dΩ, tags, ν_func_map, σ_func_map, source_current_func, ω_source)
    solution = solve_fem_problem(problem, U, V)
    Az0 = solution[1]
    return solution, Az0, Ω, ν_func_map, σ_func_map, tags
end
