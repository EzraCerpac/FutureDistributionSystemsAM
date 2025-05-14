using Gridap
using Gridap.FESpaces
using Gridap.MultiField
using LinearAlgebra
using WriteVTK # For existing solve_fem_problem
using Serialization # For new save/load functions
include("problems.jl") 

"""
    setup_fe_spaces(model, order::Int, field_type::Type, dirichlet_tag::String, dirichlet_value)

Sets up the Test and Trial Finite Element spaces with Dirichlet boundary conditions.
Handles single-field (Real) or multi-field (for complex split) cases.
"""
function setup_fe_spaces(model, order::Int, field_type::Type, dirichlet_tag::String, dirichlet_value)
    if field_type == ComplexF64
        # Multi-field setup for u + iv
        println("Setting up multi-field spaces (Real, Imag) for Complex problem.")
        reffe = ReferenceFE(lagrangian, Float64, order)
        
        # Define test spaces for real and imaginary parts
        V_re = TestFESpace(model, reffe; dirichlet_tags=[dirichlet_tag])
        V_im = TestFESpace(model, reffe; dirichlet_tags=[dirichlet_tag])
        V = MultiFieldFESpace([V_re, V_im])

        # Define trial spaces with potentially complex Dirichlet values split
        u_D = real(field_type(dirichlet_value)) # Real part of BC
        v_D = imag(field_type(dirichlet_value)) # Imag part of BC
        U_re = TrialFESpace(V_re, u_D)
        U_im = TrialFESpace(V_im, v_D)
        U = MultiFieldFESpace([U_re, U_im])
        
        return U, V
    elseif field_type <: Real
         # Single-field setup
        println("Setting up single-field space.")
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
        # Complex solution case - extract real and imaginary parts
        u_field = solution[1]  # Real part
        v_field = solution[2]  # Imaginary part
        
        # Convert to cell arrays
        u_cell = get_cell_dof_values(u_field)
        v_cell = get_cell_dof_values(v_field)
        
        # Convert to matrices
        u = convert_to_matrix(u_cell)
        v = convert_to_matrix(v_cell)
        
        return u + 1im * v  # Combine real and imaginary parts
    else
        # Real-valued solution case
        u_cell = get_cell_dof_values(solution)
        u = convert_to_matrix(u_cell)
        return u
    end
end

"""
    convert_to_matrix(cell_values)

Helper function to convert cell values to a matrix format.
"""
function convert_to_matrix(cell_values)
    # Flatten the cell array into a vector
    values = vcat(collect.(cell_values)...)
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


"""
    get_fem_matrices_and_vector(
        model::DiscreteModel,
        order::Int,
        dirichlet_tag::String,
        dirichlet_value_scalar::Float64, # Scalar real Dirichlet value for the base space
        Ω::Triangulation,
        dΩ::Measure,
        ν_cellfield::CellField, # Reluctivity as a CellField
        σ_cellfield::CellField, # Conductivity as a CellField
        Js_re_cellfield::CellField # Real part of source current as a CellField
    )

Assembles the stiffness matrix S, mass matrix M, and real part of the load vector f_re
for the complex equation (S + jωM)A = Js, where A is complex and Js is assumed real here.

S_ij = ∫( ν * ∇φ_j ⋅ ∇φ_i ) dΩ
M_ij = ∫( σ * φ_j ⋅ φ_i ) dΩ
f_re_i = ∫( J_s_real ⋅ v ) dΩ

Returns S_matrix, M_matrix, f_re_vector, and the trial FE space U_scalar_trial.
"""
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
    # Test space defines where Dirichlet conditions are applied (on entities with dirichlet_tag)
    V_scalar_test = TestFESpace(model, reffe_scalar, conformity=:H1, dirichlet_tags=[dirichlet_tag])
    # Trial space applies the actual dirichlet_value_scalar to the DOFs identified by V_scalar_test
    U_scalar_trial = TrialFESpace(V_scalar_test, dirichlet_value_scalar)

    # Stiffness matrix S: ∫( ν * ∇u ⋅ ∇v ) dΩ
    a_S_form(u,v) = ∫( ν_cellfield * (∇(u) ⋅ ∇(v)) ) * dΩ
    S_matrix = assemble_matrix(a_S_form, U_scalar_trial, V_scalar_test)

    # Mass matrix M: ∫( σ * u ⋅ v ) dΩ
    a_M_form(u,v) = ∫( σ_cellfield * u ⋅ v ) * dΩ
    M_matrix = assemble_matrix(a_M_form, U_scalar_trial, V_scalar_test)

    # Load vector f_re: ∫( J_s_real ⋅ v ) dΩ
    l_f_re_form(v) = ∫( Js_re_cellfield * v ) * dΩ
    f_re_vector = assemble_vector(l_f_re_form, V_scalar_test)
    
    return S_matrix, M_matrix, f_re_vector, U_scalar_trial
end

"""
    save_data_serialized(filepath::String, data)

Serializes and saves data to the specified filepath.
Creates the directory if it does not exist.
"""
function save_data_serialized(filepath::String, data)
    mkpath(dirname(filepath)) # Ensure directory exists
    open(filepath, "w") do f
        serialize(f, data)
    end
    println("Saved data to $(filepath)")
end

"""
    load_data_serialized(filepath::String)

Loads and deserializes data from the specified filepath.
"""
function load_data_serialized(filepath::String)
    local data
    open(filepath, "r") do f
        data = deserialize(f)
    end
    println("Loaded data from $(filepath)")
    return data
end
