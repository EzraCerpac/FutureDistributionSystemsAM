using Gridap
using Gridap.FESpaces
using Gridap.MultiField
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
