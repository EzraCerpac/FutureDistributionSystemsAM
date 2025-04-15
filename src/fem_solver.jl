using Gridap
using Gridap.FESpaces
include("problems.jl") 

"""
    setup_fe_spaces(model, reffe, dirichlet_tag::String, dirichlet_value)

Sets up the Test and Trial Finite Element spaces with Dirichlet boundary conditions.
"""
function setup_fe_spaces(model, reffe, dirichlet_tag::String, dirichlet_value)
    V = TestFESpace(model, reffe, dirichlet_tags=[dirichlet_tag])
    U = TrialFESpace(V, dirichlet_value)
    return U, V
end

"""
    solve_fem_problem(
        problem::WeakFormProblem, 
        U::SingleFieldFESpace, 
        V::SingleFieldFESpace
    )

Assembles and solves the Finite Element problem defined by the weak form.
"""
function solve_fem_problem(
    problem::WeakFormProblem, 
    U::SingleFieldFESpace, 
    V::SingleFieldFESpace
    )

    # Extract bilinear and linear forms from the problem definition
    a = problem.a
    b = problem.b

    # Create the FE operator
    op = AffineFEOperator(a, b, U, V)

    # Solve the linear FE system
    ls = LUSolver() # Or another appropriate solver
    solver = LinearFESolver(ls)
    solution = solve(solver, op)

    return solution
end
