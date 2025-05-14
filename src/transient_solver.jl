module TransientSolver

using Gridap
using Gridap.ODEs # Base ODEs module
using Gridap.ODEs.TransientFETools # For FE specific transient tools in v0.17
using Gridap.ODEs.ODETools # For general ODE solver tools in v0.17

# FESpace, Measure, CellField, FEFunction are assumed to come from `using Gridap`

export setup_transient_operator, solve_transient_problem

"""
    setup_transient_operator(
        Ug_transient::TransientFETools.TransientTrialFESpace, 
        V0_test::FESpace, 
        dΩ::Measure,
        σ_cf::CellField, 
        ν_cf::CellField,
        Js_t_func::Function # Source current Js(x,t) as Js_t_func(t)(x)
    ) -> TransientFETools.TransientFEOperator # Changed to TransientFEOperator

Sets up the transient FE operator for the 1D magnetodynamics problem:
σ ∂Az/∂t + ∇⋅(-ν∇Az) = Js(x,t)
using the more general TransientFEOperator for Gridap v0.17.
"""
function setup_transient_operator(
    Ug_transient::TransientFETools.TransientTrialFESpace, 
    V0_test::FESpace, 
    dΩ::Measure,
    σ_cf::CellField,
    ν_cf::CellField,
    Js_t_func::Function 
)
    # Weak form components from the PDE: σ ∂u/∂t + K u = f(t)
    # m_form_kernel(dut, v) = ∫(σ_cf * v * dut)dΩ
    # a_form_kernel(u, v)   = ∫(ν_cf * ∇(v) ⋅ ∇(u))dΩ
    # l_form_kernel(v, t)   = ∫(v * Js_t_func(t))dΩ

    # Define res(t, u, v) = m(∂t(u),v) + a(u,v) - l(t,v)
    # u is the unknown Az. v is the test function.
    # ∂t(u) is the time derivative of Az.
    
    # Mass term for m(∂t(u),v)
    # Arguments for m are (t, u_t, v) where u_t is time derivative of solution
    m(t, u_t, v) = ∫( σ_cf * v * u_t )*dΩ

    # Stiffness term for a(u,v)
    a(t, u, v) = ∫( ν_cf * (∇(v) ⋅ ∇(u)) )*dΩ

    # Forcing term l(t,v)
    l(t, v) = ∫( v * Js_t_func(t) )*dΩ

    # Residual: res(t,u,v) = a(u,v) + m(∂t(u),v) - l(t,v)
    # Gridap's TransientFEOperator expects res(t,u,v) where u is Tuple (uh, uh_t) for second order ODEs
    # For first order ODE: σ u_t + K u = f  =>  res(t, (u, u_t)) = K u + σ u_t - f
    # The arguments to res for TransientFEOperator are (t, u, v) where u is a TrialFESpace FEFunction.
    # The time derivative is implicitly handled by ∂t(u) if u is from TransientTrialFESpace.
    
    # For TransientFEOperator res(t,u,v), u is the solution FEFunction.
    # We need to use ∂t(u) to represent the time derivative term.
    res(t, u, v) = ∫( σ_cf * v * ∂t(u) + ν_cf * (∇(v) ⋅ ∇(u)) - v * Js_t_func(t) )*dΩ

    # Jacobian of residual with respect to u: ∂res/∂u (du)
    # ∂/∂u ( σ_cf * v * ∂t(u) + ν_cf * (∇(v) ⋅ ∇(u)) - v * Js_t_func(t) ) (du)
    # = ν_cf * ∇(v) ⋅ ∇(du)
    jac_u(t, u, du, v) = ∫( ν_cf * (∇(v) ⋅ ∇(du)) )*dΩ

    # Jacobian of residual with respect to ∂t(u): ∂res/∂(∂t(u)) (du_t)
    # ∂/∂(∂t(u)) ( σ_cf * v * ∂t(u) + ν_cf * (∇(v) ⋅ ∇(u)) - v * Js_t_func(t) ) (du_t)
    # = σ_cf * v * du_t
    jac_ut(t, u, du_t, v) = ∫( σ_cf * v * du_t )*dΩ
    
    op = TransientFETools.TransientFEOperator(res, jac_u, jac_ut, Ug_transient, V0_test)
    return op
end

"""
    solve_transient_problem(
        op::TransientFETools.TransientFEOperator, # Changed to TransientFEOperator
        odesolver::ODETools.ODESolver, 
        t0::Float64, 
        tF::Float64, 
        uh0::FEFunction
    )
"""
function solve_transient_problem(
    op::TransientFETools.TransientFEOperator, 
    odesolver::ODETools.ODESolver,
    t0::Float64, 
    tF::Float64, 
    uh0::FEFunction
)
    # Correct argument order for the 5-argument solve candidate:
    # solve(::ODETools.ODESolver, ::TransientFETools.TransientFEOperator, init_condition::Any, t0::Real, tF::Real)
    solution_iterable = solve(odesolver, op, uh0, t0, tF) 
    return solution_iterable
end

end # module TransientSolver 