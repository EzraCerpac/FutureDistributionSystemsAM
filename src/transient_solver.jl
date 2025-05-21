module TransientSolver

using Gridap
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using LinearAlgebra

using ..MagnetostaticsFEM: load_mesh_and_tags, get_material_tags, define_reluctivity, define_conductivity, define_current_density


export setup_transient_operator, solve_transient_problem
export prepare_and_solve_transient_1d # New export

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

"""
    prepare_and_solve_transient_1d(
        mesh_file::String,
        order::Int,
        dirichlet_tag::String,
        dirichlet_bc_function::Function, # Single BC function with multiple dispatch methods
         # Module like MagnetostaticsFEM
        μ0::Float64,
        μr_core::Float64,
        σ_core::Float64,
        J0_amplitude::Float64,
        ω_source::Float64,
        t0::Float64,
        tF::Float64,
        Δt::Float64,
        θ_method::Float64
    ) -> Tuple{Any, FEFunction, Triangulation, CellField, CellField, Function}

Prepares the FE model, sets up, and solves the transient 1D magnetodynamics problem.
Returns the solution iterable, initial condition Az0, Triangulation Ω,
CellFields ν_cf and σ_cf, and the time-dependent source function Js_t_func.
This function encapsulates the setup and solve steps from `examples/transient_1d.jl`.
"""
function prepare_and_solve_transient_1d(
    mesh_file::String,
    order_fem::Int, # Renamed from order to avoid conflict with reffe
    dirichlet_tag::String,
    dirichlet_bc_function::Function, # Single BC function with multiple dispatch methods
    μ0::Float64,
    μr_core::Float64,
    σ_core::Float64,
    J0_amplitude::Float64,
    ω_source::Float64,
    t0::Float64,
    tF::Float64,
    Δt::Float64,
    θ_method::Float64
)
    println("--- Preparing Transient 1D Simulation ---")
    println("Loading mesh and defining domains/materials...")
    model, labels, tags = load_mesh_and_tags(mesh_file)
    Ω = Triangulation(model)
    degree_quad = 2*order_fem # Quadrature degree
    dΩ = Measure(Ω, degree_quad)

    material_tags_dict = get_material_tags(labels)
    ν_func_map = define_reluctivity(material_tags_dict, μ0, μr_core)
    σ_func_map = define_conductivity(material_tags_dict, σ_core)

    cell_tags_cf = CellField(tags, Ω) # Helper CellField of tags
    ν_cf = Operation(ν_func_map)(cell_tags_cf)
    σ_cf = Operation(σ_func_map)(cell_tags_cf)

    spatial_js_profile_func = define_current_density(material_tags_dict, J0_amplitude)
    # Js_t_func now uses the dirichlet_bc_function style for consistency if needed,
    # though for current it's Js_t_func(t)(x)
    Js_t_func(t) = x -> spatial_js_profile_func(cell_tags_cf(x)) * sin(ω_source * t)

    println("Setting up transient FE spaces...")
    reffe = ReferenceFE(lagrangian, Float64, order_fem)
    V0_test = TestFESpace(model, reffe, dirichlet_tags=[dirichlet_tag])
    # Pass the single, multi-dispatch BC function
    Ug_transient = TransientTrialFESpace(V0_test, dirichlet_bc_function) 

    println("Defining initial condition...")
    Az0 = zero(Ug_transient(t0)) # Uses Ug_transient(t0)

    println("Setting up transient operator...")
    # The Js_t_func is passed directly to the operator
    transient_op = setup_transient_operator(Ug_transient, V0_test, dΩ, σ_cf, ν_cf, Js_t_func)

    println("Setting up ODE solver...")
    linear_solver_for_ode = LUSolver()
    odesolver = ThetaMethod(linear_solver_for_ode, Δt, θ_method)

    println("Solving transient problem from t=$(t0) to t=$(tF) with Δt=$(Δt)...")
    solution_transient_iterable = solve_transient_problem(transient_op, odesolver, t0, tF, Az0)
    println("Transient solution obtained (iterable).")

    return solution_transient_iterable, Az0, Ω, ν_cf, σ_cf, Js_t_func, model, cell_tags_cf, labels
end

end # module TransientSolver 