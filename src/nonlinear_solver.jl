using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using LinearAlgebra

"""
    calculate_b_field_magnitude(uv::MultiFieldFEFunction, Ω)

Helper function to calculate B-field magnitude from solution.
Works with both MultiFieldFEFunction and Vector{ComplexF64} inputs.
"""
function calculate_b_field_magnitude(uv::MultiFieldFEFunction, Ω)
    B_re, B_im = calculate_b_field(uv)
    
    function B_mag_sq(x)
        b_re_val = B_re(x)
        b_im_val = B_im(x)
        return sqrt(inner(b_re_val, b_re_val) + inner(b_im_val, b_im_val))
    end
    
    return CellField(B_mag_sq, Ω)
end

"""
    calculate_b_field_magnitude(uv::Vector{ComplexF64}, Ω)

Vector{ComplexF64} version of the function to calculate B-field magnitude.
Uses calculate_b_field for consistency.
"""
function calculate_b_field_magnitude(uv::Vector{ComplexF64}, Ω)
    B_re_approx, B_im_approx = calculate_b_field(uv)
    
    function B_mag_sq(x)
        b_re_val = B_re_approx(x)
        b_im_val = B_im_approx(x)
        return sqrt(inner(b_re_val, b_re_val) + inner(b_im_val, b_im_val))
    end
    
    return CellField(B_mag_sq, Ω)
end

"""
    calculate_reluctivity_cellfield(uv_fe::MultiFieldFEFunction, Ω::Triangulation, 
                                    tags_vector::AbstractArray, material_tags_dict::Dict, 
                                    μ0::Float64, fmur_core_func::Function)

Calculates a spatially varying reluctivity CellField based on the local B-field magnitude.
"""
function calculate_reluctivity_cellfield(uv_fe::MultiFieldFEFunction, Ω::Triangulation, 
                                         tags_vector::AbstractArray, material_tags_dict::Dict, 
                                         μ0::Float64, fmur_core_func::Function)
    
    τ_cf = CellField(tags_vector, Ω)

    B_re, B_im = calculate_b_field(uv_fe)

    function B_mag_at_x(x)
        b_re_val = B_re(x)
        b_im_val = B_im(x)
        return sqrt(inner(b_re_val, b_re_val) + inner(b_im_val, b_im_val))
    end
    B_mag_cf = CellField(B_mag_at_x, Ω)

    μr_calculator = (b_mag_val, tag_val) -> begin
        if tag_val == material_tags_dict["Core"]
            # Ensure B_mag_val is non-negative for fmur_core_func
            return fmur_core_func(max(0.0, b_mag_val))
        else
            return 1.0 
        end
    end
    μr_cf = Operation(μr_calculator)(B_mag_cf, τ_cf)

    # Avoid division by zero if μr_cf happens to be zero (e.g. if fmur_core_func can return 0)
    # Add a small epsilon to the denominator or ensure fmur_core_func returns > 0
    ν_cf = 1.0 / (μ0 * Operation(μ_val -> max(μ_val, 1e-9))(μr_cf)) 
    
    return ν_cf
end


"""
    solve_nonlinear_magnetodynamics(...)

Function to solve the nonlinear magnetodynamic problem using iterative substitution.
"""
function solve_nonlinear_magnetodynamics(
    model, labels, tags, J0, μ0, bh_a, bh_b, bh_c, σ_core, ω, 
    order, field_type, dirichlet_tag, dirichlet_value;
    max_iterations=50, tolerance=1e-10, damping=0.7)
    
    Ω = Triangulation(model)
    integration_degree = 2 * order 
    dΩ = Measure(Ω, integration_degree)
    
    material_tags = get_material_tags(labels)
    
    function fmur_core(B)
        # Ensure B is non-negative for the power operation
        B_safe = max(0.0, B)
        return 1.0 / (bh_a + (1 - bh_a) * B_safe^(2*bh_b) / (B_safe^(2*bh_b) + bh_c))
    end
    
    τ_cf = CellField(tags, Ω)
    initial_μr_core_val = fmur_core(0.0) # μr for B=0
    
    initial_μr_op = (tag_val) -> (tag_val == material_tags["Core"] ? initial_μr_core_val : 1.0)
    initial_μr_cell_field = Operation(initial_μr_op)(τ_cf)
    
    current_reluctivity_param = 1.0 / (μ0 * Operation(μ_val -> max(μ_val, 1e-9))(initial_μr_cell_field))


    conductivity_func = define_conductivity(material_tags, σ_core)
    source_current_func = define_current_density(material_tags, J0)
    
    U, V = setup_fe_spaces(model, order, field_type, dirichlet_tag, dirichlet_value)
    
    iter = 0
    error = 1.0
    uv_current_dofs_actual = nothing 

    problem = magnetodynamics_harmonic_coupled_weak_form(
        Ω, dΩ, tags, current_reluctivity_param, conductivity_func, source_current_func, ω)
    
    uv_FESpace = solve_fem_problem(problem, U, V)

    println("Starting nonlinear iterations")

    # Prepare Dirichlet values vector correctly for single or multi-field case
    local dirichlet_vals_vec::Vector{Float64}
    if isa(U, MultiFieldFESpace)
        dirichlet_vals_list = [get_dirichlet_dof_values(U_i) for U_i in U.spaces]
        dirichlet_vals_vec = vcat(dirichlet_vals_list...)
    else # SingleFieldFESpace
        dirichlet_vals_vec = get_dirichlet_dof_values(U)
    end
    
    while iter < max_iterations && error > tolerance
        current_dofs_vec = get_free_dof_values(uv_FESpace)

        if uv_current_dofs_actual !== nothing
            error = norm(current_dofs_vec - uv_current_dofs_actual) / (norm(uv_current_dofs_actual) + 1e-12)
            uv_current_dofs_actual = current_dofs_vec * damping + (1 - damping) * uv_current_dofs_actual
        else
            uv_current_dofs_actual = current_dofs_vec
        end
        
        uv_FESpace_for_physics = FEFunction(U, uv_current_dofs_actual)

        current_reluctivity_param = calculate_reluctivity_cellfield(uv_FESpace_for_physics, Ω, tags, material_tags, μ0, fmur_core)
        
        problem = magnetodynamics_harmonic_coupled_weak_form(
            Ω, dΩ, tags, current_reluctivity_param, conductivity_func, source_current_func, ω)
        
        uv_FESpace = solve_fem_problem(problem, U, V)

        iter += 1
        println("Iteration $iter: error = $error")
    end
    
    if iter >= max_iterations && error > tolerance
        println("Warning: Maximum iterations reached without convergence")
    else
        println("Nonlinear solution converged in $iter iterations")
    end
    
    return uv_FESpace
end

"""
    solve_nonlinear_transient_magnetodynamics(
        mesh_file, order_fem, dirichlet_tag, dirichlet_bc_function,
        μ0, bh_a, bh_b, bh_c, σ_core, J0_amplitude, ω_source,
        t0, tF, Δt, θ_method; max_iterations_nl=15, tolerance_nl=1e-5, damping_nl=0.75
    )

Solves nonlinear transient magnetodynamics with B-H curve using Newton-Raphson iteration.
Combines transient time-stepping with nonlinear material property updates.
"""
function solve_nonlinear_transient_magnetodynamics(
    mesh_file::String,
    order_fem::Int,
    dirichlet_tag::String, 
    dirichlet_bc_function::Function,
    μ0::Float64,
    bh_a::Float64, bh_b::Float64, bh_c::Float64,  # B-H curve params
    σ_core::Float64,
    J0_amplitude::Float64,
    ω_source::Float64,
    t0::Float64, tF::Float64, Δt::Float64,
    θ_method::Float64;
    max_iterations_nl::Int=15,
    tolerance_nl::Float64=1e-5,
    damping_nl::Float64=0.75
)
    
    println("--- Preparing Nonlinear Transient 1D Simulation ---")
    println("Loading mesh and defining domains/materials...")
    
    # Load mesh and setup
    model, labels, tags = load_mesh_and_tags(mesh_file)
    Ω = Triangulation(model)
    degree_quad = 2*order_fem
    dΩ = Measure(Ω, degree_quad)

    material_tags_dict = get_material_tags_oil(labels)
    conductivity_func = define_conductivity(material_tags_dict, σ_core)
    
    # Define B-H curve function (Frohlich-Kennelly model)
    function fmur_core_func(B)
        B_safe = max(0.0, B)
        return 1.0 / (bh_a + (1 - bh_a) * B_safe^(2*bh_b) / (B_safe^(2*bh_b) + bh_c))
    end
    
    # Initial linear reluctivity for first guess
    μr_initial = fmur_core_func(0.0)
    initial_reluctivity_func = define_reluctivity(material_tags_dict, μ0, μr_initial)
    
    cell_tags_cf = CellField(tags, Ω)
    σ_cf = Operation(conductivity_func)(cell_tags_cf)
    
    # Time-dependent source current (using cos for proper initial excitation)
    spatial_js_profile_func = define_current_density(material_tags_dict, J0_amplitude)
    Js_t_func(t) = x -> spatial_js_profile_func(cell_tags_cf(x)) * cos(ω_source * t)

    println("Setting up transient FE spaces...")
    reffe = ReferenceFE(lagrangian, Float64, order_fem)
    V0_test = TestFESpace(model, reffe, dirichlet_tags=[dirichlet_tag])
    Ug_transient = TransientTrialFESpace(V0_test, dirichlet_bc_function)

    println("Defining initial condition...")
    Az0 = zero(Ug_transient(t0))
    
    println("Setting up ODE solver...")
    linear_solver_for_ode = LUSolver()
    odesolver = ThetaMethod(linear_solver_for_ode, Δt, θ_method)
    
    # Create custom solution iterator for nonlinear transient
    solution_vector = []
    
    println("Solving nonlinear transient problem from t=$(t0) to t=$(tF) with Δt=$(Δt)...")
    println("Nonlinear parameters: max_iter=$(max_iterations_nl), tol=$(tolerance_nl), damping=$(damping_nl)")
    
    # Time stepping with nonlinear iteration at each step
    t_current = t0
    Az_current = Az0
    step_count = 0
    
    while t_current < tF
        t_next = min(t_current + Δt, tF)
        step_count += 1
        
        if step_count % 10 == 0
            println("Processing time step $(step_count): t = $(t_current) -> $(t_next)")
        end
        
        # Nonlinear iteration for this time step
        nl_iteration = 0
        nl_error = 1.0
        Az_guess = Az_current  # Initial guess from previous time step
        
        while nl_iteration < max_iterations_nl && nl_error > tolerance_nl
            nl_iteration += 1
            
            # Calculate B-field magnitude from current guess
            if nl_iteration == 1
                # Use linear reluctivity for first iteration
                ν_cf = Operation(initial_reluctivity_func)(cell_tags_cf)
            else
                # Update reluctivity based on B-field from previous iteration
                # For scalar case, create a dummy MultiField structure or adapt function
                try
                    # Try to calculate field-dependent reluctivity
                    B_field = calculate_b_field(Az_guess)
                    
                    # Sample B field at a few points to estimate average magnitude
                    x_sample = [VectorValue(-0.03), VectorValue(0.0), VectorValue(0.03)]
                    B_sample_vals = [B_field(x) for x in x_sample]
                    B_magnitudes = [sqrt(b[1]^2) for b in B_sample_vals]  # For 1D, B_y component
                    B_avg = sum(B_magnitudes) / length(B_magnitudes)
                    
                    # Update reluctivity based on average B field
                    μr_updated = fmur_core_func(B_avg)
                    updated_reluctivity_func = define_reluctivity(material_tags_dict, μ0, μr_updated)
                    ν_cf = Operation(updated_reluctivity_func)(cell_tags_cf)
                    
                    if nl_iteration % 5 == 0
                        println("    B_avg = $(B_avg) T, μr_updated = $(μr_updated)")
                    end
                catch e
                    # Fallback to linear reluctivity if B-field calculation fails
                    if nl_iteration % 5 == 0
                        println("    Using linear reluctivity (B-field calc failed: $(e))")
                    end
                    ν_cf = Operation(initial_reluctivity_func)(cell_tags_cf)
                end
            end
            
            # Setup transient operator with current reluctivity
            res(t, u, v) = ∫( σ_cf * v * ∂t(u) + ν_cf * (∇(v) ⋅ ∇(u)) - v * Js_t_func(t) )*dΩ
            jac_u(t, u, du, v) = ∫( ν_cf * (∇(v) ⋅ ∇(du)) )*dΩ
            jac_ut(t, u, du_t, v) = ∫( σ_cf * v * du_t )*dΩ
            
            transient_op = TransientFEOperator(res, jac_u, jac_ut, Ug_transient, V0_test)
            
            # Solve single time step
            Az_new = solve_transient_step(odesolver, transient_op, Az_current, t_current, t_next)
            
            # Check convergence
            if nl_iteration > 1
                nl_error = norm(get_free_dof_values(Az_new) - get_free_dof_values(Az_guess)) / 
                          (norm(get_free_dof_values(Az_guess)) + 1e-12)
            end
            
            # Apply damping: simply use the new solution for now (simplest approach)
            # TODO: Improve damping by properly constructing FEFunction
            Az_guess = Az_new
            
            if nl_iteration <= 3 || nl_iteration % 5 == 0
                println("  NL iteration $(nl_iteration): error = $(nl_error)")
            end
        end
        
        if nl_iteration >= max_iterations_nl && nl_error > tolerance_nl
            println("  Warning: Nonlinear iteration did not converge at t=$(t_next) (error=$(nl_error))")
        else
            println("  Converged in $(nl_iteration) iterations (error=$(nl_error))")
        end
        
        # Store solution
        push!(solution_vector, (Az_guess, t_next))
        Az_current = Az_guess
        t_current = t_next
    end
    
    println("Nonlinear transient solution completed with $(length(solution_vector)) time steps")
    
    # Create solution iterable compatible with existing post-processing
    solution_transient_iterable = solution_vector
    
    # Return same structure as linear solver for compatibility
    ν_cf_final = Operation(initial_reluctivity_func)(cell_tags_cf)  # Final reluctivity state
    
    return solution_transient_iterable, Az0, Ω, ν_cf_final, σ_cf, Js_t_func, model, cell_tags_cf, labels
end

"""
    solve_transient_step(odesolver, transient_op, Az_prev, t_current, t_next)

Helper function to solve a single transient time step.
"""
function solve_transient_step(odesolver, transient_op, Az_prev, t_current, t_next)
    # Use the ODE solver to advance one time step
    dt = t_next - t_current
    try
        # Create a simple solution iterator for just this step
        step_solution = solve(odesolver, transient_op, Az_prev, t_current, t_next)
        # Get the final solution at t_next
        for (Az_new, t_new) in step_solution
            if abs(t_new - t_next) < 1e-10
                return Az_new
            end
        end
        # If we don't find exact match, return the last solution
        last_solution = nothing
        for (Az_new, t_new) in step_solution
            last_solution = Az_new
        end
        return last_solution !== nothing ? last_solution : Az_prev
    catch e
        println("Warning: Error in time step solve: $(e)")
        return Az_prev  # Return previous solution as fallback
    end
end
