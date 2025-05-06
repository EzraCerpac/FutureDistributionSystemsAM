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
