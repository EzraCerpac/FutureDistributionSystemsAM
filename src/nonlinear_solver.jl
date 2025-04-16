using Gridap
using LinearAlgebra

"""
    calculate_b_field_magnitude(uv::MultiFieldFEFunction, Ω)

Helper function to calculate B-field magnitude from solution.
Works with both MultiFieldFEFunction and Vector{ComplexF64} inputs.
"""
function calculate_b_field_magnitude(uv::MultiFieldFEFunction, Ω)
    # Get the real and imaginary parts of B from the multifield solution
    B_re, B_im = calculate_b_field(uv)
    
    # Calculate B magnitude at each integration point
    function B_mag_sq(x)
        b_re = B_re(x)
        b_im = B_im(x)
        return sqrt(inner(b_re, b_re) + inner(b_im, b_im))
    end
    
    return B_mag_sq
end

"""
    calculate_b_field_magnitude(uv::Vector{ComplexF64}, Ω)

Vector{ComplexF64} version of the function to calculate B-field magnitude.
Uses calculate_b_field for consistency.
"""
function calculate_b_field_magnitude(uv::Vector{ComplexF64}, Ω)
    # Use calculate_b_field to get the B field components
    B_re_approx, B_im_approx = calculate_b_field(uv)
    
    # Create function to calculate magnitude
    function B_mag_sq(x)
        b_re = B_re_approx(x)
        b_im = B_im_approx(x)
        return sqrt(inner(b_re, b_re) + inner(b_im, b_im))
    end
    
    return B_mag_sq
end

"""
    calc_fnu(uv::MultiFieldFEFunction, tags, material_tags, μ0, fmur_core)

Calculates the reluctivity based on B-field with an efficient approach.
Inspired by the notebook implementation but adapted for Gridap's FE functions.
"""
function calc_fnu(uv::MultiFieldFEFunction, tags, material_tags, μ0, fmur_core)
    # Get the real and imaginary parts of B from the multifield solution
    B_re, B_im = calculate_b_field(uv)
    
    # Create a function to calculate B magnitude at each point
    function B_mag(x)
        b_re = B_re(x)
        b_im = B_im(x)
        return sqrt(inner(b_re, b_re) + inner(b_im, b_im))
    end
    
    # Map the elements to reluctivity based on their material tag
    function reluctivity_at_tag(tag)
        if tag == material_tags["Core"]
            # For core material, we need to sample B at multiple points and average
            # For 1D case, we can use simple averaging
            sample_points = [VectorValue(x) for x in range(-0.03, 0.03, length=30)]
            B_samples = [B_mag(x) for x in sample_points]
            avg_B = sum(B_samples) / length(B_samples)
            
            # Calculate μr using the BH curve
            μr = fmur_core(avg_B)
            return 1.0 / (μ0 * μr)
        else
            # Non-core materials have μr = 1
            return 1.0 / μ0
        end
    end
    
    # Create array of reluctivity values for each element
    fnu_values = [reluctivity_at_tag(tag) for tag in tags]
    
    return fnu_values
end

"""
    create_reluctivity_function(material_tags, μ0, fmur_core, B_values)

Creates a reluctivity function based on calculated B-field values.
"""
function create_reluctivity_function(material_tags, μ0, fmur_core, B_values, e_group)
    # Create mapping from tags to B values
    tag_to_B = Dict{Int, Float64}()
    
    for (i, tag) in enumerate(e_group)
        if haskey(tag_to_B, tag)
            # Average B values for same material
            tag_to_B[tag] = (tag_to_B[tag] + B_values[i]) / 2
        else
            tag_to_B[tag] = B_values[i]
        end
    end
    
    # Create the reluctivity function
    function updated_reluctivity(tag)
        if tag == material_tags["Core"]
            B = get(tag_to_B, tag, 0.0)
            μr = fmur_core(B)
            return 1.0 / (μ0 * μr)
        else
            # Non-core materials have μr = 1
            return 1.0 / μ0
        end
    end
    
    return updated_reluctivity
end

"""
    solve_nonlinear_magnetodynamics(...)

Function to solve the nonlinear magnetodynamic problem using iterative substitution.
"""
function solve_nonlinear_magnetodynamics(
    model, labels, tags, J0, μ0, bh_a, bh_b, bh_c, σ_core, ω, 
    order, field_type, dirichlet_tag, dirichlet_value;
    max_iterations=50, tolerance=1e-6, damping=0.7)
    
    # Setup triangulation and measures
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*order)
    
    # Get material tags dictionary
    material_tags = get_material_tags(labels)
    
    # Define function for relative permeability based on BH curve
    function fmur_core(B)
        return 1.0 / (bh_a + (1 - bh_a) * B^(2*bh_b) / (B^(2*bh_b) + bh_c))
    end
    
    # Initial material properties (linear)
    μr_initial = 1000.0  # Initial high permeability as starting point
    reluctivity_func_initial = define_reluctivity(material_tags, μ0, μr_initial)
    conductivity_func = define_conductivity(material_tags, σ_core)
    source_current_func = define_current_density(material_tags, J0)
    
    # Store a reference to the initial reluctivity function
    reluctivity_func = reluctivity_func_initial
    
    # Setup FE spaces
    U, V = setup_fe_spaces(model, order, field_type, dirichlet_tag, dirichlet_value)
    
    # Initialize nonlinear iteration
    iter = 0
    error = 1.0
    uv_prev = nothing
    uv = nothing

    # Define the weak form problem with current material properties
    problem = magnetodynamics_1d_harmonic_coupled_weak_form(
        Ω, dΩ, tags, reluctivity_func, conductivity_func, source_current_func, ω)
    
    # Solve the FEM problem
    uv_FESpace = solve_fem_problem(problem, U, V)
    
    println("Starting nonlinear iterations")
    
    while iter < max_iterations && error > tolerance
        uv_current = extract_solution_values(uv_FESpace)
        println(uv_FESpace)
        println(uv_current)
        
        # Apply damping for better convergence
        if uv !== nothing
            # Store previous solution for error calculation
            uv_prev = uv
            
            uv = uv_current * damping + (1 - damping) * uv_prev
        else
            # First iteration
            uv_prev = uv_current
            uv = uv_current
        end
        
        # Calculate error (approximation using L2 norm)
        if iter > 0
            # Calculate L2 norm of difference (approximation)
            error = norm(uv - uv_prev)
        end
        
        # Update reluctivity function based on B-field solution
        if iter > 0
            # Calculate reluctivity directly from the solution using the fixed function
            fnu_values = calc_fnu(uv_FESpace, tags, material_tags, μ0, fmur_core)
            
            # Create updated reluctivity function using anonymous function assignment
            reluctivity_func = tag -> begin
                for (i, t) in enumerate(tags)
                    if t == tag
                        return fnu_values[i]
                    end
                end
                # Default return for tags not found
                return 1.0 / μ0
            end
        end
        
        # Define the weak form problem with current material properties
        problem = magnetodynamics_1d_harmonic_coupled_weak_form(
            Ω, dΩ, tags, reluctivity_func, conductivity_func, source_current_func, ω)
        
        # Solve the FEM problem
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
