using Gridap
using LinearAlgebra

"""
    calculate_b_field_magnitude(uv, Ω)

Helper function to calculate B-field magnitude from solution.
"""
function calculate_b_field_magnitude(uv, Ω)
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
    reluctivity_func = define_reluctivity(material_tags, μ0, μr_initial)
    conductivity_func = define_conductivity(material_tags, σ_core)
    source_current_func = define_current_density(material_tags, J0)
    
    # Setup FE spaces
    U, V = setup_fe_spaces(model, order, field_type, dirichlet_tag, dirichlet_value)
    
    # Initialize nonlinear iteration
    iter = 0
    error = 1.0
    uv_prev = nothing
    uv = nothing
    
    println("Starting nonlinear iterations")
    
    while iter < max_iterations && error > tolerance
        # Define the weak form problem with current material properties
        problem = magnetodynamics_1d_harmonic_coupled_weak_form(
            Ω, dΩ, tags, reluctivity_func, conductivity_func, source_current_func, ω)
        
        # Solve the FEM problem
        uv_current = solve_fem_problem(problem, U, V)
        
        # Apply damping for better convergence
        if uv !== nothing
            # Store previous solution for error calculation
            uv_prev = uv
            
            # Apply damping (this is trickier with Gridap objects)
            # For now, we'll solve without damping as Gridap doesn't easily support
            # arithmetic operations between FE functions
            uv = uv_current
        else
            # First iteration
            uv_prev = uv_current
            uv = uv_current
        end
        
        # Calculate error (approximation using L2 norm)
        if iter > 0
            # Calculate L2 norm of difference (approximation)
            u_prev = uv_prev[1]  # Real part
            v_prev = uv_prev[2]  # Imaginary part
            u_curr = uv[1]  # Real part
            v_curr = uv[2]  # Imaginary part
            
            # Calculate error as integral of squared difference
            error_u = sqrt(sum(∫((u_curr - u_prev)^2)dΩ))
            error_v = sqrt(sum(∫((v_curr - v_prev)^2)dΩ))
            error = sqrt(error_u^2 + error_v^2)
        end
        
        # Update reluctivity function based on B-field solution
        B_magnitude = calculate_b_field_magnitude(uv, Ω)
        
        # Create a new function that updates reluctivity at each position
        function updated_reluctivity(tag)
            if tag == material_tags["Core"]
                # Sample points in the core (simplified approach)
                n_samples = 100
                sample_points = [VectorValue(x) for x in range(-0.02, 0.02, length=n_samples)]
                B_samples = [B_magnitude(x) for x in sample_points]
                avg_B = sum(B_samples) / n_samples
                
                # Calculate new permeability
                μr = fmur_core(avg_B)
                return 1.0 / (μ0 * μr)
            else
                return reluctivity_func(tag)
            end
        end
        
        # Update reluctivity function
        reluctivity_func = updated_reluctivity
        
        iter += 1
        println("Iteration $iter: error = $error")
    end
    
    if iter >= max_iterations && error > tolerance
        println("Warning: Maximum iterations reached without convergence")
    else
        println("Nonlinear solution converged in $iter iterations")
    end
    
    return uv
end
