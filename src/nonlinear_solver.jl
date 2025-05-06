using Gridap
using Gridap.Geometry
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
    calc_fnu(uv_fe::MultiFieldFEFunction, Ω::Triangulation, integration_degree::Int, tags::AbstractArray, material_tags::Dict, μ0::Float64, fmur_core::Function)

Calculates the reluctivity based on B-field with an efficient approach.
Each cell gets its own reluctivity based on the average B-field in that cell.
Uses an explicit integration_degree to define quadrature points for averaging.
"""
function calc_fnu(uv_fe::MultiFieldFEFunction, Ω::Triangulation, integration_degree::Int, tags::AbstractArray, material_tags::Dict, μ0::Float64, fmur_core::Function)
    B_re, B_im = calculate_b_field(uv_fe)
    
    function B_mag_pointwise(x)
        b_re_val = B_re(x)
        b_im_val = B_im(x)
        return sqrt(inner(b_re_val, b_re_val) + inner(b_im_val, b_im_val))
    end

    fnu_values = zeros(Float64, length(tags))
    
    # Construct CellQuadrature explicitly using the triangulation and integration_degree
    cell_quad = CellQuadrature(Ω, integration_degree)
    # Get CellPoint object from CellQuadrature
    cell_point_obj = get_cell_points(cell_quad)
    # Get the array of physical quadrature points from CellPoint
    # phys_quad_points is an array of arrays: phys_quad_points[cell_id][point_id_in_cell]
    phys_quad_points = get_array(cell_point_obj)

    grid = get_grid(Ω) # Get the underlying grid from the triangulation
    cell_maps = get_cell_map(grid)
    reffes = get_reffes(grid)
    cell_types = get_cell_type(grid)

    for cell_id in 1:num_cells(Ω)
        tag = tags[cell_id]
        if tag == material_tags["Core"]
            points_in_cell = phys_quad_points[cell_id]
            if isempty(points_in_cell)
                cell_map = cell_maps[cell_id]
                reffe = reffes[cell_types[cell_id]]
                ref_polytope = get_polytope(reffe)
                ref_centroid = get_centroid(ref_polytope)
                phys_centroid = evaluate(cell_map, ref_centroid)
                avg_B = B_mag_pointwise(phys_centroid)
            else
                B_samples_at_quad_points = [B_mag_pointwise(p) for p in points_in_cell]
                avg_B = sum(B_samples_at_quad_points) / length(B_samples_at_quad_points)
            end
            
            μr = fmur_core(avg_B)
            fnu_values[cell_id] = 1.0 / (μ0 * μr)
        else
            fnu_values[cell_id] = 1.0 / μ0 
        end
    end
    
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
    max_iterations=50, tolerance=1e-10, damping=0.7)
    
    # Setup triangulation and measures
    Ω = Triangulation(model)
    integration_degree = 2 * order
    dΩ = Measure(Ω, integration_degree)
    
    # Get material tags dictionary
    material_tags = get_material_tags(labels)
    
    # Define function for relative permeability based on BH curve
    function fmur_core(B)
        return 1.0 / (bh_a + (1 - bh_a) * B^(2*bh_b) / (B^(2*bh_b) + bh_c))
    end
    
    # Initial material properties (linear)
    μr_initial = 1000.0  # Initial high permeability as starting point
    reluctivity_func_initial = define_reluctivity(material_tags, μ0, μr_initial)
    
    # Convert initial reluctivity function to a CellField based on cell tags
    initial_reluctivity_values = [reluctivity_func_initial(t) for t in tags]
    current_reluctivity_param = CellField(initial_reluctivity_values, Ω)

    conductivity_func = define_conductivity(material_tags, σ_core)
    source_current_func = define_current_density(material_tags, J0)
    
    # Setup FE spaces
    U, V = setup_fe_spaces(model, order, field_type, dirichlet_tag, dirichlet_value)
    
    # Initialize nonlinear iteration
    iter = 0
    error = 1.0
    uv_prev = nothing
    uv = nothing

    # Define the weak form problem with current material properties
    problem = magnetodynamics_harmonic_coupled_weak_form(
        Ω, dΩ, tags, current_reluctivity_param, conductivity_func, source_current_func, ω)
    
    # Solve the FEM problem
    uv_FESpace = solve_fem_problem(problem, U, V)

    println("Starting nonlinear iterations")
    
    while iter < max_iterations && error > tolerance
        uv_current_sol = uv_FESpace
        
        uv_current_dofs = extract_solution_values(uv_current_sol)

        if uv !== nothing
            uv_prev = uv
            uv = uv_current_dofs * damping + (1 - damping) * uv_prev
        else
            uv_prev = uv_current_dofs 
            uv = uv_current_dofs
        end
        
        if iter > 0
            error = norm(uv - uv_prev) / norm(uv_prev)
        end
        
        # Pass integration_degree instead of dΩ to calc_fnu
        fnu_values = calc_fnu(uv_FESpace, Ω, integration_degree, tags, material_tags, μ0, fmur_core)
        current_reluctivity_param = CellField(fnu_values, Ω)
        
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
