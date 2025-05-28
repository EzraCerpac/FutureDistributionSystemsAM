module Visualisation

using Plots
using Gridap
using Gridap.MultiField: MultiFieldFEFunction
using Gridap.CellData: CellField, Operation
using Gridap.Visualization
using Gridap.FESpaces: TestFESpace, FEFunction, FESpace
using Gridap.ReferenceFEs: lagrangian
using Gridap.Geometry: get_triangulation, get_node_coordinates, num_cell_dims, get_grid, DiscreteModel
using Printf
using LaTeXStrings
using WriteVTK

# To access PostProcessing.calculate_b_field. This assumes MagnetostaticsFEM.jl correctly includes and uses PostProcessing.
# This makes Visualisation dependent on PostProcessing being available in its scope.
# An alternative is to always require B_field to be passed in if Visualisation is to be kept fully independent.
using ..PostProcessing

# Helper function to extract scalar value from FE evaluations
_extract_val(v) = isa(v, Gridap.Fields.ForwardDiff.Dual) ? Gridap.Fields.ForwardDiff.value(v) : (isa(v, AbstractArray) && !isempty(v) ? first(v) : v)

export plot_contour_2d, create_field_animation, plot_time_signal, plot_fft_spectrum, plot_line_1d, create_transient_animation, plot_harmonic_magnitude_1d, plot_harmonic_animation_1d

# Hardcoded geometry parameters for now
a_len = 100.3e-3; b_len = 73.15e-3; c_len = 27.5e-3
xa1 = -a_len/2; xb1 = -b_len/2; xc1 = -c_len/2
xc2 = c_len/2; xb2 = b_len/2; xa2 = a_len/2
boundaries = [xa1, xb1, xc1, xc2, xb2, xa2]
x_int = collect(range(-0.1, 0.1, length=1000))
coord = [VectorValue(x_) for x_ in x_int]
x_min_plot = minimum(x_int); x_max_plot = maximum(x_int)
midpoints = [(x_min_plot + xa1)/2, (xa1 + xb1)/2, (xb1 + xc1)/2, (xc1 + xc2)/2, (xc2 + xb2)/2, (xb2 + xa2)/2, (xa2 + x_max_plot)/2]
region_labels = ["Air", "Core", "Coil L", "Core", "Coil R", "Core", "Air"]

function plot_harmonic_magnitude_1d(Az_mag, B_mag, Jeddy_mag, ν_field; output_path=nothing)
    # Evaluate magnitudes at interpolation points
    Az_mag_vals = Az_mag(coord)
    B_mag_vals = B_mag(coord)
    Jeddy_mag_vals = Jeddy_mag(coord)
    ν_vals = ν_field(coord) # Evaluate the reluctivity field
    μ_vals = 1 ./ ν_vals # Convert reluctivity to permeability

    # Plot Magnitudes
    p1 = plot(x_int * 1e2, Az_mag_vals * 1e5, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"|A_z(x)|\ \mathrm{[mWb/cm]}", color=:black, lw=1, legend=false, title=L"|A_z|" *" Magnitude")
    p2 = plot(x_int * 1e2, B_mag_vals * 1e3, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"|B_y(x)|\ \mathrm{[mT]}", color=:black, lw=1, legend=false, title=L"|B_y|" *" Magnitude")
    p3 = plot(x_int * 1e2, Jeddy_mag_vals * 1e-4, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"|J_{eddy}(x)|\ \mathrm{[A/cm^2]}", color=:black, lw=1, legend=false, title=L"|J_{eddy}|" *" Magnitude")
    p4 = plot(x_int * 1e2, μ_vals, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"\mu(x)\ \mathrm{[m/H]}", color=:black, lw=1, legend=false, title="Permeability (Linear)")

    # Add annotations
    for p in [p1, p2, p3, p4]
        vline!(p, boundaries * 1e2, color=:grey, linestyle=:dash, label="")
        plot_ylims = Plots.ylims(p)
        label_y = plot_ylims[1] - 0.08 * (plot_ylims[2] - plot_ylims[1])
        annotate!(p, [(midpoints[i]*1e2, label_y, text(region_labels[i], 8, :center, :top)) for i in eachindex(midpoints)])
    end

    plt_mag = plot(p1, p2, p3, p4, layout=(4,1), size=(800, 1200))
    savefig(plt_mag, joinpath(output_path, "harmonic_magnitude_1d.pdf"))
    display(plt_mag)
    return plt_mag
end

function plot_harmonic_animation_1d(A_re::FEFunction, A_im::FEFunction, Az_mag, B_re, B_im, B_mag, J_eddy_re, J_eddy_im, J_eddy_mag, ν_field, ω::Float64; output_path=nothing, fps=15)
    freq = ω / (2 * pi)
    T_period = 1/freq
    t_vec = range(0, T_period, length=100)

    ν_vals = ν_field(coord) # Evaluate the reluctivity field
    μ_vals = 1 ./ ν_vals # Convert reluctivity to permeability

    # Get magnitude limits for consistent y-axis scaling
    Az_max = maximum(Az_mag(coord))
    By_max = maximum(B_mag(coord))
    J_eddy_max = maximum(J_eddy_mag(coord))

    anim = @animate for t_step in t_vec
        # Calculate instantaneous real value: Re( (u+iv) * exp(jωt) ) = u*cos(ωt) - v*sin(ωt)
        cos_wt = cos(ω * t_step)
        sin_wt = sin(ω * t_step)
        
        Az_inst = A_re * cos_wt - A_im * sin_wt
        B_re_inst = B_re * cos_wt - B_im * sin_wt # Instantaneous B_re
        Jeddy_inst = J_eddy_re * cos_wt - J_eddy_im * sin_wt
        
        # Evaluate at interpolation points
        Az_inst_vals = Az_inst(coord)
        B_re_inst_vals = B_re_inst(coord)
        By_inst_vals = [b[1] for b in B_re_inst_vals] # Extract y-component
        Jeddy_inst_vals = Jeddy_inst(coord)

        # Plot instantaneous real parts at time t
        p1_t = plot(x_int * 1e2, Az_inst_vals * 1e5, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"A_z(x,t)\ \mathrm{[mWb/cm]}", color=:blue, lw=1, legend=false, title=@sprintf("Time-Harmonic (t = %.2e s)", t_step), ylims=(-Az_max*1.1e5, Az_max*1.1e5))
        p2_t = plot(x_int * 1e2, By_inst_vals * 1e3, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"B_y(x,t)\ \mathrm{[mT]}", color=:blue, lw=1, legend=false, ylims=(-By_max*1.1e3, By_max*1.1e3))
        p3_t = plot(x_int * 1e2, Jeddy_inst_vals * 1e-4, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"J_{eddy}(x,t)\ \mathrm{[A/cm^2]}", color=:red, lw=1, legend=false, ylims=(-J_eddy_max*1.1e-4, J_eddy_max*1.1e-4))
        p4_t = plot(x_int * 1e2, μ_vals, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"\mu(x)\ \mathrm{[m/H]}", color=:black, lw=1, legend=false, title="Permeability (Linear)")

        # Add annotations
        for p in [p1_t, p2_t, p3_t, p4_t]
            vline!(p, boundaries * 1e2, color=:grey, linestyle=:dash, label="")
            plot_ylims = Plots.ylims(p)
            label_y = plot_ylims[1] - 0.08 * (plot_ylims[2] - plot_ylims[1])
            annotate!(p, [(midpoints[i]*1e2, label_y, text(region_labels[i], 8, :center, :top)) for i in eachindex(midpoints)])
        end
        
        plot(p1_t, p2_t, p3_t, p4_t, layout=(4,1), size=(800, 1200))
    end

    movie = gif(anim, joinpath(output_path, "harmonic_animation_1d.gif"), fps = fps)
    display(movie)
    return movie
end


# Helper function to get a representative grid of points for min/max calculation
function _get_evaluation_points(Ω; num_points_per_dim=30)
    dim = num_cell_dims(Ω)
    model_coords = get_node_coordinates(Ω) # Get all node coordinates
    
    min_coords = [minimum(getindex.(model_coords, i)) for i in 1:dim]
    max_coords = [maximum(getindex.(model_coords, i)) for i in 1:dim]
    
    if dim == 1
        return [VectorValue(x) for x in range(min_coords[1], max_coords[1], length=num_points_per_dim)]
    elseif dim == 2
        x_range = range(min_coords[1], max_coords[1], length=num_points_per_dim)
        y_range = range(min_coords[2], max_coords[2], length=num_points_per_dim)
        points = Vector{Point{dim, Float64}}()
        for x_val in x_range
            for y_val in y_range
                push!(points, Point(x_val, y_val))
            end
        end
        return points
    else
        error("Evaluation points for dim $dim not implemented for min/max calculation.")
    end
end

# Internal helper function to calculate transient eddy currents
function _calculate_transient_jeddy(Az_current, Az_previous, σ_cf, efectivo_Δt, Ω_domain, initial_step::Bool=false)
    if initial_step || efectivo_Δt <= 1e-9 # Handle initial step or very small/zero Δt
        return CellField(0.0, Ω_domain)
    else
        return -σ_cf .* (Az_current .- Az_previous) ./ efectivo_Δt
    end
end

"""
    plot_contour_2d(Ω, field_to_plot; title="Contour Plot", nlevels=20, output_path=nothing, size=(600,600), clims=nothing)

Generates a 2D contour plot of a given FEFunction or CellField using Plots.jl.
Only displays and saves the plot if output_path is specified.

# Arguments
- `Ω`: The Triangulation (used for extracting points).
- `field_to_plot`: The FEFunction or CellField to plot.
- `title`: Optional title for the plot.
- `nlevels`: Optional number of contour levels.
- `output_path`: Optional path (including filename, e.g., "output/plot.png") to save the plot.
- `size`: Optional tuple specifying plot dimensions (width, height).
- `clims`: Optional tuple specifying color limits for the plot.
"""
function plot_contour_2d(Ω, field_to_plot; title="Contour Plot", nlevels=20, output_path=nothing, size=(600,600), clims=nothing)
    # println("Generating contour plot (this might take a while for large meshes)...") # Reduced verbosity

    field_for_plotting = field_to_plot
    Uh_p1 = nothing # Initialize Uh_p1

    # Check if the field needs interpolation (e.g., if it's a CellField from composition)
    # Force interpolation to ensure we have a standard FEFunction on a P1 space
    # println("Attempting interpolation of field to P1 space for plotting...") # Reduced verbosity
    try
        reffe_p1 = ReferenceFE(lagrangian, Float64, 1)
        # Check if Ω is just a Triangulation or a DiscreteModel
        model_for_fes_rigging = get_grid(Ω) isa DiscreteModel ? get_grid(Ω) : Ω
        Uh_p1 = FESpace(model_for_fes_rigging, reffe_p1) # Use model_for_fes_rigging
        field_for_plotting = interpolate(field_to_plot, Uh_p1)
        # println("Interpolation successful.") # Reduced verbosity
    catch e_interp
        # println("Warning: Could not interpolate field onto P1 space. Using original field. Error: $e_interp") # Reduced verbosity
        field_for_plotting = field_to_plot # Fallback to original field
    end
    
    try
        # Get the coordinates of the mesh nodes (cell vertices)
        coords = get_node_coordinates(Ω)
        
        # Extract x and y coordinates
        x_coords = [point[1] for point in coords]
        y_coords = [point[2] for point in coords]
        
        # Min/max for plot bounds
        x_min, x_max = minimum(x_coords), maximum(x_coords)
        y_min, y_max = minimum(y_coords), maximum(y_coords)
        
        # Create a regular grid for evaluation
        n_grid = 75  # Number of points in each dimension (reduced from 100 for speed)
        x_grid = range(x_min, x_max, length=n_grid)
        y_grid = range(y_min, y_max, length=n_grid)
        
        # Evaluate the field on the grid
        z_values = zeros(n_grid, n_grid)
        
        for (i, x_val_loop) in enumerate(x_grid) # Renamed x to x_val_loop to avoid conflict
            for (j, y_val_loop) in enumerate(y_grid) # Renamed y to y_val_loop to avoid conflict
                point = VectorValue(x_val_loop, y_val_loop)
                try
                    # Evaluate field at the point
                    val = field_for_plotting(point)
                    z_values[j, i] = _extract_val(val)
                catch e
                    # Point might be outside the mesh
                    z_values[j, i] = NaN
                end
            end
        end
        
        # Create contour plot using the regular grid data
        plt = contourf(x_grid, y_grid, z_values, 
                      levels=nlevels, color=:viridis, 
                      title=title, aspect_ratio=:equal, 
                      size=size,
                      clims=clims) # Added clims
        
        # Only display and save the plot if output_path is specified
        if output_path !== nothing
            # Display the plot
            display(plt)
            
            # Save the plot
            try
                savefig(plt, output_path)
                println("Contour plot saved to $output_path")
            catch e_save
                println("Error saving contour plot to '$output_path': $e_save")
            end
        end
        
        # Always return the plot object
        return plt
    catch e
        # println("Error generating contour plot: $e") # Reduced verbosity
        # println("If the error persists, consider exporting to VTK and using Paraview.") # Reduced verbosity
    end
    return nothing # Return nothing if error occurs
end

# Function to extract node coordinates from a triangulation
# function get_node_coordinates(Ω) # Now imported from Gridap.Geometry
#     # Get the underlying grid from the triangulation
#     grid = Ω.grid
    
#     # Extract node coordinates from the grid
#     # This might need adjustment based on the specific Gridap version and grid type
#     coords = grid.node_coordinates
    
#     return coords
# end

# function calculate_b_field is not defined in this file. Assuming it's imported or available in scope.
# If not, it should be passed as an argument or handled appropriately.
# For create_field_animation, it seems to rely on a global/imported calculate_b_field.

function create_field_animation(
    Ω, uv_complex, ω, output_path; # Renamed uv to uv_complex for clarity
    B_field_complex=nothing, # Renamed B_field to B_field_complex
    J_eddy_complex=nothing,  # Renamed J_eddy to J_eddy_complex
    nframes=100,
    fps=15,
    plotinfo=Dict() # Keep for future use
)
    println("Creating harmonic field animation...")
    
    # Extract real and imaginary parts
    u = uv_complex[1]  # Real part of Az
    v = uv_complex[2]  # Imaginary part of Az
    
    # Calculate B field if not provided
    if B_field_complex === nothing
        println("B_field_complex not provided, attempting to calculate using PostProcessing.calculate_b_field...")
        try
            B_re_calc, B_im_calc = PostProcessing.calculate_b_field(uv_complex)
        catch e_bfield
            println("Warning: Could not calculate B-field using PostProcessing.calculate_b_field. Error: $e_bfield. B-field will not be plotted.")
            B_re_calc, B_im_calc = nothing, nothing
        end
    else
        B_re_calc, B_im_calc = B_field_complex
    end
    
    J_eddy_re_calc, J_eddy_im_calc = J_eddy_complex === nothing ? (nothing, nothing) : J_eddy_complex

    dim = num_cell_dims(Ω)
    
    if dim == 1
        error("1D harmonic animation not implemented yet in this function.")
    else  # 2D case
        println("Creating 2D harmonic animation...")
        
        T_period = 2π / ω
        t_vec = range(0, T_period, length=nframes)
        
        anim = @animate for t_step in t_vec
            cos_wt = cos(ω * t_step)
            sin_wt = sin(ω * t_step)
            
            Az_inst = u * cos_wt - v * sin_wt
            
            plots_to_combine = []
            
            title_az_str = Printf.@sprintf("A_z (t=%.3fs)", t_step)
            p1 = plot_contour_2d(Ω, Az_inst, 
                    title=L"%$title_az_str",
                    output_path=nothing, 
                    size=(600,500), clims=get(plotinfo, :Az_clims, nothing))
            push!(plots_to_combine, p1)

            if B_re_calc !== nothing && B_im_calc !== nothing
                B_vec_inst = B_re_calc * cos_wt - B_im_calc * sin_wt
                B_inst_mag = Operation(b -> sqrt(inner(b,b)))(B_vec_inst)
                title_b_str = Printf.@sprintf("|B| (t=%.3fs)", t_step)
                p2 = plot_contour_2d(Ω, B_inst_mag, 
                        title=L"%$title_b_str", 
                        output_path=nothing, 
                        size=(600,500), clims=get(plotinfo, :B_clims, nothing))
                push!(plots_to_combine, p2)
            end
                    
            if J_eddy_re_calc !== nothing && J_eddy_im_calc !== nothing
                J_eddy_inst = J_eddy_re_calc * cos_wt - J_eddy_im_calc * sin_wt
                title_jeddy_str = Printf.@sprintf("J_{eddy} (t=%.3fs)", t_step)
                p3 = plot_contour_2d(Ω, J_eddy_inst, 
                      title=L"%$title_jeddy_str",
                      output_path=nothing, 
                      size=(600,500), clims=get(plotinfo, :Jeddy_clims, nothing))
                push!(plots_to_combine, p3)
            end
            
            num_plots = length(plots_to_combine)
            if num_plots > 0
                plot(plots_to_combine..., layout=(num_plots,1), size=(700, 400*num_plots), 
                        plot_title=@sprintf("Time-Harmonic Fields (ω=%.2f rad/s)", ω))
            end
        end
        
        gif_path = output_path
        if !endswith(lowercase(output_path), ".gif")
            gif_path = output_path * ".gif"
        end
        
        try
            gif(anim, gif_path, fps=fps)
            println("2D Harmonic Animation saved to: $gif_path")
        catch e
            println("Error saving GIF: $e. Make sure Plots.jl backend supports GIF or save individual frames.")
        end
        return anim
    end
    return nothing
end

"""
    plot_time_signal(time_vector::AbstractVector, signal_vector::AbstractVector; 
                     title_str::String="Time Signal", xlabel_str::AbstractString=L"Time (s)", ylabel_str::AbstractString="Amplitude", 
                     output_path::Union{String,Nothing}=nothing)

Plots a 1D time signal.
"""
function plot_time_signal(time_vector::AbstractVector, signal_vector::AbstractVector; 
                          title_str::String="Time Signal", xlabel_str::AbstractString=L"Time (s)", ylabel_str::AbstractString="Amplitude", 
                          output_path::Union{String,Nothing}=nothing)
    plt = plot(time_vector, signal_vector, title=title_str, xlabel=xlabel_str, ylabel=ylabel_str, legend=false)
    if output_path !== nothing
        savefig(plt, output_path)
        println("Time signal plot saved to: $output_path")
    end
    display(plt)
    return plt
end

"""
    plot_fft_spectrum(frequencies::AbstractVector, magnitudes::AbstractVector; 
                        title_str::String="FFT Spectrum", xlabel_str::AbstractString=L"Frequency (Hz)", ylabel_str::AbstractString="Magnitude", 
                        xlims_val=nothing, output_path::Union{String,Nothing}=nothing)

Plots an FFT spectrum.
"""
function plot_fft_spectrum(frequencies::AbstractVector, magnitudes::AbstractVector; 
                           title_str::String="FFT Spectrum", xlabel_str::AbstractString=L"Frequency (Hz)", ylabel_str::AbstractString="Magnitude", 
                           xlims_val=nothing, output_path::Union{String,Nothing}=nothing)
    plt = plot(frequencies, magnitudes, title=title_str, xlabel=xlabel_str, ylabel=ylabel_str, legend=false, seriestype=:stem)
    if xlims_val !== nothing
        xlims!(plt, xlims_val)
    end
    if output_path !== nothing
        savefig(plt, output_path)
        println("FFT spectrum plot saved to: $output_path")
    end
    display(plt)
    return plt
end

"""
    plot_line_1d(Ω_1d, field_to_plot; title_str="Line Plot", npoints=100, output_path=nothing, size=(600,400), y_label="Value", ylims=nothing)

Generates a 1D line plot of a given FEFunction or CellField.
"""
function plot_line_1d(Ω_1d, field_to_plot; title_str="Line Plot", npoints=100, output_path=nothing, size=(600,400), y_label="Value", ylims=nothing)
    
    field_for_plotting = field_to_plot
    try
        reffe_p1 = ReferenceFE(lagrangian, Float64, 1)
        model_for_fes_rigging = get_grid(Ω_1d) isa DiscreteModel ? get_grid(Ω_1d) : Ω_1d
        Uh_p1 = FESpace(model_for_fes_rigging, reffe_p1)
        field_for_plotting = interpolate(field_to_plot, Uh_p1)
    catch e_interp
    end

    try
        coords = get_node_coordinates(Ω_1d)
        x_coords = [point[1] for point in coords]
        x_min, x_max = minimum(x_coords), maximum(x_coords)
        
        x_eval = range(x_min, x_max, length=npoints)
        y_eval = zeros(npoints)

        for (i, x_val) in enumerate(x_eval)
            point_1d = VectorValue(x_val) 
            try
                val = field_for_plotting(point_1d)
                y_eval[i] = _extract_val(val)
            catch e_eval
                y_eval[i] = NaN 
            end
        end
        
        plt = plot(x_eval, y_eval, title=title_str, xlabel=L"x \mathrm{(m)}", ylabel=y_label, size=size, legend=false, ylims=ylims)
        
        if output_path !== nothing
            display(plt)
            try
                savefig(plt, output_path)
                println("1D Line plot saved to $output_path")
            catch e_save
                println("Error saving 1D line plot to '$output_path': $e_save")
            end
        end
        return plt
    catch e
        return nothing
    end
end

"""
    create_transient_animation(
        Ω,
        solution_iterable_input, 
        σ_cf::Union{CellField, Function, Number},
        Δt::Float64, 
        Az0::FEFunction,
        output_path::String;
        fps::Int=10,
        npoints_1d::Int=100, 
        nlevels_2d::Int=15,  
        consistent_axes::Bool=true,
        num_eval_points_minmax::Int=20 
    )

Creates a GIF animation of the transient simulation results using PostProcessing functions.
Includes Az, B, and J_eddy calculated through the enhanced post-processing pipeline.
"""
function create_transient_animation(
    Ω,
    solution_iterable_input, 
    σ_cf::Union{CellField, Function, Number}, 
    Δt::Float64, 
    Az0::FEFunction,
    output_path::String;
    fps::Int=10,
    npoints_1d::Int=100,
    nlevels_2d::Int=15,
    consistent_axes::Bool=true,
    num_eval_points_minmax::Int=20
)
    println("Creating transient field animation using enhanced PostProcessing pipeline...")
    
    dim = num_cell_dims(Ω)

    # Use the enhanced post-processing function to get all fields
    println("Processing transient solution with B-field and J_eddy calculations...")
    processed_steps = PostProcessing.process_transient_solution(solution_iterable_input, Az0, Ω, σ_cf, Δt)
    
    # Prepare for P1 interpolation for plotting
    reffe_p1_anim = ReferenceFE(lagrangian, Float64, 1)
    model_for_fes_rigging_anim = get_grid(Ω) isa DiscreteModel ? get_grid(Ω) : Ω
    Uh_p1_cache_and_plot = FESpace(model_for_fes_rigging_anim, reffe_p1_anim)

    # Cache interpolated solutions for plotting
    println("Interpolating processed solutions for plotting...")
    solution_cache = []
    for (step_idx, (Az_n, B_n, J_eddy_n, tn)) in enumerate(processed_steps)
        try
            Az_n_interp = interpolate(Az_n, Uh_p1_cache_and_plot)
            # Note: B_n and J_eddy_n will be recalculated from Az_n_interp to ensure consistency
            push!(solution_cache, (Az_n_interp, tn))
            if step_idx % 50 == 0
                println("Cached processed step $(step_idx) at t=$(tn)")
            end
        catch e_interp_cache
            println("Warning: Could not interpolate processed solution at t=$(tn) for cache. Error: $e_interp_cache. Using original.")
            push!(solution_cache, (Az_n, tn))
        end
    end
    println("Finished caching $(length(solution_cache)) processed solution steps.")

    ylims_az, ylims_b, ylims_jeddy = nothing, nothing, nothing 
    clims_az, clims_b, clims_jeddy = nothing, nothing, nothing 

    if consistent_axes && !isempty(solution_cache)
        println("Calculating global min/max for consistent axes (using cached P1 solutions)...")
        eval_points = _get_evaluation_points(Ω, num_points_per_dim=num_eval_points_minmax)

        min_az_vals, max_az_vals = Float64[], Float64[]
        min_b_vals, max_b_vals = Float64[], Float64[]
        min_jeddy_vals, max_jeddy_vals = Float64[], Float64[]

        # Iterate through cached solutions and use processed results for limits calculation
        for (idx_cache, (Az_n_cache, tn_cache)) in enumerate(solution_cache) 
            # Get corresponding processed results
            (_, B_n_processed, J_eddy_n_processed, _) = processed_steps[idx_cache]
            B_n_cache = B_n_processed
            J_eddy_n_cache = J_eddy_n_processed

            for pt in eval_points
                try push!(min_az_vals, _extract_val(Az_n_cache(pt))); push!(max_az_vals, _extract_val(Az_n_cache(pt))) catch; end
                if dim == 1
                    # For Bx, B_n_cache is a vector, extract component
                    try 
                        b_val_pt = B_n_cache(pt)
                        if isa(b_val_pt, VectorValue) && length(b_val_pt) > 0
                            push!(min_b_vals, _extract_val(b_val_pt[1]))
                            push!(max_b_vals, _extract_val(b_val_pt[1]))
                        end
                    catch; end
                else # dim == 2, |B|
                    try 
                        b_val_pt = B_n_cache(pt)
                        if isa(b_val_pt, VectorValue) # Ensure it's a VectorValue before inner product
                             push!(min_b_vals, _extract_val(sqrt(inner(b_val_pt,b_val_pt))))
                             push!(max_b_vals, _extract_val(sqrt(inner(b_val_pt,b_val_pt))))
                        end
                    catch; end
                end
                try push!(min_jeddy_vals, _extract_val(J_eddy_n_cache(pt))); push!(max_jeddy_vals, _extract_val(J_eddy_n_cache(pt))) catch; end
            end
        end

        if !isempty(min_az_vals) && !isempty(max_az_vals) ylims_az = (minimum(min_az_vals), maximum(max_az_vals)); clims_az = ylims_az; else ylims_az=nothing; clims_az=nothing; end
        if !isempty(min_b_vals) && !isempty(max_b_vals) ylims_b = (minimum(min_b_vals), maximum(max_b_vals)); clims_b = ylims_b; else ylims_b=nothing; clims_b=nothing; end
        if !isempty(min_jeddy_vals) && !isempty(max_jeddy_vals) ylims_jeddy = (minimum(min_jeddy_vals), maximum(max_jeddy_vals)); clims_jeddy = ylims_jeddy; else ylims_jeddy=nothing; clims_jeddy=nothing; end
        println("Global limits: Az=$ylims_az, B=$ylims_b, Jeddy=$ylims_jeddy")
    end

    anim = @animate for (idx, (Az_n_from_cache, tn)) in enumerate(solution_cache) 
        # Az_n_from_cache is already an interpolated FEFunction
        Az_n = Az_n_from_cache 

        # Get corresponding processed fields from the enhanced post-processing
        (_, B_n, J_eddy_n, _) = processed_steps[idx]

        plot_title_str = Printf.@sprintf("Time: %.4f s", tn)

        if dim == 1
            Bx_n_field = Operation(b_vec -> b_vec[1])(B_n)
            title_az_1d = L"A_z(x,t)"
            title_bx_1d = L"B_x(x,t)"
            title_jeddy_1d = L"J_{eddy}(x,t)"

            # Restore calls to plot_line_1d, Az_n is already P1 interpolated from cache
            p1 = plot_line_1d(Ω, Az_n, title_str=title_az_1d, npoints=npoints_1d, y_label=L"A_z \mathrm{(Wb/m)}", output_path=nothing, ylims=ylims_az) 
            p2 = plot_line_1d(Ω, Bx_n_field, title_str=title_bx_1d, npoints=npoints_1d, y_label=L"B_x \mathrm{(T)}", output_path=nothing, ylims=ylims_b)
            p3 = plot_line_1d(Ω, J_eddy_n, title_str=title_jeddy_1d, npoints=npoints_1d, y_label=L"J_{eddy} \mathrm{(A/m^2)}", output_path=nothing, ylims=ylims_jeddy)
            
            plot(p1, p2, p3, layout=(3,1), plot_title=plot_title_str, size=(700,1000))

        elseif dim == 2
            B_mag_n = Operation(b_vec -> sqrt(inner(b_vec, b_vec)))(B_n)
            
            title_az_2d_str = Printf.@sprintf("A_z (Wb/m) (t=%.3fs)", tn)
            title_b_2d_str = Printf.@sprintf("|B| (T) (t=%.3fs)", tn)
            title_jeddy_2d_str = Printf.@sprintf("J_{eddy} (A/m^2) (t=%.3fs)", tn)

            # Az_n is already P1 interpolated from cache
            p1 = plot_contour_2d(Ω, Az_n, title=L"%$title_az_2d_str", nlevels=nlevels_2d, output_path=nothing, clims=clims_az)
            p2 = plot_contour_2d(Ω, B_mag_n, title=L"%$title_b_2d_str", nlevels=nlevels_2d, output_path=nothing, clims=clims_b)
            p3 = plot_contour_2d(Ω, J_eddy_n, title=L"%$title_jeddy_2d_str", nlevels=nlevels_2d, output_path=nothing, clims=clims_jeddy)
            
            plot(p1, p2, p3, layout=(3,1), plot_title=plot_title_str, size=(700,1200))
        else
            error("Animation for dimension $dim not implemented.")
        end
    end

    gif_path = output_path
    if !endswith(lowercase(output_path), ".gif")
        gif_path = output_path * ".gif"
    end

    println("Transient animation saved to: $gif_path")
    try
        display(gif(anim, gif_path, fps=fps))
    catch e
        println("Error saving GIF: $e. Make sure Plots.jl backend supports GIF or save individual frames.")
    end
    return anim
end

end # module Visualisation