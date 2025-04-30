module Visualisation

using Plots
using Gridap
using Gridap.MultiField: MultiFieldFEFunction
using Gridap.CellData: CellField
using Gridap.Visualization
using Gridap.FESpaces: TestFESpace, FEFunction, FESpace
using Gridap.ReferenceFEs: lagrangian
using Gridap.Geometry: get_triangulation
using Printf
using LaTeXStrings

export plot_contour_2d, create_field_animation

"""
    plot_contour_2d(Ω, field_to_plot; title="Contour Plot", nlevels=20, output_path=nothing, size=(600,600))

Generates a 2D contour plot of a given FEFunction or CellField using Plots.jl.
Only displays and saves the plot if output_path is specified.

# Arguments
- `Ω`: The Triangulation (used for extracting points).
- `field_to_plot`: The FEFunction or CellField to plot.
- `title`: Optional title for the plot.
- `nlevels`: Optional number of contour levels.
- `output_path`: Optional path (including filename, e.g., "output/plot.png") to save the plot.
- `size`: Optional tuple specifying plot dimensions (width, height).
"""
function plot_contour_2d(Ω, field_to_plot; title="Contour Plot", nlevels=20, output_path=nothing, size=(600,600))
    println("Generating contour plot (this might take a while for large meshes)...")

    field_for_plotting = field_to_plot
    Uh_p1 = nothing # Initialize Uh_p1

    # Check if the field needs interpolation (e.g., if it's a CellField from composition)
    # Force interpolation to ensure we have a standard FEFunction on a P1 space
    println("Attempting interpolation of field to P1 space for plotting...")
    try
        reffe_p1 = ReferenceFE(lagrangian, Float64, 1)
        Uh_p1 = FESpace(Ω, reffe_p1)
        field_for_plotting = interpolate(field_to_plot, Uh_p1)
        println("Interpolation successful.")
    catch e_interp
        println("Warning: Could not interpolate field onto P1 space. Using original field. Error: $e_interp")
        field_for_plotting = field_to_plot
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
        n_grid = 100  # Number of points in each dimension
        x_grid = range(x_min, x_max, length=n_grid)
        y_grid = range(y_min, y_max, length=n_grid)
        
        # Evaluate the field on the grid
        z_values = zeros(n_grid, n_grid)
        
        for (i, x) in enumerate(x_grid)
            for (j, y) in enumerate(y_grid)
                point = VectorValue(x, y)
                try
                    # Evaluate field at the point
                    z_values[j, i] = field_for_plotting(point)
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
                      size=size)
        
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
        println("Error generating contour plot: $e")
        println("If the error persists, consider exporting to VTK and using Paraview.")
    end
end

# Function to extract node coordinates from a triangulation
function get_node_coordinates(Ω)
    # Get the underlying grid from the triangulation
    grid = Ω.grid
    
    # Extract node coordinates from the grid
    # This might need adjustment based on the specific Gridap version and grid type
    coords = grid.node_coordinates
    
    return coords
end

function create_field_animation(
    Ω, uv, ω, output_path;
    B_field=nothing,
    J_eddy=nothing,
    nframes=100,
    fps=15,
    grid_points=100,
    plotinfo=Dict()
)
    println("Creating field animation...")
    
    # Extract real and imaginary parts
    u = uv[1]  # Real part
    v = uv[2]  # Imaginary part
    
    # Calculate B field if not provided
    if B_field === nothing
        println("Calculating B field components...")
        B_re, B_im = calculate_b_field(uv)
    else
        B_re, B_im = B_field
    end
    
    # Get problem dimension
    dim = num_cell_dims(Ω)
    
    # Create evaluation points
    if dim == 1
        # 1D animation code (not implemented here)
    else  # 2D case
        println("Creating 2D animation...")
        
        # Calculate time period
        T_period = 2π / ω
        t_vec = range(0, T_period, length=nframes)
        
        # Precompute field magnitudes for consistent color scales
        Az_mag = CellField((x) -> sqrt(u(x)^2 + v(x)^2), Ω)
        B_mag = CellField((x) -> sqrt(inner(B_re(x), B_re(x)) + inner(B_im(x), B_im(x))), Ω)
        
        # Create animation
        anim = @animate for t_step in t_vec
            # Calculate instantaneous values at this time step
            cos_wt = cos(ω * t_step)
            sin_wt = sin(ω * t_step)
            
            # Create instantaneous fields
            Az_inst = CellField((x) -> u(x) * cos_wt - v(x) * sin_wt, Ω)
            B_inst_mag = CellField((x) -> begin
                B_re_val = B_re(x) * cos_wt - B_im(x) * sin_wt
                return sqrt(inner(B_re_val, B_re_val))
            end, Ω)
            
            # Create eddy current field if provided
            J_eddy_inst = nothing
            if J_eddy !== nothing
                J_eddy_re, J_eddy_im = J_eddy
                J_eddy_inst = CellField((x) -> J_eddy_re(x) * cos_wt - J_eddy_im(x) * sin_wt, Ω)
            end
            
            # Create contour plots
            p1 = plot_contour_2d(Ω, Az_inst, 
                    title=@sprintf("Az(t=%.3fs)", t_step),
                    output_path=nothing, 
                    size=(600,500))
            
            p2 = plot_contour_2d(Ω, B_inst_mag, 
                    title=@sprintf("|B|(t=%.3fs)", t_step),
                    output_path=nothing, 
                    size=(600,500))
                    
            if J_eddy_inst !== nothing
                p3 = plot_contour_2d(Ω, J_eddy_inst, 
                      title=@sprintf("Jeddy(t=%.3fs)", t_step),
                      output_path=nothing, 
                      size=(600,500))
                # Combine all three plots
                plot(p1, p2, p3, layout=(3,1), size=(700, 1200), 
                     title=@sprintf("Time-Harmonic Fields (t=%.3fs)", t_step))
            else
                # Just combine Az and B plots
                plot(p1, p2, layout=(2,1), size=(700, 800),
                     title=@sprintf("Time-Harmonic Fields (t=%.3fs)", t_step))
            end
        end
        
        # Save the animation
        gif_path = output_path
        if !endswith(lowercase(output_path), ".gif")
            gif_path = output_path * ".gif"
        end
        
        
        println("2D Animation saved to: $gif_path")
        
        return gif(anim, gif_path, fps=fps)
    end
end

end # module Visualisation