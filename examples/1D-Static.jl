# %% [markdown]
# # Exercise 1: 1D Magnetostatics

# %%
include(joinpath(dirname(@__DIR__), "config.jl"))
paths = get_project_paths("examples")

# Ensure the module is reloaded if changed
if isdefined(Main, :MagnetostaticsFEM)
    println("Reloading MagnetostaticsFEM...")
    # A simple way to force reload in interactive sessions
    try; delete!(LOAD_PATH, joinpath(paths["SRC_DIR"], "src")); catch; end
    try; delete!(Base.loaded_modules, Base.PkgId(Base.UUID("f8a2b3c4-d5e6-f7a8-b9c0-d1e2f3a4b5c6"), "MagnetostaticsFEM")); catch; end
end
include(joinpath(paths["SRC_DIR"], "MagnetostaticsFEM.jl"))

using LinearAlgebra
using Plots
using LaTeXStrings
using Gridap
using .MagnetostaticsFEM
using Plots.PlotMeasures

# %% [markdown]
# ## Define Parameters and Paths

# %%
# Model Parameters (same as harmonic case but no frequency dependence)
J0 = 2.2e4       # Source current density [A/m²]
μ0 = 4e-7 * pi  # Vacuum permeability [H/m]
μr_core = 5000.0 # Relative permeability of the core

# FEM Parameters
order = 4
field_type = Float64  # Real field for static case
dirichlet_tag = "D"
dirichlet_value = 0.0  # Real boundary condition

# Paths
mesh_file = joinpath(paths["GEO_DIR"], "coil_geo_new.msh")
output_file_base = joinpath(paths["OUTPUT_DIR"], "magnetostatics_1d")

println("Mesh file: ", mesh_file)
println("Output directory: ", paths["OUTPUT_DIR"])

# %% [markdown]
# ## Setup FEM Problem

# %%
# Load mesh and tags
model, labels, tags = load_mesh_and_tags(mesh_file)

# Get material tags dictionary
material_tags = get_material_tags_oil(labels)
println("Material tags: ", material_tags)

# Set up triangulation and measures
Ω = Triangulation(model)
dΩ = Measure(Ω, 2*order)

# Define material property functions (no conductivity needed for static case)
reluctivity_func = define_reluctivity(material_tags, μ0, μr_core)
source_current_func = define_current_density(material_tags, J0)

# Setup FE spaces (single real field)
U, V = setup_fe_spaces(model, order, field_type, dirichlet_tag, dirichlet_value)

# Define the weak form problem for magnetostatics
problem = magnetostatics_weak_form(Ω, dΩ, tags, reluctivity_func, source_current_func)

# %% [markdown]
# ## Solve FEM Problem

# %%
# Solve the linear FE system
Az = solve_fem_problem(problem, U, V)

# %% [markdown]
# ## Post-processing

# %%
# Compute B-field from Az
B = calculate_b_field(Az)

# Calculate reluctivity field for visualization
τ_cell_field = CellField(tags, Ω)
ν_field_linear = Operation(reluctivity_func)(τ_cell_field)

# Save results to VTK format
save_results_vtk(Ω, output_file_base, 
    Dict(
        "Az" => Az, 
        "B" => B,
        "ν_linear" => ν_field_linear
    ))

# %%
# Define geometry boundaries and plotting setup
x_int = collect(range(-0.1, 0.1, length=1000))
coord = [VectorValue(x_) for x_ in x_int]
a_len = 100.3e-3

# Define geometry boundaries for plotting (based on 1d_mesh_w_oil_reservois.jl)
b_len = 73.15e-3; c_len = 27.5e-3
reservoir_width = 3 * (2*a_len - b_len)  # Oil reservoir width from mesh generator

# For electromagnetic plots (narrow range): Oil | Core | Coil L | Core | Coil R | Core | Oil
xa2 = -a_len/2           # Left core boundary
xb1 = -b_len/2           # Left coil boundary  
xc1 = -c_len/2           # Core center left
xc2 = c_len/2            # Core center right
xb2 = b_len/2            # Right coil boundary
xa3 = a_len/2            # Right core boundary
boundaries_em = [xa2, xb1, xc1, xc2, xb2, xa3]  # 6 boundaries for electromagnetic plots

# Calculate midpoints for electromagnetic plots (7 regions: Oil-centered view)
x_min_em = -0.1; x_max_em = 0.1  # Electromagnetic plot range
midpoints_em = [
    (x_min_em + xa2)/2,    # Oil (left)
    (xa2 + xb1)/2,         # Core (left)
    (xb1 + xc1)/2,         # Coil L
    (xc1 + xc2)/2,         # Core (center)
    (xc2 + xb2)/2,         # Coil R
    (xb2 + xa3)/2,         # Core (right)
    (xa3 + x_max_em)/2     # Oil (right)
]
region_labels_em = ["Oil", "Core", "Coil L", "Core", "Coil R", "Core", "Oil"]

# Define color scheme function
function get_region_color(region_name)
    if region_name == "Oil"
        return :orange
    elseif region_name == "Air"
        return :blue
    elseif region_name == "Transformer"
        return :green
    elseif region_name == "Core"
        return :green
    elseif occursin("Coil", region_name)
        return :purple
    else
        return :black
    end
end

# Define background tinting function
function add_region_backgrounds!(p, x_boundaries, region_labels, x_range)
    for i in 1:(length(x_boundaries)+1)
        x_start = i == 1 ? x_range[1] : x_boundaries[i-1]
        x_end = i == length(x_boundaries)+1 ? x_range[2] : x_boundaries[i]
        
        region_name = i <= length(region_labels) ? region_labels[i] : "Air"
        base_color = get_region_color(region_name)
        
        # Add very light background tint
        plot_ylims = Plots.ylims(p)
        vspan!(p, [x_start*1e2, x_end*1e2], color=base_color, alpha=0.1, label="")
    end
end

# %% [markdown]
# ## Visualization

# %%
# Define points for visualization
x_int = collect(range(-0.1, 0.1, length=1000))
coord = [VectorValue(x_) for x_ in x_int]

# Evaluate fields at interpolation points
Az_vals = Az(coord)
B_vals = B(coord)
By_vals = [b[1] for b in B_vals] # Extract y-component
ν_vals_linear = ν_field_linear(coord) # Evaluate the linear reluctivity field
μ_vals_linear = 1 ./ ν_vals_linear # Convert reluctivity to permeability

# Plot Static Fields
p1 = plot(x_int * 1e2, Az_vals * 1e5, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"A_z(x)\ \mathrm{[mWb/cm]}", color=:black, lw=1.5, legend=false, title=L"A_z" *" (Static)", bottom_margin=8mm)
p2 = plot(x_int * 1e2, By_vals * 1e3, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"B_y(x)\ \mathrm{[mT]}", color=:black, lw=1.5, legend=false, title=L"B_y" *" (Static)", bottom_margin=8mm)
p3 = plot(x_int * 1e2, μ_vals_linear, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"\mu(x)\ \mathrm{[H/m]}", color=:black, lw=1.5, legend=false, title="Permeability (Linear)", bottom_margin=8mm)

# Add annotations with improved legibility (electromagnetic plots)
for p in [p1, p2, p3]
    # Add background tinting for electromagnetic plots
    add_region_backgrounds!(p, boundaries_em, region_labels_em, [x_min_em, x_max_em])
    
    vline!(p, boundaries_em * 1e2, color=:grey, linestyle=:dash, alpha=0.6, label="")
    plot_ylims = Plots.ylims(p)
    label_y = plot_ylims[1] - 0.06 * (plot_ylims[2] - plot_ylims[1])  # Reduced offset to prevent cutoff
    
    # Add region labels with color scheme
    for i in eachindex(midpoints_em)
        color = get_region_color(region_labels_em[i])
        annotate!(p, midpoints_em[i]*1e2, label_y, text(region_labels_em[i], 9, color, :center, :top))
    end
end

plt_static = plot(p1, p2, p3, layout=(3,1), size=(800, 900), bottom_margin=10mm, left_margin=5mm, right_margin=5mm)
savefig(plt_static, joinpath(paths["OUTPUT_DIR"], "magnetostatics_1d_fields.pdf"))
display(plt_static)

println("✓ Magnetostatic simulation completed successfully")
println("Results saved to: ", output_file_base)
println("Plots saved to: ", joinpath(paths["OUTPUT_DIR"], "magnetostatics_1d_fields.pdf"))