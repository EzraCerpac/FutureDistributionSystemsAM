# %% [markdown]
# # Exercise 2: 1D Magnetodynamics (Time-Harmonic)

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
using Printf # For animation title formatting

# %% [markdown]
# ## Define Parameters and Paths

# %%
# Model Parameters
J0 = 2.2e4       # Source current density [A/m²] (Assumed Real)
μ0 = 4e-7 * pi  # Vacuum permeability [H/m]
μr_core = 5000.0 # Relative permeability of the core
σ_core = 1e7    # Conductivity of the core [S/m]
freq = 50    # Frequency [Hz]
ω = 2 * pi * freq # Angular frequency [rad/s]

# FEM Parameters
order = 3
field_type = ComplexF64 # Still use ComplexF64 marker for setup_fe_spaces
dirichlet_tag = "D"
dirichlet_value = 0.0 + 0.0im # Dirichlet BC for A = u + iv

# Paths
mesh_file = joinpath(paths["GEO_DIR"], "coil_geo_new.msh")
output_file_base = joinpath(paths["OUTPUT_DIR"], "magnetodynamics_harmonic_coupled")

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

# Define material property functions
reluctivity_func = define_reluctivity(material_tags, μ0, μr_core)
conductivity_func = define_conductivity(material_tags, σ_core)
source_current_func = define_current_density(material_tags, J0) # Real source

# Setup FE spaces (multi-field: Real, Imag parts)
U, V = setup_fe_spaces(model, order, field_type, dirichlet_tag, dirichlet_value)

# Define the weak form problem for the coupled system
problem = magnetodynamics_harmonic_coupled_weak_form(Ω, dΩ, tags, reluctivity_func, conductivity_func, source_current_func, ω)

# %% [markdown]
# ## Solve FEM Problem

# %%
# Solve the real coupled linear FE system
uv = solve_fem_problem(problem, U, V) # uv is a MultiFieldFEFunction

# Extract real and imaginary parts
u = uv[1] # Real part of Az
v = uv[2] # Imag part of Az


# %% [markdown]
# ## Post-processing

# %%
# Compute B-field (Real and Imag parts)
B_re, B_im = calculate_b_field(uv)

# Compute Eddy Currents (Real and Imag parts)
# Ensure conductivity_func here is for electrical conductivity
J_eddy_re, J_eddy_im = calculate_eddy_current(uv, conductivity_func, ω, Ω, tags)

# Define helper functions for magnitude squared
mag_sq_scalar(re, im) = re*re + im*im
mag_sq_vector(re, im) = inner(re, re) + inner(im, im)

# Calculate Magnitudes for saving/plotting using composition
Az_mag = sqrt ∘ (mag_sq_scalar ∘ (u, v))
B_mag = sqrt ∘ (mag_sq_vector ∘ (B_re, B_im))
Jeddy_mag_squared = mag_sq_vector ∘ (J_eddy_re, J_eddy_im)
Jeddy_mag = sqrt ∘ (mag_sq_scalar ∘ (J_eddy_re, J_eddy_im))

# Calculate reluctivity field for visualization (linear case)
# reluctivity_func is already defined: maps tag to reluctivity value
τ_cell_field = CellField(tags, Ω) # 'tags' is the vector of cell tags, Ω is Triangulation
ν_field_linear = Operation(reluctivity_func)(τ_cell_field)

# Save results to VTK format
save_results_vtk(Ω, output_file_base, 
    Dict(
        "Az_re" => u, "Az_im" => v, "Az_mag" => Az_mag,
        "B_re" => B_re, "B_im" => B_im, "B_mag" => B_mag,
        "Jeddy_re" => J_eddy_re, "Jeddy_im" => J_eddy_im, "Jeddy_mag" => Jeddy_mag,
        "ν_linear" => ν_field_linear
    ))

# %%
kappa_core = 80.0
kappa_coil = 385.0   
kappa_air = 0.024
kappa_oil = 0.136

x_int = collect(range(-0.1, 0.1, length=1000))
coord = [VectorValue(x_) for x_ in x_int]
a_len = 100.3e-3



heat_conductivity_func = define_heat_conductivity(material_tags, kappa_core, kappa_coil, kappa_air, kappa_oil)

# Define a CellField for the eddy current losses
τ_cell_field = CellField(tags, Ω)

# FIX: Use electrical conductivity for eddy current loss density
σ_elec_field = Operation(conductivity_func)(τ_cell_field) # [S/m]

# Loss density: Q = 0.5 * σ_elec * (Jeddy_re^2 + Jeddy_im^2)
Ke = 1.0      # Constant (adjust as needed)
mt = 2.0e-3   # Magnetic material thickness [m] (adjust as needed)
V  = 1.0e-6   # Representative volume [m³] (adjust as needed)

B_vals = B_mag(coord)
B_peak = maximum(B_vals)

# TODO: fix this with an integral
eddy_loss_density = (x) -> (π^2 / 6) * Ke * (B_peak^2) * (mt^2) * (freq^2) * (a_len^2/3) / 1e-7


boundary_flux_func = tag -> begin
    if tag == "Boundary"
        return -10.0  # Heat flux leaving the boundary (negative for outward flux)
    else
        return 0.0
    end
end

# Now solve the heat equation using this CellField directly
T = solve_heatdynamics(
    model, tags, order, dirichlet_tag, 0.0,
    eddy_loss_density, heat_conductivity_func
)

x_int = collect(range(-0.5, 0.5, length=1000))
coord = [VectorValue(x_) for x_ in x_int]

# For plotting, evaluate at coordinates
# plot(x_int * 1e2, eddy_loss_density(coord), xlabel="x [cm]", ylabel="Q [W/m³]", title="Eddy Current Loss Density")

heat_plot = plot(x_int * 1e2, T(coord), xlabel="x [cm]", ylabel="ΔT [K]", title="Temperature Rise due to Eddy Currents")
display(heat_plot)

# %% [markdown]
# ## Visualization (Magnitudes)

# %%
# Define geometry boundaries for plotting
a_len = 100.3e-3; b_len = 73.15e-3; c_len = 27.5e-3
xa1 = -a_len/2; xb1 = -b_len/2; xc1 = -c_len/2
xc2 = c_len/2; xb2 = b_len/2; xa2 = a_len/2
boundaries = [xa1, xb1, xc1, xc2, xb2, xa2]

# %%
# Define points for visualization
x_int = collect(range(-0.1, 0.1, length=1000))
coord = [VectorValue(x_) for x_ in x_int]

# Evaluate magnitudes at interpolation points
Az_mag_vals = Az_mag(coord)
B_mag_vals = B_mag(coord)
Jeddy_mag_vals = Jeddy_mag(coord)
ν_vals_linear = ν_field_linear(coord) # Evaluate the linear reluctivity field
μ_vals_linear = 1 ./ ν_vals_linear # Convert reluctivity to permeability

# Calculate midpoints for region labels
x_min_plot = minimum(x_int); x_max_plot = maximum(x_int)
midpoints = [(x_min_plot + xa1)/2, (xa1 + xb1)/2, (xb1 + xc1)/2, (xc1 + xc2)/2, (xc2 + xb2)/2, (xb2 + xa2)/2, (xa2 + x_max_plot)/2]
region_labels = ["Air", "Core", "Coil L", "Core", "Coil R", "Core", "Air"]

# Plot Magnitudes
p1 = plot(x_int * 1e2, Az_mag_vals * 1e5, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"|A_z(x)|\ \mathrm{[mWb/cm]}", color=:black, lw=1, legend=false, title=L"|A_z|" *" Magnitude")
p2 = plot(x_int * 1e2, B_mag_vals * 1e3, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"|B_y(x)|\ \mathrm{[mT]}", color=:black, lw=1, legend=false, title=L"|B_y|" *" Magnitude")
p3 = plot(x_int * 1e2, Jeddy_mag_vals * 1e-4, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"|J_{eddy}(x)|\ \mathrm{[A/cm^2]}", color=:black, lw=1, legend=false, title=L"|J_{eddy}|" *" Magnitude")
p4 = plot(x_int * 1e2, μ_vals_linear, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"\mu(x)\ \mathrm{[m/H]}", color=:black, lw=1, legend=false, title="Permeability (Linear)")

# Add annotations
for p in [p1, p2, p3, p4]
    vline!(p, boundaries * 1e2, color=:grey, linestyle=:dash, label="")
    plot_ylims = Plots.ylims(p)
    label_y = plot_ylims[1] - 0.08 * (plot_ylims[2] - plot_ylims[1])
    annotate!(p, [(midpoints[i]*1e2, label_y, text(region_labels[i], 8, :center, :top)) for i in eachindex(midpoints)])
end

plt_mag = plot(p1, p2, p3, p4, layout=(4,1), size=(800, 1200))
savefig(plt_mag, joinpath(paths["OUTPUT_DIR"], "magnetodynamics_harmonic_coupled_magnitudes.pdf"))
display(plt_mag)

# %% [markdown]
# ## Visualization (Animation)

# %%
# Create animation over one period
T_period = 1/freq
t_vec = range(0, T_period, length=100)

anim = @animate for t_step in t_vec
    # Calculate instantaneous real value: Re( (u+iv) * exp(jωt) ) = u*cos(ωt) - v*sin(ωt)
    cos_wt = cos(ω * t_step)
    sin_wt = sin(ω * t_step)
    
    Az_inst = u * cos_wt - v * sin_wt
    B_re_inst = B_re * cos_wt - B_im * sin_wt # Instantaneous B_re
    Jeddy_inst = J_eddy_re * cos_wt - J_eddy_im * sin_wt
    
    # Evaluate at interpolation points
    Az_inst_vals = Az_inst(coord)
    B_re_inst_vals = B_re_inst(coord)
    By_inst_vals = [b[1] for b in B_re_inst_vals] # Extract y-component
    Jeddy_inst_vals = Jeddy_inst(coord)
    
    # Get magnitude limits for consistent y-axis scaling
    Az_max = maximum(Az_mag_vals)
    By_max = maximum(B_mag_vals)
    Jeddy_max = maximum(Jeddy_mag_vals)

    # Plot instantaneous real parts at time t
    p1_t = plot(x_int * 1e2, Az_inst_vals * 1e5, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"A_z(x,t)\ \mathrm{[mWb/cm]}", color=:blue, lw=1, legend=false, title=@sprintf("Time-Harmonic (t = %.2e s)", t_step), ylims=(-Az_max*1.1e5, Az_max*1.1e5))
    p2_t = plot(x_int * 1e2, By_inst_vals * 1e3, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"B_y(x,t)\ \mathrm{[mT]}", color=:blue, lw=1, legend=false, ylims=(-By_max*1.1e3, By_max*1.1e3))
    p3_t = plot(x_int * 1e2, Jeddy_inst_vals * 1e-4, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"J_{eddy}(x,t)\ \mathrm{[A/cm^2]}", color=:red, lw=1, legend=false, ylims=(-Jeddy_max*1.1e-4, Jeddy_max*1.1e-4))
    p4_t = plot(x_int * 1e2, μ_vals_linear, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"\mu(x)\ \mathrm{[m/H]}", color=:black, lw=1, legend=false, title="Permeability (Linear)")

    # Add annotations
    for p in [p1_t, p2_t, p3_t, p4_t]
        vline!(p, boundaries * 1e2, color=:grey, linestyle=:dash, label="")
        plot_ylims = Plots.ylims(p)
        label_y = plot_ylims[1] - 0.08 * (plot_ylims[2] - plot_ylims[1])
        annotate!(p, [(midpoints[i]*1e2, label_y, text(region_labels[i], 8, :center, :top)) for i in eachindex(midpoints)])
    end
    
    plot(p1_t, p2_t, p3_t, p4_t, layout=(4,1), size=(800, 1200))
end

gif(anim, joinpath(paths["OUTPUT_DIR"], @sprintf("magnetodynamics_harmonic_coupled_animation(f=%.2e).gif", freq)), fps = 15)

# %%



