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
order = 4
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

# For thermal plots (wide range): Air | Oil | Transformer | Oil | Air
xa1 = -reservoir_width/2  # Left oil boundary
xa4 = reservoir_width/2   # Right oil boundary
boundaries_thermal = [xa1, xa2, xa3, xa4]  # 4 boundaries for thermal plots (5 regions)

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

# Calculate midpoints for thermal plots (5 regions: detailed view)
x_min_thermal = -1.0; x_max_thermal = 1.0  # Thermal plot range
# Define oil region boundaries (based on mesh geometry)
oil_left_start = xa1    # Left oil boundary
oil_left_end = xa2      # Left oil ends at left core boundary
oil_right_start = xa3   # Right oil starts at right core boundary
oil_right_end = xa4     # Right oil boundary

offset_avoid_overlap = 0.15  # Small offset to avoid overlap in oil regions
midpoints_thermal = [
    (x_min_thermal + oil_left_start)/2,     # Air (left)
    (oil_left_start + oil_left_end)/2 - offset_avoid_overlap,      # Oil (left) - positioned to avoid overlap
    (oil_left_end + oil_right_start)/2,     # Transformer (core + coils)
    (oil_right_start + oil_right_end)/2 + offset_avoid_overlap,    # Oil (right) - positioned to avoid overlap
    (oil_right_end + x_max_thermal)/2       # Air (right)
]
region_labels_thermal = ["Air", "Oil", "Transformer", "Oil", "Air"]

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
eddy_loss_density = (x) -> begin
    B_local = B_mag(VectorValue(x[1])) # Ensure x is properly indexed for evaluation
    (π^2 / 6) * Ke * (B_local^2) * (mt^2) * (freq^2) * (a_len^2/3) / 1e-7
end

hysterisis_loss_density = (x) -> 0.0  

eddy_loss_vals = [eddy_loss_density(x) for x in coord]
# Plot the eddy loss density with simplified thermal annotations
eddy_loss_plot = plot(x_int * 1e2, eddy_loss_vals, xlabel="x [cm]", ylabel="Eddy Loss Density [W/m³]", title="Eddy Current Loss Density", color=:blue, lw=1.5, legend=false)

# Add simplified mesh annotations to eddy loss plot
vline!(eddy_loss_plot, boundaries_thermal * 1e2, color=:grey, linestyle=:dash, alpha=0.6, label="")
plot_ylims = Plots.ylims(eddy_loss_plot)
label_y = plot_ylims[1] - 0.10 * (plot_ylims[2] - plot_ylims[1])
for i in eachindex(midpoints_thermal)
    color = if region_labels_thermal[i] == "Transformer"
        :red
    elseif region_labels_thermal[i] == "Oil"
        :blue
    else  # Air
        :black
    end
    annotate!(eddy_loss_plot, midpoints_thermal[i]*1e2, label_y, text(region_labels_thermal[i], 10, color, :center, :top))
end

savefig(eddy_loss_plot, joinpath(paths["OUTPUT_DIR"], "eddy_loss_density.png"))
display(eddy_loss_plot)

# Set dissipation coefficient and ambient temperature for the loss region (e.g., air)
dissipation_coeff = 0.25  # Increase for stronger exponential decay in the loss region
T_ambient = 273.15 + 20.0 # Ambient temperature in Kelvin (20°C)
T_ref_K = 273.15 + 110.0 # Reference temperature for AAF calculation

dissipation_func = define_loss_region_dissipation(material_tags, dissipation_coeff; region_tag_name="Air")

# Now solve the heat equation using the dissipation term in the specified region
T = solve_heatdynamics_with_dissipation(
    model, tags, order, dirichlet_tag, 0.0,
    eddy_loss_density, heat_conductivity_func,
    dissipation_func, T_ambient
)

x_int = collect(range(-1.0, 1.0, length=1000))
coord = [VectorValue(x_) for x_ in x_int]

T_vals = T(coord) # T is already in Kelvin
T_peak = minimum(T_vals)

# Define activation energy and gas constant for AAF calculation
E = 110e3 # Activation energy [J/mol], typical for transformer oil
R = 8.314 # Gas constant [J/(mol·K)]

# Oil region boundaries (in meters)
oil_xmin = -0.191175 # Oil region starts at -6.3725 cm
oil_xmax = -0.05015  # Oil region ends at -5.015 cm
oil_xmin_r = 0.05015   # Right oil region starts at 5.015 cm
oil_xmax_r = 0.191175  # Right oil region ends at 19.1175 cm

# Calculate AAF only in oil regions (left and right), NaN elsewhere
AAF_vals = [(((x[1] ≥ oil_xmin) && (x[1] ≤ oil_xmax)) ||
             ((x[1] ≥ oil_xmin_r) && (x[1] ≤ oil_xmax_r))) && T_val > 0.0 ?
             exp(E/R * (1/T_ref_K - 1/T_val)) : NaN
            for (x, T_val) in zip(coord, T_vals)]

# Plot and save the AAF (aging) plot with simplified thermal annotations
AAF_plot = plot(x_int * 1e2, AAF_vals, xlabel="x [cm]", ylabel="Aging Acceleration Factor",
     title="Oil Aging Acceleration Factor (Oil Only)", lw=2.5, color=:green)

# Add simplified mesh annotations to AAF plot
vline!(AAF_plot, boundaries_thermal * 1e2, color=:grey, linestyle=:dash, alpha=0.6, label="")
plot_ylims = Plots.ylims(AAF_plot)
label_y_aaf = plot_ylims[1] - 0.10 * (plot_ylims[2] - plot_ylims[1])
for i in eachindex(midpoints_thermal)
    color = if region_labels_thermal[i] == "Transformer"
        :red
    elseif region_labels_thermal[i] == "Oil"
        :blue
    else  # Air
        :black
    end
    annotate!(AAF_plot, midpoints_thermal[i]*1e2, label_y_aaf, text(region_labels_thermal[i], 10, color, :center, :top))
end

savefig(AAF_plot, joinpath(paths["OUTPUT_DIR"], "aging_acceleration_factor.png"))
display(AAF_plot)

# Plot and save the temperature profile with simplified thermal annotations
heat_plot = plot(x_int * 1e2, T_vals, xlabel="x [cm]", ylabel="Temperature [K]", title="Temperature Profile", lw=2.5, color=:red)

# Add simplified mesh annotations to temperature plot
vline!(heat_plot, boundaries_thermal * 1e2, color=:grey, linestyle=:dash, alpha=0.6, label="")
plot_ylims = Plots.ylims(heat_plot)
label_y_temp = plot_ylims[1] - 0.10 * (plot_ylims[2] - plot_ylims[1])
for i in eachindex(midpoints_thermal)
    color = if region_labels_thermal[i] == "Transformer"
        :red
    elseif region_labels_thermal[i] == "Oil"
        :blue
    else  # Air
        :black
    end
    annotate!(heat_plot, midpoints_thermal[i]*1e2, label_y_temp, text(region_labels_thermal[i], 10, color, :center, :top))
end

savefig(heat_plot, joinpath(paths["OUTPUT_DIR"], "temperature_profile.png"))
display(heat_plot)


# %% [markdown]
# ## Visualization (Magnitudes)

# %%

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

# Plot Magnitudes
p1 = plot(x_int * 1e2, Az_mag_vals * 1e5, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"|A_z(x)|\ \mathrm{[mWb/cm]}", color=:black, lw=1, legend=false, title=L"|A_z|" *" Magnitude")
p2 = plot(x_int * 1e2, B_mag_vals * 1e3, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"|B_y(x)|\ \mathrm{[mT]}", color=:black, lw=1, legend=false, title=L"|B_y|" *" Magnitude")
p3 = plot(x_int * 1e2, Jeddy_mag_vals * 1e-4, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"|J_{eddy}(x)|\ \mathrm{[A/cm^2]}", color=:black, lw=1, legend=false, title=L"|J_{eddy}|" *" Magnitude")
p4 = plot(x_int * 1e2, μ_vals_linear, xlabel=L"x\ \mathrm{[cm]}", ylabel=L"\mu(x)\ \mathrm{[m/H]}", color=:black, lw=1, legend=false, title="Permeability (Linear)")

# Add annotations with improved legibility (electromagnetic plots)
for p in [p1, p2, p3, p4]
    vline!(p, boundaries_em * 1e2, color=:grey, linestyle=:dash, alpha=0.6, label="")
    plot_ylims = Plots.ylims(p)
    label_y = plot_ylims[1] - 0.10 * (plot_ylims[2] - plot_ylims[1])
    # Add region labels with better formatting
    for i in eachindex(midpoints_em)
        color = if region_labels_em[i] == "Oil"
            :blue
        elseif region_labels_em[i] == "Core"
            :red
        elseif occursin("Coil", region_labels_em[i])
            :orange
        else
            :black
        end
        annotate!(p, midpoints_em[i]*1e2, label_y, text(region_labels_em[i], 9, color, :center, :top))
    end
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

    # Add annotations with improved legibility (electromagnetic animation)
    for p in [p1_t, p2_t, p3_t, p4_t]
        vline!(p, boundaries_em * 1e2, color=:grey, linestyle=:dash, alpha=0.6, label="")
        plot_ylims = Plots.ylims(p)
        label_y = plot_ylims[1] - 0.10 * (plot_ylims[2] - plot_ylims[1])
        # Add region labels with better formatting
        for i in eachindex(midpoints_em)
            color = if region_labels_em[i] == "Oil"
                :blue
            elseif region_labels_em[i] == "Core"
                :red
            elseif occursin("Coil", region_labels_em[i])
                :orange
            else
                :black
            end
            annotate!(p, midpoints_em[i]*1e2, label_y, text(region_labels_em[i], 9, color, :center, :top))
        end
    end
    
    plot(p1_t, p2_t, p3_t, p4_t, layout=(4,1), size=(800, 1200))
end

gif(anim, joinpath(paths["OUTPUT_DIR"], @sprintf("magnetodynamics_harmonic_coupled_animation(f=%.2e).gif", freq)), fps = 15)
