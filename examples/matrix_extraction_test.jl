# %% [markdown]
# # Exercise 2: 1D Magnetodynamics (Time-Harmonic)

# %%
include(joinpath(dirname(@__DIR__), "config.jl"))
paths = get_project_paths("examples")
include("../src/MagnetostaticsFEM.jl")

using LinearAlgebra
using Plots
using LaTeXStrings
using Gridap
using .MagnetostaticsFEM
using Printf # For animation title formatting
using SparseArrays # For nnz function

# %% [markdown]
# ## Define Parameters and Paths

# %%
# Model Parameters
J0 = 2.2e4       # Source current density [A/m²] (Assumed Real)
μ0 = 4e-7 * pi  # Vacuum permeability [H/m]
μr_core = 50000.0 # Relative permeability of the core
σ_core = 1e7    # Conductivity of the core [S/m]
freq = 50    # Frequency [Hz]
ω = 2 * pi * freq # Angular frequency [rad/s]

# FEM Parameters
order = 2
field_type = ComplexF64 # Still use ComplexF64 marker for setup_fe_spaces
dirichlet_tag = "D"
dirichlet_value = 0.0 + 0.0im # Dirichlet BC for A = u + iv

# Paths
mesh_file = joinpath(paths["GEO_DIR"], "coil_geo.msh")
output_file_base = joinpath(OUT_DIR, "magnetodynamics_harmonic_coupled")

println("Mesh file: ", mesh_file)

# %% [markdown]
# ## Setup FEM Problem

# %%
# Load mesh and tags
model, labels, tags = MagnetostaticsFEM.load_mesh_and_tags(mesh_file)

# Get material tags dictionary
material_tags = MagnetostaticsFEM.get_material_tags(labels)

# Set up triangulation and measures
Ω = Triangulation(model)
dΩ = Measure(Ω, 2*order)

# Define material property functions
reluctivity_func = MagnetostaticsFEM.define_reluctivity(material_tags, μ0, μr_core)
conductivity_func = MagnetostaticsFEM.define_conductivity(material_tags, σ_core)
source_current_func = MagnetostaticsFEM.define_current_density(material_tags, J0) # Real source

# Setup FE spaces (multi-field: Real, Imag parts)
U, V = MagnetostaticsFEM.setup_fe_spaces(model, order, field_type, dirichlet_tag, dirichlet_value)

# Define the weak form problem for the coupled system
problem = MagnetostaticsFEM.magnetodynamics_harmonic_coupled_weak_form(Ω, dΩ, tags, reluctivity_func, conductivity_func, source_current_func, ω)

# %% [markdown]
# ## Solve FEM Problem

# %%
# Solve the real coupled linear FE system
uv = MagnetostaticsFEM.solve_fem_problem(problem, U, V) # uv is a MultiFieldFEFunction

# Extract real and imaginary parts
u = uv[1] # Real part of Az
v = uv[2] # Imag part of Az


# %% [markdown]
# ## Post-processing

# %%
# Compute B-field (Real and Imag parts)
B_re, B_im = MagnetostaticsFEM.calculate_b_field(uv)

# Compute Eddy Currents (Real and Imag parts)
# Ensure conductivity_func here is for electrical conductivity
J_eddy_re, J_eddy_im = MagnetostaticsFEM.calculate_eddy_current(uv, conductivity_func, ω, Ω, tags)

# Define helper functions for magnitude squared
mag_sq_scalar(re, im) = re*re + im*im
mag_sq_vector(re, im) = inner(re, re) + inner(im, im)

# Calculate Magnitudes for saving/plotting using composition
Az_mag = sqrt ∘ (mag_sq_scalar ∘ (u, v))
B_mag = sqrt ∘ (mag_sq_vector ∘ (B_re, B_im))
Jeddy_mag_squared = mag_sq_vector ∘ (J_eddy_re, J_eddy_im)
Jeddy_mag = sqrt ∘ (mag_sq_scalar ∘ (J_eddy_re, J_eddy_im))

# Calculate reluctivity field for visualization (linear case) and matrix-extraction
# reluctivity_func is already defined: maps tag to reluctivity value
τ_cell_field = CellField(tags, Ω) # 'tags' is the vector of cell tags, Ω is Triangulation
ν_field_linear = Operation(MagnetostaticsFEM.define_reluctivity(material_tags, μ0, μr_core))(τ_cell_field)
σ_cell_field = Operation(MagnetostaticsFEM.define_conductivity(material_tags, σ_core))(τ_cell_field)
Js_re_cell_field = Operation(MagnetostaticsFEM.define_current_density(material_tags, J0))(τ_cell_field)


# Get S, M, f_re and the scalar trial space
S_matrix, M_matrix, f_re_vector, U_scalar_trial = MagnetostaticsFEM.get_fem_matrices_and_vector(
    model, order,
    dirichlet_tag, real(dirichlet_value), # dirichlet_value is 0.0+0.0im
    Ω, dΩ,
    ν_field_linear,   # Reluctivity CellField
    σ_cell_field,     # Conductivity CellField
    Js_re_cell_field  # Source Current (real part) CellField
)

println("Stiffness Matrix S: size $(size(S_matrix)), non-zeros $(nnz(S_matrix))")
println("Mass Matrix M: size $(size(M_matrix)), non-zeros $(nnz(M_matrix))")
println("Load Vector f_re: length $(length(f_re_vector))")

# Save matrices and vector
MagnetostaticsFEM.save_data_serialized(joinpath(MATRICES_DIR, "S_matrix.dat"), S_matrix)
MagnetostaticsFEM.save_data_serialized(joinpath(MATRICES_DIR, "M_matrix.dat"), M_matrix)
MagnetostaticsFEM.save_data_serialized(joinpath(MATRICES_DIR, "f_re_vector.dat"), f_re_vector)

# Load matrices and vector (example)
S_loaded = MagnetostaticsFEM.load_data_serialized(joinpath(MATRICES_DIR, "S_matrix.dat"))
M_loaded = MagnetostaticsFEM.load_data_serialized(joinpath(MATRICES_DIR, "M_matrix.dat"))
f_re_loaded = MagnetostaticsFEM.load_data_serialized(joinpath(MATRICES_DIR, "f_re_vector.dat"))

# Basic check of loaded data
@assert S_matrix ≈ S_loaded "Loaded S matrix does not match original."
@assert M_matrix ≈ M_loaded "Loaded M matrix does not match original."
@assert f_re_vector ≈ f_re_loaded "Loaded f_re vector does not match original."
println("Successfully saved and loaded S, M, f_re.")

# %% [markdown]
# ## Solve (S + jωM)u = f for different frequencies and Compare

# %%
# f_im is zero because source current is real
ndofs = size(S_loaded, 1)
f_im_vector = zeros(Float64, ndofs)
# Construct the complex load vector f = f_re + j*f_im
f_complex_vec = Complex.(f_re_loaded, f_im_vector) # Ensure it's complex vector

# Test with different ω values
ω_values_test = [ω/2, ω, ω*2] # Test original, half, and double frequency

for current_ω_test in ω_values_test
    println("\nSolving for ω = $(current_ω_test) rad/s using S, M, f:")

    # Form the complex system matrix A_complex = S + jωM
    A_complex = S_loaded + im * current_ω_test * M_loaded

    # Solve the complex linear system
    u_complex_dofs = A_complex \ f_complex_vec

    println("Solved system. First 3 DoFs (complex): $(u_complex_dofs[1:min(3,end)])")

    # If current_ω_test is the original ω, compare with the original solution from solve_fem_problem
    if abs(current_ω_test - ω) < 1e-9 # Compare floating point numbers carefully
        println("Comparing with original solution (from solve_fem_problem at ω = $(ω)):")

        u_orig_dofs = get_free_dof_values(u) # Real part from original solution
        v_orig_dofs = get_free_dof_values(v) # Imaginary part from original solution

        u_re_from_SMf = real(u_complex_dofs)
        u_im_from_SMf = imag(u_complex_dofs)

        # Print max differences for verification
        max_diff_re = maximum(abs.(u_re_from_SMf - u_orig_dofs))
        max_diff_im = maximum(abs.(u_im_from_SMf - (-v_orig_dofs))) # or abs.(u_im_from_SMf + v_orig_dofs)
        println("Max difference in real part: $(max_diff_re)")
        println("Max difference in imaginary part (u_im_SMf vs -v_orig): $(max_diff_im)")

        # Add assertions for automated testing (adjust tolerance as needed)
        # Tolerance might need to be adjusted based on solver precision and problem conditioning
        tolerance = 1e-8
        @assert max_diff_re < tolerance "Real parts do not match original solution within tolerance $(tolerance)."
        @assert max_diff_im < tolerance "Imaginary parts (after sign correction) do not match original solution within tolerance $(tolerance)."
        println("Solution from S, M, f matches original solution (with sign correction for imaginary part) at ω = $(ω).")

        # Optional: Reconstruct FEFunctions for visualization (if needed later)
        # Az_re_from_SMf_fe = FEFunction(U_scalar_trial, u_re_from_SMf)
        # Az_im_from_SMf_fe = FEFunction(U_scalar_trial, u_im_from_SMf)
        # println("FEFunctions reconstructed from S,M,f solution.")
    end
end

# Save results to VTK format
fields_for_vtk = Dict{String, Any}(
    "Az_re" => u, "Az_im" => v, "Az_mag" => Az_mag,
    "B_re" => B_re, "B_im" => B_im, "B_mag" => B_mag,
    "Jeddy_re" => J_eddy_re, "Jeddy_im" => J_eddy_im, "Jeddy_mag" => Jeddy_mag,
    "ν_linear" => ν_field_linear # ν_field_linear was from Operation(MagnetostaticsFEM.define_reluctivity(...))(τ_cell_field)
)
MagnetostaticsFEM.save_results_vtk(Ω, output_file_base, fields_for_vtk)

# %%

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
savefig(plt_mag, joinpath(OUT_DIR, "magnetodynamics_harmonic_coupled_magnitudes.pdf"))
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

gif(anim, joinpath(OUT_DIR, @sprintf("magnetodynamics_harmonic_coupled_animation(f=%.2e).gif", freq)), fps = 15)

# %%
