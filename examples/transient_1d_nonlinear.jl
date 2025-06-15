# %% [markdown]
# # Exercise 4: 1D Transient Magnetodynamics (Nonlinear)

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
using Printf
using Plots.PlotMeasures

# %% [markdown]
# ## Define Parameters and Paths

# %%
# Model Parameters (enhanced for nonlinear behavior)
J0_amplitude = 2.2e4  # Source current density amplitude [A/m²]
μ0 = 4e-7 * pi     # Vacuum permeability [H/m]
μr_core = 5000.0    # Initial relative permeability of the core
σ_core = 1e7       # Conductivity of the core [S/m]
freq = 50.0          # Frequency of the source current [Hz]
ω_source = 2 * pi * freq # Angular frequency [rad/s]

# B-H curve parameters (Frohlich-Kennelly model)
bh_a = 1 / 12000         # Parameter a in B-H curve (affects low field behavior)  
bh_b = 4.          # Parameter b in B-H curve (affects saturation transition)
bh_c = 7.5          # Parameter c in B-H curve (affects saturation level)

# FEM Parameters
order_fem = 4
dirichlet_tag = "D"

# Define a single dirichlet_bc_func with multiple dispatch a la original script
dirichlet_bc_func(t::Float64) = x -> 0.0  # For TransientTrialFESpace g(t) -> h(x)
dirichlet_bc_func(x::Point, t::Real) = 0.0 # For Gridap internals needing g(x,t) (e.g. for derivatives)

# --- Transient Simulation Parameters ---
t0 = 0.0
periods_to_simulate = 5  # More periods for better frequency resolution
tF = periods_to_simulate / freq # Simulate for 5 periods
num_steps_per_period = 50  # Higher sampling rate to reach 1000 Hz (Nyquist = 1000 Hz needs 2000 Hz sampling)
num_periods_collect_fft = 3 # Use last N periods for FFT to avoid initial transient effects
Δt_val = (1/freq) / num_steps_per_period # Time step size, renamed to Δt_val to avoid conflict with module Δt
θ_method = 0.5 # Crank-Nicolson (0.5), BE (1.0), FE (0.0)

# Paths
mesh_file = joinpath(paths["GEO_DIR"], "coil_geo_new.msh")
output_dir = joinpath(paths["OUTPUT_DIR"], "transient_1d_nonlinear_results")
if !isdir(output_dir)
    mkpath(output_dir)
end

println("Mesh file: ", mesh_file)
println("Output directory: ", output_dir)

# %% [markdown]
# ## Geometry Setup and Color Scheme

# %%
# Define geometry boundaries for plotting (based on 1d_mesh_w_oil_reservois.jl)
a_len = 100.3e-3; b_len = 73.15e-3; c_len = 27.5e-3
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
# ## Solve Nonlinear Transient Problem

# %%
# Call the nonlinear transient solver
println("Solving nonlinear transient magnetodynamics with B-H curve...")
solution_transient_iterable, Az0_out, Ω_out, ν_cf_out, σ_cf_out, Js_t_func_out, model_out, tags_cf_out, labels_out = 
    MagnetostaticsFEM.solve_nonlinear_transient_magnetodynamics(
        mesh_file,
        order_fem, 
        dirichlet_tag,
        dirichlet_bc_func,  
        μ0,
        bh_a, bh_b, bh_c,  # B-H curve parameters
        σ_core,
        J0_amplitude,
        ω_source,
        t0,
        tF,
        Δt_val,
        θ_method;
        max_iterations_nl=10,  # Reduced for faster testing
        tolerance_nl=1e-4,     # Slightly relaxed tolerance
        damping_nl=0.8         # Strong damping for stability
    )

# %% [markdown]
# ## Post-processing and Visualization

# %%
# Extract signal at probe point
x_probe = VectorValue(-0.03) 
steps_for_fft_start_time = tF - (num_periods_collect_fft / freq)

# Fast signal extraction (skip expensive VTK saving for speed)
println("Extracting signal at probe point (fast mode)...")
time_steps_for_fft = Float64[]
time_signal_data = Float64[]

global step_count = 0
for (Az_n, tn) in solution_transient_iterable
    global step_count += 1
    if tn >= steps_for_fft_start_time
        push!(time_steps_for_fft, tn)
        try
            probe_val = Az_n(x_probe)
            if isa(probe_val, AbstractArray) && length(probe_val) == 1
                push!(time_signal_data, first(probe_val))
            elseif isa(probe_val, Number)
                push!(time_signal_data, Float64(probe_val))
            else
                push!(time_signal_data, NaN)
            end
        catch e
            println("Warning: Could not evaluate at probe point for t=$(tn)")
            push!(time_signal_data, NaN)
        end
    end
    if step_count % 20 == 0
        println("Extracted signal from step $(step_count) at t=$(tn)")
    end
end
println("Signal extraction completed for $(length(time_signal_data)) points")

# %%
# Process extracted signal
valid_indices = .!isnan.(time_signal_data)
time_steps_for_fft = time_steps_for_fft[valid_indices]
time_signal_data = time_signal_data[valid_indices]

if isempty(time_steps_for_fft)
    error("No time points collected for FFT. Check simulation time (tF=$(tF), Δt=$(Δt_val)), collection window, or probe point.\nCollected $(length(time_signal_data)) points before NaN filter.")
end

# Plot time signal with improved styling (consistent units with 1D-Harmonic.jl)
time_plot = plot(time_steps_for_fft, time_signal_data * 1e5, 
    xlabel="Time [s]", ylabel=L"A_z\ \mathrm{[mWb/cm]}", 
    title="Az (Nonlinear) at x=$(x_probe[1]) (last $(num_periods_collect_fft) periods)",
    lw=2, color=:blue, legend=false, bottom_margin=8mm)
savefig(time_plot, joinpath(output_dir, "transient_1d_nonlinear_signal.pdf"))
display(time_plot)

# %%
# FFT Analysis
println("Performing FFT analysis on nonlinear transient results...")
sampling_rate = 1/Δt_val

fft_frequencies, fft_magnitudes = MagnetostaticsFEM.perform_fft(time_signal_data, sampling_rate)

# Plot FFT with stem plot (vertical stripes) - consistent units with 1D-Harmonic.jl
max_freq_plot = 1000  # Plot up to 1000 Hz as requested
fft_plot = plot(fft_frequencies, fft_magnitudes * 1e5,
    xlabel="Frequency [Hz]", ylabel=L"Magnitude\ \mathrm{[mWb/cm]}",
    title="FFT Spectrum of Az (Nonlinear) at x=$(x_probe[1])",
    xlims=(0, max_freq_plot), seriestype=:sticks, lw=3, color=:blue, legend=false, bottom_margin=8mm)
savefig(fft_plot, joinpath(output_dir, "transient_1d_nonlinear_fft.pdf"))
display(fft_plot)

if !isempty(fft_magnitudes) && !isempty(fft_frequencies)
    max_magnitude_fft, idx_max = findmax(fft_magnitudes)
    if idx_max <= length(fft_frequencies)
        peak_frequency_fft = fft_frequencies[idx_max]
        println("FFT Analysis Results for Az (Nonlinear) at x=$(x_probe[1]):")
        println("  - Peak Amplitude (from FFT): $(max_magnitude_fft * 1e5) mWb/cm")
        println("  - Frequency at Peak: $(peak_frequency_fft) Hz")
        
        freq_resolution = sampling_rate / length(time_signal_data) 
        if abs(peak_frequency_fft - freq) < freq_resolution * 1.5 # Allow some tolerance for nonlinear case
            println("  - FFT peak frequency is close to source fundamental frequency of $(freq) Hz.")
        else
            println("  - WARNING: FFT peak frequency ($(peak_frequency_fft) Hz) does NOT closely match source fundamental frequency ($(freq) Hz). Resolution: $(freq_resolution) Hz")
        end
        
        # Look for harmonics (characteristic of nonlinear behavior)
        println("\\n  - Harmonic Analysis:")
        for harmonic in 2:5
            target_freq = harmonic * freq
            if target_freq <= maximum(fft_frequencies)
                # Find closest frequency to harmonic
                closest_idx = argmin(abs.(fft_frequencies .- target_freq))
                harmonic_magnitude = fft_magnitudes[closest_idx]
                harmonic_freq = fft_frequencies[closest_idx]
                if abs(harmonic_freq - target_freq) < freq_resolution * 2
                    harmonic_ratio = harmonic_magnitude / max_magnitude_fft * 100
                    println("    - $(harmonic)nd harmonic: $(harmonic_magnitude * 1e5) mWb/cm ($(harmonic_ratio)% of fundamental)")
                end
            end
        end
    else
        println("FFT analysis error: index of max magnitude is out of bounds for frequencies.")
    end
else
    println("FFT analysis could not be performed (no magnitudes or frequencies found).")
end

# %% [markdown] 
# ## Enhanced Visualization with Mesh Annotations

# %%
# Create enhanced plots of transient fields at specific time points
x_int = collect(range(-0.1, 0.1, length=1000))
coord = [VectorValue(x_) for x_ in x_int]

# Enhanced field visualization with mesh annotations  
# For TransientFESolution, we need to collect solutions first or use the last one
let solution_collection = collect(solution_transient_iterable)
    if length(solution_collection) > 5  # Reduced threshold for nonlinear case
        peak_time_idx = Int(round(length(solution_collection) * 0.75))  # 3/4 through simulation
        Az_peak, _ = solution_collection[peak_time_idx]
        
        # Evaluate fields
        Az_vals = Az_peak(coord)
        B_peak = calculate_b_field(Az_peak)
        B_vals = B_peak(coord)
        By_vals = [b[1] for b in B_vals]
        
        # Plot transient fields with annotations
        p1 = plot(x_int * 1e2, Az_vals * 1e5, xlabel=L"x\\ \\mathrm{[cm]}", ylabel=L"A_z(x,t)\\ \\mathrm{[mWb/cm]}", 
                  color=:blue, lw=1.5, legend=false, title="Transient Az (Nonlinear, t = peak)", bottom_margin=8mm)
        p2 = plot(x_int * 1e2, By_vals * 1e3, xlabel=L"x\\ \\mathrm{[cm]}", ylabel=L"B_y(x,t)\\ \\mathrm{[mT]}", 
                  color=:blue, lw=1.5, legend=false, title="Transient By (Nonlinear, t = peak)", bottom_margin=8mm)
        
        # Add annotations with color scheme and background tinting
        for p in [p1, p2]
            # Add background tinting
            add_region_backgrounds!(p, boundaries_em, region_labels_em, [x_min_em, x_max_em])
            
            vline!(p, boundaries_em * 1e2, color=:grey, linestyle=:dash, alpha=0.6, label="")
            plot_ylims = Plots.ylims(p)
            label_y = plot_ylims[1] - 0.06 * (plot_ylims[2] - plot_ylims[1])
            
            # Add region labels with color scheme
            for i in eachindex(midpoints_em)
                color = get_region_color(region_labels_em[i])
                annotate!(p, midpoints_em[i]*1e2, label_y, text(region_labels_em[i], 9, color, :center, :top))
            end
        end
        
        plt_transient = plot(p1, p2, layout=(2,1), size=(800, 600), 
                            bottom_margin=10mm, left_margin=5mm, right_margin=5mm)
        savefig(plt_transient, joinpath(output_dir, "transient_1d_nonlinear_fields_annotated.pdf"))
        display(plt_transient)
    else
        println("Not enough time steps for field visualization")
    end
end

# %%
# Generate Enhanced Animation (disabled for speed - enable if needed)
if false  # Set to true if you want animation (slow!)
    transient_animation_path = joinpath(output_dir, "transient_1d_nonlinear_enhanced_animation.gif")
    MagnetostaticsFEM.create_transient_animation(
        Ω_out, 
        solution_transient_iterable, 
        σ_cf_out,
        Δt_val,
        Az0_out,
        transient_animation_path, 
        fps=10,
        consistent_axes=true
    )
    println("Animation saved to: ", transient_animation_path)
else
    println("Animation generation skipped for speed (set flag to true if needed)")
end

println("\n✓ Nonlinear Transient 1D simulation completed successfully!")
println("Results saved to: ", output_dir)
println("Enhanced plots saved with mesh annotations and color coding")
println("\nNote: This version uses true nonlinear B-H curve material properties.")
println("Frohlich-Kennelly model parameters: a=$(bh_a), b=$(bh_b), c=$(bh_c)")
println("Magnetic saturation effects should be visible in harmonic generation.")
