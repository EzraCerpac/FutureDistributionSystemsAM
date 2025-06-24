# %% [markdown]
# # Exercise 4: 1D Transient Magnetodynamics (Nonlinear)

# %%
include(joinpath(dirname(@__DIR__), "config.jl"))
paths = get_project_paths("examples")

# Ensure the module is reloaded if changed
if isdefined(Main, :MagnetostaticsFEM)
    println("Reloading MagnetostaticsFEM...")
    # A simple way to force reload in interactive sessions
    try
        delete!(LOAD_PATH, joinpath(paths["SRC_DIR"], "src"))
    catch
    end
    try
        delete!(Base.loaded_modules, Base.PkgId(Base.UUID("f8a2b3c4-d5e6-f7a8-b9c0-d1e2f3a4b5c6"), "MagnetostaticsFEM"))
    catch
    end
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
# # Paths
mesh_file = joinpath(paths["GEO_DIR"], "coil_geo.msh")
output_dir = joinpath(paths["FIGURES_DIR"], "transient_nonlinear_results")
if !isdir(output_dir)
    mkpath(output_dir)
end

# %%
# Model Parameters (enhanced for nonlinear behavior)
J0_amplitude = 2.2e4   # Fine-tuned current density for ~35 mWb/cm amplitude with nonlinear effects [A/m²]
μ0 = 4e-7 * pi     # Vacuum permeability [H/m]
μr_core = 5000.0    # Initial relative permeability of the core
σ_core = 1e7       # Conductivity of the core [S/m]
freq = 50.0          # Frequency of the source current [Hz]
ω_source = 2 * pi * freq # Angular frequency [rad/s]

# Debug: Print model parameters
println("=== Model Parameters ===")
println("  J0_amplitude = $(J0_amplitude) A/m²")
println("  μ0 = $(μ0) H/m")
println("  σ_core = $(σ_core) S/m")
println("  freq = $(freq) Hz")

# B-H curve parameters (Frohlich-Kennelly model: μr = 1/(a + (1-a)*B^(2b)/(B^(2b) + c)))
bh_a = 1 / 15000.0     # Parameter a: μr(B→0) = 1/a = 5000 (low field behavior)
bh_b = 1.5          # Parameter b: saturation transition sharpness
bh_c = 4500.        # Parameter c: very low saturation field to match achievable B levels

# Debug: Print B-H curve parameters
println("=== B-H Curve Parameters ===")
println("  bh_a = $(bh_a) → μr(B→0) = $(1/bh_a)")
println("  bh_b = $(bh_b)")
println("  bh_c = $(bh_c) → B_knee ≈ $(round(bh_c^(1/(2*bh_b)), digits=6)) T")
println("  Expected μr at B_knee: $(1/(bh_a + (1-bh_a)*0.5))")

# Display BH curve
function fmur_core(B)
    return 1.0 / (bh_a + (1 - bh_a) * B^(2*bh_b) / (B^(2*bh_b) + bh_c))
end

B_vals = collect(0:0.01:3.)  # B field values in T
μr_vals = fmur_core.(B_vals)  # Relative permeability values
H_vals = B_vals ./ (μ0 .* μr_vals)  # H field values in A/m

p1 = plot(B_vals, μr_vals, xlabel="B [T]", ylabel="μr", label="Relative Permeability", title="BH Curve Properties")
p2 = plot(H_vals, B_vals, xlabel="H [A/m]", ylabel="B [T]", label="B(H) Curve")
bh_plot = plot(p1, p2, layout=(1,2), size=(900, 400))
savefig(bh_plot, joinpath(output_dir, "bh_curve.png"))

# FEM Parameters
order_fem = 4
dirichlet_tag = "D"

# Define a single dirichlet_bc_func with multiple dispatch a la original script
dirichlet_bc_func(t::Float64) = x -> 0.0  # For TransientTrialFESpace g(t) -> h(x)
dirichlet_bc_func(x::Point, t::Real) = 0.0 # For Gridap internals needing g(x,t) (e.g. for derivatives)

# --- Transient Simulation Parameters ---
t0 = 0.0
periods_to_simulate = 10  # Quick test with 2 periods
tF = periods_to_simulate / freq # Simulate for 2 periods
num_steps_per_period = 100  # Smaller steps for faster testing (250 Hz Nyquist)
num_periods_collect_fft = 5 # Use last 10 period for FFT
Δt_val = (1 / freq) / num_steps_per_period # Time step size, renamed to Δt_val to avoid conflict with module Δt
θ_method = 0.5 # Crank-Nicolson (0.5), BE (1.0), FE (0.0)

println("Mesh file: ", mesh_file)
println("Output directory: ", output_dir)

# %% [markdown]
# ## Geometry Setup and Color Scheme

# %%
# Define geometry boundaries for plotting (based on 1d_mesh_w_oil_reservois.jl)
a_len = 100.3e-3;
b_len = 73.15e-3;
c_len = 27.5e-3;
reservoir_width = 3 * (2 * a_len - b_len)  # Oil reservoir width from mesh generator

# For electromagnetic plots (narrow range): Oil | Core | Coil L | Core | Coil R | Core | Oil
xa2 = -a_len / 2           # Left core boundary
xb1 = -b_len / 2           # Left coil boundary
xc1 = -c_len / 2           # Core center left
xc2 = c_len / 2            # Core center right
xb2 = b_len / 2            # Right coil boundary
xa3 = a_len / 2            # Right core boundary
boundaries_em = [xa2, xb1, xc1, xc2, xb2, xa3]  # 6 boundaries for electromagnetic plots

# Calculate midpoints for electromagnetic plots (7 regions: Oil-centered view)
x_min_em = -0.1;
x_max_em = 0.1;  # Electromagnetic plot range
midpoints_em = [
    (x_min_em + xa2) / 2,    # Oil (left)
    (xa2 + xb1) / 2,         # Core (left)
    (xb1 + xc1) / 2,         # Coil L
    (xc1 + xc2) / 2,         # Core (center)
    (xc2 + xb2) / 2,         # Coil R
    (xb2 + xa3) / 2,         # Core (right)
    (xa3 + x_max_em) / 2     # Oil (right)
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
        x_end = i == length(x_boundaries) + 1 ? x_range[2] : x_boundaries[i]

        region_name = i <= length(region_labels) ? region_labels[i] : "Air"
        base_color = get_region_color(region_name)

        # Add very light background tint
        plot_ylims = Plots.ylims(p)
        vspan!(p, [x_start * 1e2, x_end * 1e2], color=base_color, alpha=0.1, label="")
    end
end

# %% [markdown]
# ## Solve Nonlinear Transient Problem

# %%
# Call the nonlinear transient solver
println("=== Starting Nonlinear Transient Solver ===")
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
        max_iterations_nl=15,  # More iterations to handle strong nonlinearity
        tolerance_nl=1e-6,     # Tighter tolerance to ensure convergence
        damping_nl=0.5         # Reduced damping to allow larger updates
    )
println("✓ Solver completed successfully!")

# %% [markdown]
# ## Post-processing and Visualization

# %%
# Multi-probe FFT analysis with averaging and individual probe plots (Nonlinear case)
steps_for_fft_start_time = tF - (num_periods_collect_fft / freq)
sampling_rate = 1/Δt_val

offset = 0.01
# Define probe points
probe_x_03 = VectorValue(-0.03)           # Individual probe at x = -0.03 m
probe_xb1_minus_01 = VectorValue(xb1 - offset)  # Individual probe at x = xb1 - 0.003 m

# Define multiple probe points for averaging (from xa2 to center)
num_averaging_probes = 51
x_coords_averaging = collect(range(xa2, xb1 - offset, length=num_averaging_probes))
probe_points_averaging = [VectorValue(x) for x in x_coords_averaging]

# All probe points (for extraction)
all_probe_points = [probe_x_03, probe_xb1_minus_01]
append!(all_probe_points, probe_points_averaging)

println("Multi-probe FFT setup (Nonlinear):")
println("  - Individual probe 1: x = $(probe_x_03[1]) m")
println("  - Individual probe 2: x = $(probe_xb1_minus_01[1]) m")
println("  - Averaging probes: $(num_averaging_probes) points from $(xa2) to 0.0 m")

# Compute multi-probe FFT with individual results and time series
frequencies, averaged_fft_magnitudes, individual_fft_results, time_series_data = MagnetostaticsFEM.compute_multi_probe_fft_average(
    solution_transient_iterable,
    all_probe_points,
    steps_for_fft_start_time,
    sampling_rate;
    return_individual_ffts=true,
    return_time_series=true
)

# Plot individual FFT at x = -0.03 m
if haskey(individual_fft_results, 1)  # First probe is x = -0.03
    freq_1, mag_1 = individual_fft_results[1]
    max_freq_plot = 1000
    
    fft_plot_x03 = plot(freq_1, mag_1 * 1e5,
        xlabel="Frequency [Hz]", ylabel=L"Magnitude\ \mathrm{[mWb/cm]}",
        title="FFT Spectrum (Nonlinear) at x = -0.03 m",
        xlims=(0, max_freq_plot), seriestype=:sticks, lw=3, color=:red, legend=false, bottom_margin=8mm)
    savefig(fft_plot_x03, joinpath(output_dir, "transient_1d_nonlinear_fft_x_minus_0p03.png"))
    display(fft_plot_x03)
    
    # Analysis for x = -0.03 probe (including harmonics)
    max_mag_1, idx_max_1 = findmax(mag_1)
    if idx_max_1 <= length(freq_1)
        peak_freq_1 = freq_1[idx_max_1]
        println("=== FFT Analysis Results (Nonlinear) at x = -0.03 m ===")
        println("  - Peak Amplitude: $(max_mag_1 * 1e5) mWb/cm")
        println("  - Peak Frequency: $(peak_freq_1) Hz")
        
        # Check for harmonics
        fundamental_idx = findmin(abs.(freq_1 .- freq))[2]
        if fundamental_idx <= length(mag_1)
            fundamental_mag = mag_1[fundamental_idx]
            third_harmonic_idx = findmin(abs.(freq_1 .- (3*freq)))[2]
            fifth_harmonic_idx = findmin(abs.(freq_1 .- (5*freq)))[2]
            
            println("  - Fundamental ($(freq) Hz): $(fundamental_mag * 1e5) mWb/cm")
            if third_harmonic_idx <= length(mag_1)
                third_mag = mag_1[third_harmonic_idx]
                third_percent = (third_mag / fundamental_mag) * 100
                println("  - 3rd Harmonic ($(3*freq) Hz): $(third_mag * 1e5) mWb/cm ($(round(third_percent, digits=1))%)")
            end
            if fifth_harmonic_idx <= length(mag_1)
                fifth_mag = mag_1[fifth_harmonic_idx]
                fifth_percent = (fifth_mag / fundamental_mag) * 100
                println("  - 5th Harmonic ($(5*freq) Hz): $(fifth_mag * 1e5) mWb/cm ($(round(fifth_percent, digits=1))%)")
            end
        end
    end
end

# Plot time signal at x = -0.03 m (Nonlinear)
if haskey(time_series_data, 1)  # First probe is x = -0.03
    times_1, signal_1 = time_series_data[1]
    
    time_plot_x03 = plot(times_1, signal_1 * 1e5,
        xlabel="Time [s]", ylabel=L"A_z\ \mathrm{[mWb/cm]}",
        title="Time Signal (Nonlinear) at x = -0.03 m",
        lw=2, color=:red, legend=false, bottom_margin=8mm)
    savefig(time_plot_x03, joinpath(output_dir, "transient_1d_nonlinear_time_signal_x_minus_0p03.png"))
    display(time_plot_x03)
    
    # Print signal statistics
    max_val_1 = maximum(abs.(signal_1))
    println("Time Signal Statistics (Nonlinear) at x = -0.03 m:")
    println("  - Max |Az|: $(max_val_1 * 1e5) mWb/cm")
    println("  - Time range: $(minimum(times_1)) to $(maximum(times_1)) s")
end

# Plot individual FFT at x = xb1 - 0.01 m
if haskey(individual_fft_results, 2)  # Second probe is x = xb1 - 0.01
    freq_2, mag_2 = individual_fft_results[2]
    
    fft_plot_xb1 = plot(freq_2, mag_2 * 1e5,
        xlabel="Frequency [Hz]", ylabel=L"Magnitude\ \mathrm{[mWb/cm]}",
        title="FFT Spectrum (Nonlinear) at x = $(round(xb1 - 0.01, digits=4)) m",
        xlims=(0, max_freq_plot), seriestype=:sticks, lw=3, color=:green, legend=false, bottom_margin=8mm)
    savefig(fft_plot_xb1, joinpath(output_dir, "transient_1d_nonlinear_fft_x_xb1_minus_0p01.png"))
    display(fft_plot_xb1)
    
    # Analysis for xb1 - 0.01 probe (including harmonics)
    max_mag_2, idx_max_2 = findmax(mag_2)
    if idx_max_2 <= length(freq_2)
        peak_freq_2 = freq_2[idx_max_2]
        println("=== FFT Analysis Results (Nonlinear) at x = $(round(xb1 - 0.01, digits=4)) m ===")
        println("  - Peak Amplitude: $(max_mag_2 * 1e5) mWb/cm")
        println("  - Peak Frequency: $(peak_freq_2) Hz")
        
        # Check for harmonics
        fundamental_idx = findmin(abs.(freq_2 .- freq))[2]
        if fundamental_idx <= length(mag_2)
            fundamental_mag = mag_2[fundamental_idx]
            third_harmonic_idx = findmin(abs.(freq_2 .- (3*freq)))[2]
            fifth_harmonic_idx = findmin(abs.(freq_2 .- (5*freq)))[2]
            
            println("  - Fundamental ($(freq) Hz): $(fundamental_mag * 1e5) mWb/cm")
            if third_harmonic_idx <= length(mag_2)
                third_mag = mag_2[third_harmonic_idx]
                third_percent = (third_mag / fundamental_mag) * 100
                println("  - 3rd Harmonic ($(3*freq) Hz): $(third_mag * 1e5) mWb/cm ($(round(third_percent, digits=1))%)")
            end
            if fifth_harmonic_idx <= length(mag_2)
                fifth_mag = mag_2[fifth_harmonic_idx]
                fifth_percent = (fifth_mag / fundamental_mag) * 100
                println("  - 5th Harmonic ($(5*freq) Hz): $(fifth_mag * 1e5) mWb/cm ($(round(fifth_percent, digits=1))%)")
            end
        end
    end
end

# Plot time signal at x = xb1 - 0.01 m (Nonlinear)
if haskey(time_series_data, 2)  # Second probe is x = xb1 - 0.01
    times_2, signal_2 = time_series_data[2]
    
    time_plot_xb1 = plot(times_2, signal_2 * 1e5,
        xlabel="Time [s]", ylabel=L"A_z\ \mathrm{[mWb/cm]}",
        title="Time Signal (Nonlinear) at x = $(round(xb1 - 0.01, digits=4)) m",
        lw=2, color=:green, legend=false, bottom_margin=8mm)
    savefig(time_plot_xb1, joinpath(output_dir, "transient_1d_nonlinear_time_signal_x_xb1_minus_0p01.png"))
    display(time_plot_xb1)
    
    # Print signal statistics
    max_val_2 = maximum(abs.(signal_2))
    println("Time Signal Statistics (Nonlinear) at x = $(round(xb1 - 0.01, digits=4)) m:")
    println("  - Max |Az|: $(max_val_2 * 1e5) mWb/cm")
    println("  - Time range: $(minimum(times_2)) to $(maximum(times_2)) s")
end

# Plot averaged FFT across multiple probes
fft_plot_averaged = plot(frequencies, averaged_fft_magnitudes * 1e5,
    xlabel="Frequency [Hz]", ylabel=L"Magnitude\ \mathrm{[mWb/cm]}",
    title="FFT Spectrum (Nonlinear, Averaged across $(num_averaging_probes) probes)",
    xlims=(0, max_freq_plot), seriestype=:sticks, lw=3, color=:blue, legend=false, bottom_margin=8mm)
savefig(fft_plot_averaged, joinpath(output_dir, "transient_1d_nonlinear_fft_averaged.png"))
display(fft_plot_averaged)

# Generate core-only averaged FFT 
println("Computing core-only averaged FFT (Nonlinear)...")
core_frequencies, core_averaged_fft_magnitudes = MagnetostaticsFEM.compute_multi_probe_fft_average(
    solution_transient_iterable,
    all_probe_points,
    steps_for_fft_start_time,
    sampling_rate;
    core_only=true
)

# Plot core-only averaged FFT and save as PDF
fft_plot_core_averaged = plot(core_frequencies, core_averaged_fft_magnitudes * 1e5,
    xlabel="Frequency [Hz]", ylabel=L"Magnitude\ \mathrm{[mWb/cm]}",
    title="FFT Spectrum (Nonlinear, Core-Only Averaged)",
    xlims=(0, max_freq_plot), seriestype=:sticks, lw=3, color=:darkgreen, legend=false, bottom_margin=8mm)
savefig(fft_plot_core_averaged, joinpath(output_dir, "transient_1d_nonlinear_fft_core_averaged.pdf"))
display(fft_plot_core_averaged)

# Analysis for core-only averaged FFT
if !isempty(core_averaged_fft_magnitudes)
    max_magnitude_core_fft, idx_max_core = findmax(core_averaged_fft_magnitudes)
    if idx_max_core <= length(core_frequencies)
        peak_frequency_core_fft = core_frequencies[idx_max_core]
        println("=== FFT Analysis Results (Nonlinear, Core-Only Averaged) ===")
        println("  - Peak Amplitude: $(max_magnitude_core_fft * 1e5) mWb/cm")
        println("  - Peak Frequency: $(peak_frequency_core_fft) Hz")
        
        # Check for harmonics in core-only averaged spectrum
        fundamental_idx = findmin(abs.(core_frequencies .- freq))[2]
        if fundamental_idx <= length(core_averaged_fft_magnitudes)
            fundamental_mag = core_averaged_fft_magnitudes[fundamental_idx]
            third_harmonic_idx = findmin(abs.(core_frequencies .- (3*freq)))[2]
            fifth_harmonic_idx = findmin(abs.(core_frequencies .- (5*freq)))[2]
            
            println("  - Fundamental ($(freq) Hz): $(fundamental_mag * 1e5) mWb/cm")
            if third_harmonic_idx <= length(core_averaged_fft_magnitudes)
                third_mag = core_averaged_fft_magnitudes[third_harmonic_idx]
                third_percent = (third_mag / fundamental_mag) * 100
                println("  - 3rd Harmonic ($(3*freq) Hz): $(third_mag * 1e5) mWb/cm ($(round(third_percent, digits=1))%)")
            end
            if fifth_harmonic_idx <= length(core_averaged_fft_magnitudes)
                fifth_mag = core_averaged_fft_magnitudes[fifth_harmonic_idx]
                fifth_percent = (fifth_mag / fundamental_mag) * 100
                println("  - 5th Harmonic ($(5*freq) Hz): $(fifth_mag * 1e5) mWb/cm ($(round(fifth_percent, digits=1))%)")
            end
        end
        
        freq_resolution = sampling_rate / 41  # Approximate number of time points
        if abs(peak_frequency_core_fft - freq) < freq_resolution * 1.5
            println("  - FFT peak frequency matches source frequency of $(freq) Hz.")
        else
            println("  - WARNING: FFT peak frequency ($(peak_frequency_core_fft) Hz) does NOT closely match source frequency ($(freq) Hz).")
        end
    end
end

# Analysis for averaged FFT (including harmonics)
if !isempty(averaged_fft_magnitudes)
    max_magnitude_fft, idx_max = findmax(averaged_fft_magnitudes)
    if idx_max <= length(frequencies)
        peak_frequency_fft = frequencies[idx_max]
        println("=== FFT Analysis Results (Nonlinear, Averaged across probes) ===")
        println("  - Peak Amplitude: $(max_magnitude_fft * 1e5) mWb/cm")
        println("  - Peak Frequency: $(peak_frequency_fft) Hz")
        println("  - Target Amplitude: 35 mWb/cm")
        println("  - Amplitude Ratio (current/target): $(round(max_magnitude_fft * 1e5 / 35, digits=2))")
        
        # Check for harmonics in averaged spectrum
        fundamental_idx = findmin(abs.(frequencies .- freq))[2]
        if fundamental_idx <= length(averaged_fft_magnitudes)
            fundamental_mag = averaged_fft_magnitudes[fundamental_idx]
            third_harmonic_idx = findmin(abs.(frequencies .- (3*freq)))[2]
            fifth_harmonic_idx = findmin(abs.(frequencies .- (5*freq)))[2]
            
            println("  - Fundamental ($(freq) Hz): $(fundamental_mag * 1e5) mWb/cm")
            if third_harmonic_idx <= length(averaged_fft_magnitudes)
                third_mag = averaged_fft_magnitudes[third_harmonic_idx]
                third_percent = (third_mag / fundamental_mag) * 100
                println("  - 3rd Harmonic ($(3*freq) Hz): $(third_mag * 1e5) mWb/cm ($(round(third_percent, digits=1))%)")
            end
            if fifth_harmonic_idx <= length(averaged_fft_magnitudes)
                fifth_mag = averaged_fft_magnitudes[fifth_harmonic_idx]
                fifth_percent = (fifth_mag / fundamental_mag) * 100
                println("  - 5th Harmonic ($(5*freq) Hz): $(fifth_mag * 1e5) mWb/cm ($(round(fifth_percent, digits=1))%)")
            end
        end
        
        freq_resolution = sampling_rate / 41  # Approximate number of time points
        if abs(peak_frequency_fft - freq) < freq_resolution * 1.5 # Allow some tolerance for nonlinear case
            println("  - FFT peak frequency is close to source fundamental frequency of $(freq) Hz.")
        else
            println("  - WARNING: FFT peak frequency ($(peak_frequency_fft) Hz) does NOT closely match source fundamental frequency ($(freq) Hz). Resolution: $(freq_resolution) Hz")
        end

        # Look for harmonics (characteristic of nonlinear behavior)
        println("\n  - Harmonic Analysis:")
        global total_harmonic_power = 0.0
        for harmonic in 2:5
            target_freq = harmonic * freq
            if target_freq <= maximum(frequencies)
                # Find closest frequency to harmonic
                closest_idx = argmin(abs.(frequencies .- target_freq))
                harmonic_magnitude = averaged_fft_magnitudes[closest_idx]
                harmonic_freq = frequencies[closest_idx]
                if abs(harmonic_freq - target_freq) < freq_resolution * 2
                    harmonic_ratio = harmonic_magnitude / max_magnitude_fft * 100
                    global total_harmonic_power += harmonic_magnitude^2
                    println("    - $(harmonic)nd harmonic: $(harmonic_magnitude * 1e5) mWb/cm ($(round(harmonic_ratio, digits=2))% of fundamental)")
                end
            end
        end

        # Calculate THD (Total Harmonic Distortion)
        global thd = sqrt(total_harmonic_power) / max_magnitude_fft * 100
        println("  - Total Harmonic Distortion (THD): $(round(thd, digits=2))%")

        if thd > 5.0
            println("  ✓ Significant nonlinear effects detected (THD > 5%)")
        else
            println("  ⚠ Limited nonlinear effects (THD < 5%) - consider increasing J0 or adjusting B-H curve")
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
                annotate!(p, midpoints_em[i] * 1e2, label_y, text(region_labels_em[i], 9, color, :center, :top))
            end
        end

        plt_transient = plot(p1, p2, layout=(2, 1), size=(800, 600),
            bottom_margin=10mm, left_margin=5mm, right_margin=5mm)
        savefig(plt_transient, joinpath(output_dir, "transient_1d_nonlinear_fields_annotated.png"))
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
