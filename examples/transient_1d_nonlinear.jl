# Transient 1D Nonlinear Magnetodynamics Example

# %% Setup
include(joinpath(dirname(@__DIR__), "config.jl"))
paths = get_project_paths("examples")
GEO_DIR = paths["GEO_DIR"]
OUT_DIR = paths["OUT_DIR"]
include("../src/MagnetostaticsFEM.jl")

using LinearAlgebra
using Plots
using LaTeXStrings
using Gridap
using .MagnetostaticsFEM
using Printf

println("--- Starting Nonlinear Transient 1D Magnetodynamics Example ---")

# %% Parameters
# --- Model Parameters ---
J0_amplitude = 2.2e4  # Source current density amplitude [A/m²]
μ0 = 4e-7 * pi     # Vacuum permeability [H/m]
# BH curve parameters for Frohlich-Kennelly model: μr(B) = 1 / (a + (1-a) * |B|^(2b) / (|B|^(2b) + c))
bh_a = 1/50000.0 # Corresponds to initial relative permeability (μr_initial = 1/a)
bh_b = 1.0       # Exponent
bh_c = 0.5       # Saturation parameter related to B_sat
σ_core = 1e7       # Conductivity of the core [S/m]
freq = 50.0          # Frequency of the source current [Hz]
ω_source = 2 * pi * freq # Angular frequency [rad/s]

# --- FEM Parameters ---
order_fem = 2
dirichlet_tag = "D"
dirichlet_bc_func(t::Float64) = x -> 0.0
dirichlet_bc_func(x::Point, t::Real) = 0.0

# --- Transient Simulation Parameters ---
t0 = 0.0
periods_to_simulate = 5
tF = periods_to_simulate / freq
num_steps_per_period = 50
Δt_val = (1/freq) / num_steps_per_period
θ_method = 0.5 # Crank-Nicolson

# --- Nonlinear Solver Parameters ---
max_iterations_nl = 15
tolerance_nl = 1e-5
damping_nl = 0.75

# --- Output Parameters ---
mesh_file = joinpath(GEO_DIR, "coil_geo.msh")
output_dir = joinpath(OUT_DIR, "transient_1d_nonlinear_results_jl")
if !isdir(output_dir)
    mkpath(output_dir) # Use mkpath for recursive directory creation
end

println("Mesh file: ", mesh_file)
println("Output directory: ", output_dir)

# %% Load mesh and tags (needed for the nonlinear solver function)
model, labels, tags_array = MagnetostaticsFEM.load_mesh_and_tags(mesh_file)

# %% Call the nonlinear transient solver
solution_transient_iterable, Az0_out, Ω_out, ν_cf_final, σ_cf_out, Js_t_func_out, model_out, tags_cf_out, labels_out =
    MagnetostaticsFEM.solve_nonlinear_transient_magnetodynamics(
        model, labels, tags_array,
        J0_amplitude, ω_source,
        μ0, bh_a, bh_b, bh_c,
        σ_core,
        order_fem,
        dirichlet_tag, dirichlet_bc_func,
        t0, tF, Δt_val, θ_method,
        max_iterations=max_iterations_nl,
        tolerance=tolerance_nl,
        damping=damping_nl
    )

# %% Post-processing:
x_probe = VectorValue(-0.02)
num_periods_collect_fft = 3
steps_for_fft_start_time = tF - (num_periods_collect_fft / freq)

time_steps_for_fft, time_signal_data = MagnetostaticsFEM.save_pvd_and_extract_signal(
    solution_transient_iterable,
    Az0_out,
    Ω_out,
    t0,
    x_probe,
    steps_for_fft_start_time,
    σ_cf_out,
    Δt_val,
    output_dir=output_dir, # Save PVDs to the new output directory
    pvd_filename_base="transient_nonlinear_solution" # New base name for PVD files
)

# %% Process extracted signal
valid_indices = .!isnan.(time_signal_data)
time_steps_for_fft = time_steps_for_fft[valid_indices]
time_signal_data = time_signal_data[valid_indices]

if isempty(time_steps_for_fft)
    error("No time points collected for FFT. Check simulation time (tF=$(tF), Δt=$(Δt_val)), collection window, or probe point. Collected $(length(time_signal_data)) points before NaN filter.")
end

MagnetostaticsFEM.plot_time_signal(time_steps_for_fft, time_signal_data,
                 title_str="Az (Nonlinear) at x=$(x_probe[1]) (last $(num_periods_collect_fft) periods)",
                 output_path=joinpath(output_dir, "transient_1d_nonlinear_signal.pdf"))

# %% FFT Analysis
println("Performing FFT analysis on nonlinear transient results...")
sampling_rate = 1/Δt_val

fft_frequencies, fft_magnitudes = MagnetostaticsFEM.perform_fft(time_signal_data, sampling_rate)

max_freq_plot = freq * 5 # Plot up to 5th harmonic
MagnetostaticsFEM.plot_fft_spectrum(fft_frequencies, fft_magnitudes,
                  title_str="FFT Spectrum of Az (Nonlinear) at x=$(x_probe[1])",
                  xlims_val=(0, max_freq_plot),
                  output_path=joinpath(output_dir, "transient_1d_nonlinear_fft.pdf"))

if !isempty(fft_magnitudes) && !isempty(fft_frequencies)
    max_magnitude_fft, idx_max = findmax(fft_magnitudes)
    if idx_max <= length(fft_frequencies)
        peak_frequency_fft = fft_frequencies[idx_max]
        println("FFT Analysis Results for Az (Nonlinear) at x=$(x_probe[1]):")
        println("  - Peak Amplitude (from FFT): $(max_magnitude_fft)")
        println("  - Frequency at Peak: $(peak_frequency_fft) Hz")

        freq_resolution = sampling_rate / length(time_signal_data)
        if abs(peak_frequency_fft - freq) < freq_resolution * 1.5 # Allow some tolerance
            println("  - FFT peak frequency is close to source fundamental frequency of $(freq) Hz.")
        else
            println("  - WARNING: FFT peak frequency ($(peak_frequency_fft) Hz) does NOT closely match source fundamental frequency ($(freq) Hz). Resolution: $(freq_resolution) Hz")
        end
    else
        println("FFT analysis error: index of max magnitude is out of bounds for frequencies.")
    end
else
    println("FFT analysis could not be performed (no magnitudes or frequencies found).")
end

println("
--- Nonlinear Transient 1D example finished successfully! ---")

# %% Generate Enhanced Animation with B-field and J_eddy
transient_animation_path = joinpath(output_dir, "transient_1d_nonlinear_enhanced_animation.gif")
MagnetostaticsFEM.create_transient_animation(
    Ω_out,
    solution_transient_iterable,
    σ_cf_out,
    Δt_val,
    Az0_out,
    transient_animation_path,
    fps=10,
    consistent_axes=true,
    # Optional: Pass the final reluctivity to visualize it if the function supports it
    # ν_cf_final=ν_cf_final
    # Note: create_transient_animation might need an update to accept and use ν_cf_final
    # For now, B-field will be calculated from Az, which implicitly uses the nonlinear ν.
)

println("Nonlinear transient animation saved to: $(transient_animation_path)")
