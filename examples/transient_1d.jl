# Transient 1D Magnetodynamics Example

# %% Setup
include(joinpath(dirname(@__DIR__), "config.jl"))
paths = get_project_paths("examples") # For OUT_DIR, GEO_DIR etc.
GEO_DIR = paths["GEO_DIR"]
OUT_DIR = paths["OUT_DIR"] # Ensure OUT_DIR is defined here
include("../src/MagnetostaticsFEM.jl")

using LinearAlgebra
using Plots
using LaTeXStrings
using Gridap
using .MagnetostaticsFEM # This brings all exported functions into scope
using Printf # For animation title formatting

println("--- Starting Transient 1D Magnetodynamics Example ---")

# %% Parameters
# --- Model Parameters ---
J0_amplitude = 2.2e4  # Source current density amplitude [A/m²]
μ0 = 4e-7 * pi     # Vacuum permeability [H/m]
μr_core = 50000.0    # Relative permeability of the core (linear assumption for transient)
σ_core = 1e7       # Conductivity of the core [S/m]
freq = 50.0          # Frequency of the source current [Hz]
ω_source = 2 * pi * freq # Angular frequency [rad/s]

# --- FEM Parameters ---
order_fem = 2 # Order for transient simulation (renamed from 'order' to avoid conflicts)
dirichlet_tag = "D"

# Define a single dirichlet_bc_func with multiple dispatch a la original script
dirichlet_bc_func(t::Float64) = x -> 0.0  # For TransientTrialFESpace g(t) -> h(x)
dirichlet_bc_func(x::Point, t::Real) = 0.0 # For Gridap internals needing g(x,t) (e.g. for derivatives)

# --- Transient Simulation Parameters ---
t0 = 0.0
periods_to_simulate = 5
tF = periods_to_simulate / freq # Simulate for 5 periods
num_steps_per_period = 50
num_periods_collect_fft = 3 # Use last N periods for FFT to avoid initial transient effects
Δt_val = (1/freq) / num_steps_per_period # Time step size, renamed to Δt_val to avoid conflict with module Δt
θ_method = 0.5 # Crank-Nicolson (0.5), BE (1.0), FE (0.0)

# --- Output Parameters ---
mesh_file = joinpath(GEO_DIR, "coil_geo.msh")
output_dir = joinpath(OUT_DIR, "transient_1d_results_jl") 
if !isdir(output_dir)
    mkdir(output_dir)
end
fft_plot_path = joinpath(output_dir, "transient_1d_fft.pdf")
time_signal_plot_path = joinpath(output_dir, "transient_1d_signal.pdf")

println("Mesh file: ", mesh_file)
println("Output directory: ", output_dir)

# %% Call the new preparation and solving function from TransientSolver
solution_transient_iterable, Az0_out, Ω_out, ν_cf_out, σ_cf_out, Js_t_func_out, model_out, tags_cf_out, labels_out = 
    MagnetostaticsFEM.prepare_and_solve_transient_1d( # Called via MagnetostaticsFEM module
        mesh_file,
        order_fem, 
        dirichlet_tag,
        dirichlet_bc_func,  
        μ0,
        μr_core,
        σ_core,
        J0_amplitude,
        ω_source,
        t0,
        tF,
        Δt_val, # Pass renamed Δt_val
        θ_method
    )

# %% Post-processing:

x_probe = VectorValue(-0.03) 
steps_for_fft_start_time = tF - (num_periods_collect_fft / freq)

time_steps_for_fft, time_signal_data = MagnetostaticsFEM.save_pvd_and_extract_signal( # Called via MagnetostaticsFEM
    solution_transient_iterable,
    Az0_out, # Pass Az0_out
    Ω_out,   # Pass Ω_out
    t0,
    x_probe,
    steps_for_fft_start_time,
    σ_cf_out, # Pass conductivity CellField for enhanced J_eddy calculation
    Δt_val,   # Pass time step for enhanced calculations;
    output_dir=output_dir
)

# %% Process extracted signal (same as before)
valid_indices = .!isnan.(time_signal_data)
time_steps_for_fft = time_steps_for_fft[valid_indices]
time_signal_data = time_signal_data[valid_indices]

if isempty(time_steps_for_fft)
    error("No time points collected for FFT. Check simulation time (tF=$(tF), Δt=$(Δt_val)), collection window, or probe point.\nCollected $(length(time_signal_data)) points before NaN filter.")
end

MagnetostaticsFEM.plot_time_signal(time_steps_for_fft, time_signal_data, 
                 title_str="Az at x=$(x_probe[1]) (last $(num_periods_collect_fft) periods)",
                 output_path=joinpath(output_dir, "transient_1d_signal.pdf"))

# %% FFT Analysis
println("Performing FFT analysis...")
sampling_rate = 1/Δt_val # Use Δt_val

fft_frequencies, fft_magnitudes = MagnetostaticsFEM.perform_fft(time_signal_data, sampling_rate)

max_freq_plot = freq * 3 
MagnetostaticsFEM.plot_fft_spectrum(fft_frequencies, fft_magnitudes,
                  title_str="FFT Spectrum of Az at x=$(x_probe[1])",
                  xlims_val=(0, max_freq_plot),
                  output_path=joinpath(output_dir, "transient_1d_fft.pdf"))

if !isempty(fft_magnitudes) && !isempty(fft_frequencies)
    max_magnitude_fft, idx_max = findmax(fft_magnitudes)
    if idx_max <= length(fft_frequencies)
        peak_frequency_fft = fft_frequencies[idx_max]
        println("FFT Analysis Results for Az at x=$(x_probe[1]):")
        println("  - Peak Amplitude (from FFT): $(max_magnitude_fft)")
        println("  - Frequency at Peak: $(peak_frequency_fft) Hz")
        
        freq_resolution = sampling_rate / length(time_signal_data) 
        if abs(peak_frequency_fft - freq) < freq_resolution 
            println("  - FFT peak frequency matches source frequency of $(freq) Hz.")
        else
            println("  - WARNING: FFT peak frequency ($(peak_frequency_fft) Hz) does NOT closely match source frequency ($(freq) Hz). Resolution: $(freq_resolution) Hz")
        end
    else
        println("FFT analysis error: index of max magnitude is out of bounds for frequencies.")
    end
else
    println("FFT analysis could not be performed (no magnitudes or frequencies found).")
end

# %% Conceptual Comparison with Frequency Domain
println("\nConceptual comparison with Frequency Domain:")
println("The peak amplitude from FFT ($(isdefined(Main, :max_magnitude_fft) ? max_magnitude_fft : "N/A")) at $(isdefined(Main, :peak_frequency_fft) ? peak_frequency_fft : "N/A") Hz")
println("can be compared with the magnitude of Az from a frequency-domain solution at $(freq) Hz using J0_amplitude.")

println("\n--- Transient 1D example finished successfully! ---")

# %% Generate Enhanced Animation with B-field and J_eddy
transient_animation_path = joinpath(output_dir, "transient_1d_enhanced_animation.gif")
MagnetostaticsFEM.create_transient_animation(
    Ω_out, 
    solution_transient_iterable, 
    σ_cf_out, # Pass conductivity CellField
    Δt_val,   # Pass time step value
    Az0_out,  # Pass initial condition FEFunction
    transient_animation_path, 
    fps=10,
    consistent_axes=true # Optional: use consistent axes
)
