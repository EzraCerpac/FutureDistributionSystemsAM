# Compare Harmonic and Transient 1D Results

include(joinpath(dirname(@__DIR__), "config.jl"))
paths = get_project_paths(@__DIR__)
GEO_DIR = paths["GEO_DIR"]
OUT_DIR = paths["OUT_DIR"]

include("../src/MagnetostaticsFEM.jl")
include("../src/comparison_utils.jl")
include("../src/visualisation.jl")

using .MagnetostaticsFEM
using .ComparisonUtils
using .Visualisation
using Printf
using Gridap: Point, VectorValue
using LaTeXStrings: latexstring

# --- PARAMETERS (adapt as needed) ---
freq = 50.0  # Hz
ω_source = 2 * pi * freq
probe_x = -0.03  # Probe location (meters)
output_dir = joinpath(OUT_DIR, "compare_harmonic_transient_1d")
if !isdir(output_dir)   
    mkdir(output_dir)
end

# --- Load or run Harmonic Example ---
println("Running harmonic 1D example...")
solution_harmonic, Az0_h, Ω_h, ν_func_map_h, σ_func_map_h, tags_h = 
    prepare_and_solve_harmonic_1d(
        joinpath(GEO_DIR, "coil_geo.msh"),
        2, ComplexF64, "D", 0.0 + 0.0im, 4e-7*pi, 50000.0, 1e7, 2.2e4, ω_source)
Az_mag_h, B_mag_h, J_eddy_mag_h, Az_re_h, Az_im_h, B_re_h, B_im_h, J_eddy_re_h, J_eddy_im_h, ν_field_linear_h = 
    process_harmonic_solution(solution_harmonic, Ω_h, ν_func_map_h, σ_func_map_h, ω_source, tags_h)

# --- Load or run Transient Example ---
println("Running transient 1D example...")
order_fem = 2
dirichlet_tag = "D"
dirichlet_bc_func(t::Float64) = x -> 0.0
dirichlet_bc_func(x::Point, t::Real) = 0.0
t0 = 0.0
periods_to_simulate = 5
tF = periods_to_simulate / freq
num_steps_per_period = 50
Δt_val = (1/freq) / num_steps_per_period
θ_method = 0.5

solution_transient_iterable, Az0_t, Ω_t, ν_cf_t, σ_cf_t, Js_t_func_t, model_t, tags_cf_t, labels_t = 
    MagnetostaticsFEM.prepare_and_solve_transient_1d(
        joinpath(GEO_DIR, "coil_geo.msh"),
        order_fem, dirichlet_tag, dirichlet_bc_func, 4e-7*pi, 50000.0, 1e7, 2.2e4, ω_source, t0, tF, Δt_val, θ_method)

# --- Extract probe signal and FFT from transient ---
x_probe = VectorValue(probe_x)
num_periods_collect_fft = 3
steps_for_fft_start_time = tF - (num_periods_collect_fft / freq)
time_steps_for_fft, time_signal_data = MagnetostaticsFEM.save_pvd_and_extract_signal(
    solution_transient_iterable, Az0_t, Ω_t, t0, x_probe, steps_for_fft_start_time, σ_cf_t, Δt_val; output_dir=output_dir)

valid_indices = .!isnan.(time_signal_data)
time_steps_for_fft = time_steps_for_fft[valid_indices]
time_signal_data = time_signal_data[valid_indices]

sampling_rate = 1 / Δt_val
fft_frequencies, fft_magnitudes = MagnetostaticsFEM.perform_fft(time_signal_data, sampling_rate)

# --- Find peak amplitude and frequency ---
idx_max_fft = argmax(fft_magnitudes)
peak_freq_fft = fft_frequencies[idx_max_fft]
peak_amp_fft = fft_magnitudes[idx_max_fft]

# --- Get harmonic amplitude at probe ---
Az_mag_h_vals = Az_mag_h([x_probe])
harmonic_amp = abs(Az_mag_h_vals[1])
harmonic_freq = freq

# --- Compare amplitudes and frequencies ---
amp_cmp = compare_amplitudes(peak_amp_fft, harmonic_amp)
freq_cmp = compare_frequencies(peak_freq_fft, harmonic_freq)

println("\n--- Amplitude Comparison ---")
@printf("Transient (FFT) peak amplitude: %.5e\n", peak_amp_fft)
@printf("Harmonic (FD) amplitude:       %.5e\n", harmonic_amp)
println("Absolute diff: ", amp_cmp.abs_diff)
println("Relative diff: ", amp_cmp.rel_diff)
println("Within tolerance: ", amp_cmp.within_tol)

println("\n--- Frequency Comparison ---")
@printf("Transient (FFT) peak freq: %.3f Hz\n", peak_freq_fft)
@printf("Harmonic (FD) freq:        %.3f Hz\n", harmonic_freq)
println("Absolute diff: ", freq_cmp.abs_diff)
println("Relative diff: ", freq_cmp.rel_diff)
println("Within tolerance: ", freq_cmp.within_tol)

# --- Plot amplitude and frequency comparison ---
plot_amplitude_comparison(peak_amp_fft, harmonic_amp; output_path=joinpath(OUT_DIR, "compare_amplitude.pdf"))
plot_frequency_comparison(peak_freq_fft, harmonic_freq; output_path=joinpath(OUT_DIR, "compare_frequency.pdf"))

# --- Compare and plot field profiles at probe grid ---
x_grid = collect(range(-0.1, 0.1, length=1000))
probe_coords = [VectorValue(x) for x in x_grid]
Az_mag_h_profile = Az_mag_h(probe_coords)

# For transient, take last time step (steady-state)
last_Az_t, _ = last(collect(solution_transient_iterable))
Az_mag_t_profile = [abs(last_Az_t(x)) for x in probe_coords]

profile_cmp = compare_profiles(Az_mag_t_profile, Az_mag_h_profile)
println("\n--- Profile Comparison (Az magnitude, last transient step vs harmonic) ---")
println("Max diff: ", profile_cmp.max_diff)
println("Mean diff: ", profile_cmp.mean_diff)
println("Relative max diff: ", profile_cmp.rel_max_diff)
println("Within tolerance: ", profile_cmp.within_tol)

plot_profile_comparison(x_grid, Az_mag_t_profile, Az_mag_h_profile; labels=("Transient (last)", "Harmonic"), output_path=joinpath(OUT_DIR, "compare_Az_profile.pdf"), ylabel=latexstring(raw"|A_z(x)|\ \mathrm{[Wb/m]}") , title_str="Az Magnitude Profile Comparison")

println("\nComparison complete. See output plots in $OUT_DIR.")
