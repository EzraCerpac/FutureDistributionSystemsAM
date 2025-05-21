# Transient 1D Magnetodynamics Example

# %% Setup
include(joinpath(dirname(@__DIR__), "config.jl"))
paths = get_project_paths("examples") # For OUT_DIR, GEO_DIR etc.
GEO_DIR = paths["GEO_DIR"]
include("../src/MagnetostaticsFEM.jl")

using LinearAlgebra
using Plots
using LaTeXStrings
using Gridap
using .MagnetostaticsFEM # This brings all exported functions into scope
using Printf # For animation title formatting
# FFTW is used by SignalProcessing module, Plots by Visualisation

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
order = 2 # Order for transient simulation
dirichlet_tag = "D"
# Dirichlet BC Az=0. Define for g(t)=(x->val) and g(x,t) to satisfy different Gridap internals.
dirichlet_bc_func(t::Float64) = x -> 0.0  # For TransientTrialFESpace construction via g(t)
dirichlet_bc_func(x::Point, t::Real) = 0.0 # For time_derivative_f (accepts ForwardDiff.Dual for t)

# --- Transient Simulation Parameters ---
t0 = 0.0
tF = 5 / freq # Simulate for 5 periods
num_steps_per_period = 50
num_periods_collect_fft = 3 # Use last N periods for FFT to avoid initial transient effects
Δt = (1/freq) / num_steps_per_period # Time step size
θ_method = 0.5 # Crank-Nicolson (0.5), BE (1.0), FE (0.0)

# --- Output Parameters ---
mesh_file = joinpath(GEO_DIR, "coil_geo.msh")
pvd_output_base = joinpath(OUT_DIR, "transient_1d_results_jl") # Added _jl to distinguish
fft_plot_path = joinpath(OUT_DIR, "transient_1d_fft.pdf")
time_signal_plot_path = joinpath(OUT_DIR, "transient_1d_signal.pdf")

println("Mesh file: ", mesh_file)
println("Output PVD base: ", pvd_output_base)

# %% Load Mesh and Define Domains/Materials
model, labels, tags = MagnetostaticsFEM.load_mesh_and_tags(mesh_file)
Ω = Triangulation(model)
degree = 2*order # Quadrature degree
dΩ = Measure(Ω, degree)

material_tags_dict = MagnetostaticsFEM.get_material_tags(labels)
ν_func_map = MagnetostaticsFEM.define_reluctivity(material_tags_dict, μ0, μr_core)
σ_func_map = MagnetostaticsFEM.define_conductivity(material_tags_dict, σ_core)

# Create CellFields for ν and σ (assumed constant in time for linear transient)
cell_tags_cf = CellField(tags, Ω) # Helper CellField of tags
ν_cf = Operation(ν_func_map)(cell_tags_cf)
σ_cf = Operation(σ_func_map)(cell_tags_cf)

# --- Define Time-Dependent Source Current Js(x,t) ---
# spatial_js_profile_func expects an integer tag, not a point.
# We need to evaluate the tag at point x using cell_tags_cf.
spatial_js_profile_func = MagnetostaticsFEM.define_current_density(material_tags_dict, J0_amplitude)
Js_t_func(t) = x -> spatial_js_profile_func(cell_tags_cf(x)) * sin(ω_source * t)

# %% Setup Transient FE Spaces
reffe = ReferenceFE(lagrangian, Float64, order)
V0_test = TestFESpace(model, reffe, dirichlet_tags=[dirichlet_tag])
# Pass the g(t) version; Julia's dispatch will pick dirichlet_bc_func(t::Float64)
Ug_transient = TransientTrialFESpace(V0_test, dirichlet_bc_func) 

# %% Initial Condition
Az0 = zero(Ug_transient(t0)) # Uses Ug_transient(t0) which calls dirichlet_bc_func(t0)

# %% Setup Transient Operator
transient_op = MagnetostaticsFEM.setup_transient_operator(Ug_transient, V0_test, dΩ, σ_cf, ν_cf, Js_t_func)

# %% Setup ODE Solver
linear_solver_for_ode = LUSolver()
odesolver = ThetaMethod(linear_solver_for_ode, Δt, θ_method)

# %% Solve Transient Problem
println("Solving transient problem from t=$(t0) to t=$(tF) with Δt=$(Δt)...")
solution_transient_iterable = MagnetostaticsFEM.solve_transient_problem(transient_op, odesolver, t0, tF, Az0)
println("Transient solution obtained (iterable).")

# %% Post-processing: Extract Signal and Save PVD
x_probe = VectorValue(0.0) # Probe point at x=0
time_signal_data = Float64[]
time_steps_for_fft = Float64[]

println("Extracting solution at probe point $(x_probe) and saving PVD...")
processed_pvd_filename_base = first(splitext(pvd_output_base))

# Ensure Gridap.Visualization is accessible for createpvd/createvtk if not fully qualified
# MagnetostaticsFEM.Visualisation should handle this if its using Gridap.Visualization

createpvd(processed_pvd_filename_base) do pvd_file
    # Save initial condition
    # Using MagnetostaticsFEM.save_transient_pvd structure as a guide for direct PVD manipulation here
    try
        pvd_file[t0] = createvtk(Ω, processed_pvd_filename_base * "_t0_snapshot", cellfields=Dict("Az" => Az0))
        println("Saved initial condition to PVD.")
    catch e_pvd_init
        println("Error saving initial condition to PVD: $e_pvd_init")
    end

    steps_for_fft_start_time = tF - (num_periods_collect_fft / freq)

    for (i, (Az_n, tn)) in enumerate(solution_transient_iterable)
        try
            # Ensure filename compatibility for createvtk, avoid issues with '.' from Printf
            tn_str_for_file = replace(Printf.@sprintf("%.4f", tn), "." => "p")
            pvd_file[tn] = createvtk(Ω, processed_pvd_filename_base * "_t$(tn_str_for_file)_snapshot", cellfields=Dict("Az" => Az_n))
        catch e_pvd_step
            println("Error saving step t=$tn to PVD: $e_pvd_step")
        end
        
        if tn >= steps_for_fft_start_time
            push!(time_steps_for_fft, tn)
            try
                push!(time_signal_data, Az_n(x_probe))
            catch e_probe
                println("Warning: Could not evaluate Az_n at probe point $(x_probe) for t=$(tn). Error: $e_probe. Storing NaN.")
                push!(time_signal_data, NaN)
            end
        end

        if i % 20 == 0 
            println("Processed PVD save for t=$(@sprintf("%.4f", tn))")
        end
    end
end
println("Finished PVD saving to $(processed_pvd_filename_base).pvd")

valid_indices = .!isnan.(time_signal_data)
time_steps_for_fft = time_steps_for_fft[valid_indices]
time_signal_data = time_signal_data[valid_indices]

if isempty(time_steps_for_fft)
    error("No time points collected for FFT. Check simulation time (tF=$(tF), Δt=$(Δt)), collection window, or probe point.\nCollected $(length(time_signal_data)) points before NaN filter.")
end

MagnetostaticsFEM.plot_time_signal(time_steps_for_fft, time_signal_data, 
                 title_str="Az at x=$(x_probe[1]) (last $(num_periods_collect_fft) periods)",
                 output_path=time_signal_plot_path)

# %% FFT Analysis
println("Performing FFT analysis...")
sampling_rate = 1/Δt 

fft_frequencies, fft_magnitudes = MagnetostaticsFEM.perform_fft(time_signal_data, sampling_rate)

max_freq_plot = freq * 3 
MagnetostaticsFEM.plot_fft_spectrum(fft_frequencies, fft_magnitudes,
                  title_str="FFT Spectrum of Az at x=$(x_probe[1])",
                  xlims_val=(0, max_freq_plot),
                  output_path=fft_plot_path)

if !isempty(fft_magnitudes) && !isempty(fft_frequencies)
    max_magnitude_fft, idx_max = findmax(fft_magnitudes)
    if idx_max <= length(fft_frequencies)
        peak_frequency_fft = fft_frequencies[idx_max]
        println("FFT Analysis Results for Az at x=$(x_probe[1]):")
        println("  - Peak Amplitude (from FFT): $(max_magnitude_fft)")
        println("  - Frequency at Peak: $(peak_frequency_fft) Hz")
        
        freq_resolution = sampling_rate / length(time_signal_data) # Approximate resolution
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

# %% End of script
