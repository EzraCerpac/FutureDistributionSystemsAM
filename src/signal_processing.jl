module SignalProcessing

using FFTW
using Statistics: mean
using Gridap: VectorValue

export perform_fft, select_periodic_window, perform_fft_periodic, extract_spatially_averaged_signal, compute_multi_probe_fft_average

"""
    perform_fft(time_signal::Vector{<:Real}, sampling_rate::Real)

Performs FFT on a real-valued time signal.

# Arguments
- `time_signal::Vector{<:Real}`: The input time signal (e.g., Az at a point over time).
- `sampling_rate::Real`: The rate at which the signal was sampled (samples per second, Hz).

# Returns
- `frequencies::Vector{Float64}`: Vector of frequencies for the FFT components.
- `fft_magnitudes::Vector{Float64}`: Vector of magnitudes of the FFT components (one-sided spectrum).
"""
function perform_fft(time_signal::Vector{<:Real}, sampling_rate::Real)
    n = length(time_signal)
    if n == 0
        return Float64[], Float64[]
    end

    # Perform FFT
    fft_result = fft(time_signal)

    # Calculate frequencies
    # FFTW.fftfreq gives frequencies for fftshift(fft(A))
    # For a one-sided spectrum of a real signal, up to Nyquist frequency
    # freqs = FFTW.fftfreq(n, sampling_rate)
    # fft_mags_full = abs.(fft_result)/n

    # We want one-sided spectrum. For real signal, it's symmetric.
    # Power is doubled for non-DC and non-Nyquist components.

    
    # Calculate one-sided spectrum frequencies
    if mod(n, 2) == 0 # even n
        frequencies = FFTW.rfftfreq(n, sampling_rate) # Correct for real FFT
        num_unique_pts = n÷2 + 1
    else # odd n
        frequencies = FFTW.rfftfreq(n, sampling_rate)
        num_unique_pts = (n+1)÷2
    end

    # Calculate magnitudes for one-sided spectrum
    # rfft output already scaled differently than fft
    fft_one_sided = rfft(time_signal) # Use rfft for real signals
    fft_magnitudes = abs.(fft_one_sided) / n # Scale by N

    # For correct amplitude, multiply by 2 for non-DC/Nyquist frequencies
    # DC component (0 Hz) is fft_magnitudes[1]
    # Nyquist component (if n is even) is fft_magnitudes[end]
    # These are not doubled.
    if n > 1
        fft_magnitudes[2:end-1] .= fft_magnitudes[2:end-1] .* 2
        if mod(n,2) == 0 # Even length, Nyquist is not doubled, it's already at end
             # fft_magnitudes[end] is Nyquist, no doubling needed by this convention for rfft.
        else # Odd length, last element is not Nyquist, so it was doubled.
            # if length(fft_magnitudes) > 1, then fft_magnitudes[end] was doubled. Ok.
        end
    end
    # If n=1, only DC, correctly scaled by /n.

    # Alternative simpler scaling for magnitude (common for just viewing spectrum shape)
    # fft_magnitudes = abs.(fft_result[1:num_unique_pts]) / n # Scale by N
    # if n > 1
    #    fft_magnitudes[2:end] .= fft_magnitudes[2:end] .* 2 # Double for non-DC for one-sided view
    #    if mod(n,2) == 0 && length(fft_magnitudes) == num_unique_pts # if n is even, don't double Nyquist
    #        fft_magnitudes[end] /= 2
    #    end
    # end

    return frequencies, fft_magnitudes
end

"""
    select_periodic_window(time_vec::Vector{<:Real}, signal::Vector{<:Real}, freq::Real; num_periods::Int=3)

Selects a window of the signal covering an integer number of periods to avoid spectral leakage in FFT.

# Arguments
- `time_vec::Vector{<:Real}`: Time points corresponding to the signal values.
- `signal::Vector{<:Real}`: The input time signal.
- `freq::Real`: Expected frequency of the signal (Hz).
- `num_periods::Int=3`: Number of periods to include in the window (default: 3).

# Returns
- `window_times::Vector{Float64}`: Time points for the selected window.
- `window_signal::Vector{Float64}`: Signal values for the selected window.
- `actual_periods::Float64`: Actual number of periods in the selected window.
"""
function select_periodic_window(time_vec::Vector{<:Real}, signal::Vector{<:Real}, freq::Real; num_periods::Int=3)
    if length(time_vec) != length(signal)
        error("Time vector and signal must have the same length")
    end
    
    if isempty(time_vec) || isempty(signal)
        return Float64[], Float64[], 0.0
    end
    
    # Calculate period
    T = 1.0 / freq
    
    # Get the time range
    t_start = minimum(time_vec)
    t_end = maximum(time_vec)
    
    # Calculate sampling interval (approximately)
    Δt = (t_end - t_start) / (length(time_vec) - 1)
    
    # Calculate samples per period (approximately)
    samples_per_period = round(Int, T / Δt)
    
    if samples_per_period < 2
        @warn "Sampling rate too low for frequency $freq Hz. Need at least 2 samples per period."
        return copy(time_vec), copy(signal), (t_end - t_start) * freq
    end
    
    # Determine window size in samples
    window_size = min(length(signal), num_periods * samples_per_period)
    
    # Use the last window_size samples for analysis
    if window_size < length(signal)
        window_signal = signal[end-window_size+1:end]
        window_times = time_vec[end-window_size+1:end]
    else
        window_signal = copy(signal)
        window_times = copy(time_vec)
    end
    
    # Calculate actual number of periods in the window
    actual_periods = (maximum(window_times) - minimum(window_times)) * freq
    
    return window_times, window_signal, actual_periods
end

"""
    perform_fft_periodic(time_vec::Vector{<:Real}, signal::Vector{<:Real}, freq::Real, sampling_rate::Real; 
                         num_periods::Int=3, remove_mean::Bool=true)

Selects a periodic window of the signal and performs FFT for accurate frequency detection.

# Arguments
- `time_vec::Vector{<:Real}`: Time points corresponding to the signal values.
- `signal::Vector{<:Real}`: The input time signal.
- `freq::Real`: Expected frequency of the signal (Hz).
- `sampling_rate::Real`: The rate at which the signal was sampled (samples per second, Hz).
- `num_periods::Int=3`: Number of periods to include in the window (default: 3).
- `remove_mean::Bool=true`: Whether to remove the mean from the signal before FFT (default: true).

# Returns
- `frequencies::Vector{Float64}`: Vector of frequencies for the FFT components.
- `fft_magnitudes::Vector{Float64}`: Vector of magnitudes of the FFT components (one-sided spectrum).
- `peak_freq::Float64`: Detected peak frequency (Hz).
- `peak_amplitude::Float64`: Amplitude at the peak frequency.
- `window_times::Vector{Float64}`: Time points for the selected window.
- `window_signal::Vector{Float64}`: Signal values for the selected window.
"""
function perform_fft_periodic(time_vec::Vector{<:Real}, signal::Vector{<:Real}, freq::Real, sampling_rate::Real; 
                            num_periods::Int=3, remove_mean::Bool=true, expected_freq_tolerance::Float64=0.2)
    # Select periodic window
    window_times, window_signal, actual_periods = select_periodic_window(time_vec, signal, freq; num_periods=num_periods)
    
    # Remove mean if requested
    if remove_mean && !isempty(window_signal)
        window_signal = window_signal .- mean(window_signal)
    end
    
    # Apply a Hann window to reduce spectral leakage
    window_length = length(window_signal)
    hann_window = 0.5 .* (1 .- cos.(2π .* (0:window_length-1) ./ (window_length-1)))
    windowed_signal = window_signal .* hann_window
    
    # Perform FFT
    frequencies, fft_magnitudes = perform_fft(windowed_signal, sampling_rate)
    
    # Find peak frequency (ignoring DC component)
    peak_idx = 1
    if length(fft_magnitudes) > 1
        # Find peak excluding DC component (index 1)
        peak_idx = argmax(fft_magnitudes[2:end]) + 1
        
        # Check if there's a peak near the expected frequency
        expected_freq_idx = findall(f -> abs(f - freq)/freq < expected_freq_tolerance, frequencies)
        
        if !isempty(expected_freq_idx)
            # Find the highest magnitude peak near the expected frequency
            expected_peak_idx = expected_freq_idx[argmax(fft_magnitudes[expected_freq_idx])]
            
            # If this peak is significant (at least 50% of the highest peak), use it
            if fft_magnitudes[expected_peak_idx] > 0.5 * fft_magnitudes[peak_idx]
                peak_idx = expected_peak_idx
            end
        end
    end
    
    peak_freq = frequencies[peak_idx]
    peak_amplitude = fft_magnitudes[peak_idx]
    
    # For debugging
    println("All frequencies: ", frequencies)
    println("All magnitudes: ", fft_magnitudes)
    println("Peak index: ", peak_idx)
    println("Frequency resolution: ", frequencies[2] - frequencies[1])
    
    return frequencies, fft_magnitudes, peak_freq, peak_amplitude, window_times, window_signal
end

"""
    extract_spatially_averaged_signal(solution_iterable, x_start::Float64, x_end::Float64, 
                                     time_start::Float64; num_spatial_points::Int=100)

Extracts a time series by spatially averaging the field values across a specified spatial range.

# Arguments
- `solution_iterable`: Iterator over (solution, time) pairs from transient solver
- `x_start::Float64`: Starting x-coordinate for spatial averaging [m]
- `x_end::Float64`: Ending x-coordinate for spatial averaging [m] 
- `time_start::Float64`: Time at which to start collecting data [s]
- `num_spatial_points::Int=100`: Number of spatial sampling points across the range

# Returns
- `time_steps::Vector{Float64}`: Time points for the collected data
- `spatially_averaged_signal::Vector{Float64}`: Spatially averaged field values over time

# Notes
- Evaluates the solution at `num_spatial_points` evenly spaced points from x_start to x_end
- For each time step ≥ time_start, computes the arithmetic mean of field values across space
- Handles evaluation failures gracefully by excluding failed points from the average
- Returns NaN for time steps where spatial averaging completely fails
"""
function extract_spatially_averaged_signal(
    solution_iterable, 
    x_start::Float64, 
    x_end::Float64, 
    time_start::Float64;
    num_spatial_points::Int = 100
)
    # Create spatial sampling points
    x_coords = collect(range(x_start, x_end, length=num_spatial_points))
    spatial_points = [VectorValue(x) for x in x_coords]
    
    # Initialize output arrays
    time_steps = Float64[]
    spatially_averaged_signal = Float64[]
    
    println("Spatial averaging setup:")
    println("  - Range: x = $(x_start) to $(x_end) m")
    println("  - Number of spatial points: $(num_spatial_points)")
    println("  - Starting data collection at t = $(time_start) s")
    
    global step_count = 0
    for (solution, t) in solution_iterable
        global step_count += 1
        
        if t >= time_start
            push!(time_steps, t)
            
            # Evaluate solution at all spatial points
            field_values = Float64[]
            successful_evaluations = 0
            
            for point in spatial_points
                try
                    field_val = solution(point)
                    # Handle different return types
                    if isa(field_val, AbstractArray) && length(field_val) == 1
                        push!(field_values, first(field_val))
                        successful_evaluations += 1
                    elseif isa(field_val, Number)
                        push!(field_values, Float64(field_val))
                        successful_evaluations += 1
                    end
                catch
                    # Skip points that fail to evaluate
                    continue
                end
            end
            
            # Compute spatial average
            if successful_evaluations > 0
                spatial_average = mean(field_values)
                push!(spatially_averaged_signal, spatial_average)
                
                # Debug output every few steps
                if step_count % 20 == 0
                    success_rate = successful_evaluations / num_spatial_points * 100
                    println("Step $(step_count), t=$(round(t, digits=5)): $(successful_evaluations)/$(num_spatial_points) points ($(round(success_rate, digits=1))%), avg = $(round(spatial_average, digits=6))")
                end
            else
                push!(spatially_averaged_signal, NaN)
                println("Warning: All spatial evaluations failed at t=$(t)")
            end
        end
    end
    
    println("Spatial averaging completed:")
    println("  - Collected $(length(time_steps)) time points")
    println("  - Data range: t = $(minimum(time_steps)) to $(maximum(time_steps)) s")
    
    return time_steps, spatially_averaged_signal
end

"""
    compute_multi_probe_fft_average(solution_iterable, probe_points::Vector, time_start::Float64, 
                                   sampling_rate::Float64; return_individual_ffts::Bool=false, 
                                   return_time_series::Bool=false, core_only::Bool=false)

Computes FFT at multiple probe locations and returns the averaged FFT spectrum.
Also optionally returns individual FFT results and time series for specific probe plotting.

# Arguments
- `solution_iterable`: Iterator over (solution, time) pairs from transient solver
- `probe_points::Vector`: Vector of VectorValue probe points to sample
- `time_start::Float64`: Time at which to start collecting data [s]
- `sampling_rate::Float64`: Sampling rate for FFT calculation [Hz]
- `return_individual_ffts::Bool=false`: Whether to return individual FFT results
- `return_time_series::Bool=false`: Whether to return individual time series data
- `core_only::Bool=false`: Whether to filter probes to only include core regions

# Returns
- `frequencies::Vector{Float64}`: Frequency vector for FFT
- `averaged_fft_magnitudes::Vector{Float64}`: FFT magnitudes averaged across all probes
- `individual_fft_results::Dict` (if return_individual_ffts=true): Dict mapping probe index to (frequencies, magnitudes)
- `time_series_data::Dict` (if return_time_series=true): Dict mapping probe index to (time_steps, signal_values)

# Notes
- Computes FFT separately at each probe location
- Averages the FFT magnitudes (not the time signals) across all probes
- Handles evaluation failures gracefully by excluding failed probes from averaging
- All individual FFTs must have the same length for averaging to work
- If core_only=true, filters probes to core regions only (xa2 to xc1 and xc2 to xa3)
"""
function compute_multi_probe_fft_average(
    solution_iterable, 
    probe_points::Vector, 
    time_start::Float64,
    sampling_rate::Float64;
    return_individual_ffts::Bool = false,
    return_time_series::Bool = false,
    core_only::Bool = false
)
    # Filter probe points to core regions if requested
    filtered_probe_points = probe_points
    if core_only
        # Define core region boundaries (from 1d_mesh_w_oil_reservois.jl geometry)
        a_len = 100.3e-3
        b_len = 73.15e-3
        c_len = 27.5e-3
        xa2 = -a_len / 2  # Left core boundary
		xb2 = -b_len / 2  # Right core boundary
        xc1 = -c_len / 2  # Core center left
        xc2 = c_len / 2   # Core center right
		xb3 = b_len / 2   # Left core boundary
        xa3 = a_len / 2   # Right core boundary
        
        # Filter probes to core regions: xa2 to xb2 and xc1 to 0 (because of symmetry)
        core_probe_indices = Int[]
        for (i, probe) in enumerate(probe_points)
            x_coord = probe[1]  # Extract x coordinate
            if (xa2 <= x_coord <= xb2) || (xc1 <= x_coord <= 0)
                push!(core_probe_indices, i)
            end
        end
        
        filtered_probe_points = probe_points[core_probe_indices]
        
        println("Core-only filtering applied:")
        println("  - Core regions: [$xa2, $xb2] and [$xc1, $xc2]")
        println("  - Original probes: $(length(probe_points))")
        println("  - Core-only probes: $(length(filtered_probe_points))")
    end
    
    println("Multi-probe FFT averaging setup:")
    println("  - Number of probe points: $(length(filtered_probe_points))")
    println("  - Probe locations: $(filtered_probe_points)")
    println("  - Starting data collection at t = $(time_start) s")
    println("  - Sampling rate: $(sampling_rate) Hz")
    println("  - Core-only filtering: $(core_only)")
    
    # Store time series for each probe
    probe_time_series = Dict{Int, Vector{Float64}}()  # probe_index -> time_signal
    time_steps = Float64[]
    
    # Initialize storage for each probe
    for i in 1:length(filtered_probe_points)
        probe_time_series[i] = Float64[]
    end
    
    # Extract time series at all probe points
    println("Extracting time series at all probe locations...")
    global step_count = 0
    for (solution, t) in solution_iterable
        global step_count += 1
        
        if t >= time_start
            if isempty(time_steps) || t > time_steps[end]  # Avoid duplicate time points
                push!(time_steps, t)
                
                # Evaluate solution at all probe points
                for (i, probe_point) in enumerate(filtered_probe_points)
                    try
                        field_val = solution(probe_point)
                        # Handle different return types
                        if isa(field_val, AbstractArray) && length(field_val) == 1
                            push!(probe_time_series[i], first(field_val))
                        elseif isa(field_val, Number)
                            push!(probe_time_series[i], Float64(field_val))
                        else
                            push!(probe_time_series[i], NaN)
                        end
                    catch
                        push!(probe_time_series[i], NaN)
                    end
                end
                
                # Debug output
                if length(time_steps) % 20 == 0
                    valid_probes = sum([!isnan(probe_time_series[i][end]) for i in 1:length(filtered_probe_points)])
                    println("Step $(step_count), t=$(round(t, digits=5)): $(valid_probes)/$(length(filtered_probe_points)) probes valid")
                end
            end
        end
    end
    
    println("Time series extraction completed:")
    println("  - Collected $(length(time_steps)) time points")
    println("  - Data range: t = $(minimum(time_steps)) to $(maximum(time_steps)) s")
    
    # Compute FFT at each probe location
    println("Computing FFT at each probe location...")
    individual_fft_results = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    valid_ffts = []
    
    for i in 1:length(filtered_probe_points)
        # Clean time series (remove NaN values)
        time_signal = probe_time_series[i]
        valid_indices = .!isnan.(time_signal)
        
        if sum(valid_indices) > 10  # Need sufficient data points
            clean_signal = time_signal[valid_indices]
            
            # Compute FFT
            frequencies, fft_magnitudes = perform_fft(clean_signal, sampling_rate)
            
            if !isempty(frequencies) && !isempty(fft_magnitudes)
                individual_fft_results[i] = (frequencies, fft_magnitudes)
                push!(valid_ffts, i)
                
                println("  Probe $(i) at $(filtered_probe_points[i]): FFT computed ($(length(clean_signal)) points)")
            else
                println("  Probe $(i) at $(filtered_probe_points[i]): FFT failed")
            end
        else
            println("  Probe $(i) at $(filtered_probe_points[i]): Insufficient valid data ($(sum(valid_indices)) points)")
        end
    end
    
    if isempty(valid_ffts)
        error("No valid FFTs computed at any probe location")
    end
    
    # Average FFT magnitudes across all valid probes
    println("Averaging FFT magnitudes across $(length(valid_ffts)) valid probes...")
    
    # Use the first valid FFT to get frequency vector (all should be the same)
    first_valid_idx = valid_ffts[1]
    frequencies, _ = individual_fft_results[first_valid_idx]
    
    # Initialize averaged magnitudes
    averaged_fft_magnitudes = zeros(Float64, length(frequencies))
    
    # Sum magnitudes from all valid FFTs
    for probe_idx in valid_ffts
        _, fft_magnitudes = individual_fft_results[probe_idx]
        
        if length(fft_magnitudes) == length(averaged_fft_magnitudes)
            averaged_fft_magnitudes .+= fft_magnitudes
        else
            println("Warning: FFT length mismatch at probe $(probe_idx), skipping from average")
        end
    end
    
    # Divide by number of valid probes to get average
    averaged_fft_magnitudes ./= length(valid_ffts)
    
    println("Multi-probe FFT averaging completed:")
    println("  - Averaged across $(length(valid_ffts)) probes")
    println("  - Frequency range: $(minimum(frequencies)) to $(maximum(frequencies)) Hz")
    
    # Prepare time series data if requested
    time_series_data = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    if return_time_series
        for i in 1:length(filtered_probe_points)
            time_signal = probe_time_series[i]
            valid_indices = .!isnan.(time_signal)
            if sum(valid_indices) > 0
                valid_times = time_steps[valid_indices]
                valid_signal = time_signal[valid_indices]
                time_series_data[i] = (valid_times, valid_signal)
            end
        end
    end
    
    # Return based on requested options
    if return_individual_ffts && return_time_series
        return frequencies, averaged_fft_magnitudes, individual_fft_results, time_series_data
    elseif return_individual_ffts
        return frequencies, averaged_fft_magnitudes, individual_fft_results
    elseif return_time_series
        return frequencies, averaged_fft_magnitudes, time_series_data
    else
        return frequencies, averaged_fft_magnitudes
    end
end

end # module SignalProcessing 
