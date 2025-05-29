module SignalProcessing

using FFTW
using Statistics: mean

export perform_fft, select_periodic_window, perform_fft_periodic

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

end # module SignalProcessing 