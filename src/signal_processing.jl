module SignalProcessing

using FFTW

export perform_fft

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

end # module SignalProcessing 