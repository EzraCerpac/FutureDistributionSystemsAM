module ComparisonUtils

using Statistics: mean

export compare_amplitudes, compare_frequencies, compare_profiles

# Compare two amplitudes (scalars)
function compare_amplitudes(transient_amp::Number, harmonic_amp::Number; tolerance::Real=1e-2)
    diff = abs(transient_amp - harmonic_amp)
    rel_diff = diff / abs(harmonic_amp)
    return (abs_diff=diff, rel_diff=rel_diff, within_tol=rel_diff <= tolerance)
end

# Compare two frequencies (scalars)
function compare_frequencies(transient_freq::Number, target_freq::Number; tolerance::Real=1e-2)
    diff = abs(transient_freq - target_freq)
    rel_diff = diff / abs(target_freq)
    return (abs_diff=diff, rel_diff=rel_diff, within_tol=rel_diff <= tolerance)
end

# Compare two field profiles (arrays or vectors), e.g. Az(x) or B(x)
function compare_profiles(profile1, profile2; tolerance::Real=1e-2)
    # Both should be arrays of the same length
    diff = abs.(profile1 .- profile2)
    max_diff = maximum(diff)
    mean_diff = mean(diff)
    norm1 = maximum(abs.(profile1))
    norm2 = maximum(abs.(profile2))
    rel_max_diff = max_diff / max(norm1, norm2)
    rel_mean_diff = mean_diff / max(norm1, norm2)
    return (max_diff=max_diff, mean_diff=mean_diff, rel_max_diff=rel_max_diff, rel_mean_diff=rel_mean_diff, within_tol=rel_max_diff <= tolerance)
end

end # module ComparisonUtils