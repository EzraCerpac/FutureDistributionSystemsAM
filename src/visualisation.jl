module Visualisation

using Plots, LaTeXStrings, Printf
using Gridap: VectorValue

export plot_line, magnetostatics_1d_plot,
       magnetodynamics_1d_plot,
       magnetodynamics_magnitude_plot,
       magnetodynamics_animation,
       compare_flux_density,
       DEFAULT_BOUNDARIES, DEFAULT_REGION_LABELS

const A_LEN = 100.3e-3
const B_LEN = 73.15e-3
const C_LEN = 27.5e-3

function get_default_boundaries()
    xa1, xb1, xc1 = -A_LEN/2, -B_LEN/2, -C_LEN/2
    xc2, xb2, xa2 =  C_LEN/2,  B_LEN/2,  A_LEN/2
    return [xa1, xb1, xc1, xc2, xb2, xa2]
end

const DEFAULT_BOUNDARIES = get_default_boundaries()
const DEFAULT_REGION_LABELS = ["Air","Core","Coil L","Core","Coil R","Core","Air"]

ensure_vector(x, coord) = try x(coord) catch; x end

"""
Generic 1D line plot with optional region annotations.
"""
function plot_line(x::Vector{<:Real}, y::Vector{<:Real}; xlabel::Union{String,LaTeXString}="", ylabel::Union{String,LaTeXString}="", title::Union{String,LaTeXString}="", boundaries::Vector{<:Real}=Float64[], region_labels::Vector{String}=String[], label_offset_ratio::Real=0.08)
    plt = plot(x, y; xlabel=xlabel, ylabel=ylabel, color=:black, lw=1, legend=false, title=title)
    if !isempty(boundaries)
        vline!(plt, boundaries; color=:grey, linestyle=:dash, label="")
        if !isempty(region_labels)
            x_min, x_max = minimum(x), maximum(x)
            bounds = [x_min; boundaries; x_max]
            mids = [(bounds[i] + bounds[i+1]) / 2 for i in 1:length(bounds)-1]
            ylims = Plots.ylims(plt)
            label_y = ylims[1] - label_offset_ratio * (ylims[2] - ylims[1])
            annotate!(plt, [(mids[i], label_y, text(region_labels[i], 8, :center, :top)) for i in 1:length(region_labels)])
        end
    end
    return plt
end

"""
Combined plot for magnetostatics 1D: vector potential and flux density.
"""
function magnetostatics_1d_plot(x::Vector{<:Real}, Az_values::Vector{<:Real}, B_values::Vector{<:Real};
        boundaries::Vector{<:Real}=DEFAULT_BOUNDARIES,
        region_labels::Vector{String}=DEFAULT_REGION_LABELS,
        output_path::String="", size=(800,600))
    p1 = plot_line(x, Az_values; xlabel=L"x \\[mathrm{[m]}\\]", ylabel=L"A_z(x) \\[mathrm{[Wb/m]}\\]", title="Magnetic Vector Potential", boundaries=boundaries, region_labels=region_labels)
    p2 = plot_line(x, B_values; xlabel=L"x \\[mathrm{[m]}\\]", ylabel=L"B_y(x) \\[mathrm{[T]}\\]", title="Magnetic Flux Density", boundaries=boundaries, region_labels=region_labels)
    plt = plot(p1, p2; layout=(2,1), size=size)
    if output_path != ""
        savefig(plt, output_path)
    end
    return plt
end

function magnetodynamics_1d_plot(x::Vector{<:Real}, u, v, B_re, B_im;
        boundaries::Vector{<:Real}=DEFAULT_BOUNDARIES,
        region_labels::Vector{String}=DEFAULT_REGION_LABELS,
        xlabel=L"x \\,[m]", ylabel1=L"A_z(x)\\,[Wb/m]", ylabel2=L"B_y(x)\\,[T]",
        title1="Re/Im A_z", title2="Re/Im B_y",
        output_path::String="", size=(800,600))
    coord = [VectorValue(x_) for x_ in x]
    u_vals   = ensure_vector(u,   coord)
    v_vals   = ensure_vector(v,   coord)
    Bre_vals = ensure_vector(B_re, coord)
    Bim_vals = ensure_vector(B_im, coord)

    p1 = plot(x, u_vals; label="Re", color=:blue, xlabel=xlabel, ylabel=ylabel1, title=title1)
    plot!(x, v_vals; label="Im", color=:red)
    p2 = plot(x, [b[1] for b in Bre_vals]; label="Re", color=:blue, xlabel=xlabel, ylabel=ylabel2, title=title2)
    plot!(x, [b[1] for b in Bim_vals]; label="Im", color=:red)

    plt = plot(p1, p2; layout=(2,1), size=size)
    if output_path != ""
        savefig(plt, output_path)
    end
    return plt
end

function magnetodynamics_magnitude_plot(x::Vector{<:Real},
        Az_mag_vals, B_mag_vals, Jeddy_mag_vals;
        boundaries::Vector{<:Real}=DEFAULT_BOUNDARIES,
        region_labels::Vector{String}=DEFAULT_REGION_LABELS,
        output_path::String="", size=(800,900))
    coord = [VectorValue(x_) for x_ in x]
    Azv = ensure_vector(Az_mag_vals, coord)
    Bv  = ensure_vector(B_mag_vals, coord)
    Jv  = ensure_vector(Jeddy_mag_vals, coord)

    p1 = plot(x, Azv; xlabel=L"x \\,[m]", ylabel=L"|A_z|", title="|A_z|", legend=false)
    p2 = plot(x, Bv;  xlabel=L"x \\,[m]", ylabel=L"|B_y|", title="|B_y|", legend=false)
    p3 = plot(x, Jv;  xlabel=L"x \\,[m]", ylabel=L"|J_{eddy}|", title="|J_{eddy}|", legend=false)

    plt = plot(p1, p2, p3; layout=(3,1), size=size)
    if output_path != ""
        savefig(plt, output_path)
    end
    return plt
end

function magnetodynamics_animation(x::Vector{<:Real}, u, v, B_re, B_im, Jeddy_re, Jeddy_im;
        boundaries::Vector{<:Real}=DEFAULT_BOUNDARIES,
        region_labels::Vector{String}=DEFAULT_REGION_LABELS,
        period::Real = 2π, fps::Int = 15, output_path::String = "")

    coord    = [VectorValue(x_) for x_ in x]
    u_vals   = ensure_vector(u, coord)
    v_vals   = ensure_vector(v, coord)
    Bre_vals = ensure_vector(B_re, coord)
    Bim_vals = ensure_vector(B_im, coord)
    Jer_vals = ensure_vector(Jeddy_re, coord)
    Jei_vals = ensure_vector(Jeddy_im, coord)

    a_min = min(minimum(u_vals), minimum(v_vals))
    a_max = max(maximum(u_vals), maximum(v_vals))
    b1    = [b[1] for b in Bre_vals]
    b2    = [b[1] for b in Bim_vals]
    b_min = min(minimum(b1), minimum(b2))
    b_max = max(maximum(b1), maximum(b2))
    j_min = min(minimum(Jer_vals), minimum(Jei_vals))
    j_max = max(maximum(Jer_vals), maximum(Jei_vals))

    anim = @animate for t in range(0, period; length=100)
        c, s = cos(t), sin(t)
        Ainst   = u_vals .* c .- v_vals .* s
        Brest   = Bre_vals .* c .- Bim_vals .* s
        Jedinst = Jer_vals .* c .- Jei_vals .* s

        p1 = plot(x, Ainst; xlabel=L"x \\,[m]", ylabel=L"A_z", title=@sprintf("A_z(t=%.2f)",t),
                  ylim=(a_min,a_max), legend=false)
        p2 = plot(x, [b[1] for b in Brest]; xlabel=L"x \\,[m]", ylabel=L"B_y", title="B_y",
                  ylim=(b_min,b_max), legend=false)
        p3 = plot(x, Jedinst; xlabel=L"x \\,[m]", ylabel=L"J_{eddy}", title="J_eddy",
                  ylim=(j_min,j_max), legend=false)

        plot(p1,p2,p3; layout=(3,1), size=(800,900))
    end

    if output_path != ""
        gif(anim, output_path; fps=fps)
    end
    return nothing
end

function compare_flux_density(x::Vector{<:Real}, B_mag_vals::Vector{<:Real}, B_mag_linear_vals::Vector{<:Real};
        boundaries::Vector{<:Real}=DEFAULT_BOUNDARIES,
        region_labels::Vector{String}=DEFAULT_REGION_LABELS,
        output_path::String="", size=(800,600))
    p1 = plot(x, B_mag_vals; xlabel=L"x \\,[m]", ylabel=L"|B_y|\\,[T]", label="Nonlinear", title="|B_y| Comparison")
    plot!(x, B_mag_linear_vals; label="Linear")
    p2 = plot(x, B_mag_vals .÷ B_mag_linear_vals; xlabel=L"x \\,[m]", ylabel=L"Ratio", label="Nonlinear/Linear", title="Nonlinear/Linear")
    plt = plot(p1, p2; layout=(2,1), size=size)
    if output_path != ""
        savefig(plt, output_path)
    end
    return plt
end

end # module