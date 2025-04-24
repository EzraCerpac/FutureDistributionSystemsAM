using Gridap
using Gridap.FESpaces
using Gridap.MultiField

function solve_1d_steady_heat_weak_form(
    Ω::Triangulation, 
    dΩ::Measure, 
    tags::AbstractArray, 
    k,
    Q,
)
    τ = CellField(tags, Ω) # Cell field for material tags

    a(u,v) = ∫( (k ∘ τ) * ∇(v) ⋅ ∇(u) )dΩ
    l(v)   = ∫( Q * v )dΩ

    return WeakFormProblem(a, l)
end

function get_Q(
    J_eddy_squared,
    σ_core,
)
    return (0.5 / σ_core) * J_eddy_squared
end

