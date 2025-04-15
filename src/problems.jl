using Gridap

"""
    WeakFormProblem(a, b)

Represents the weak form of a PDE, consisting of a bilinear form `a(u, v)` 
and a linear form `b(v)`.
"""
struct WeakFormProblem
    a::Function # Bilinear form a(u, v)
    b::Function # Linear form b(v)
end

"""
    magnetostatics_1d_weak_form(Ω, dΩ, tags, reluctivity_func, current_density_func)

Creates the weak form definition for the 1D magnetostatic problem.

# Arguments
- `Ω::Triangulation`: The triangulation of the domain.
- `dΩ::Measure`: The integration measure on the domain.
- `tags::AbstractArray`: Cell array of material tags.
- `reluctivity_func`: Function mapping tag to reluctivity.
- `current_density_func`: Function mapping tag to current density.

# Returns
- `WeakFormProblem`: An instance containing the bilinear and linear forms.
"""
function magnetostatics_1d_weak_form(
    Ω::Triangulation, 
    dΩ::Measure, 
    tags::AbstractArray, 
    reluctivity_func, 
    current_density_func
    )
    
    τ = CellField(tags, Ω) # Cell field for material tags

    a(u,v) = ∫( (reluctivity_func ∘ τ) * ∇(u) ⋅ ∇(v) )dΩ
    b(v)   = ∫( (current_density_func ∘ τ) * v )dΩ

    return WeakFormProblem(a, b)
end

# You could add other problem definitions here, e.g.:
# function thermal_stationary_1d_weak_form(...) ... end
# function thermal_transient_2d_weak_form(...) ... end

