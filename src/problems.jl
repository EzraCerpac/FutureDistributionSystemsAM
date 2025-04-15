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

"""
    magnetodynamics_1d_harmonic_weak_form(
        Ω, dΩ, tags, 
        reluctivity_func, conductivity_func, current_density_func, 
        omega::Float64
    )

Creates the weak form definition for the 1D time-harmonic magnetodynamics problem.

# Arguments
- `Ω::Triangulation`: The triangulation of the domain.
- `dΩ::Measure`: The integration measure on the domain.
- `tags::AbstractArray`: Cell array of material tags.
- `reluctivity_func`: Function mapping tag to reluctivity.
- `conductivity_func`: Function mapping tag to conductivity.
- `current_density_func`: Function mapping tag to source current density.
- `omega::Float64`: Angular frequency (2πf).

# Returns
- `WeakFormProblem`: An instance containing the bilinear and linear forms.
"""
function magnetodynamics_1d_harmonic_weak_form(
    Ω::Triangulation, 
    dΩ::Measure, 
    tags::AbstractArray, 
    reluctivity_func, 
    conductivity_func,
    current_density_func,
    omega::Float64
    )
    
    τ = CellField(tags, Ω) # Cell field for material tags

    # Bilinear form: ∫(ν ∇u ⋅ ∇v)dΩ + ∫(jωσ u v)dΩ
    a(u,v) = ∫( (reluctivity_func ∘ τ) * ∇(u) ⋅ ∇(v) )dΩ + 
             ∫( (1im * omega * conductivity_func ∘ τ) * u * v )dΩ
             
    # Linear form: ∫(J_source v)dΩ
    b(v)   = ∫( (current_density_func ∘ τ) * v )dΩ

    return WeakFormProblem(a, b)
end

"""
    magnetodynamics_1d_harmonic_coupled_weak_form(
        Ω, dΩ, tags, 
        reluctivity_func, conductivity_func, source_current_func, 
        omega::Float64
    )

Creates the weak form definition for the 1D time-harmonic magnetodynamics problem
by splitting the complex variable A = u + iv into a coupled real system.

# Returns
- `WeakFormProblem`: An instance containing the bilinear and linear forms for the coupled system (u, v).
"""
function magnetodynamics_1d_harmonic_coupled_weak_form(
    Ω::Triangulation, 
    dΩ::Measure, 
    tags::AbstractArray, 
    reluctivity_func, 
    conductivity_func,
    source_current_func, # Assumed real
    omega::Float64
    )
    
    τ = CellField(tags, Ω) 
    ν = reluctivity_func ∘ τ
    σ = conductivity_func ∘ τ
    J_r = source_current_func ∘ τ # Real source current

    # Bilinear form for the coupled system (u, v) tested with (ϕ₁, ϕ₂)
    # a((u,v), (ϕ₁,ϕ₂)) = ∫(ν∇u⋅∇ϕ₁)dΩ + ∫(ωσv ϕ₁)dΩ  (Eq 1)
    #                  + ∫(ν∇v⋅∇ϕ₂)dΩ - ∫(ωσu ϕ₂)dΩ  (Eq 2)
    a((u,v), (ϕ₁,ϕ₂)) = ∫( ν*∇(u)⋅∇(ϕ₁) + (omega*σ)*v*ϕ₁ +  # Eq 1 terms
                           ν*∇(v)⋅∇(ϕ₂) - (omega*σ)*u*ϕ₂ )dΩ  # Eq 2 terms
             
    # Linear form for the coupled system (u, v) tested with (ϕ₁, ϕ₂)
    # b((ϕ₁,ϕ₂)) = ∫(J_r ϕ₁)dΩ (from Eq 1)
    b((ϕ₁,ϕ₂)) = ∫( J_r * ϕ₁ )dΩ

    return WeakFormProblem(a, b)
end

