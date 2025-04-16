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

Creates the weak form for 1D magnetostatics.
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
    magnetodynamics_1d_harmonic_weak_form(Ω, dΩ, tags, reluctivity_func, conductivity_func, current_density_func, omega)

Creates the weak form for 1D time-harmonic magnetodynamics.
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
    magnetodynamics_1d_harmonic_coupled_weak_form(Ω, dΩ, tags, reluctivity_func, conductivity_func, source_current_func, ω)

Creates the weak form for 1D time-harmonic magnetodynamics using coupled real/imaginary parts.
"""
function magnetodynamics_1d_harmonic_coupled_weak_form(
    Ω::Triangulation, 
    dΩ::Measure, 
    tags::AbstractArray, 
    reluctivity_func, 
    conductivity_func, 
    source_current_func, 
    ω::Float64
    )

    τ = CellField(tags, Ω) 
    ν = reluctivity_func ∘ τ
    σ = conductivity_func ∘ τ
    J_r = source_current_func ∘ τ # Real source current

    # Bilinear form for the coupled system (u, v) tested with (ϕ₁, ϕ₂)
    # a((u,v), (ϕ₁,ϕ₂)) = ∫(ν∇u⋅∇ϕ₁)dΩ + ∫(ωσv ϕ₁)dΩ  (Eq 1)
    #                  + ∫(ν∇v⋅∇ϕ₂)dΩ - ∫(ωσu ϕ₂)dΩ  (Eq 2)
    a((u,v), (ϕ₁,ϕ₂)) = ∫( ν*∇(u)⋅∇(ϕ₁) + (ω*σ)*v*ϕ₁ +  # Eq 1 terms
                           ν*∇(v)⋅∇(ϕ₂) - (ω*σ)*u*ϕ₂ )dΩ  # Eq 2 terms
             
    # Linear form for the coupled system (u, v) tested with (ϕ₁, ϕ₂)
    # b((ϕ₁,ϕ₂)) = ∫(J_r ϕ₁)dΩ (from Eq 1)
    b((ϕ₁,ϕ₂)) = ∫( J_r * ϕ₁ )dΩ

    return WeakFormProblem(a, b)
end

"""
    magnetodynamics_2d_harmonic_coupled_weak_form(Ω, dΩ, tags, reluctivity_func, conductivity_func, source_current_func, ω)

Creates the weak form for 2D time-harmonic magnetodynamics using coupled real/imaginary parts.
"""
function magnetodynamics_2d_harmonic_coupled_weak_form(
    Ω::Triangulation,
    dΩ::Measure,
    tags::AbstractArray,
    reluctivity_func,
    conductivity_func,
    source_current_func,
    ω::Float64
    )

    τ = CellField(tags, Ω)
    ν = reluctivity_func ∘ τ
    σ = conductivity_func ∘ τ
    J0 = source_current_func ∘ τ

    # Define the individual terms for the coupled system
    a11(u,ϕ) = ∫( ν*∇(u)⋅∇(ϕ) )dΩ
    a12(v,ϕ) = ∫( -ω*σ*v*ϕ )dΩ
    a21(u,ψ) = ∫( ω*σ*u*ψ )dΩ
    a22(v,ψ) = ∫( ν*∇(v)⋅∇(ψ) )dΩ

    b1(ϕ) = ∫( J0*ϕ )dΩ
    b2(ψ) = ∫( 0*ψ )dΩ

    terms = [a11, a12, a21, a22]
    lterms = [b1, b2]

    return WeakFormProblem(terms, lterms)
end

