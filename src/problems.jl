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
    magnetostatics_weak_form(Ω, dΩ, tags, reluctivity_func, current_density_func)

Creates the weak form for magnetostatics.
"""
function magnetostatics_weak_form(
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
    magnetodynamics_harmonic_weak_form(Ω, dΩ, tags, reluctivity_func, conductivity_func, current_density_func, omega)

Creates the weak form for time-harmonic magnetodynamics.
"""
function magnetodynamics_harmonic_weak_form(
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
    magnetodynamics_harmonic_coupled_weak_form(Ω, dΩ, tags, reluctivity_cellfield, conductivity_func, source_current_func, ω)

Creates the weak form for time-harmonic magnetodynamics using coupled real/imaginary parts.
`reluctivity_cellfield` is a CellField representing reluctivity for each cell.
`conductivity_func` and `source_current_func` are functions of material tags.
"""
function magnetodynamics_harmonic_coupled_weak_form(
    Ω::Triangulation, 
    dΩ::Measure, 
    tags::AbstractArray, 
    reluctivity_cellfield::CellField,
    conductivity_func::Function,
    source_current_func::Function,
    ω::Float64
    )

    τ = CellField(tags, Ω) # Cell field for material tags (used for conductivity and source)
    ν = reluctivity_cellfield   # Use CellField directly
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
    magnetodynamics_harmonic_coupled_weak_form(Ω, dΩ, tags, reluctivity_func, conductivity_func, source_current_func, ω)

Creates the weak form for time-harmonic magnetodynamics using coupled real/imaginary parts.
"""
function magnetodynamics_harmonic_coupled_weak_form(
    Ω::Triangulation, 
    dΩ::Measure, 
    tags::AbstractArray, 
    reluctivity_func::Function,
    conductivity_func::Function, 
    source_current_func::Function,
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

function heat_problem_weak_form(
    Ω::Triangulation, 
    dΩ::Measure, 
    tags::AbstractArray, 
    diffusivity_func, 
    source_func
    )
    
    τ = CellField(tags, Ω)
    k = diffusivity_func ∘ τ  # Thermal conductivity
    Q = source_func ∘ τ      # Heat source (loss density)

    # Modified weak form to ensure non-negative temperature difference
    a(u,v) = ∫( k * ∇(u) ⋅ ∇(v) )dΩ
    b(v)   = ∫( Q * v )dΩ

    return WeakFormProblem(a, b)
end
