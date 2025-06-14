using Gridap
using LinearAlgebra
include("problem_definition.jl")

function define_heat_conductivity(material_tags::Dict{String, Int}, k_core::Float64, k_coil::Float64, k_air::Float64, k_oil::Float64; core_tag_name="Core", coil_tags=["Coil1", "Coil2"], air_tag_name="Air", oil_tag_name="Oil")
    function conductivity(tag)
        if haskey(material_tags, core_tag_name) && tag == material_tags[core_tag_name]
            return k_core
        elseif any(haskey(material_tags, coil_tag) && tag == material_tags[coil_tag] for coil_tag in coil_tags)
            return k_coil
        elseif haskey(material_tags, air_tag_name) && tag == material_tags[air_tag_name]
            return k_air
        elseif haskey(material_tags, oil_tag_name) && tag == material_tags[oil_tag_name]
            return k_oil
        else
            @warn "Heat conductivity not defined for tag $(tag), returning 0.0"
            return 0.0
        end
    end
    return conductivity
end

function define_heat_source(material_tags, loss_func)
    function heat_source(tag)
        if haskey(material_tags, "Core") && tag == material_tags["Core"]
            return loss_func(tag)
        else
            return 0.0
        end
    end
    return heat_source
end

# Returns a function of tag that is dissipation_coeff in the specified region, 0 elsewhere
function define_loss_region_dissipation(material_tags::Dict{String, Int}, dissipation_coeff::Float64; region_tag_name="Air")
    function region_dissipation(tag)
        if haskey(material_tags, region_tag_name) && tag == material_tags[region_tag_name]
            return dissipation_coeff
        else
            return 0.0
        end
    end
    return region_dissipation
end

function solve_heatdynamics(
    model, tags, order, dirichlet_tag, dirichlet_value, 
    loss_func, conductivity_func)

    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*order)

    # Setup heat source from loss function
    function heat_source(tag)
        return max(0.0, loss_func(tag))  # Ensure non-negative heat source
    end

    # Setup FE spaces with ambient temperature as reference
    U = FESpace(model, ReferenceFE(lagrangian, Float64, order), 
                conformity=:H1, dirichlet_tags=[dirichlet_tag])
    
    # Set Dirichlet BC to ambient temperature
    U0 = TrialFESpace(U, dirichlet_value)

    # Solve the heat problem
    problem = heat_problem_weak_form(Ω, dΩ, tags, conductivity_func, heat_source)
    
    # Solve with temperature offset from ambient
    θ = solve_fem_problem(problem, U0, U)

    # The solution θ represents temperature rise above ambient
    # Add ambient temperature back to get actual temperature
    T = θ

    return T
end

# Extended heat solver with optional dissipation in a specified region
function solve_heatdynamics_with_dissipation(
    model, tags, order, dirichlet_tag, dirichlet_value, 
    loss_func, conductivity_func,
    dissipation_func, T_ambient
)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*order)

    U = FESpace(model, ReferenceFE(lagrangian, Float64, order), 
                conformity=:H1)
    U0 = TrialFESpace(U)

    problem = heat_problem_weak_form_with_dissipation(
        Ω, dΩ, tags, conductivity_func, loss_func, dissipation_func, T_ambient)
    θ = solve_fem_problem(problem, U0, U)
    T = θ
    return T
end