using Gridap
using LinearAlgebra
include("problem_definition.jl")

function define_heat_conductivity(material_tags::Dict{String, Int}, k_core::Float64, k_coil::Float64, k_air::Float64; core_tag_name="Core", coil_tags=["Coil1", "Coil2"], air_tag_name="Air")
    function conductivity(tag)
        if haskey(material_tags, core_tag_name) && tag == material_tags[core_tag_name]
            return k_core
        elseif any(haskey(material_tags, coil_tag) && tag == material_tags[coil_tag] for coil_tag in coil_tags)
            return k_coil
        elseif haskey(material_tags, air_tag_name) && tag == material_tags[air_tag_name]
            return k_air
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


function solve_heatdynamics(
    model, tags, order, dirichlet_tag, dirichlet_value, 
    loss_func, conductivity_func)

    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2*order)

    # Use loss_func directly as the heat source term
    Q = loss_func

    U, V = setup_fe_spaces(Ω, order, Float64, dirichlet_tag, dirichlet_value)

    problem = heat_problem_weak_form(Ω, dΩ, tags, conductivity_func, Q)

    uv_FESpace = solve_fem_problem(problem, U, V)
    return uv_FESpace
end