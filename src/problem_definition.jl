using Gridap

"""
    define_reluctivity(material_tags::Dict{String, Int}, μ0::Float64, μr_core::Float64)

Returns a function `reluctivity(tag)` that maps a material tag to its magnetic reluctivity.
"""
function define_reluctivity(material_tags::Dict{String, Int}, μ0::Float64, μr_core::Float64)
    
    function permeability(tag)
        if tag == material_tags["Air"] || tag == material_tags["Coil1"] || tag == material_tags["Coil2"]
            return μ0
        elseif tag == material_tags["Core"]
            return μ0 * μr_core
        else
            # Default or error for unknown tags
            @warn "Permeability not defined for tag $(tag), returning μ0"
            return μ0 
        end
    end

    reluctivity(tag) = 1.0 / permeability(tag)
    return reluctivity
end

"""
    define_current_density(material_tags::Dict{String, Int}, J_coil::Float64)

Returns a function `current_density(tag)` that maps a material tag to its current density.
"""
function define_current_density(material_tags::Dict{String, Int}, J_coil::Float64)
    
    function current_density(tag)
        if tag == material_tags["Coil1"]
            return J_coil
        elseif tag == material_tags["Coil2"]
            return -J_coil
        else
            return 0.0
        end
    end
    return current_density
end
