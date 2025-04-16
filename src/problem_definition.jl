using Gridap

"""
    define_reluctivity(material_tags::Dict{String, Int}, μ0::Float64, μr_core::Float64)

Returns a function `reluctivity(tag)` that maps a material tag to its magnetic reluctivity.
"""
function define_reluctivity(material_tags::Dict{String, Int}, μ0::Float64, μr_core::Float64; 
                          core_tag_name="Core")
    
    function permeability(tag)
        if haskey(material_tags, core_tag_name) && tag == material_tags[core_tag_name]
            return μ0 * μr_core
        elseif haskey(material_tags, "Air") && tag == material_tags["Air"] || 
               haskey(material_tags, "Coil1") && tag == material_tags["Coil1"] || 
               haskey(material_tags, "Coil2") && tag == material_tags["Coil2"] ||
               haskey(material_tags, "HV winding phase 1 left") && tag == material_tags["HV winding phase 1 left"] ||
               # Add other coil tags as needed
               false
            return μ0
        else
            @warn "Permeability not defined for tag $(tag), returning μ0"
            return μ0 
        end
    end

    reluctivity(tag) = 1.0 / permeability(tag)
    return reluctivity
end

"""
    define_nonlinear_reluctivity(material_tags::Dict{String, Int}, bh_a::Float64, bh_b::Float64, bh_c::Float64, μ0::Float64)

Returns a function `reluctivity(tag, B)` that maps a material tag and B-field magnitude to magnetic reluctivity.
"""
function define_nonlinear_reluctivity(material_tags::Dict{String, Int}, bh_a::Float64, bh_b::Float64, bh_c::Float64, μ0::Float64)
    function reluctivity(tag, B)
        if haskey(material_tags, "Core") && tag == material_tags["Core"]
            μr = 1.0 / (bh_a + (1 - bh_a) * B^(2*bh_b) / (B^(2*bh_b) + bh_c))
            return 1.0 / (μ0 * μr)
        elseif haskey(material_tags, "Air") && tag == material_tags["Air"] || 
               haskey(material_tags, "Coil1") && tag == material_tags["Coil1"] || 
               haskey(material_tags, "Coil2") && tag == material_tags["Coil2"]
            return 1.0 / μ0
        else
            @warn "Reluctivity not defined for tag $(tag), returning air reluctivity"
            return 1.0 / μ0
        end
    end
    
    return reluctivity
end

"""
    update_reluctivity_from_field(original_func, B_field, tags)

Returns an updated reluctivity function that incorporates B-field information.
"""
function update_reluctivity_from_field(original_func, B_field, tags)
    function updated_reluctivity(tag)
        # Find the average B magnitude for this tag
        elements_with_tag = findall(t -> t == tag, tags)
        if isempty(elements_with_tag)
            # No elements with this tag, return original value
            return original_func(tag, 0.0)
        end
            
        # Calculate average B magnitude for this material
        avg_B = 0.0
        count = 0
        for idx in elements_with_tag
            if idx <= length(B_field)
                avg_B += norm(B_field[idx])
                count += 1
            end
        end
        
        if count > 0
            avg_B /= count
        end
            
        # Return reluctivity for this tag with the average B magnitude
        return original_func(tag, avg_B)
    end
    
    return updated_reluctivity
end

"""
    define_current_density(material_tags::Dict{String, Int}, J_coil::Float64; coil_tags_pos=["Coil1"], coil_tags_neg=["Coil2"])

Returns a function `current_density(tag)` that maps a material tag to its current density.
"""
function define_current_density(material_tags::Dict{String, Int}, J_coil::Float64; 
                               coil_tags_pos=["Coil1"], coil_tags_neg=["Coil2"])
    
    function current_density(tag)
        # Check positive coil tags
        for coil_tag in coil_tags_pos
            if haskey(material_tags, coil_tag) && tag == material_tags[coil_tag]
                return J_coil
            end
        end
        
        # Check negative coil tags
        for coil_tag in coil_tags_neg
            if haskey(material_tags, coil_tag) && tag == material_tags[coil_tag]
                return -J_coil
            end
        end
        
        # Default case - no current
        return 0.0
    end
    
    return current_density
end

"""
    define_conductivity(material_tags::Dict{String, Int}, σ_core::Float64; core_tag_name="Core")

Returns a function `conductivity(tag)` that maps a material tag to its electrical conductivity.
Assumes only the core is conductive.
"""
function define_conductivity(material_tags::Dict{String, Int}, σ_core::Float64; core_tag_name="Core")
    
    function conductivity(tag)
        if haskey(material_tags, core_tag_name) && tag == material_tags[core_tag_name]
            return σ_core
        else
            # Assume other materials are non-conductive
            return 0.0 
        end
    end
    return conductivity
end
