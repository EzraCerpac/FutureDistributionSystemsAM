using Gridap

"""
    define_reluctivity(material_tags::Dict{String, Int}, μ0::Float64, μr_core::Float64; 
                          core_tag_name="Core", air_tag_name="Air")

Returns a function `reluctivity(tag)` that maps a material tag to its magnetic reluctivity.
Uses `air_tag_name` to identify the non-magnetic, non-conductive regions.
"""
function define_reluctivity(material_tags::Dict{String, Int}, μ0::Float64, μr_core::Float64; 
                          core_tag_name="Core", air_tag_name="Air")
    
    # Get the actual tag IDs, handling cases where they might be missing
    core_tag_id = get(material_tags, core_tag_name, -1)
    air_tag_id = get(material_tags, air_tag_name, -1)

    function permeability(tag)
        if haskey(material_tags, core_tag_name) && tag == material_tags[core_tag_name]
            return μ0 * μr_core
        elseif haskey(material_tags, air_tag_name) && tag == material_tags[air_tag_name]
             return μ0
        # Check if the tag belongs to any winding (assuming names contain HV or LV)
        # This requires the material_tags dict to contain all individual winding tags
        elseif any(name -> occursin("HV", name) || occursin("LV", name), keys(filter(p -> p.second == tag, material_tags)))
             return μ0 # Windings are typically in air/oil
        else
            # Default to air/oil permeability if tag is unknown or not explicitly handled
            return μ0 
        end
    end

    reluctivity(tag) = 1.0 / permeability(tag)
    return reluctivity
end

"""
    define_nonlinear_reluctivity(material_tags::Dict{String, Int}, bh_a::Float64, bh_b::Float64, bh_c::Float64, μ0::Float64;
                                     core_tag_name="Core", air_tag_name="Air")

Returns a function `reluctivity(tag, B)` that maps a material tag and B-field magnitude to magnetic reluctivity.
"""
function define_nonlinear_reluctivity(material_tags::Dict{String, Int}, bh_a::Float64, bh_b::Float64, bh_c::Float64, μ0::Float64;
                                     core_tag_name="Core", air_tag_name="Air")
    
    core_tag_id = get(material_tags, core_tag_name, -1)
    air_tag_id = get(material_tags, air_tag_name, -1)
    ν_air = 1.0 / μ0

    function reluctivity(tag, B)
        if tag == core_tag_id
            # Avoid division by zero or NaN if B is very small
            B_eff = max(B, 1e-9) 
            μr = 1.0 / (bh_a + (1 - bh_a) * B_eff^(2*bh_b) / (B_eff^(2*bh_b) + bh_c))
            return 1.0 / (μ0 * μr)
        elseif tag == air_tag_id
            return ν_air
        # Check windings
        elseif any(name -> occursin("HV", name) || occursin("LV", name), keys(filter(p -> p.second == tag, material_tags)))
             return ν_air
        else
            # Default to air reluctivity
            return ν_air
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
    define_conductivity(material_tags::Dict{String, Int}, σ_core::Float64; core_tag_name="Core", air_tag_name="Air")

Returns a function `conductivity(tag)` that maps a material tag to its electrical conductivity.
Assumes only the core is conductive. Uses `air_tag_name` for consistency.
"""
function define_conductivity(material_tags::Dict{String, Int}, σ_core::Float64; core_tag_name="Core", air_tag_name="Air")
    
    core_tag_id = get(material_tags, core_tag_name, -1)
    air_tag_id = get(material_tags, air_tag_name, -1) # Get air tag ID

    function conductivity(tag)
        if tag == core_tag_id
            return σ_core
        else
            # Assume other materials (air, oil, windings) are non-conductive
            return 0.0 
        end
    end
    return conductivity
end
