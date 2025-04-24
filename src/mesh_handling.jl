using Gridap
using GridapGmsh
using Gridap.Geometry: get_face_tag, get_face_labeling, get_tag_from_name

"""
    load_mesh_and_tags(mesh_file::String)

Loads a Gmsh mesh file, creates a Gridap model, and extracts face labeling and tags.
"""
function load_mesh_and_tags(mesh_file::String)
    model = GmshDiscreteModel(mesh_file)
    labels = get_face_labeling(model)
    dimension = num_cell_dims(model)
    tags = get_face_tag(labels, dimension)
    return model, labels, tags
end

"""
    get_material_tags(labels)

Extracts specific material tags ("Air", "Core", "Coil1", "Coil2") from the face labeling.
Assumes Gmsh physical group names match these strings.
"""
function get_material_tags(labels)
    tag_air = get_tag_from_name(labels, "Air")
    tag_core = get_tag_from_name(labels, "Core")
    # Adjust names based on actual Gmsh physical group names
    tag_coil1 = get_tag_from_name(labels, "Coil left") 
    tag_coil2 = get_tag_from_name(labels, "Coil right")
    return Dict(
        "Air" => tag_air,
        "Core" => tag_core,
        "Coil1" => tag_coil1,
        "Coil2" => tag_coil2
    )
end

"""
    get_material_tags_2d(labels)

Extracts specific material tags for the 2D transformer model.
Assumes Gmsh physical group names match the specified strings.
"""
function get_material_tags_2d(labels)
    tags = Dict{String, Int}()
    tag_names = [
        "Oil", "Core",
        "HV1l", "HV1r", "HV2l", "HV2r", "HV3l", "HV3r",
        "LV1l", "LV1r", "LV2l", "LV2r", "LV3l", "LV3r",
        "HV windings", "LV windings", "Enclosure"
    ]
    
    # Note: "Enclosure" is a boundary tag (dim 1), others are surface tags (dim 2)
    # We only fetch surface tags here for material properties.
    # Boundary tags are handled separately (e.g., for Dirichlet conditions).
    surface_tag_names = filter(name -> name != "Enclosure", tag_names)

    for name in surface_tag_names
        try
            tags[name] = get_tag_from_name(labels, name)
        catch e
            println("Warning: Could not find tag '$name' in the mesh labels.")
        end
    end

    # Add specific aliases if needed, e.g., for generic functions
    if haskey(tags, "Oil")
        tags["Air"] = tags["Oil"] # Treat Oil as Air for permeability/conductivity
    end
    # Add aliases for winding groups if needed by generic functions
    if haskey(tags, "HV windings")
        # Example: If a function expects "Coil1", "Coil2", etc.
        # tags["Coil1"] = tags["HV windings"] 
    end
     if haskey(tags, "LV windings")
        # tags["Coil2"] = tags["LV windings"]
    end

    # Add individual winding tags for source definition
    winding_tags = filter(name -> occursin("HV", name) || occursin("LV", name), tag_names)
    for name in winding_tags
         if !haskey(tags, name) # Avoid overwriting if already added (e.g., "HV windings")
            try
                tags[name] = get_tag_from_name(labels, name)
            catch e
                 println("Warning: Could not find winding tag '$name' in the mesh labels.")
            end
         end
    end


    return tags
end

