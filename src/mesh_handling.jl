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

