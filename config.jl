# Configuration file for the Future Distribution Systems AM project
# This file defines common paths and settings that can be imported in notebooks

"""
Get standard project paths relative to a specified working directory.
If no working directory is specified, uses the directory containing this file.
"""
function get_project_paths(working_dir=nothing)
    root_dir = dirname(@__FILE__)
    if isnothing(working_dir)
        # Default to directory containing this config file
        base_dir = root_dir
    else
        base_dir = joinpath(root_dir, working_dir)
    end
    
    # Main project directories
    paths = Dict(
        "ROOT_DIR" => root_dir,
        "BASE_DIR" => base_dir,
        "SRC_DIR" => joinpath(root_dir, "src"),
        "TA_EXAMPLE_1D_DIR" => joinpath(root_dir, "ta_example_1d"),
        "TA_EXAMPLE_2D_DIR" => joinpath(root_dir, "ta_example_2d_distribution_transformer"),
        "REPORT_DIR" => joinpath(root_dir, "report")
    )
    
    # Sub-directories
    paths["GEO_DIR"] = joinpath(paths["BASE_DIR"], "geo")
    paths["REPORT_FIGURE_DIR"] = joinpath(paths["REPORT_DIR"], "Figures")
    paths["OUTPUT_DIR"] = joinpath(paths["BASE_DIR"], "output")
    
    # Create directories if they don't exist
    for (_, path) in paths
        mkpath(path)
    end
    
    return paths
end


# Default exports when including this file
export get_project_paths