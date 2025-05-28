# Configuration file for the Future Distribution Systems AM project
# This file defines common paths and settings that can be imported in notebooks

ROOT_DIR = dirname(@__FILE__)
SRC_DIR = joinpath(ROOT_DIR, "src")
OUT_DIR = joinpath(ROOT_DIR, "out")
MATRICES_DIR = joinpath(OUT_DIR, "matrices")
REPORT_DIR = joinpath(ROOT_DIR, "report")
FIGURES_DIR = joinpath(REPORT_DIR, "Figures")


"""
Get standard project paths relative to a specified working directory.
If no working directory is specified, uses the directory containing this file.
"""
function get_project_paths(working_dir=nothing)
    if isnothing(working_dir)
        # Default to directory containing this config file
        base_dir = ROOT_DIR
    else
        base_dir = joinpath(ROOT_DIR, working_dir)
    end
    
    # Main project directories
    paths = Dict(
        "ROOT_DIR" => ROOT_DIR,
        "BASE_DIR" => base_dir,
        "SRC_DIR" => SRC_DIR,
        "OUT_DIR" => OUT_DIR,
        "MATRICES_DIR" => MATRICES_DIR,
        "REPORT_DIR" => REPORT_DIR,
        "FIGURES_DIR" => FIGURES_DIR,
        "TA_EXAMPLE_1D_DIR" => joinpath(ROOT_DIR, "ta_example_1d"),
        "TA_EXAMPLE_2D_DIR" => joinpath(ROOT_DIR, "ta_example_2d_distribution_transformer"),
    )
    
    # Sub-directories
    paths["GEO_DIR"] = joinpath(paths["BASE_DIR"], "geo")
    paths["REPORT_FIGURE_DIR"] = FIGURES_DIR
    paths["OUTPUT_DIR"] = joinpath(paths["BASE_DIR"], "output")
    
    # Create directories if they don't exist
    for (_, path) in paths
        mkpath(path)
    end
    
    return paths
end


# Default exports when including this file
export get_project_paths, ROOT_DIR, SRC_DIR, OUT_DIR, MATRICES_DIR, REPORT_DIR, FIGURES_DIR