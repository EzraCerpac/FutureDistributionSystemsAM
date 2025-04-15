using Gridap
using WriteVTK

"""
    calculate_b_field(Az::FEFunction)

Calculates the magnetic flux density B from the magnetic vector potential Az.
Assumes B = -∇(Az) for the 1D case (By = -dAz/dx).
"""
function calculate_b_field(Az::FEFunction)
    # In 1D, ∇(Az) gives (dAz/dx,), so B = (0, -dAz/dx, 0) effectively.
    # Gridap's ∇ on a scalar FEFunction in 1D returns a VectorValue field.
    B = -∇(Az) 
    return B
end

"""
    save_results_vtk(Ω, output_file_base::String, fields::Dict{String, <:Union{FEFunction, CellField}})

Saves the specified fields to a VTK file.
"""
function save_results_vtk(Ω::Triangulation, output_file_base::String, fields::Dict{String, <:Union{FEFunction, CellField}})
    vtk_filename = output_file_base # No need for .vtu extension, writevtk handles it
    writevtk(Ω, vtk_filename, cellfields=fields)
    println("Results saved to $(vtk_filename).vtu")
end
