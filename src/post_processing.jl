using Gridap
using WriteVTK
using Gridap.MultiField: MultiFieldFEFunction

"""
    calculate_b_field(Az) -> B

Calculates B = -∇(Az) for scalar Az (magnetostatics).
"""
function calculate_b_field(Az::FEFunction)
    # Assumes Az is scalar (e.g., Float64)
    B = -∇(Az)
    return B
end

"""
    calculate_b_field(uv::MultiFieldFEFunction) -> (B_re, B_im)

Calculates real and imaginary parts of B = -∇(u) - i∇(v) for coupled solution uv = (u, v).
"""
function calculate_b_field(uv::MultiFieldFEFunction)
    u = uv[1] # Real part of Az
    v = uv[2] # Imag part of Az
    B_re = -∇(u)
    B_im = -∇(v)
    return B_re, B_im
end

"""
    save_results_vtk(Ω, filenamebase, fields::Dict)

Saves FE results to a VTK file. Handles potential MultiFieldFEFunction inputs by saving components.
"""
function save_results_vtk(Ω, filenamebase, fields::Dict)
    vtk_fields = Dict{String, Any}()
    for (name, field) in fields
        if isa(field, MultiFieldFEFunction)
            # Example: Split 'uv' into 'u' and 'v'
            # This part might need adjustment based on how you want to name components
            try 
                vtk_fields[name * "_re"] = field[1]
                vtk_fields[name * "_im"] = field[2]
            catch e
                println("Warning: Could not split MultiFieldFEFunction '$name' for VTK output. Error: $e")
                vtk_fields[name] = field # Save the multifield object itself (may not be ideal for viz)
            end
        else
            vtk_fields[name] = field
        end
    end
    
    try
        writevtk(Ω, filenamebase, cellfields=vtk_fields)
        println("Results saved to $filenamebase.vtu")
    catch e
        println("Error saving VTK file: $e")
        # Optionally save components individually if the combined dict fails
        for (name, field) in vtk_fields
             try
                 writevtk(Ω, filenamebase * "_$name", cellfields=Dict(name => field))
                 println("Saved component $name separately.")
             catch e_comp
                 println("Failed to save component $name. Error: $e_comp")
             end
        end
    end
end

"""
    calculate_eddy_current(Az::FEFunction, conductivity_func, omega::Float64, Ω::Triangulation, tags::AbstractArray)

Calculates the eddy current density J_eddy = jω σ Az.
"""
function calculate_eddy_current(Az::FEFunction, conductivity_func, omega::Float64, Ω::Triangulation, tags::AbstractArray)
    τ = CellField(tags, Ω)
    J_eddy = (1im * omega * conductivity_func ∘ τ) * Az
    return J_eddy
end

"""
    calculate_eddy_current(uv::MultiFieldFEFunction, conductivity_func, omega::Float64, Ω::Triangulation, tags::AbstractArray) -> (J_eddy_re, J_eddy_im)

Calculates real and imaginary parts of eddy current J_eddy = -ωσv + jωσu for coupled solution uv = (u, v).
"""
function calculate_eddy_current(uv::MultiFieldFEFunction, conductivity_func, omega::Float64, Ω::Triangulation, tags::AbstractArray)
    u = uv[1] # Real part of Az
    v = uv[2] # Imag part of Az
    τ = CellField(tags, Ω)
    σ = conductivity_func ∘ τ
    
    J_eddy_re = -(omega * σ) * v
    J_eddy_im = (omega * σ) * u
    return J_eddy_re, J_eddy_im
end
