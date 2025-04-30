using Gridap
using WriteVTK # Still needed for single file saving
using Gridap.MultiField: MultiFieldFEFunction
using Gridap.Visualization: createpvd, createvtk, vtk_save, VTKCellData # Ensure createvtk is imported

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
    calculate_b_field(Az::Vector{ComplexF64}) -> (B_re_approx, B_im_approx)

Approximates B-field components from a vector of complex Az values.
"""
function calculate_b_field(Az::Vector{ComplexF64})
    # Raise not implemented error for now
    error("calculate_b_field for Vector{ComplexF64} is not implemented yet.")
end

"""
    save_results_vtk(Ω, filenamebase, fields::Dict; save_time_series::Bool=false, ω::Union{Float64, Nothing}=nothing, nframes::Int=20)

Saves FE results to a VTK file or a PVD time series.

If `save_time_series` is false (default): Saves components of MultiFieldFEFunction (e.g., `_re`, `_im`) or scalar fields to a single VTK file using `WriteVTK.writevtk`.
If `save_time_series` is true: Saves instantaneous values over one period (2π/ω) to a PVD collection referencing multiple VTK files using `Gridap.Visualization.createpvd` and `createvtk`. Requires `ω` to be provided.
"""
function save_results_vtk(Ω, filenamebase, fields::Dict; save_time_series::Bool=false, ω::Union{Float64, Nothing}=nothing, nframes::Int=20)

    if save_time_series
        if ω === nothing
            error("Angular frequency ω must be provided when save_time_series is true.")
        end
        if nframes <= 0
             error("nframes must be positive for time series saving.")
        end

        println("Saving results as time series PVD...")
        T_period = 2π / ω
        t_vec = range(0, T_period, length=nframes)

        # Use createpvd with a do-block
        createpvd(filenamebase) do pvd
            for (i, t_step) in enumerate(t_vec)
                cos_wt = cos(ω * t_step)
                sin_wt = sin(ω * t_step)
                vtk_fields_inst = Dict{String, Any}()
                processed_bases = Set{String}()

                # --- Logic for calculating instantaneous fields (vtk_fields_inst) ---
                for (name, field) in fields
                    if isa(field, MultiFieldFEFunction)
                        if length(field) >= 2
                            inst_val = field[1] * cos_wt - field[2] * sin_wt
                            vtk_fields_inst[name] = inst_val
                            push!(processed_bases, name)
                        else
                            println("Warning: MultiFieldFEFunction '$name' has < 2 components, cannot compute instantaneous value for t=$t_step.")
                        end
                    elseif endswith(name, "_re")
                        base_name = name[1:end-3]
                        if base_name in processed_bases continue end
                        name_im = base_name * "_im"
                        if haskey(fields, name_im)
                            field_re = field
                            field_im = fields[name_im]
                            try
                                inst_val = field_re * cos_wt - field_im * sin_wt
                                vtk_fields_inst[base_name] = inst_val
                                push!(processed_bases, base_name)
                            catch e
                                println("Warning: Could not combine '$name' and '$name_im' for t=$t_step. Error: $e")
                                if !(name in keys(vtk_fields_inst)) vtk_fields_inst[name] = field_re end
                                if !(name_im in keys(vtk_fields_inst)) vtk_fields_inst[name_im] = field_im end
                            end
                        else
                            if !(name in keys(vtk_fields_inst)) vtk_fields_inst[name] = field end
                        end
                    elseif endswith(name, "_im")
                        base_name = name[1:end-3]
                        if base_name in processed_bases continue end
                        name_re = base_name * "_re"
                        if !haskey(fields, name_re)
                            try
                                inst_val = -field * sin_wt
                                vtk_fields_inst[base_name] = inst_val
                                push!(processed_bases, base_name)
                            catch e
                                println("Warning: Could not compute instantaneous value for '$name' (imaginary only) for t=$t_step. Error: $e")
                                if !(name in keys(vtk_fields_inst)) vtk_fields_inst[name] = field end
                            end
                        end
                    else
                        if !(name in processed_bases) && !(name in keys(vtk_fields_inst))
                            vtk_fields_inst[name] = field
                        end
                    end
                end
                # --- End of logic for calculating vtk_fields_inst ---

                # Generate filename for this time step's VTU file
                filename_t = filenamebase * "_t$(lpad(i, 4, '0'))"

                try
                    # Use createvtk to generate the VTKFile object and add it to the PVD
                    pvd[t_step] = createvtk(Ω, filename_t, cellfields=vtk_fields_inst, order=2)
                    println("Saved frame $i (t=$t_step) to $filename_t.vtu")
                catch e
                    println("Error saving VTK file for time step $t_step (frame $i): $e")
                    showerror(stdout, e)
                    Base.show_backtrace(stdout, catch_backtrace())
                    println()
                end
            end
        end # End of createpvd block (automatically saves the PVD file)

        println("Time series PVD collection saved to $filenamebase.pvd")
        return # Exit after saving time series

    else
        # --- Original logic for saving a single VTK file using WriteVTK ---
        println("Saving results to single VTK file...")
        vtk_fields = Dict{String, Any}()
        for (name, field) in fields
            if isa(field, MultiFieldFEFunction)
                try
                    if length(field) >= 2
                        vtk_fields[name * "_re"] = field[1]
                        vtk_fields[name * "_im"] = field[2]
                    else
                         println("Warning: MultiFieldFEFunction '$name' has fewer than 2 components, saving as is.")
                         vtk_fields[name] = field
                    end
                catch e
                    println("Warning: Could not split MultiFieldFEFunction '$name' for VTK output. Error: $e")
                    vtk_fields[name] = field
                end
            else
                vtk_fields[name] = field
            end
        end

        try
            # Use WriteVTK.writevtk for the single file case
            WriteVTK.writevtk(Ω, filenamebase, cellfields=vtk_fields, order=2)
            println("Results saved to $filenamebase.vtu")
        catch e
            println("Error saving single VTK file: $e")
            println("Attempting to save components individually...")
            for (name, field) in vtk_fields
                 try
                     WriteVTK.writevtk(Ω, filenamebase * "_$name", cellfields=Dict(name => field), order=2)
                     println("Saved component $name separately to $(filenamebase)_$name.vtu")
                 catch e_comp
                     println("Failed to save component $name. Error: $e_comp")
                 end
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
