module PostProcessing

using Gridap
using Printf
using WriteVTK # Still needed for single file saving (though not directly by new func)
using Gridap.MultiField: MultiFieldFEFunction
using Gridap.Visualization: createpvd, createvtk, vtk_save, VTKCellData # Ensure createvtk is imported

export calculate_b_field, save_results_vtk, calculate_eddy_current # Existing exports
export save_pvd_and_extract_signal # New export

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

function save_results_vtk(Ω::Triangulation, output_file_base::String, fields_dict::Dict{String, Any}; append::Bool=false)
    # Ensure the output directory exists
    output_dir = dirname(output_file_base)
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    base_name = first(splitext(output_file_base))
    full_output_path = base_name * ".vtk"
    
    try
        Gridap.writevtk(Ω, full_output_path; cellfields=fields_dict)
        println("Successfully saved results to $(full_output_path)")
    catch e
        println("Error saving single VTK file: $(e)")
        println("Attempting to save components individually...")
        # Fallback: Save each component to a separate file if the combined save fails
        for (name, field) in fields_dict
            individual_output_path = base_name * "_" * name * ".vtk"
            try
                # Removing explicit append here as well.
                Gridap.writevtk(Ω, individual_output_path; cellfields=Dict(name => field))
                println("Successfully saved $(name) to $(individual_output_path)")
            catch individual_e
                println("Failed to save component $(name). Error: $(individual_e)")
            end
        end
    end
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

"""
    save_pvd_and_extract_signal(
        solution_iterable, 
        Az0::FEFunction, 
        Ω::Triangulation, 
        pvd_filename_base::String, 
        t0::Float64, 
        x_probe::Union{Point, VectorValue}, 
        steps_for_fft_start_time::Float64,
        freq::Float64,
        num_periods_collect_fft::Real # Can be Int or Float
    ) -> Tuple{Vector{Float64}, Vector{Float64}}

Saves transient solution snapshots to a PVD file and extracts the time signal of Az at a probe point.
This function is adapted from the post-processing block in `examples/transient_1d.jl`.
"""
function save_pvd_and_extract_signal(
    solution_iterable, 
    Az0::FEFunction, 
    Ω::Triangulation, 
    pvd_filename_base::String, 
    t0::Float64, 
    x_probe::Union{Point, VectorValue}, 
    steps_for_fft_start_time::Float64,
)
    time_signal_data = Float64[]
    time_steps_for_fft = Float64[]

    println("Extracting solution at probe point $(x_probe) and saving PVD to $(pvd_filename_base).pvd...")
    
    # Ensure Gridap.Visualization is accessible for createpvd/createvtk
    createpvd(pvd_filename_base) do pvd_file
        # Save initial condition
        try
            # Using a slightly more robust naming for snapshot files
            pvd_file[t0] = createvtk(Ω, pvd_filename_base * "_t_initial_snapshot", cellfields=Dict("Az" => Az0))
            println("Saved initial condition to PVD.")
        catch e_pvd_init
            println("Error saving initial condition to PVD: $e_pvd_init")
        end

        for (i, (Az_n, tn)) in enumerate(solution_iterable)
            try
                # Ensure filename compatibility for createvtk, avoid issues with '.' from Printf
                tn_str_for_file = replace(Printf.@sprintf("%.4f", tn), "." => "p")
                pvd_file[tn] = createvtk(Ω, pvd_filename_base * "_t_$(tn_str_for_file)_snapshot", cellfields=Dict("Az" => Az_n))
            catch e_pvd_step
                println("Error saving step t=$tn to PVD: $e_pvd_step")
            end
            
            if tn >= steps_for_fft_start_time
                push!(time_steps_for_fft, tn)
                try
                    probe_val = Az_n(x_probe)
                    # Ensure probe_val is scalar Float64 if Az_n(x_probe) returns a single-element array or similar
                    if isa(probe_val, AbstractArray) && length(probe_val) == 1
                        push!(time_signal_data, first(probe_val))
                    elseif isa(probe_val, Number)
                        push!(time_signal_data, Float64(probe_val))
                    else
                        println("Warning: Probe value at t=$(tn) is not a scalar Number or single-element array. Type: $(typeof(probe_val)). Storing NaN.")
                        push!(time_signal_data, NaN)
                    end
                catch e_probe
                    println("Warning: Could not evaluate Az_n at probe point $(x_probe) for t=$(tn). Error: $e_probe. Storing NaN.")
                    push!(time_signal_data, NaN)
                end
            end

            if i % 20 == 0 
                println("Processed PVD save for t=$(@sprintf("%.4f", tn))")
            end
        end
    end
    println("Finished PVD saving to $(pvd_filename_base).pvd")
    
    return time_steps_for_fft, time_signal_data
end

end # module PostProcessing
