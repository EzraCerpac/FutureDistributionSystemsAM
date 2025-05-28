module PostProcessing

using Gridap
using Printf
using WriteVTK # Still needed for single file saving (though not directly by new func)
using Gridap.MultiField: MultiFieldFEFunction
using Gridap.Visualization: createpvd, createvtk, vtk_save, VTKCellData # Ensure createvtk is imported
using Gridap.FESpaces: FEFunction # Added for type hint

export calculate_b_field, save_results_vtk, calculate_eddy_current
export save_pvd_and_extract_signal, save_transient_pvd
export calculate_transient_jeddy, process_transient_solution, process_harmonic_solution

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
    calculate_transient_jeddy(Az_current::FEFunction, Az_previous::FEFunction, σ_cf::Union{CellField, Function, Number}, Δt::Float64, Ω::Triangulation; initial_step::Bool=false) -> CellField

Calculates transient eddy current density J_eddy = -σ * (Az_current - Az_previous) / Δt.
For the initial step or when Δt ≤ 1e-9, returns zero current.
"""
function calculate_transient_jeddy(Az_current::FEFunction, Az_previous::FEFunction, σ_cf::Union{CellField, Function, Number}, Δt::Float64, Ω::Triangulation; initial_step::Bool=false)
    if initial_step || Δt <= 1e-9
        return CellField(0.0, Ω)
    else
        return -σ_cf .* (Az_current .- Az_previous) ./ Δt
    end
end

"""
    process_transient_solution(solution_iterable, Az0::FEFunction, Ω::Triangulation, σ_cf::Union{CellField, Function, Number}, Δt::Float64) -> Vector{Tuple}

Processes a transient solution iterable to calculate B-field and J_eddy for each time step.
Returns a vector of tuples: [(Az_n, B_n, J_eddy_n, tn), ...] where:
- Az_n: Vector potential at time step n
- B_n: Magnetic field B = -∇(Az_n) 
- J_eddy_n: Eddy current density
- tn: Time at step n
"""
function process_transient_solution(solution_iterable, Az0::FEFunction, Ω::Triangulation, σ_cf::Union{CellField, Function, Number}, Δt::Float64)
    processed_steps = []
    
    # Process initial condition
    B0 = calculate_b_field(Az0)
    J_eddy_0 = CellField(0.0, Ω)  # Zero at initial time
    push!(processed_steps, (Az0, B0, J_eddy_0, 0.0))
    
    Az_prev = Az0
    t_prev = 0.0
    
    for (step_idx, (Az_n, tn)) in enumerate(solution_iterable)
        # Calculate B-field
        B_n = calculate_b_field(Az_n)
        
        # Calculate effective time step
        Δt_eff = tn - t_prev
        if Δt_eff <= 1e-9
            Δt_eff = Δt  # Fallback to nominal Δt
        end
        
        # Calculate J_eddy for this step
        J_eddy_n = calculate_transient_jeddy(Az_n, Az_prev, σ_cf, Δt_eff, Ω; initial_step=false)
        
        push!(processed_steps, (Az_n, B_n, J_eddy_n, tn))
        
        # Update previous values for next iteration
        Az_prev = Az_n
        t_prev = tn
        
        if step_idx % 50 == 0
            println("Processed transient step $(step_idx) at t=$(tn)")
        end
    end
    
    println("Processed $(length(processed_steps)) total transient steps including initial condition.")
    return processed_steps
end

function process_harmonic_solution(solution::MultiFieldFEFunction, Ω::Triangulation, reluctivity_func, conductivity_func, ω::Float64, tags::AbstractArray)
    # A field components
    Az_re = solution[1]
    Az_im = solution[2]
    
    # Compute B-field (Real and Imag parts)
    B_re, B_im = calculate_b_field(solution)

    # Compute Eddy Currents (Real and Imag parts)
    # Ensure conductivity_func here is for electrical conductivity
    J_eddy_re, J_eddy_im = calculate_eddy_current(solution, conductivity_func, ω, Ω, tags)

    # Define helper functions for magnitude squared
    mag_sq_scalar(re, im) = re*re + im*im
    mag_sq_vector(re, im) = inner(re, re) + inner(im, im)

    # Calculate Magnitudes for saving/plotting using composition
    Az_mag = sqrt ∘ (mag_sq_scalar ∘ (Az_re, Az_im))
    B_mag = sqrt ∘ (mag_sq_vector ∘ (B_re, B_im))
    Jeddy_mag_squared = mag_sq_vector ∘ (J_eddy_re, J_eddy_im)
    Jeddy_mag = sqrt ∘ (mag_sq_scalar ∘ (J_eddy_re, J_eddy_im))

    τ_cell_field = CellField(tags, Ω) # 'tags' is the vector of cell tags, Ω is Triangulation
    ν_field_linear = Operation(reluctivity_func)(τ_cell_field)

    return Az_mag, B_mag, Jeddy_mag, Az_re, Az_im, B_re, B_im, J_eddy_re, J_eddy_im, ν_field_linear
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
        σ_cf::Union{CellField, Function, Number}, 
        Δt::Float64
    ) -> Tuple{Vector{Float64}, Vector{Float64}}

Saves transient solution snapshots to a PVD file with B-field and J_eddy calculations, and extracts the time signal of Az at a probe point.
Enhanced version that includes B-field and eddy current calculations in VTK output.
"""
function save_pvd_and_extract_signal(
    solution_iterable, 
    Az0::FEFunction, 
    Ω::Triangulation, 
    pvd_filename_base::String, 
    t0::Float64, 
    x_probe::Union{Point, VectorValue}, 
    steps_for_fft_start_time::Float64,
    σ_cf::Union{CellField, Function, Number}, 
    Δt::Float64
)
    time_signal_data = Float64[]
    time_steps_for_fft = Float64[]

    println("Extracting solution at probe point $(x_probe) and saving enhanced PVD with B-field and J_eddy to $(pvd_filename_base).pvd...")
    
    # Process transient solution to get B and J_eddy for all steps
    processed_steps = process_transient_solution(solution_iterable, Az0, Ω, σ_cf, Δt)
    
    # Ensure Gridap.Visualization is accessible for createpvd/createvtk
    createpvd(pvd_filename_base) do pvd_file
        # Process each step including initial condition
        for (step_idx, (Az_n, B_n, J_eddy_n, tn)) in enumerate(processed_steps)
            try
                # Ensure filename compatibility for createvtk
                tn_str_for_file = replace(Printf.@sprintf("%.4f", tn), "." => "p")
                filename_base = pvd_filename_base * "_t_$(tn_str_for_file)_snapshot"
                
                # Create comprehensive field dictionary for VTK output
                vtk_fields = Dict{String, Any}()
                vtk_fields["Az"] = Az_n
                
                # Add B-field components
                dim = num_cell_dims(Ω)
                if dim == 1
                    # For 1D, B is a vector but we typically want Bx component
                    vtk_fields["Bx"] = Operation(b_vec -> b_vec[1])(B_n)
                    vtk_fields["B_magnitude"] = Operation(b_vec -> sqrt(inner(b_vec, b_vec)))(B_n)
                elseif dim == 2
                    # For 2D, add both components and magnitude
                    vtk_fields["Bx"] = Operation(b_vec -> b_vec[1])(B_n)
                    vtk_fields["By"] = Operation(b_vec -> b_vec[2])(B_n) 
                    vtk_fields["B_magnitude"] = Operation(b_vec -> sqrt(inner(b_vec, b_vec)))(B_n)
                end
                
                # Add eddy current
                vtk_fields["J_eddy"] = J_eddy_n
                
                pvd_file[tn] = createvtk(Ω, filename_base, cellfields=vtk_fields)
                
                if step_idx == 1
                    println("Saved initial condition (t=$(tn)) with enhanced fields to PVD.")
                elseif step_idx % 20 == 0
                    println("Saved enhanced step $(step_idx-1) (t=$(tn)) to PVD.")
                end
                
            catch e_pvd_step
                println("Error saving enhanced step t=$tn to PVD: $e_pvd_step")
            end
            
            # Extract probe signal for FFT analysis (skip initial condition for signal extraction)
            if step_idx > 1 && tn >= steps_for_fft_start_time
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
        end
    end
    println("Finished PVD saving to $(pvd_filename_base).pvd")
    
    return time_steps_for_fft, time_signal_data
end

"""
    save_transient_pvd(Ω_static::Triangulation, uh_solution_iterable, pvd_filename_base::String; uh0::Union{FEFunction, Nothing}=nothing)

Saves transient simulation results to a PVD file collection.
Iterates through the `uh_solution_iterable` (typically from `solve(odesolver, ...)`).
`uh0` is the optional initial condition FEFunction to save as time step 0.
"""
function save_transient_pvd(Ω_static::Triangulation, uh_solution_iterable, pvd_filename_base::String; uh0::Union{FEFunction, Nothing}=nothing)
    output_dir = dirname(pvd_filename_base)
    if output_dir != "" && !isdir(output_dir)
        mkpath(output_dir)
    end
    base_name_no_ext = first(splitext(pvd_filename_base))

    println("Saving transient results to PVD collection: $(base_name_no_ext).pvd")
    
    createpvd(base_name_no_ext) do pvd
        if uh0 !== nothing
            vtk_file_t0 = joinpath(output_dir, "$(first(splitext(basename(base_name_no_ext))))_t0")
            pvd[0.0] = createvtk(Ω_static, vtk_file_t0, cellfields=Dict("Az" => uh0))
            println("Saved initial condition to $(vtk_file_t0).vtu")
        end
        
        for (i, (uh_n, t_n)) in enumerate(uh_solution_iterable)
            vtk_file_tn = joinpath(output_dir, "$(first(splitext(basename(base_name_no_ext))))_t$(@sprintf("%.4f", t_n))")
            pvd[t_n] = createvtk(Ω_static, vtk_file_tn, cellfields=Dict("Az" => uh_n))
            println("Saved frame for t=$(@sprintf("%.4f", t_n)) to $(vtk_file_tn).vtu")
        end
    end
    println("PVD collection saved.")
end

end # module PostProcessing
