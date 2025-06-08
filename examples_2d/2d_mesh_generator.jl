using  Gmsh: gmsh

# Import project configuration
include(joinpath(dirname(dirname(@__FILE__)), "config.jl"))

# Get paths
paths = get_project_paths("ta_example_1d")

println("Base directory: ", paths["BASE_DIR"])
println("Geometry directory: ", paths["GEO_DIR"])
println("Output directory: ", paths["OUTPUT_DIR"])

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)    # Enable printing information to terminal

## Simulation domain
wdom = 0.2;
hdom = 0.2;

## Core
wcore = 100.3e-3;
hcore = 118.8e-3;

## Coil
wcoil = 22.825e-3;
hcoil = 93.7e-3;

# Mesh density
lc1 = 2e-2;    # At x0 and x1
lc2 = 5e-3;   # At the core

function gmsh_add_rectangle(mid, width, height, lc)
    geo = gmsh.model.geo;
    
    # Corner points
    p1 = geo.addPoint(mid[1] - width / 2, mid[2] - height / 2, 0, lc);
    p2 = geo.addPoint(mid[1] + width / 2, mid[2] - height / 2, 0, lc);
    p3 = geo.addPoint(mid[1] + width / 2, mid[2] + height / 2, 0, lc);
    p4 = geo.addPoint(mid[1] - width / 2, mid[2] + height / 2, 0, lc);
    points = [p1, p2, p3, p4];
    
    # Lines
    l1 = geo.addLine(p1, p2);
    l2 = geo.addLine(p2, p3);
    l3 = geo.addLine(p3, p4);
    l4 = geo.addLine(p4, p1);
    lines = [l1, l2, l3, l4];
    
    # Curve loop
    loop = geo.addCurveLoop(lines);
    
    return loop, lines, points;
end


gmsh.model.add("inductor")          # Create a new model
geo = gmsh.model.geo;

## Simulation domain
sim_domain, enclosure_lines, _ = gmsh_add_rectangle([0,0], wdom, hdom, lc1)

## Core 
core_lp, _, _  = gmsh_add_rectangle([0, 0], wcore, hcore, lc2)     

## Coil
coil_left_reg, _, _ = gmsh_add_rectangle([-24.9875e-3, 0], wcoil, hcoil, lc2) 
coil_right_reg, _, _ = gmsh_add_rectangle([24.9875e-3, 0], wcoil, hcoil, lc2) 

## Surfaces
geo.addPlaneSurface([sim_domain, core_lp], 1)
geo.addPlaneSurface([core_lp, coil_left_reg, coil_right_reg], 2)
geo.addPlaneSurface([coil_left_reg], 3)
geo.addPlaneSurface([coil_right_reg], 4)

geo.synchronize()

## Physical Groups
geo.addPhysicalGroup(2, [1], 1)  # Enclosure boundary
geo.addPhysicalGroup(2, [2], 2)         # Core
geo.addPhysicalGroup(2, [3], 3)         # Coil left
geo.addPhysicalGroup(2, [4], 4)         # Coil right

geo.addPhysicalGroup(1, enclosure_lines, 1)  # Enclosure boundary

gmsh.model.setPhysicalName(2, 1, "Air")
gmsh.model.setPhysicalName(2, 2, "Core")    
gmsh.model.setPhysicalName(2, 3, "Coil left")
gmsh.model.setPhysicalName(2, 4, "Coil right")

gmsh.model.setPhysicalName(1, 1, "Enclosure")

# Generate Mesh
geo.synchronize()
gmsh.model.mesh.generate(2)

gmsh.write(joinpath(paths["GEO_DIR"], "2D_simplified_transformer.msh"))

gmsh.fltk.run()