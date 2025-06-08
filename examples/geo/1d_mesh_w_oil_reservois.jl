using  Gmsh: gmsh

# Import project configuration
include(joinpath(dirname(dirname(@__FILE__)), "../config.jl"))

# Get paths
paths = get_project_paths("ta_example_1d")

println("Base directory: ", paths["BASE_DIR"])
println("Geometry directory: ", paths["GEO_DIR"])
println("Output directory: ", paths["OUTPUT_DIR"])

gmsh.finalize()
gmsh.initialize()


# Simulation domain
x0 = -0.5;
x1 = 0.5;

# Core dimensions
a = 100.3e-3;
b = 73.15e-3;
c = 27.5e-3;

# Oil reservoirs
reservoir_width = 2*a-b ;

# Mesh density
lc1 = 0.1;    # At x0 and x1
lc2 = 1e-3;   # At the core


gmsh.option.setNumber("General.Terminal", 1)     # Enable printing information to terminal
gmsh.model.add("inductor")                       # Create a new model
geo = gmsh.model.geo;

## Points

# domain
geo.addPoint(x0, 0, 0, lc1, 1)
geo.addPoint(x1, 0, 0, lc1, 2)



# oil - core - coil - core - oil
geo.addPoint(-reservoir_width/2, 0, 0, lc2, 3)
geo.addPoint(-a/2, 0, 0, lc2, 4)
geo.addPoint(-b/2, 0, 0, lc2, 5)
geo.addPoint(-c/2, 0, 0, lc2, 6)
geo.addPoint(c/2, 0, 0, lc2, 7)
geo.addPoint(b/2, 0, 0, lc2, 8)
geo.addPoint(a/2, 0, 0, lc2, 9)
geo.addPoint(reservoir_width/2, 0, 0, lc2, 10)

# Lines
geo.addLine(1, 3, 1)    # air
geo.addLine(3, 4, 2)    # oil
geo.addLine(4, 5, 3)    # core
geo.addLine(5, 6, 4)    # coil
geo.addLine(6, 7, 5)    # core
geo.addLine(7, 8, 6)    # coil
geo.addLine(8, 9, 7)    # core
geo.addLine(9, 10, 8)   # oil
geo.addLine(10, 2, 9)   # air

# If the geometry is not synchronized before creating physical groups, a warning will be displayed:
#    "Unknown entity of dimension %d and tag %d in physical group %d"
# Synchronizing at this point ensures that the geometry data and model agree on the set of entities.
geo.synchronize() 

# Physical properties
gmsh.model.addPhysicalGroup(1, [1, 9], 1)       # Air
gmsh.model.addPhysicalGroup(1, [3, 5, 7], 2)    # Core
gmsh.model.addPhysicalGroup(1, [4], 3)          # Coil left
gmsh.model.addPhysicalGroup(1, [6], 4)          # Coil right
gmsh.model.addPhysicalGroup(1, [2, 8], 5)       # Oil

gmsh.model.addPhysicalGroup(0, [1, 2], 6)       # Dirichlet boundary condition at x0 and x1

gmsh.model.setPhysicalName(1, 1, "Air")
gmsh.model.setPhysicalName(1, 2, "Core")
gmsh.model.setPhysicalName(1, 3, "Coil left")
gmsh.model.setPhysicalName(1, 4, "Coil right")
gmsh.model.setPhysicalName(1, 5, "Oil")
gmsh.model.setPhysicalName(0, 6, "D") 

# Generate mesh
gmsh.model.mesh.generate(1)

gmsh.write(joinpath(paths["GEO_DIR"], "coil_geo_new.msh"))

gmsh.fltk.run()

# Current density
J = 2.2e4;
fsource(group_id) = J * (group_id == 3) - J * (group_id == 4);

# Permeability
mu0 = 4e-7 * pi; # Permeability of vacuum
mu_r = 1500;
fmu(group_id) = mu0 + (mu_r - 1) * mu0 * (group_id == 2);
fnu(group_id) = 1 / fmu(group_id);

# Conductivity
sigma_core = 0.2;
fsigma(group_id) = sigma_core * (group_id == 2);