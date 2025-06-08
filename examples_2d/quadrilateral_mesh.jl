using  Gmsh: gmsh

# Import project configuration
include(joinpath(dirname(dirname(@__FILE__)), "config.jl"))

# Get paths
paths = get_project_paths("examples_2d")

println("Base directory: ", paths["BASE_DIR"])
println("Geometry directory: ", paths["GEO_DIR"])
println("Output directory: ", paths["OUTPUT_DIR"])

gmsh.finalize()
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)    # Enable printing information to terminal

## Simulation domain
wdom = 0.2;
hdom = 0.2;

## Core
wcore = 100.3e-3;
hcore = 118.8e-3;

# Core gap dimensions (left and right are identical)
wgap = 22.825e-3;
hgap = 93.7e-3;
mgap = 25.1625e-3;

wlm = 2 *(mgap - wgap / 2);
wll = wcore / 2 - wgap - wlm / 2;
hlm = hgap;
hll = (hcore - hgap) / 2;

## Coil
wcoil = wgap;
hcoil = hgap;

# Mesh densities
lc1 = 5e-3;      # Enclosure & core outer
lc2 = 2e-3;      # Core inner

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

gmsh.model.add("2d_power_transformer")
geo  = gmsh.model.geo;
mesh = gmsh.model.mesh;

## Simulation domain
sim_domain, enclosure_lines, _ = gmsh_add_rectangle([0,0], wdom, hdom, lc1)


## Core 
p1  = geo.addPoint(-wcore / 2, -hcore / 2, 0, lc1)
p2  = geo.addPoint(-wcore / 2 + wll, -hcore / 2, 0, lc1)
p3  = geo.addPoint(-wlm / 2, -hcore / 2, 0, lc1)
p4  = geo.addPoint(+wlm / 2, -hcore / 2, 0, lc1)
p5  = geo.addPoint(+wcore / 2 - wll, -hcore / 2, 0, lc1)
p6  = geo.addPoint(+wcore / 2, -hcore / 2, 0, lc1)
p7  = geo.addPoint(+wcore / 2, -hcore / 2 + hll, 0, lc1)
p8  = geo.addPoint(+wcore / 2, +hcore / 2 - hll, 0, lc1)
p9  = geo.addPoint(+wcore / 2, +hcore / 2, 0, lc1)
p10 = geo.addPoint(+wcore / 2 - wll, +hcore / 2, 0, lc1)
p11 = geo.addPoint(+wlm / 2, +hcore / 2, 0, lc1)
p12 = geo.addPoint(-wlm / 2, +hcore / 2, 0, lc1)
p13 = geo.addPoint(-wcore / 2 + wll, +hcore / 2, 0, lc1)
p14 = geo.addPoint(-wcore / 2, +hcore / 2, 0, lc1)
p15 = geo.addPoint(-wcore / 2, +hcore / 2 - hll, 0, lc1)
p16 = geo.addPoint(-wcore / 2, -hcore / 2 + hll, 0, lc1)

p17 = geo.addPoint(-wcore / 2 + wll, -hcore / 2 + hll, 0, lc2)
p18 = geo.addPoint(-wlm / 2, -hcore / 2 + hll, 0, lc2)
p19 = geo.addPoint(+wlm / 2, -hcore / 2 + hll, 0, lc2)
p20 = geo.addPoint(+wcore / 2 - wll, -hcore / 2 + hll, 0, lc2)
p21 = geo.addPoint(+wcore / 2 - wll, hcore / 2 - hll, 0, lc2)
p22 = geo.addPoint(+wlm / 2, hcore / 2 - hll, 0, lc2)
p23 = geo.addPoint(-wlm / 2, hcore / 2 - hll, 0, lc2)
p24 = geo.addPoint(-wcore / 2 + wll, hcore / 2 - hll, 0, lc2)

l1 = geo.addLine(p1, p2)
l2 = geo.addLine(p2, p3)
l3 = geo.addLine(p3, p4)
l4 = geo.addLine(p4, p5)
l5 = geo.addLine(p5, p6)

l6  = geo.addLine(p16, p17)
l7  = geo.addLine(p17, p18)
l8  = geo.addLine(p18, p19)
l9  = geo.addLine(p19, p20)
l10 = geo.addLine(p20, p7)

l11 = geo.addLine(p15, p24)
l12 = geo.addLine(p24, p23)
l13 = geo.addLine(p23, p22)
l14 = geo.addLine(p22, p21)
l15 = geo.addLine(p21, p8)

l16 = geo.addLine(p14, p13)
l17 = geo.addLine(p13, p12)
l18 = geo.addLine(p12, p11)
l19 = geo.addLine(p11, p10)
l20 = geo.addLine(p10, p9)

l21 = geo.addLine(p14, p15)
l22 = geo.addLine(p15, p16)
l23 = geo.addLine(p16, p1)

l24 = geo.addLine(p13, p24)
l25 = geo.addLine(p24, p17)
l26 = geo.addLine(p17, p2)

l27 = geo.addLine(p12, p23)
l28 = geo.addLine(p23, p18)
l29 = geo.addLine(p18, p3)

l30 = geo.addLine(p11, p22)
l31 = geo.addLine(p22, p19)
l32 = geo.addLine(p19, p4)

l33 = geo.addLine(p10, p21)
l34 = geo.addLine(p21, p20)
l35 = geo.addLine(p20, p5)

l36 = geo.addLine(p9, p8)
l37 = geo.addLine(p8, p7)
l38 = geo.addLine(p7, p6)

core_lp  = geo.addCurveLoop([l1, l2, l3, l4, l5, l38, l37, l36, l20, l19, l18, l17, l16, l21, l22, l23], 2, true)
cgap1_lp = geo.addCurveLoop([l7, l28, l12, l25], 3, true)
cgap2_lp = geo.addCurveLoop([l9, l34, l14, l31], 4, true)

core_lines1 = [l1, l26, l6, l23]; core1_lp = geo.addCurveLoop(core_lines1, 5, true)
core_lines2 = [l2, l29, l7, l26]; core2_lp = geo.addCurveLoop(core_lines2, 6, true)
core_lines3 = [l3, l32, l8, l29]; core3_lp = geo.addCurveLoop(core_lines3, 7, true)
core_lines4 = [l4, l35, l9, l32]; core4_lp = geo.addCurveLoop(core_lines4, 8, true)
core_lines5 = [l5, l38, l10, l35]; core5_lp = geo.addCurveLoop(core_lines5, 9, true)

core_lines6 = [l6, l25, l11, l22]; core6_lp = geo.addCurveLoop(core_lines6, 10, true)
core_lines7 = [l8, l31, l13, l28]; core7_lp = geo.addCurveLoop(core_lines7, 11, true)
core_lines8 = [l10, l37, l15, l34]; core8_lp = geo.addCurveLoop(core_lines8, 12, true)

core_lines9 = [l11, l24, l16, l21]; core9_lp = geo.addCurveLoop(core_lines9, 13, true)
core_lines10 = [l12, l27, l17, l24]; core10_lp = geo.addCurveLoop(core_lines10, 14, true)
core_lines11 = [l13, l30, l18, l27]; core11_lp = geo.addCurveLoop(core_lines11, 15, true)
core_lines12 = [l14, l33, l19, l30]; core12_lp = geo.addCurveLoop(core_lines12, 16, true)
core_lines13 = [l15, l36, l20, l33]; core13_lp = geo.addCurveLoop(core_lines13, 17, true)


## Coil
coil_left_lines = [l25, l7, l28, l12]; coil_left_lp = geo.addCurveLoop(coil_left_lines, 18, true)
coil_right_lines = [l31, l9, l34, l14]; coil_right_lp = geo.addCurveLoop(coil_right_lines, 19, true)

## Surfaces
air_surf = geo.addPlaneSurface([sim_domain, core_lp])
# geo.addPlaneSurface([core_lp, coil_left_reg, coil_right_reg], 2)
# geo.addPlaneSurface([coil_left_lp], 2)
# geo.addPlaneSurface([coil_right_lp], 3)

core_surf1 = geo.addPlaneSurface([core1_lp])
core_surf2 = geo.addPlaneSurface([core2_lp])
core_surf3 = geo.addPlaneSurface([core3_lp])
core_surf4 = geo.addPlaneSurface([core4_lp])
core_surf5 = geo.addPlaneSurface([core5_lp])
core_surf6 = geo.addPlaneSurface([core6_lp])
core_surf7 = geo.addPlaneSurface([core7_lp])
core_surf8 = geo.addPlaneSurface([core8_lp])
core_surf9 = geo.addPlaneSurface([core9_lp])
core_surf10 = geo.addPlaneSurface([core10_lp])
core_surf11 = geo.addPlaneSurface([core11_lp])
core_surf12 = geo.addPlaneSurface([core12_lp])
core_surf13 = geo.addPlaneSurface([core13_lp])

coil_left_surf = geo.addPlaneSurface([coil_left_lp])
coil_right_surf = geo.addPlaneSurface([coil_right_lp])

geo.synchronize()

## Physical Groups
geo.addPhysicalGroup(2, [air_surf], 1)  # air
geo.addPhysicalGroup(2, [core_surf1, core_surf2, core_surf3, core_surf4, core_surf5, core_surf6, core_surf7, core_surf8, core_surf9, core_surf10, core_surf11, core_surf12, core_surf13], 2) # Core   
geo.addPhysicalGroup(2, [coil_left_surf], 3)         # Coil left
geo.addPhysicalGroup(2, [coil_right_surf], 4)         # Coil right

geo.addPhysicalGroup(1, enclosure_lines, 1)  # Enclosure boundary

gmsh.model.setPhysicalName(2, 1, "Air")
gmsh.model.setPhysicalName(2, 2, "Core")    
gmsh.model.setPhysicalName(2, 3, "Coil left")
gmsh.model.setPhysicalName(2, 4, "Coil right")

gmsh.model.setPhysicalName(1, 1, "Enclosure")

geo.synchronize()

## Define structured meshes
N1 = 10;
N2 = 10;
N3 = 10;
alpha = 0.1;
alpha2 = 0.1;

mesh.setTransfiniteCurve(core_lines1[2], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines1[3], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines1[1], N1)
mesh.setTransfiniteCurve(core_lines1[4], N1)
mesh.setTransfiniteSurface(core_surf1, "Left")

mesh.setTransfiniteCurve(core_lines2[2], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines2[4], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines2[1], N2)
mesh.setTransfiniteCurve(core_lines2[3], N2)
mesh.setTransfiniteSurface(core_surf2, "Left")

mesh.setTransfiniteCurve(core_lines3[2], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines3[3], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines3[4], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines3[1], N1)
mesh.setTransfiniteSurface(core_surf3, "Left")

mesh.setTransfiniteCurve(core_lines4[2], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines4[4], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines4[1], N2)
mesh.setTransfiniteCurve(core_lines4[3], N2)
mesh.setTransfiniteSurface(core_surf4, "Left")

mesh.setTransfiniteCurve(core_lines5[3], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines5[4], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines5[1], N1)
mesh.setTransfiniteCurve(core_lines5[2], N1)
mesh.setTransfiniteSurface(core_surf5, "Left")

mesh.setTransfiniteCurve(core_lines6[1], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines6[3], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines6[2], N3, "Bump", alpha2)
mesh.setTransfiniteCurve(core_lines6[4], N3, "Bump", alpha2)
mesh.setTransfiniteSurface(core_surf6, "Left")

mesh.setTransfiniteCurve(core_lines7[1], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines7[3], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines7[2], N3, "Bump", alpha2)
mesh.setTransfiniteCurve(core_lines7[4], N3, "Bump", alpha2)
mesh.setTransfiniteSurface(core_surf7, "Left")

mesh.setTransfiniteCurve(core_lines8[1], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines8[3], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines8[2], N3, "Bump", alpha2)
mesh.setTransfiniteCurve(core_lines8[4], N3, "Bump", alpha2)
mesh.setTransfiniteSurface(core_surf8, "Left")

mesh.setTransfiniteCurve(core_lines9[1], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines9[2], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines9[3], N1)
mesh.setTransfiniteCurve(core_lines9[4], N1)
mesh.setTransfiniteSurface(core_surf9, "Left")

mesh.setTransfiniteCurve(core_lines10[2], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines10[4], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines10[1], N2)
mesh.setTransfiniteCurve(core_lines10[3], N2)
mesh.setTransfiniteSurface(core_surf10, "Left")

mesh.setTransfiniteCurve(core_lines11[1], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines11[2], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines11[4], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines11[3], N1)
mesh.setTransfiniteSurface(core_surf11, "Left")

mesh.setTransfiniteCurve(core_lines12[2], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines12[4], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines12[1], N2)
mesh.setTransfiniteCurve(core_lines12[3], N2)
mesh.setTransfiniteSurface(core_surf12, "Left")

mesh.setTransfiniteCurve(core_lines13[1], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines13[4], N1, "Bump", alpha)
mesh.setTransfiniteCurve(core_lines13[2], N1)
mesh.setTransfiniteCurve(core_lines13[3], N1)
mesh.setTransfiniteSurface(core_surf13, "Left")

mesh.setRecombine(2, core_surf1)
mesh.setRecombine(2, core_surf2)
mesh.setRecombine(2, core_surf3)
mesh.setRecombine(2, core_surf4)
mesh.setRecombine(2, core_surf5)
mesh.setRecombine(2, core_surf6)
mesh.setRecombine(2, core_surf7)
mesh.setRecombine(2, core_surf8)
mesh.setRecombine(2, core_surf9)
mesh.setRecombine(2, core_surf10)
mesh.setRecombine(2, core_surf11)
mesh.setRecombine(2, core_surf12)
mesh.setRecombine(2, core_surf13)

mesh.setRecombine(2, coil_left_surf)
mesh.setRecombine(2, coil_right_surf)
mesh.setRecombine(2, air_surf)

# ## Generate Mesh
gmsh.model.mesh.generate(2)

gmsh.write(joinpath(paths["GEO_DIR"], "2D_quad_transformer.msh"))

gmsh.fltk.run()