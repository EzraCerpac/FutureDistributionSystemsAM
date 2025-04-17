using Gridap
using Gridap.FESpaces
using Gridap.MultiField

function solve_1d_steady_heat(model, k, Q, T_dirichlet)
    order = 2
    reffe = ReferenceFE(lagrangian, Float64, order)
    V0 = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
    U = TrialFESpace(V0, T_dirichlet)

    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2 * order)

    # Convert scalar to constant CellField if needed
    k_field = isa(k, Number) ? CellField(x -> k, Ω) : k
    Q_field = isa(Q, Number) ? CellField(x -> Q, Ω) : Q

    a(u,v) = ∫( k_field * ∇(v) ⋅ ∇(u) ) * dΩ
    l(v)   = ∫( Q_field * v ) * dΩ

    op = AffineFEOperator(U, V0, a, l)
    T = solve(op)
    return T
end