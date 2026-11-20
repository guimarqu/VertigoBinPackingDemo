# Identical blocks in a compact formulation: an index set with one element
# that stands for up to `ub` identical copies of a subproblem. This is the
# modelling proposed in nablarise/Vertigo.jl#137; it lives here until Vertigo
# has it.
#
#   K = identical(:bins, nb_items)
#   @variable(model, x[k in K, i in items], Bin)
#   ...
#   decomp, sp_map = @dantzig_wolfe model begin
#       x[k, _] => subproblem(k)
#       ...
#   end
#   apply_multiplicities!(decomp, sp_map)
#
# `subproblem(k)` receives the one element of `K`, which carries the
# multiplicity, so every annotation of the subproblem agrees on it.

using JuMP
using Vertigo
import MathOptInterface as MOI

"""
    IdenticalBlock

The one element of an `identical` index set: a name, and the bounds on the
number of copies of the block that a solution uses.
"""
struct IdenticalBlock
    name::Symbol
    lb::Int
    ub::Int
end

Base.show(io::IO, k::IdenticalBlock) = print(io, k.name, "{", k.lb, "..", k.ub, "}")

# The DML sorts subproblem ids and indices.
Base.isless(a::IdenticalBlock, b::IdenticalBlock) = isless((a.name, a.lb, a.ub), (b.name, b.lb, b.ub))

"""
    identical(name, ub; lb = 0) -> Vector{IdenticalBlock}

An index set with one element, standing for between `lb` and `ub` identical
copies of a block. Throws an `ArgumentError` unless `0 <= lb <= ub`.

The compact model then describes one representative copy, and JuMP sees one
copy only: solving `model` directly does not solve the problem.
"""
function identical(name::Symbol, ub::Int; lb::Int=0)
    if !(0 <= lb <= ub)
        throw(ArgumentError("identical($name): the bounds ($lb, $ub) must satisfy 0 <= lb <= ub"))
    end
    return [IdenticalBlock(name, lb, ub)]
end

"""
    apply_multiplicities!(decomp, sp_map)

Set the convexity bounds of every subproblem whose id is an `IdenticalBlock`
to its `(lb, ub)`, in the master constraints and in the decomposition alike.
The DML gives every subproblem `(1, 1)` (Vertigo.jl#84), so this goes through
Vertigo internals.
"""
function apply_multiplicities!(decomp, sp_map)
    master = master_model(decomp)
    for (id, sp_id) in sp_map
        if id isa IdenticalBlock
            lb, ub = Float64(id.lb), Float64(id.ub)
            with_typed_ci(ci -> MOI.set(master, MOI.ConstraintSet(), ci, MOI.GreaterThan(lb)),
                decomp.convexity_lb[sp_id])
            with_typed_ci(ci -> MOI.set(master, MOI.ConstraintSet(), ci, MOI.LessThan(ub)),
                decomp.convexity_ub[sp_id])
            old = decomp.subproblems[sp_id]
            decomp.subproblems[sp_id] = Vertigo.Reformulation.SubproblemData(
                old.variables, old.original_costs, old.coupling_coeffs, old.fixed_cost, lb, ub)
        end
    end
    return nothing
end
