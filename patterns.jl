# A master written over the columns of pattern families, compiled to Vertigo's
# DWReformulationBuilder. This is the modelling layer proposed in
# nablarise/Vertigo.jl#124; it lives here until Vertigo has one.
#
#   F = Dict(p => patterns(master, MyPatterns(p)) for p in profiles)
#   @variable(master, λ[p in profiles, q in F[p]] >= 0, ColumnOf(q))
#   @constraint(master, [p in profiles], sum(λ[p, q] for q in F[p]) <= p.count)
#   @constraint(master, [i in items], sum(q.x[i] * λ[p, q] for p in profiles, q in F[p]) >= 1)
#   @objective(master, Min, sum(q.cost * λ[p, q] for p in profiles, q in F[p]))
#
# `q` is the one symbolic pattern of its family: its fields are copies, in the
# master, of the subproblem variables. `q.x[i] * λ[p, q]` is then an ordinary
# bilinear term of one JuMP model, and compilation reads the coefficient of
# each subproblem variable in each row from it.

using JuMP
using Vertigo
import MathOptInterface as MOI

# ── Pattern families ─────────────────────────────────────────────────────────

"""
    PatternFamily(name, data, build!)

The feasible solutions of the subproblem model that `build!(model, data)`
writes. Two families are the same when their name and data are the same.
"""
struct PatternFamily
    name::Symbol
    data::Any
    build!::Function
end
Base.:(==)(a::PatternFamily, b::PatternFamily) = a.name == b.name && a.data == b.data
Base.hash(f::PatternFamily, h::UInt) = hash((f.name, f.data), h)

"""
    @pattern Name(arg, model) begin ... end

Define `Name(arg) -> PatternFamily`, whose subproblem model is written by the
block, with `arg` and `model` in scope.
"""
macro pattern(sig, body)
    name, arg, model = sig.args[1], sig.args[2], sig.args[3]
    argname = arg isa Expr ? arg.args[1] : arg
    return esc(quote
        $name($arg) = $PatternFamily(
            $(QuoteNode(name)), $argname,
            ($model, $argname) -> begin
                $body
                nothing
            end,
        )
    end)
end

# ── The symbolic pattern of a family in a master ─────────────────────────────

mutable struct Pattern
    family::PatternFamily
    sp_model::Model
    copy_of::Dict{VariableRef,VariableRef}      # subproblem variable => copy in the master
    original_of::Dict{VariableRef,VariableRef}  # copy in the master => subproblem variable
end

# `q.name` is the container `name` of the subproblem, mapped to the copies;
# `q.cost` is the subproblem objective over the copies.
function Base.getproperty(q::Pattern, s::Symbol)
    if s in fieldnames(Pattern)
        return getfield(q, s)
    end
    if s === :cost
        return _substitute(objective_function(q.sp_model), q.copy_of)
    end
    return map(v -> q.copy_of[v], q.sp_model[s])
end

Base.show(io::IO, q::Pattern) = print(io, "q∈", q.family.name)

function _substitute(aff::AffExpr, var_map)
    out = AffExpr(aff.constant)
    for (v, c) in aff.terms
        add_to_expression!(out, c, var_map[v])
    end
    return out
end

"""
    PatternSet

The patterns of a family. Iterating it yields its one symbolic pattern: the
columns themselves are never enumerated, column generation produces them.
"""
struct PatternSet
    pattern::Pattern
end
Base.iterate(s::PatternSet) = (s.pattern, nothing)
Base.iterate(::PatternSet, ::Nothing) = nothing
Base.length(::PatternSet) = 1
Base.eltype(::Type{PatternSet}) = Pattern

"""
    patterns(master, family) -> PatternSet

The patterns of `family` in `master`, built once per master and family.
"""
function patterns(master::Model, family::PatternFamily)
    sets = get!(master.ext, :pattern_sets, Dict{PatternFamily,PatternSet}())
    return get!(sets, family) do
        sp_model = Model()
        family.build!(sp_model, family.data)
        prefix = "$(family.name)[$(length(sets) + 1)]"
        copy_of = Dict(
            v => @variable(master, base_name = "$prefix.$(name(v))")
            for v in all_variables(sp_model)
        )
        original_of = Dict(c => v for (v, c) in copy_of)
        PatternSet(Pattern(family, sp_model, copy_of, original_of))
    end
end

# ── λ tagged with its pattern ────────────────────────────────────────────────

"""
    ColumnOf(q)

Mark a master variable as the weight of the columns of `q`'s family:
`@variable(master, λ[p in P, q in F[p]] >= 0, ColumnOf(q))`.
"""
struct ColumnOf
    pattern::Pattern
end

struct ColumnVariable <: JuMP.AbstractVariable
    info::JuMP.VariableInfo
    pattern::Pattern
end

function JuMP.build_variable(::Function, info::JuMP.VariableInfo, c::ColumnOf)
    return ColumnVariable(info, c.pattern)
end

function JuMP.add_variable(model::Model, v::ColumnVariable, name::String="")
    λ = JuMP.add_variable(model, JuMP.ScalarVariable(v.info), name)
    get!(model.ext, :column_pattern, Dict{VariableRef,Pattern}())[λ] = v.pattern
    return λ
end

# ── Compilation ──────────────────────────────────────────────────────────────

# One master row: for each family, the coefficient of its columns as an affine
# expression of its subproblem variables.
struct CompiledRow
    name::String
    sense::Symbol
    rhs::Float64
    coeffs::Dict{PatternFamily,AffExpr}
end

function _column_coefficients(expr, columns, copies, families)
    coeffs = Dict(f => AffExpr(0.0) for f in families)
    quad = expr isa QuadExpr ? expr : QuadExpr(convert(AffExpr, expr))
    for (pair, c) in quad.terms
        λ, x = haskey(columns, pair.a) ? (pair.a, pair.b) : (pair.b, pair.a)
        if !haskey(columns, λ) || !haskey(copies, x)
            throw(ArgumentError(
                "$(pair.a) * $(pair.b): a product must be a column times a variable of its pattern"
            ))
        end
        q = columns[λ]
        if copies[x] !== q
            throw(ArgumentError("$x is a variable of another pattern than $λ"))
        end
        add_to_expression!(coeffs[q.family], c, q.original_of[x])
    end
    for (v, c) in quad.aff.terms
        if haskey(columns, v)
            coeffs[columns[v].family].constant += c
        elseif haskey(copies, v)
            throw(ArgumentError("$v, a variable of a pattern, appears without its column"))
        else
            throw(ArgumentError("$v: master variables other than columns are not supported"))
        end
    end
    return coeffs
end

_sense(::MOI.GreaterThan) = :ge
_sense(::MOI.LessThan) = :le
_sense(::MOI.EqualTo) = :eq

# The family whose convexity constraint the row is: its coefficient is the
# constant 1 for that family and the constant 0 for every other.
function _convexity_family(coeffs, families)
    if any(c -> !isempty(c.terms), values(coeffs))
        return nothing
    end
    ones = [f for f in families if coeffs[f].constant == 1.0]
    rest_zero = all(f -> f in ones || coeffs[f].constant == 0.0, families)
    return length(ones) == 1 && rest_zero ? only(ones) : nothing
end

function _read_rows(master, columns, copies, families)
    rows = CompiledRow[]
    for cref in all_constraints(master; include_variable_in_set_constraints=false)
        obj = constraint_object(cref)
        coeffs = _column_coefficients(obj.func, columns, copies, families)
        push!(rows, CompiledRow(name(cref), _sense(obj.set), MOI.constant(obj.set), coeffs))
    end
    return rows
end

function _read_costs(master, columns, copies, families)
    if objective_sense(master) != MIN_SENSE
        throw(ArgumentError("only a minimisation master is supported"))
    end
    objective = objective_function(master)
    constant = objective isa QuadExpr ? objective.aff.constant : convert(AffExpr, objective).constant
    if constant != 0.0
        throw(ArgumentError("a constant in the master objective is not supported"))
    end
    return _column_coefficients(objective, columns, copies, families)
end

"""
    compile_master(master, optimizer) -> DWReformulation

Turn a JuMP master written over pattern sets into a decomposition that
Vertigo's column generation solves. Convexity rows are recognised by their
coefficients; every other row becomes a coupling constraint, where the
coefficient of a subproblem variable is read from the bilinear terms. The
constant of a column's cost becomes the fixed cost of its subproblem.

Throws an `ArgumentError` for a family without an upper convexity row, a
family with two lower or two upper convexity rows, a maximisation master, a
constant in the master objective, a constant column coefficient outside a
convexity row, and master variables that are not columns.
"""
function compile_master(master::Model, optimizer)
    sets = master.ext[:pattern_sets]
    if !haskey(master.ext, :column_pattern)
        throw(ArgumentError("no column: declare λ with `ColumnOf(q)`"))
    end
    families = collect(keys(sets))
    columns = master.ext[:column_pattern]
    copies = Dict(c => s.pattern for s in values(sets) for c in keys(s.pattern.original_of))
    rows = _read_rows(master, columns, copies, families)
    costs = _read_costs(master, columns, copies, families)

    vertigo_master = Model(optimizer)
    set_silent(vertigo_master)
    @objective(vertigo_master, Min, 0)
    builder = DWReformulationBuilder{Tuple{Int,Int}}(minimize=true)

    conv_lb = Dict{PatternFamily,Tuple{Float64,ConstraintRef}}()
    conv_ub = Dict{PatternFamily,Tuple{Float64,ConstraintRef}}()
    coupling = Tuple{ConstraintRef,Dict{PatternFamily,AffExpr}}[]
    for row in rows
        cstr = _vertigo_row(vertigo_master, row.sense, row.rhs)
        family = _convexity_family(row.coeffs, families)
        if family !== nothing && row.sense in (:ge, :le)
            bounds = row.sense === :ge ? conv_lb : conv_ub
            if haskey(bounds, family)
                throw(ArgumentError("$(row.name): a second convexity bound on $(family.name)"))
            end
            bounds[family] = (row.rhs, cstr)
        elseif any(c -> c.constant != 0.0, values(row.coeffs))
            throw(ArgumentError(
                "$(row.name): a constant column coefficient is only supported in a convexity constraint"
            ))
        else
            add_coupling_constraint!(builder, index(cstr), row.rhs)
            push!(coupling, (cstr, row.coeffs))
        end
    end

    sp_ids = Dict(f => PricingSubproblemId(k) for (k, f) in enumerate(families))
    for f in families
        if !haskey(conv_ub, f)
            throw(ArgumentError("$(f.name): no upper convexity constraint"))
        end
        sp_model = sets[f].pattern.sp_model
        set_optimizer(sp_model, optimizer)
        set_silent(sp_model)
        cost = costs[f]
        lb = haskey(conv_lb, f) ? conv_lb[f][1] : 0.0
        add_subproblem!(builder, sp_ids[f], cost.constant, lb, conv_ub[f][1])
        # Pricing rewrites only the coefficients of the subproblem objective:
        # a constant left there would be counted on top of the fixed cost.
        set_objective_function(sp_model, cost - cost.constant)
        for v in all_variables(sp_model)
            add_sp_variable!(builder, sp_ids[f], index(v), coefficient(cost, v))
        end
    end
    for (cstr, coeffs) in coupling, f in families
        for (v, c) in coeffs[f].terms
            add_coupling_coefficient!(builder, sp_ids[f], index(v), index(cstr), c)
        end
    end

    decomp = build(builder)
    set_models!(
        decomp, backend(vertigo_master),
        Dict{PricingSubproblemId,Any}(sp_ids[f] => backend(sets[f].pattern.sp_model) for f in families),
        Dict{PricingSubproblemId,TaggedCI}(sp_ids[f] => TaggedCI(index(conv_ub[f][2])) for f in families),
        Dict{PricingSubproblemId,TaggedCI}(f_id => TaggedCI(index(conv_lb[f][2])) for (f, f_id) in sp_ids if haskey(conv_lb, f)),
    )
    return decomp
end

function _vertigo_row(model, sense, rhs)
    if sense === :ge
        return @constraint(model, 0 >= rhs)
    elseif sense === :le
        return @constraint(model, 0 <= rhs)
    end
    return @constraint(model, 0 == rhs)
end
