using JuMP
using HiGHS
using Vertigo
import MathOptInterface as MOI

include("identical.jl")
include("patterns.jl")

# ── Instance ──────────────────────────────────────────────────────────────────

struct BinPackingInstance
    name::String
    capacity::Float64
    weights::Vector{Float64}
end

"""
    parse_bpp(path; items = nothing) -> BinPackingInstance

Read an instance in the BPP format of BPPLIB: the number of items, the bin
capacity, then one weight per line. With `items`, keeps that many items
spread evenly over the file: the weights are sorted there, so the first ones
would all be large.
"""
function parse_bpp(path::AbstractString; items::Union{Nothing,Int}=nothing)
    values = parse.(Float64, filter(!isempty, strip.(readlines(path))))
    n, capacity = Int(values[1]), values[2]
    weights = values[3:(2 + n)]
    keep = items === nothing ? (1:n) : round.(Int, range(1, n; length=items))
    return BinPackingInstance(first(splitext(basename(path))), capacity, weights[keep])
end

# ── Compact formulation with identical bins ───────────────────────────────────

"""
    compact_decomposition(inst, nb_bins) -> DWReformulation

The compact model of one representative bin, `K = identical(:bins, nb_bins)`,
decomposed by the DML: `x[k, i]` puts item `i` in the bin, `y[k]` opens it.
The one subproblem stands for up to `nb_bins` bins.
"""
function compact_decomposition(inst::BinPackingInstance, nb_bins::Int)
    items = eachindex(inst.weights)
    K = identical(:bins, nb_bins)

    model = Model(HiGHS.Optimizer)
    @variable(model, x[k in K, i in items], Bin)
    @variable(model, y[k in K], Bin)
    @constraint(model, knapsack[k in K], sum(inst.weights[i] * x[k, i] for i in items) <= inst.capacity * y[k])
    @constraint(model, cover[i in items], sum(x[k, i] for k in K) >= 1)
    # With one element in K, sum(y[k] for k in K) is the variable y[k] itself:
    # the DML would copy a single-variable objective, which column generation
    # cannot modify (Vertigo.jl#137). The 1.0 keeps an AffExpr.
    @objective(model, Min, sum(1.0 * y[k] for k in K))

    decomp, sp_map = @dantzig_wolfe model begin
        x[k, _] => subproblem(k)
        y[k] => subproblem(k)
        knapsack[k] => subproblem(k)
        cover[_] => master()
    end
    apply_multiplicities!(decomp, sp_map)
    return decomp
end

# ── Pattern formulation ───────────────────────────────────────────────────────

# One kind of bin: its capacity, how many are available, the item weights.
struct BinProfile
    capacity::Float64
    nb_bins::Int
    weights::Vector{Float64}
end

@pattern BinFeasiblePatterns(profile::BinProfile, subproblem_model) begin
    items = eachindex(profile.weights)
    @variable(subproblem_model, assign_item[i in items], Bin)
    @constraint(
        subproblem_model,
        sum(profile.weights[i] * assign_item[i] for i in items) <= profile.capacity
    )
    @objective(subproblem_model, Min, 1.0)
end

"""
    pattern_master(profiles, items) -> Model

The master over the patterns of each bin profile: every item is covered, a
profile is used at most `nb_bins` times, and each bin costs its pattern's
cost, 1.
"""
function pattern_master(profiles, items)
    master = Model()
    F = Dict(p => patterns(master, BinFeasiblePatterns(p)) for p in profiles)

    @variable(master, λ[p in profiles, q in F[p]] >= 0, Int, ColumnOf(q))
    @constraint(master, convexity_ub[p in profiles], sum(λ[p, q] for q in F[p]) <= p.nb_bins)
    @constraint(master, cover[i in items],
        sum(q.assign_item[i] * λ[p, q] for p in profiles, q in F[p]) >= 1)
    @objective(master, Min, sum(q.cost * λ[p, q] for p in profiles, q in F[p]))
    return master
end

# ── The same decomposition, written by hand ──────────────────────────────────

"""
    hand_built_decomposition(inst, nb_bins) -> DWReformulation

The decomposition `pattern_master` compiles to, written directly with
`DWReformulationBuilder`: one subproblem, a knapsack over the items; its
fixed cost 1 is the cost of a bin.
"""
function hand_built_decomposition(inst::BinPackingInstance, nb_bins::Int)
    items = eachindex(inst.weights)
    master = Model(HiGHS.Optimizer)
    set_silent(master)
    @constraint(master, cover[i in items], 0 >= 1)
    @constraint(master, convexity_ub, 0 <= nb_bins)
    @objective(master, Min, 0)

    sp = Model(HiGHS.Optimizer)
    set_silent(sp)
    @variable(sp, z[i in items], Bin)
    @constraint(sp, sum(inst.weights[i] * z[i] for i in items) <= inst.capacity)
    @objective(sp, Min, 0)

    sp_id = PricingSubproblemId(1)
    builder = DWReformulationBuilder{Tuple{Int,Int}}(minimize=true)
    add_subproblem!(builder, sp_id, 1.0, 0.0, Float64(nb_bins))
    for i in items
        add_sp_variable!(builder, sp_id, index(z[i]), 0.0)
        add_coupling_coefficient!(builder, sp_id, index(z[i]), index(cover[i]), 1.0)
        add_coupling_constraint!(builder, index(cover[i]), 1.0)
    end
    decomp = build(builder)
    set_models!(
        decomp, backend(master),
        Dict{PricingSubproblemId,Any}(sp_id => backend(sp)),
        Dict{PricingSubproblemId,TaggedCI}(sp_id => TaggedCI(index(convexity_ub))),
        Dict{PricingSubproblemId,TaggedCI}(),
    )
    return decomp
end

# ── Root column generation ────────────────────────────────────────────────────

function solve_root(decomp; log::Bool)
    ws = ColGenWorkspace(
        decomp, ColumnPool(),
        Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        ColGenConfig(silent=!log),
    )
    runner = log ? Vertigo.ColGen.ColGenLoggerWorkspace(ws) : ws
    elapsed = @elapsed out = run_column_generation(runner)
    return out, elapsed
end

function report(label, out, elapsed)
    bound = out.incumbent_dual_bound
    bins = bound === nothing ? "-" : string(ceil(Int, bound - 1e-6))
    println(
        rpad(label, 10), rpad(string(out.status), 19),
        "LP ", rpad(something(out.master_lp_obj, "-"), 20),
        "dual bound ", rpad(something(bound, "-"), 20),
        "bins ≥ ", rpad(bins, 6), round(elapsed; digits=1), " s"
    )
end

# ── Command line ──────────────────────────────────────────────────────────────

const USAGE = """
usage: julia --project=. binpacking.jl INSTANCE [--items N] [--bins K] [--log]

  INSTANCE    a file in the BPP format of BPPLIB
  --items N   keep N items spread over the instance
  --bins K    at most K bins (default: one per item)
  --log       print the column generation log
"""

function parse_args(args)
    opts = Dict{String,Any}("items" => nothing, "bins" => nothing, "log" => false)
    path = nothing
    k = 1
    while k <= length(args)
        arg = args[k]
        if arg in ("--items", "--bins")
            opts[arg[3:end]] = parse(Int, args[k+1])
            k += 2
        elseif arg == "--log"
            opts["log"] = true
            k += 1
        elseif arg in ("-h", "--help") || startswith(arg, "--") || path !== nothing
            print(USAGE)
            exit(arg in ("-h", "--help") ? 0 : 1)
        else
            path = arg
            k += 1
        end
    end
    if path === nothing
        print(USAGE)
        exit(1)
    end
    return path, opts
end

function (@main)(args)
    path, opts = parse_args(args)
    inst = parse_bpp(path; items=opts["items"])
    nb_bins = something(opts["bins"], length(inst.weights))
    println(inst.name, ": ", length(inst.weights), " items, capacity ", inst.capacity,
        ", at most ", nb_bins, " bins")

    profile = BinProfile(inst.capacity, nb_bins, inst.weights)
    master = pattern_master([profile], eachindex(inst.weights))
    results = [
        "compact" => solve_root(compact_decomposition(inst, nb_bins); log=opts["log"]),
        "patterns" => solve_root(compile_master(master, HiGHS.Optimizer); log=opts["log"]),
        "by hand" => solve_root(hand_built_decomposition(inst, nb_bins); log=opts["log"]),
    ]
    for (label, (out, elapsed)) in results
        report(label, out, elapsed)
    end
    lps = [out.master_lp_obj for (_, (out, _)) in results]
    if any(isnothing, lps) ? !all(isnothing, lps) : maximum(lps) - minimum(lps) > 1e-6
        println("the formulations disagree")
        return 1
    end
    return 0
end
