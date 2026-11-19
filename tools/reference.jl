# Independent reference: enumerate every subset of items that fits in a bin,
# then solve the master LP and IP with HiGHS. Only for small instances
# (2^items subsets).
#
#   julia --project=. tools/reference.jl instances/Falkenauer_u120_00.txt <items> [<bins>]
using JuMP, HiGHS
module Demo   # in a module, so that the @main of binpacking.jl is not the entry point
include("../binpacking.jl")
end
using .Demo: parse_bpp

function reference(path, items, bins)
    inst = parse_bpp(path; items)
    w, n = inst.weights, length(inst.weights)
    bins = something(bins, n)
    patterns = Vector{Int}[]
    for mask in 1:(2^n - 1)
        chosen = [i for i in 1:n if (mask >> (i - 1)) & 1 == 1]
        if sum(w[chosen]) <= inst.capacity
            push!(patterns, chosen)
        end
    end
    res = map((false, true)) do integer
        m = Model(HiGHS.Optimizer)
        set_silent(m)
        @variable(m, λ[eachindex(patterns)] >= 0, integer = integer)
        @constraint(m, [i in 1:n], sum(λ[p] for p in eachindex(patterns) if i in patterns[p]) >= 1)
        @constraint(m, sum(λ) <= bins)
        @objective(m, Min, sum(λ))
        optimize!(m)
        termination_status(m) == OPTIMAL ? objective_value(m) : termination_status(m)
    end
    println("items=$n bins=$bins capacity=$(inst.capacity) patterns=$(length(patterns)) LP=$(res[1]) IP=$(res[2])")
end

reference(ARGS[1], parse(Int, ARGS[2]), length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing)
