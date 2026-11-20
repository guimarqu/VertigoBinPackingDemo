# Vertigo bin packing demo

Root column generation for bin packing with
[Vertigo.jl](https://github.com/nablarise/Vertigo.jl), on the Falkenauer
instances of [BPPLIB](https://github.com/mdelorme2/BPPLIB). The same
decomposition is built three ways, and the script fails if they disagree:

- **Compact** (`identical.jl`): the compact model of one bin, indexed by a set
  of identical blocks, decomposed by `@dantzig_wolfe`. This is the modelling
  proposed in Vertigo.jl#137; it lives here until Vertigo has it.
- **Patterns** (`patterns.jl`): the master is written in plain JuMP over
  families of patterns, as proposed in Vertigo.jl#124, now on hold in favour of
  the compact one.
- **By hand** (`hand_built_decomposition` in `binpacking.jl`): the master and
  the pricing subproblem are fed to `DWReformulationBuilder` directly. This is
  the reference for the other two.

Only the root is solved. See the known limits below for why.

## The compact formulation

```julia
K = identical(:bins, nb_bins)          # one element k̂, between 0 and nb_bins copies

@variable(model, x[k in K, i in items], Bin)
@variable(model, y[k in K], Bin)
@constraint(model, knapsack[k in K], sum(w[i] * x[k, i] for i in items) <= C * y[k])
@constraint(model, cover[i in items], sum(x[k, i] for k in K) >= 1)
@objective(model, Min, sum(1.0 * y[k] for k in K))

decomp, sp_map = @dantzig_wolfe model begin
    x[k, _] => subproblem(k)
    y[k] => subproblem(k)
    knapsack[k] => subproblem(k)
    cover[_] => master()
end
apply_multiplicities!(decomp, sp_map)
```

- `K` has one element, which stands for every bin. The model describes one
  representative bin, and `x[k̂, i]` reads as the number of bins holding item
  `i`. JuMP itself sees one bin only: `optimize!(model)` does not solve the
  problem.
- `subproblem(k)` receives that element, which carries the bounds on the number
  of bins. Every annotation of the subproblem gets the same element, so they
  agree on it.
- `apply_multiplicities!` writes those bounds into the decomposition and the
  master, because the DML gives every subproblem `(1, 1)` for now.
- A bin costs its variable `y[k̂]`, so the subproblem has no fixed cost.

## The pattern formulation

```julia
@pattern BinFeasiblePatterns(profile::BinProfile, subproblem_model) begin
    items = eachindex(profile.weights)
    @variable(subproblem_model, assign_item[i in items], Bin)
    @constraint(subproblem_model, sum(profile.weights[i] * assign_item[i] for i in items) <= profile.capacity)
    @objective(subproblem_model, Min, 1.0)
end

master = Model()
F = Dict(p => patterns(master, BinFeasiblePatterns(p)) for p in profiles)
@variable(master, λ[p in profiles, q in F[p]] >= 0, Int, ColumnOf(q))
@constraint(master, convexity_ub[p in profiles], sum(λ[p, q] for q in F[p]) <= p.nb_bins)
@constraint(master, cover[i in items], sum(q.assign_item[i] * λ[p, q] for p in profiles, q in F[p]) >= 1)
@objective(master, Min, sum(q.cost * λ[p, q] for p in profiles, q in F[p]))
```

- `q in F[p]` is never enumerated. Iterating a pattern set yields one symbolic
  pattern, so `λ` has one entry per family. Column generation produces the
  actual columns.
- The fields of `q` are copies, in the master, of the subproblem variables.
  `q.assign_item[i] * λ[p, q]` is then a bilinear term of one JuMP model, and
  `compile_master` reads from it the coefficient of `assign_item[i]` in
  `cover[i]`.
- `ColumnOf(q)` tells which family each `λ` belongs to.
- A row whose coefficient is the constant 1 for one family is that family's
  convexity constraint. Its bound is the number of bins of that profile.
- The constant of `q.cost` (1 per bin) becomes the subproblem's fixed cost,
  and is removed from the pricing model.

## Requirements

- Julia 1.11 or later.
- A checkout of Vertigo.jl next to this one, in `../Vertigo`, on `main`.
  `Project.toml` points at it by path.
- `curl` and `bsdtar`, to fetch the instances. `bsdtar` comes with macOS. On
  Debian or Ubuntu, it is in `libarchive-tools`.

## Setup

```sh
./tools/fetch_instances.sh                     # 160 Falkenauer instances into instances/
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

BPPLIB is published under CC BY-NC-ND 4.0, so the instances are downloaded
from BPPLIB at a pinned commit rather than copied into this repository.

## Run

```sh
julia --project=. binpacking.jl instances/Falkenauer_u120_00.txt
julia --project=. binpacking.jl instances/Falkenauer_u120_00.txt --items 18 --log
```

| Option | What it does |
|---|---|
| `--items N` | Keep N items spread over the instance. The weights are sorted in the files, so the first ones would all be large. |
| `--bins K` | Allow at most K bins (default: one per item) |
| `--log` | Print Vertigo's column generation log |

The output gives, for each formulation, the status, the master LP value, the
dual bound, the number of bins it implies (the dual bound rounded up), and the
time.

## Results

Root column generation on an Apple M3 Pro, Julia 1.12, Vertigo.jl 8c1058d.
The three formulations give the same root LP value, to within 1e-11.

| Instance | Items | Capacity | Root LP | Bins ≥ | BPPLIB optimum | Compact | Patterns | By hand |
|---|---|---|---|---|---|---|---|---|
| Falkenauer_t60_00 | 60 | 1000 | 20.0 | 20 | 20 | 57 s | 45 s | 44 s |
| Falkenauer_u120_00 | 120 | 150 | 47.266 | 48 | 48 | 86 s | 87 s | 86 s |

Two larger instances were run with the pattern and hand-built formulations
only, before the compact one existed: Falkenauer_t120_00 (root LP 40.0,
optimum 40) and Falkenauer_u250_00 (root LP 98.553, optimum 99), in about
125 s each.

On these instances, the root bound rounded up already reaches the optimum.

## Checking a result

`tools/reference.jl` enumerates every subset of items that fits in a bin, then
solves the master LP and IP with HiGHS. This gives a reference that does not
depend on Vertigo, for small instances (2^items subsets):

```sh
julia --project=. tools/reference.jl instances/Falkenauer_u120_00.txt 14      # LP 5.6, IP 6
julia --project=. tools/reference.jl instances/Falkenauer_u120_00.txt 18      # LP 7.1515, IP 8
julia --project=. tools/reference.jl instances/Falkenauer_u120_00.txt 14 5    # infeasible
```

`binpacking.jl` with the same `--items` and `--bins` gives the same LP values
with the three formulations, and `master_infeasible` for the last one.

## Known limits

| What | Why | Tracked in |
|---|---|---|
| Only the root is solved | Branch-and-price would close nodes as integral without branching. In the compact formulation, `x[k̂, i]` is always 1, so the projection is integral while the columns are fractional; a pattern master has no original variables at all. | Vertigo.jl#128 |
| Patterns and by hand: a bin must cost less than an artificial variable (10000) | There the cost of a bin is a fixed cost, which Phase 1 adds to reduced costs; with a larger cost, column generation ends `master_infeasible`. The compact formulation pays it through `y[k̂]` instead. | Vertigo.jl#126 |
| Several profiles: one without a feasible pattern stops column generation | Even with a lower bound of 0 on its number of bins | Vertigo.jl#127 |
| `apply_multiplicities!` writes into Vertigo's decomposition | The DML gives every subproblem the convexity bounds `(1, 1)` | Vertigo.jl#137, #84 |
| The compact objective is written `sum(1.0 * y[k] for k in K)` | With one element in `K`, `sum(y[k] for k in K)` is a single variable; the DML keeps it as is, and column generation cannot modify such an objective | Vertigo.jl#137 |
| The pattern layer accepts only some masters | Minimisation only, no constant in the objective, no master variable other than the columns, at most one lower and one upper convexity row per family. `Int` on `λ` is not used at the root. | Vertigo.jl#124 |
| HiGHS is pinned to 1.24 | HiGHS.jl 1.27.0 reports wrong MIP objective values, and pricing reads them as reduced costs | guimarqu/VertigoCVRPDemo#1 |
| Pricing is a HiGHS MIP solved at every iteration | No dedicated knapsack algorithm, so the larger instances take minutes | — |

## Credits

Instances from BPPLIB: M. Delorme, M. Iori, and S. Martello. BPPLIB: A library
for bin packing and cutting stock problems. *Optimization Letters*,
12(2):235–250, 2018. The Falkenauer instances come from E. Falkenauer. A
hybrid grouping genetic algorithm for bin packing. *Journal of Heuristics*,
2(1):5–30, 1996.
