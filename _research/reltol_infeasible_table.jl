#========================================================
INFEASIBLE SEEDS (max forecast > population) from feasibility_summary.csv:

 1. 365-day totals per arm (51 locations x 3 beta forms) for the original
    Tsit5, the delta/pop-fixed Tsit5 (delta_pop_), the solver arms and the
    six reltol-sweep arms. Note: the delta_pop_ arm only has data for the
    configurations that were run.
 2. The 29 "unstable" (beta form, train length, location) combos from the
    original Tsit5 runs, with the plain-Tsit5-with-delta/pop-fix control
    (tsit5_ prefix) added next to the existing solver columns. Writes
    instability_feasibility_combined_v2.csv.
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using CSV
using DataFrames
using Printf

feas = CSV.read(projectdir("data", "exp_pro", "feasibility_summary.csv"), DataFrame)
feas.prefix = [replace(first(split(s, "UDE_single")), r"_$" => "") for s in feas.sim_name]
println("prefixes found: ", sort(unique(feas.prefix)))

const FORMS = ["beta_exp", "beta_rational", "beta_mixed"]
const ARMS = [("", "Tsit5, original (delta/pop optimised)"),
              ("delta_pop", "Tsit5 (1e-3), delta/pop fix, original data"),
              ("tsit5", "Tsit5 + delta/pop fix (only the 29 unstable combos rerun)"),
              ("rosenbrock23", "Rosenbrock23"), ("vern7", "Vern7"), ("autotsit5", "AutoTsit5 (1e-3)"),
              ("tsit5_rtol1e-5", "Tsit5 (1e-5)"), ("autotsit5_rtol1e-5", "AutoTsit5 (1e-5)"),
              ("tsit5_rtol1e-6", "Tsit5 (1e-6)"), ("autotsit5_rtol1e-6", "AutoTsit5 (1e-6)"),
              ("tsit5_rtol1e-10", "Tsit5 (1e-10)"), ("autotsit5_rtol1e-10", "AutoTsit5 (1e-10)")]

println("\n365-day infeasible seeds (all 51 locations x 100 seeds, where run)\n")
println("| Arm | Beta form | Locations | Seeds | Infeasible | % |\n| --- | --- | --- | --- | --- | --- |")
for (prefix, lab) in ARMS
    sub = filter(r -> r.prefix == prefix && r.train_length == 365 && r.noise == 0.0, feas)
    isempty(sub) && (println("| $(lab) | — | no data | | | |"); continue)
    for bf in FORMS
        s = filter(:beta => ==(bf), sub)
        isempty(s) && continue
        println("| $(lab) | $(bf) | $(nrow(s)) | $(sum(s.n_seeds)) | $(sum(s.n_infeasible)) | $(@sprintf("%.2f", 100 * sum(s.n_infeasible) / sum(s.n_seeds))) |")
    end
end

# 29-combo table
combo = CSV.read(projectdir("_research", "crps_tables", "instability_feasibility_combined.csv"), DataFrame)
ctrl = filter(r -> r.prefix == "tsit5" && r.noise == 0.0, feas)
lookup = Dict((r.beta, r.train_length, r.location) => r.n_infeasible for r in eachrow(ctrl))
combo.tsit5_deltapop = [get(lookup, (r.beta_form, r.train_length, r.location), missing) for r in eachrow(combo)]
select!(combo, :beta_form, :train_length, :location, :n_seeds, :tsit5, :tsit5_deltapop, :rosenbrock23, :vern7, :autotsit5)
CSV.write(projectdir("_research", "crps_tables", "instability_feasibility_combined_v2.csv"), combo)

println("\nInfeasible seeds at the 29 originally-unstable combos\n")
println("| Beta form | Train length | State | Seeds | Tsit5 (original) | Tsit5 + delta/pop fix | Rosenbrock23 | Vern7 | AutoTsit5 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for r in eachrow(combo)
    println("| $(r.beta_form) | $(r.train_length) | $(r.location) | $(r.n_seeds) | $(r.tsit5) | $(r.tsit5_deltapop) | $(r.rosenbrock23) | $(r.vern7) | $(r.autotsit5) |")
end
println("| **Total** | | | $(sum(combo.n_seeds)) | $(sum(combo.tsit5)) | $(sum(skipmissing(combo.tsit5_deltapop))) | $(sum(combo.rosenbrock23)) | $(sum(combo.vern7)) | $(sum(combo.autotsit5)) |")
