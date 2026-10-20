#========================================================
COMPUTATION TIME for the reltol-sweep arms — same measure as
solver_timing_summary.jl (per-seed elapsed_seconds stored in every
results.jld2, n up to 3 beta forms * 51 locations * 100 seeds = 15300).

Only the six NEW arms (Tsit5 / AutoTsit5 at reltol 1e-5, 1e-6, 1e-10) are
scanned here; the default-tolerance (1e-3) rows are taken from the
existing solver_timing_*.csv (Tsit5 = the delta_pop_ runs, AutoTsit5 =
autotsit5_), so those ~30k files aren't re-read. Writes
reltol_timing_overall.csv / reltol_timing_by_form.csv covering all 8 arms
and prints a markdown table.
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using DataFrames
using CSV
using Statistics
using Printf

const BETA_FORMS = ["beta_exp", "beta_rational", "beta_mixed"]
const TRAIN_LENGTH = 365
const NOISE = 0.0

# (solver label, reltol label, sim_name prefix); order = reltol outer, solver inner
const NEW_ARMS = [("Tsit5", "1e-5", "tsit5_rtol1e-5_"), ("AutoTsit5", "1e-5", "autotsit5_rtol1e-5_"),
                  ("Tsit5", "1e-6", "tsit5_rtol1e-6_"), ("AutoTsit5", "1e-6", "autotsit5_rtol1e-6_"),
                  ("Tsit5", "1e-10", "tsit5_rtol1e-10_"), ("AutoTsit5", "1e-10", "autotsit5_rtol1e-10_")]

function elapsed_times(prefix, beta_form)
    sim_name = "$(prefix)UDE_single_beta=$(beta_form)_adam=2500_lbfgs=2000_traindata=$(TRAIN_LENGTH)_noise=$(NOISE)"
    sim_dir = datadir("exp_pro", "sims", "ude_single", sim_name)
    isdir(sim_dir) || return Float64[]
    times = Float64[]
    for loc_dir in readdir(sim_dir; join=true)
        isdir(loc_dir) || continue
        startswith(basename(loc_dir), "synthetic_") || continue
        for seed_dir in readdir(loc_dir; join=true)
            isdir(seed_dir) || continue
            startswith(basename(seed_dir), "simulation_seed=") || continue
            result_path = joinpath(seed_dir, "results.jld2")
            isfile(result_path) || continue
            d = JLD2.load(result_path)
            haskey(d, "elapsed_seconds") && push!(times, d["elapsed_seconds"])
        end
    end
    return times
end

per_form_rows = NamedTuple[]
overall_rows = NamedTuple[]

for (solver, reltol, prefix) in NEW_ARMS
    all_times = Float64[]
    for bf in BETA_FORMS
        t = elapsed_times(prefix, bf)
        println(stderr, "$(solver) $(reltol) / $(bf): n=$(length(t))")
        isempty(t) && continue
        push!(per_form_rows, (; solver, reltol, beta_form=bf, n=length(t), mean_seconds=mean(t),
                                median_seconds=median(t), total_compute_hours=sum(t) / 3600))
        append!(all_times, t)
    end
    isempty(all_times) || push!(overall_rows, (; solver, reltol, n=length(all_times), mean_seconds=mean(all_times),
                                                median_seconds=median(all_times), total_compute_hours=sum(all_times) / 3600))
end

# default-tolerance (1e-3) rows from the existing solver timing tables
old_overall = CSV.read(projectdir("_research", "crps_tables", "solver_timing_overall.csv"), DataFrame)
old_form    = CSV.read(projectdir("_research", "crps_tables", "solver_timing_by_form.csv"), DataFrame)
for s in ("Tsit5", "AutoTsit5")
    r = old_overall[old_overall.solver .== s, :][1, :]
    push!(overall_rows, (; solver=s, reltol="1e-3", n=r.n, mean_seconds=r.mean_seconds,
                           median_seconds=r.median_seconds, total_compute_hours=r.total_compute_hours))
    for rr in eachrow(old_form[old_form.solver .== s, :])
        push!(per_form_rows, (; solver=s, reltol="1e-3", beta_form=rr.beta_form, n=rr.n, mean_seconds=rr.mean_seconds,
                                median_seconds=rr.median_seconds, total_compute_hours=rr.total_compute_hours))
    end
end

reltol_order = Dict("1e-3" => 1, "1e-5" => 2, "1e-6" => 3, "1e-10" => 4)
solver_order = Dict("Tsit5" => 1, "AutoTsit5" => 2)
sorter(df) = sort(df, [order(:reltol, by=r -> reltol_order[r]), order(:solver, by=s -> solver_order[s])])
overall_df  = sorter(DataFrame(overall_rows))
per_form_df = sorter(DataFrame(per_form_rows))
CSV.write(projectdir("_research", "crps_tables", "reltol_timing_overall.csv"), overall_df)
CSV.write(projectdir("_research", "crps_tables", "reltol_timing_by_form.csv"), per_form_df)

commas(x) = replace(@sprintf("%.0f", x), r"(?<=\d)(?=(\d{3})+$)" => ",")
println("| Solver (reltol) | n seeds | Mean (s) | Median (s) | Total compute-hours for 3 beta forms, 51 locations, 100 seeds |")
println("| --- | --- | --- | --- | --- |")
for r in eachrow(overall_df)
    println("| $(r.solver) ($(r.reltol)) | $(commas(r.n)) | $(commas(r.mean_seconds)) | $(commas(r.median_seconds)) | $(commas(r.total_compute_hours)) |")
end
