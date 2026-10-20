#========================================================
SOLVER COMPUTATION-TIME SUMMARY — per-seed training time (the
elapsed_seconds field every results.jld2 already stores, from run_model's
own t_start/time() bracket in ude_model_single_traj.jl), aggregated per
solver and per (solver, beta form). One row per solver overall (all beta
forms and locations pooled, n up to 3*51*100 = 15300) plus one row per
(solver, beta form) for consistency with the other solver-comparison
tables. Mean, median, and total compute-hours (sum of elapsed_seconds /
3600) -- total matters here since it's the real HPC cost difference
between solvers, not just the per-seed average.
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using DataFrames
using CSV
using Statistics

const SOLVER_PREFIXES = ["rosenbrock23_", "vern7_", "autotsit5_", "delta_pop_"]
# "delta_pop_" stands in as the Tsit5 arm -- see solver_violin_plots.jl header.
const SOLVER_DISPLAY_NAME = Dict("rosenbrock23_" => "Rosenbrock23", "vern7_" => "Vern7",
                                  "autotsit5_" => "AutoTsit5", "delta_pop_" => "Tsit5")
const BETA_FORMS = ["beta_exp", "beta_rational", "beta_mixed"]
const TRAIN_LENGTH = 365
const NOISE = 0.0

function elapsed_times(prefix, beta_form)
    sim_name = "$(prefix)UDE_single_beta=$(beta_form)_adam=2500_lbfgs=2000_traindata=$(TRAIN_LENGTH)_noise=$(NOISE)"
    sim_dir = datadir("exp_pro", "sims", "test", "ude_single", sim_name)
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
            haskey(d, "elapsed_seconds") || continue
            push!(times, d["elapsed_seconds"])
        end
    end
    return times
end

per_form_rows = NamedTuple[]
overall_rows = NamedTuple[]

for prefix in SOLVER_PREFIXES
    all_times = Float64[]
    for bf in BETA_FORMS
        t = elapsed_times(prefix, bf)
        println("$(SOLVER_DISPLAY_NAME[prefix]) / $(bf): n=$(length(t))")
        if !isempty(t)
            push!(per_form_rows, (; solver=SOLVER_DISPLAY_NAME[prefix], beta_form=bf, n=length(t),
                                    mean_seconds=mean(t), median_seconds=median(t),
                                    total_compute_hours=sum(t) / 3600))
        end
        append!(all_times, t)
    end
    if !isempty(all_times)
        push!(overall_rows, (; solver=SOLVER_DISPLAY_NAME[prefix], n=length(all_times),
                                mean_seconds=mean(all_times), median_seconds=median(all_times),
                                total_compute_hours=sum(all_times) / 3600))
    end
end

per_form_df = DataFrame(per_form_rows)
overall_df = DataFrame(overall_rows)

CSV.write(projectdir("_research", "crps_tables", "solver_timing_by_form.csv"), per_form_df)
CSV.write(projectdir("_research", "crps_tables", "solver_timing_overall.csv"), overall_df)

println("\n=== Per (solver, beta form) ===")
show(per_form_df, allrows=true, allcols=true)
println("\n\n=== Overall per solver ===")
show(overall_df, allrows=true, allcols=true)
println("\n\nSaved to: ", projectdir("_research", "crps_tables"))
