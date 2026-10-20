#========================================================
RUNTIME — STABLE (STIRLING) VS EXACT-SUM NB NLL, MEASURED ON THE HPC

Compares the elapsed training time per seed (elapsed_seconds in results.jld2)
of the stable_exact timing run (noise = 0.1, seeds 1–5, locations with the
smallest / median / largest total count) with the same seeds of the stable
sweep. Both use the same seeds, so the same NN initialisation and phi_init,
and the same loss up to rounding; only the evaluation of the gamma ratio
differs (Stirling's formula vs the exact sum over k = 0..y-1).

Caveat: the runs did not share a set-up. The stable sweep ran 100 seeds on 8
threads per location, the timing run 5 seeds on 5 threads, on whichever nodes
SLURM gave them, so the times include different contention. The ratio is an
order of magnitude, not a precise factor.

Writes data/exp_pro/test/nll_formulation_runtime.csv
=========================================================#

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using CSV
using DataFrames
using Statistics
using Printf

const NOISE = 0.1
const SEEDS = 1:5
run_dir(suffix) = datadir("exp_pro", "sims", "test", "ude_single",
                          "autotsit5_UDE_single_beta=beta_exp_adam=2500_lbfgs=2000_traindata=365_noise=$(NOISE)$(suffix)")
const EXACT  = run_dir("_estphi_randinit_stableexact_newinit")
const STABLE = run_dir("_estphi_randinit_stable_newinit")
const SYNTH  = datadir("exp_pro", "synthetic_data", "synthetic_trajectories_HQ_beta_exp")

function seed_info(dir, loc, seed)
    f = joinpath(dir, "synthetic_$(loc)", "simulation_seed=$(seed)", "results.jld2")
    isfile(f) || return (missing, missing, missing)
    d = JLD2.load(f, "elapsed_seconds", "train_losses", "phi_hat")
    return (d[1], length(d[2]), d[3])
end

rows = []
for loc_folder in sort(readdir(EXACT))
    loc   = replace(loc_folder, "synthetic_" => "")
    total = sum(JLD2.load(joinpath(SYNTH, loc_folder, "noise=$(NOISE).jld2"))["infectious"][1:365])
    for seed in SEEDS
        te, ie, pe = seed_info(EXACT, loc, seed)
        ts, is, ps = seed_info(STABLE, loc, seed)
        push!(rows, (location=loc, total_count=total, seed=seed,
                     hours_exact=te / 3600, iterations_exact=ie, phi_hat_exact=pe,
                     hours_stable=ts / 3600, iterations_stable=is, phi_hat_stable=ps,
                     seconds_per_iteration_exact=te / ie, seconds_per_iteration_stable=ts / is,
                     slowdown_per_iteration=(te / ie) / (ts / is)))
    end
end
df = sort(DataFrame(rows), [:total_count, :seed])

out = datadir("exp_pro", "test", "nll_formulation_runtime.csv")
CSV.write(out, df)
show(select(df, :location, :seed, :hours_exact, :hours_stable, :iterations_exact, :iterations_stable,
            :phi_hat_exact, :phi_hat_stable, :slowdown_per_iteration); allrows=true); println()

println("\nPer location (median over seeds):")
for g in groupby(df, :location; sort=false)
    @printf("%s (total count %.2g): exact %.2f h/seed, stable %.3f h/seed; per iteration %.0fx slower\n",
            g.location[1], g.total_count[1], median(g.hours_exact), median(g.hours_stable), median(g.slowdown_per_iteration))
end
println("Saved $(out)")
