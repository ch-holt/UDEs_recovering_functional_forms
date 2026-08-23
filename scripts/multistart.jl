#========================================================
SCRIPT TO DO MULTISTART AS A POST-PROCESSING STEP
=========================================================#  
using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using DataFrames
using CSV

# Call module
using UDE_FUNCTIONAL_FORMS

# Where does this function belong (C&P from summarise_results.jl)
function summarise_results(sim_name_dir::String, population::Real)

    rows = []
    # Each seed run is stored in simulation_seed=* subfolders
    for sim_folder in readdir(sim_name_dir; join=true)
        !isdir(sim_folder) && continue
        !occursin("simulation_seed=", basename(sim_folder)) && continue

        results_file = joinpath(sim_folder, "results.jld2")
        !isfile(results_file) && continue

        data = JLD2.load(results_file)

        seed            = get(data, "seed",            missing)
        loss_traj_noisy = get(data, "loss_traj_noisy", missing)
        loss_traj_true  = get(data, "loss_traj_true",  missing)
        loss_beta       = get(data, "loss_beta",       missing)
        loss_I_grid     = get(data, "loss_I_grid",     missing)
        val_losses      = get(data, "val_losses",      missing)
        best_val_loss = ismissing(val_losses) ? missing : minimum(val_losses)

        # "prediction" is the long-term solve (3 years, row 3 = infectious compartment);
        # check feasibility over the full saved horizon.
        prediction   = get(data, "prediction", missing)
        max_forecast = ismissing(prediction) ? missing : maximum(prediction[3, :])
        feasible     = ismissing(max_forecast) ? missing : (max_forecast <= population)

        push!(rows, (
            sim_folder      = basename(sim_folder),
            seed            = seed,
            nmse_traj_noisy = loss_traj_noisy,
            nmse_traj_true  = loss_traj_true,
            nmse_beta       = loss_beta,
            nmse_I_grid     = loss_I_grid,
            best_val_loss   = best_val_loss,
            max_forecast    = max_forecast,
            feasible        = feasible,
        ))
    end

    df = DataFrame(rows)
    sort!(df, [:seed])


    return df
end


beta_function = beta_exp

for train_length in [7,14,21,28,35,42,49,56,60, 63, 70, 77, 84, 91, 100, 125, 150,175, 200, 365]
    for location in ["MA"]
        for noise in [0.0]
            println("Processing location: $(location), train_length: $(train_length), noise: $(noise)")

            model_name   = "ude_single"
            sim          = "UDE_single_beta=$(beta_function)_adam=2500_lbfgs=2000_traindata=$(train_length)_noise=$(noise)"
            sim_name_dir = datadir("exp_pro", "sims", "ude_single", sim, "synthetic_$(location)")
            population   = POPULATION[location]

            # Extract and save summary once per noise level
            df       = summarise_results(sim_name_dir, population)
            out_path = joinpath(sim_name_dir, "results_summary.csv")
            CSV.write(out_path, df)
            println("Saved summary to: $out_path")

            # First check forecasts over the full 3-year horizon are reasonable (i.e. less than the population size).
            # Seeds with missing predictions are treated as infeasible (can't verify them), and are
            # dropped here so they never enter the val-loss-based multistart cut below.
            n_rows_before    = nrow(df)
            feasible_mask    = coalesce.(df.feasible, false)
            infeasible_seeds = df[.!feasible_mask, :seed]
            df               = df[feasible_mask, :]
            println("Removed $(n_rows_before - nrow(df)) infeasible seed(s) " *
                    "(max_forecast > population=$(population)): $(infeasible_seeds)")

            # Sort by validation loss (ascending) once
            sort!(df, :best_val_loss, rev=false)
            n_rows = size(df, 1)

            for MS_limit in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                n_rows_MS     = Int(floor(n_rows * MS_limit))
                seeds_to_keep = df[1:end-n_rows_MS, :seed]

                seeds_to_keep_path = joinpath(sim_name_dir, "seeds_to_keep_MS=$(MS_limit).jld2")
                @save seeds_to_keep_path seeds_to_keep
                println("Saved usable seeds to: $seeds_to_keep_path")
            end
        end
    end
end

