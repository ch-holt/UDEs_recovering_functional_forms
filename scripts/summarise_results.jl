using DrWatson
@quickactivate "UDE_FUNCTIONAL_FORMS"

using JLD2
using DataFrames
using CSV

function summarise_results(sim_name_dir::String)

    rows = []

    # Each location has its own subfolder (e.g. synthesised_MA)
    for loc_folder in readdir(sim_name_dir; join=true)
        !isdir(loc_folder) && continue
        location = basename(loc_folder)

        # Each seed run is stored in simulation_v*_seed=* subfolders
        for sim_folder in readdir(loc_folder; join=true)
            !isdir(sim_folder) && continue
            !occursin("simulation_v", basename(sim_folder)) && continue

            results_file = joinpath(sim_folder, "results.jld2")
            !isfile(results_file) && continue

            data = JLD2.load(results_file)

            seed       = get(data, "seed",        missing)
            loss_traj  = get(data, "loss_traj",   missing)
            loss_beta  = get(data, "loss_beta",   missing)
            loss_I_grid = get(data, "loss_I_grid", missing)

            push!(rows, (
                location    = location,
                sim_folder  = basename(sim_folder),
                seed        = seed,
                nmse_traj   = loss_traj,
                nmse_beta   = loss_beta,
                nmse_I_grid = loss_I_grid,
            ))
        end
    end

    df = DataFrame(rows)
    sort!(df, [:location, :seed])

    out_path = joinpath(sim_name_dir, "results_summary.csv")
    CSV.write(out_path, df)
    println("Saved summary to: $out_path")
    return df
end


summarise_results(datadir("exp_pro", "sims", "ude_single", "UDE_single_beta=beta_exp_adam=2500_learning_rate=0.001_lbfgs=2000_bias=true_number_of_nn_input=1_finalactivation=softplus_nmse_change"))
