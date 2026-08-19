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
function summarise_results(sim_name_dir::String)

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

        push!(rows, (
            sim_folder      = basename(sim_folder),
            seed            = seed,
            nmse_traj_noisy = loss_traj_noisy,
            nmse_traj_true  = loss_traj_true,
            nmse_beta       = loss_beta,
            nmse_I_grid     = loss_I_grid,
            best_val_loss   = best_val_loss,
        ))
    end

    df = DataFrame(rows)
    sort!(df, [:seed])


    return df
end




for train_length in [365]
    for location in ["AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","IN","MA","OK","OR","VA","WI","WV"]
    println("Processing location: $(location)")
        for MS_limit in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            println("Processing training length: $(train_length)")
            # Define simulation
            model_name = "ude_single"
            sim = "UDE_single_beta=beta_exp_adam=2500_lbfgs=2000_traindata=$(train_length)"
            sim_name_dir = datadir("exp_pro", "sims", "ude_single", sim, "synthetic_$(location)")

            # Extract summary
            df= summarise_results(sim_name_dir)

            # Save full results to csv - for easy viewing
            out_path = joinpath(sim_name_dir, "results_summary.csv")
            CSV.write(out_path, df)
            println("Saved summary to: $out_path") 

            # Sort df from highest to lowest loss (validation)
            sort!(df, :best_val_loss, rev=false)

            # Define % of the number of rows to keep
            n_rows = size(df, 1)
            n_rows_MS = Int(floor(n_rows * MS_limit))

            # Create list of seeds in the top % of the results
            seeds_to_keep = df[1:end-n_rows_MS, :seed]

            # Write into a jld2 file    
            seeds_to_keep_path = joinpath(sim_name_dir, "seeds_to_keep_MS=$(MS_limit).jld2")
            @save seeds_to_keep_path seeds_to_keep  
            println("Saved usable seeds to: $seeds_to_keep_path") 
        end
    end
end

