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
    # Each seed run is stored in simulation_v*_seed=* subfolders
    for sim_folder in readdir(sim_name_dir; join=true)
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
            sim_folder  = basename(sim_folder),
            seed        = seed,
            nmse_traj   = loss_traj,
            nmse_beta   = loss_beta,
            nmse_I_grid = loss_I_grid,
        ))
    end

    df = DataFrame(rows)
    sort!(df, [:seed])


    return df
end


# Define simulation
location = "MA"
model_name = "ude_single"
sim = "UDE_single_beta=beta_exp_adam=2500_learning_rate=0.001_lbfgs=2000_number_of_nn_input=1_finalactivation=softplus_val=55_128"
sim_name_dir = datadir("exp_pro", "sims", "ude_single", sim, "synthesised_$(location)")
MS_limit=0.3

# Extract summary
df= summarise_results(sim_name_dir)

# Save full results to csv - for easy viewing
out_path = joinpath(sim_name_dir, "results_summary.csv")
CSV.write(out_path, df)
println("Saved summary to: $out_path") 

# Sort df from highest to lowest loss (nmse_traj)
sort!(df, :nmse_traj, rev=false)

# Define % of the number of rows to keep
n_rows = size(df, 1)
n_rows_MS = Int(floor(n_rows * MS_limit))

# Create list of seeds in the top % of the results
seeds_to_keep = df[1:end-n_rows_MS, :seed]

# Write into a jld2 file    
seeds_to_keep_path = joinpath(sim_name_dir, "seeds_to_keep_MS=$(MS_limit).jld2")
@save seeds_to_keep_path seeds_to_keep  
println("Saved usable seeds to: $seeds_to_keep_path") 



