#=============================================================
SCRIPT TO ANALYSE THE PERFORMANCE OF THE UDE MODEL IN TRAINING
=============================================================#
using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
include(joinpath(@__DIR__, "functions.jl"))
using .Functions
using JLD2
using Plots
using Statistics
using StatsBase
using Distributions
using ComponentArrays
using DataFrames
include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

#=============================================================
FUNCTION TO GENERATE PLOTS OF SIMULATIONS
=============================================================#

function plot_simulation(sim_name, plot_title)

    # Load the observed data
    dataset = JLD2.load(datadir("synthetic_trajectories_exponential", "synthesised_MA.jld2"))
    # Just use infectious trajectory
    obs = dataset["infectious"]
    days = dataset["days"]

    # Define the root file path
    root = datadir("sims", "ude_single", sim_name)
    # Create empty array to store predictions with initial mortalities 0
    all_predictions = []
    # Read all files/folders in the root directory
    for filename in readdir(root)
        # Only include directories
        if isdir(joinpath(root, filename))
            # Extract results, predictions and losses
            results = JLD2.load(datadir("sims", "ude_single", sim_name, filename, "results.jld2"))
            pred = results["prediction"]
            # Extract the predicted infectious trajectory for the training data
            i_traj = pred[3, 1:length(obs)]
            push!(all_predictions, i_traj)
        end
    end

    # create a prediction matrix that stacks each simulation on top of each other
    # e.g. 100 rows (one for each simulation) and 123 columns (one for each day of the training data)
    prediction_matrix = hcat(all_predictions...)'

    # Find the median prediction and IQR across all simulations
    # Find the mdeian aggregating over the rows (i.e. across all simulations) for each column (i.e. for each day of the training data)
    median_prediction = median(prediction_matrix, dims=1)
    # Convert to a vector
    median_prediction = vec(median_prediction)

    # Find the upper and lower quantiles across all simulations for each day of the training data
    # Compute per-column explicitly - (creating views rather than copies)
    upper_quantile = [quantile(view(prediction_matrix, :, j), 0.75) for j in axes(prediction_matrix, 2)]
    lower_quantile = [quantile(view(prediction_matrix, :, j), 0.25) for j in axes(prediction_matrix, 2)]

    # Evaluate MSE using the median
    mse = Functions.loss_mse(median_prediction, obs)

    # Save to a JLD2 file
    save(datadir("sims", "ude_single", sim_name, "summary.jld2"), "median_prediction", median_prediction, "upper_quantile", upper_quantile, "lower_quantile", lower_quantile, "mse", mse)

    # Create plot
    x = days[1:length(prediction_matrix[1, :])]
    pl = scatter(x, obs[1:length(prediction_matrix[1, :])], color=:black, markersize=2,
    label="Data", xlabel="Day", ylabel="Infection prevalence", title="$plot_title", legend=:topright)
    x_annot = x[end] - 0.02 * (x[end] - x[1])
    y_annot = maximum(obs[1:length(prediction_matrix[1, :])]) * 0.8
    annotate!(pl, x_annot, y_annot, text("MSE: $(round(mse, digits=4))", 9, :right))
    plot!(pl, x, median_prediction, color=:red, linewidth=2, ribbon = ((median_prediction - lower_quantile), (upper_quantile - median_prediction)), label="Median prediction")
    display(pl)

    # Save the plot
    savefig(pl, datadir("sims", "ude_single", sim_name, "prediction_plot.png"))

    return pl

end

#========================================================
PRODUCE ENSEMBLE PLOTS
=========================================================# 
sim_name = "270526_add_regularisation_and_lfbgs"
plot_title = "Single-trajectory UDE model rational beta function"
plot_simulation(sim_name, plot_title)

#=============================================================
FUNCTION TO PLOT APPROXIMATED FUNCTION AGAINST SYNTHESISED DATA 
=============================================================#

function plot_individual_traj(sim_num, sim_name, synthesised_data, beta_function, location)

    # Load the observed data and varying parameters
    dataset = JLD2.load(datadir("synthetic_trajectories_exponential", synthesised_data))

    varying_p = ComponentArray(
        population = dataset["varying_p"]["population"],
        prevalence = dataset["varying_p"]["prevalence"],
        delta = dataset["varying_p"]["delta"],
        R0_reproduction = dataset["varying_p"]["R0_reproduction"],
        zeta = dataset["varying_p"]["zeta"]
    )

    # Derive beta0 specific to current trajectory
    beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)

    # Just use infectious trajectory
    obs = dataset["infectious"]
    I_grid = collect(range(0, 1; length=1000))
    days = dataset["days"]

    # Define the root file path
    root = datadir("sims", "ude_multiple", sim_name, sim_num, synthesised_data, "results.jld2")
    plot_dir = dirname(root)

    # Extract predictions for epidemic trajectory
    results = JLD2.load(root)
    pred = results["prediction"]

    # Extract the predicted infectious trajectory for the training data
    i_traj = pred[3, 1:length(obs)]

    # Extract beta trajectory
    beta_pred = reshape(results["beta_prediction"], :, 1)

    # Define beta function
    if beta_function == "exponential"
        true_beta = reshape(Functions.beta_exp(location, obs),:,1)
        true_beta_against_xhat = reshape(Functions.beta_exp(location, I_grid*varying_p.population),:,1)
    elseif beta_function == "rational"
        true_beta = reshape(Functions.beta_rational(location, obs),:,1)
        true_beta_against_xhat = reshape(Functions.beta_rational(location, I_grid*varying_p.population),:,1)
    end

    
    # Create trajectory plot
    traj_plot = plot(days[1:length(i_traj)], obs[1:length(i_traj)], color=:black, markersize=2, label="Data", 
    xlabel="Day", ylabel="Infectious individuals", title="Infectious trajectory for $(sim_name) $(location)", legend=:topright)
    plot!(traj_plot, days[1:length(i_traj)], i_traj, color=:red, linewidth=2, label="Predicted trajectory")
    display(traj_plot)

    # Save the plot
    savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

    # Create beta plot
    beta_plot = plot(days[1:length(beta_pred)], true_beta, color=:blue, linewidth=2, label="True beta", 
    xlabel="Day", ylabel="Beta", title="Beta trajectory for $(sim_name) $(location)", legend=:topright)
    plot!(beta_plot, days[1:length(beta_pred)], beta_pred, color=:red, linewidth=2, label="Predicted beta")
    display(beta_plot)

    # Save the plot
    savefig(beta_plot, joinpath(plot_dir, "beta_plot.png"))

    # Create beta plot against x_hat
    delta_norm =(log(p_all.delta) - log(1e-6))/(log(1e-2) - log(1e-6))
    beta0_norm = (p_all.R0_reproduction - 1.2)/(6.0 - 1.2)
    zeta_norm = p_all.zeta/0.05

    nn_input = vcat(fill(beta0_norm, 1, 1000), fill(zeta_norm, 1, 1000), 
        fill(delta_norm, 1, 1000), reshape(I_grid, 1, :))

    # Retrieve NN parameters that resulted in the lowest error on the training data
    training_results = JLD2.load(DrWatson.datadir("sims", "ude_multiple", sim_name, sim_num, "training_results.jld2"))
    p_trained = training_results["p_trained"]

    # Define the NN architecture
    hidden_dims = 5

    # Retrieve nn architecture
    beta_network = Lux.Chain(Lux.Dense(4=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                            Lux.Dense(hidden_dims=>1, softplus))

    # Initialise parameters
    p_nn_temp, st_nn = Lux.setup(rng, beta_network)

    # Evaluate neural network and extract approximation
    y_hat = vec(beta_network(nn_input, p_trained, st_nn)[1])
    

    beta_against_xhat_plot = plot(I_grid, true_beta_against_xhat, color=:blue, linewidth=2, label="True beta", 
    xlabel="Day", ylabel="Beta", title="Beta trajectory for $(sim_name) $(location)", legend=:topright)
    plot!(beta_against_xhat_plot, I_grid, y_hat, color=:red, linewidth=2, label="Predicted beta")
    display(beta_against_xhat_plot)

    # Save the plot
    savefig(beta_against_xhat_plot, joinpath(plot_dir, "beta_against_xhat_plot.png"))

    return traj_plot, beta_plot, beta_against_xhat_plot
end

#========================================================
PRODUCE PLOTS OF SIMULATIONS
=========================================================# 


sim_num = "simulation_v9"
sim_name = "exponential_with_time_input_nolbfgs_090626"
for location in keys(POPULATION)
    synthesised_data = "synthesised_$(location).jld2"
    plot_individual_traj(sim_num, sim_name, synthesised_data, "exponential", location)
end


