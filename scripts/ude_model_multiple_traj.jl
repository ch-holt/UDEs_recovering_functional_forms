#========================================================
SCRIPT TO TRAIN THE UDE MODEL FOR MULTIPLE TRAJECTORIES
=========================================================#  
using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using Lux
using JLD2
using ComponentArrays
using DataFrames
using DiffEqFlux, Zygote
using Optimisers
using Optimization
using Optim
using OptimizationOptimJL
using DifferentialEquations
using Plots
using Random; rng = Random.default_rng()

# Call module
using UDE_FUNCTIONAL_FORMS


#========================================================
MAIN FUNCTION TO TRAIN THE UDE AND SAVE THE RESULTS
=========================================================# 

function run_model(locations, beta_function; maxiters_adam, maxiters_lbfgs, number_of_nn_input)
    println("Starting run: on thread $(Threads.threadid())")

    # Initialise parameters
    nn_params, st = Lux.setup(rng, beta_network)
    nn_params = ComponentArray(nn_params)
    nn_params = Float64.(nn_params)

    trajectories = load_trajectories(locations, beta_function)
    p_trained, losses_final = train_ude_multiple_datasets(nn_params, predict_ude, trajectories; maxiters_adam, maxiters_lbfgs)
    
    # Save the trained parameters and losses for the combined trajectories

    # Create numbered simulation folders to allow multiple runs of a single set of hyperparameters 
    model_iteration = 1
    while isdir(datadir("exp_pro","sims", model_name, sim_name, "simulation_v$(model_iteration)"))
        model_iteration += 1
    end
    foldername = "simulation_v$model_iteration"

    save(datadir("exp_pro","sims", model_name, sim_name, foldername, "training_results.jld2"), 
    "p_trained", p_trained, "losses_final", losses_final)

    # Within the plots folder create a folder for each trajectory
    if !isdir(plotsdir("sims", model_name, sim_name, foldername)) 
        mkpath(plotsdir("sims", model_name, sim_name, foldername))
    end

    # Plot losses across iterations
    loss_plot = plot(losses_final, yscale=:log10, xlabel="Iteration", ylabel="Total loss across trajectories (log scale)", title="Training loss across iterations", legend=false)
    savefig(loss_plot, plotsdir("sims", model_name, sim_name, foldername, "training_loss_plot.png"))

    # Evaluate the trained model on each trajectory and save the results
    # Loop through all simulations

    # Read all files/folders in the root directory
    for location in locations
        filename = "synthesised_$(location)"
        # Extract trajectory of infectious individuals
        dataset = JLD2.load(datadir("exp_pro","synthetic_data", "synthetic_trajectories_beta_exp", filename*".jld2"))
        data = dataset["infectious"]
        days = dataset["days"]

        varying_p = ComponentArray(
            population = dataset["varying_p"]["population"],
            prevalence = dataset["varying_p"]["prevalence"],
            delta = dataset["varying_p"]["delta"],
            R0_reproduction = dataset["varying_p"]["R0_reproduction"],
            zeta = dataset["varying_p"]["zeta"]
        )

        # Derive beta0 specific to current trajectory
        beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)

        # Update the parameters for the current trajectory to include the varying parameters
        p_all = ComponentArray(
            nn_params = p_trained,
            population = varying_p.population,
            prevalence = varying_p.prevalence,
            beta0 = beta0,
            zeta = varying_p.zeta,
            R0_reproduction = varying_p.R0_reproduction,
            delta = varying_p.delta
        )

        # Define initial state
        I0 = max(1.0, p_all.prevalence * p_all.population)
        S0 = p_all.population - E0 - I0 - R0_recovered - D0
        init_state = [S0, E0, I0, R0_recovered, D0]

        # Evaluate prediction for the trained parameters on the current trajectory
        long_term_prob= remake(prob_ude, u0 = init_state, p = p_all)
        long_term_pred = solve(long_term_prob, Tsit5(), saveat=1, dense = false)
        
        # Convert to a 1 x N matrix
        x_hat = long_term_pred[3, 1:length(data)]

        # Define the neural network input for the current trajectory 
        # Keep beta, zeta, delta constant over time but normalise
        nT = length(x_hat)
        
        # Define input for SR via I_grid
        I_grid = collect(range(0, 1; length=1000))
        nI = length(I_grid)
        if number_of_nn_input == 4
            delta_norm = (log(p_all.delta) - log(1e-6))/(log(1e-2) - log(1e-6))
            beta0_norm = (p_all.R0_reproduction - 1.2)/(6.0 - 1.2)
            zeta_norm  = p_all.zeta/0.05
            nn_input = Float64.(vcat(fill(beta0_norm, 1, nT), fill(zeta_norm, 1, nT),
                                     fill(delta_norm, 1, nT), reshape(x_hat ./ p_all.population, 1, nT)))
            y_hat_input = Float64.(vcat(fill(beta0_norm, 1, nI), fill(zeta_norm, 1, nI),
                                     fill(delta_norm, 1, nI), reshape(I_grid, 1, nI)))
        elseif number_of_nn_input == 1
            nn_input = Float64.(reshape(x_hat ./ p_all.population, 1, nT))
            y_hat_input = Float64.(reshape(I_grid, 1, nI))
        end

        # Evaluate neural network and extract approximation
        beta_traj = vec(beta_network(nn_input, p_trained, st_nn)[1])

        # Within this folder create a folder for each trajectory
        if !isdir(datadir("exp_pro","sims", model_name, sim_name, foldername, filename)) 
            mkpath(datadir("exp_pro","sims", model_name, sim_name, foldername, filename))
        end
        
        root = datadir("exp_pro","sims", model_name, sim_name, foldername, filename, "results.jld2")
        # In this folder save the infectious trajectory results and the beta results for this trajectory
        JLD2.save(root,
            "p", p_trained, "losses", losses_final, "prediction", Array(long_term_pred), "beta_prediction", beta_traj,
            "days", days)
    
        #========================
        CREATE PLOTS
        ========================#

        plot_dir = plotsdir("sims", model_name, sim_name, foldername, filename)
        if !isdir(plot_dir) 
            mkpath(plot_dir)
        end

        # Create trajectory plot
        traj_plot = plot(days[1:length(data)], data[1:length(data)], color=:black, markersize=2, label="Data", 
        xlabel="Day", ylabel="Infectious individuals", title="Infectious trajectory for $(location)", legend=:topright)
        plot!(traj_plot, days[1:length(data)], x_hat, color=:red, linewidth=2, label="Predicted trajectory")

        # Save the plot
        savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

        # Create beta plot

        # Define beta function
        true_beta = beta_function(location, data)
        true_beta_against_xhat = beta_function(location, I_grid*p_all.population)

        loss = loss_nmse(beta_traj, true_beta, maximum(true_beta)-minimum(true_beta))

        beta_plot = plot(days[1:length(beta_traj)], true_beta, color=:blue, linewidth=2, label="True beta", 
        xlabel="Day", ylabel="Beta", title="Beta trajectory for $(location)", legend=:topright)
        plot!(beta_plot, days[1:length(beta_traj)], beta_traj, color=:red, linewidth=2, label="Predicted beta")
        annotate!(beta_plot, days[round(Int, length(beta_traj)/2)], maximum(true_beta), text("NMSE: $(round(loss, digits=4))", :black))

        # Save the plot
        savefig(beta_plot, joinpath(plot_dir, "beta_plot.png"))

        # Save the plot
        savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

        # Create beta plot against x_hat
        
        # Evaluate neural network and extract approximation
        y_hat = vec(beta_network(y_hat_input, p_trained, st_nn)[1])
        
        loss_I_grid = loss_nmse(y_hat, true_beta_against_xhat, maximum(true_beta_against_xhat)-minimum(true_beta_against_xhat))

        beta_against_xhat_plot = plot(I_grid, true_beta_against_xhat, color=:blue, linewidth=2, label="True beta", 
        xlabel="I/N", ylabel="Beta", title="Beta trajectory for $(location)", legend=:topright)
        plot!(beta_against_xhat_plot, I_grid, y_hat, color=:red, linewidth=2, label="Predicted beta")
        annotate!(beta_against_xhat_plot, I_grid[round(Int, length(y_hat)/2)], maximum(true_beta_against_xhat), text("NMSE: $(round(loss_I_grid, digits=4))", :black))

        # Save the plot
        savefig(beta_against_xhat_plot, joinpath(plot_dir, "beta_against_xhat_plot.png"))

    end
    println("Finished run: $(foldername) on thread $(Threads.threadid())")
	return nothing
end


#========================================================
DEFINE HYPERPARAMETERS
=========================================================# 

# Number of data points used for training (total number of entries in the dataset)
const train_length = 365
const maxiters = 2500
# Define the timespan for the ODE solver
tspan = [1, train_length]

#========================================================
DEFINE INITIAL STATE
=========================================================#

# Latent period of 3 days represented by incubation rate sigma
const sigma = 1/3 
# Infectious period of 10 days represented by recovery rate gamma
const gamma = 1/10

# Define initial state same as the generated data
# Retrieve fixed parameters
const E0 = 1.0
const R0_recovered = 0.0
const D0 = 0.0

u0 = [0.0, E0, 0.0, R0_recovered, D0]

#========================================================
SET UP MODEL
=========================================================#

hidden_dims = 5
input_size = 4
output_size = 1
activation_function = gelu
final_activation_function = sigmoid

beta_network, p_nn_temp, st_nn = build_neural_network(rng, hidden_dims, input_size, output_size, 
                            activation_function, final_activation_function)

seird_nn! = make_seird_nn(beta_network, st_nn, sigma, gamma, input_size)
prob_ude = ODEProblem(seird_nn!, u0, tspan, p_nn_temp)
predict_ude = make_predict_ude(prob_ude, train_length)

#========================================================
DEFINE HYPERPARAMETERS
=========================================================#

beta_function = beta_exp
maxiters_adam = 5000
maxiters_lbfgs = 2000
number_of_nn_input = 4

# Define strings for file names and directory for results
model_name = "ude_multiple"
locations = ["MA"]
sim_name ="UDE_multiple_beta=$(beta_function)_adam=$(maxiters_adam)_lbfgs=$(maxiters_lbfgs)_number_of_nn_input=$(number_of_nn_input)_locations=$(join(locations, "_"))"
if !isdir(datadir("exp_pro","sims", model_name, sim_name)) 
	mkpath(datadir("exp_pro","sims", model_name, sim_name))
end

for i = 1:1
    run_model(locations, beta_function; maxiters_adam=maxiters_adam, maxiters_lbfgs=maxiters_lbfgs, number_of_nn_input=number_of_nn_input)
end


