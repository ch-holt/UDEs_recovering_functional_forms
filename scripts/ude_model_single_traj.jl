#========================================================
SCRIPT TO TRAIN THE UDE MODEL FOR A SINGLE TRAJECTORY
=========================================================#  
using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using Lux
using ComponentArrays
using DataFrames
using DiffEqFlux, Zygote
using Optimisers
using DifferentialEquations
using Plots
using JLD2
using Optimization
using OptimizationOptimJL
using Random

# Call module
using UDE_FUNCTIONAL_FORMS


#========================================================
MAIN FUNCTION TO TRAIN THE UDE AND SAVE THE RESULTS
=========================================================# 

function run_model(beta_function, data, u0, seed, predict_ude, beta_network, prob_ude; maxiters_adam, maxiters_lbfgs, number_of_nn_inputs, adam_learning_rate, learning_bias_bool)
    println("Starting run: on thread $(Threads.threadid())")
    rng = Random.seed!(seed)

    # Initialise parameters
    p, st = Lux.setup(rng, beta_network)
    p = ComponentArray(p)
    p = Float64.(p)

    # Combine all parameters into a single object for optimisation
    p_init = ComponentArray(
        nn_params = p,
        gamma = gamma,
        sigma = sigma,
        delta = delta,
        tmax = train_length,
        population = population
    )

    p_trained, train_losses_final, val_losses_final = train_ude_single_dataset(p_init, predict_ude, data, u0; maxiters_adam=maxiters_adam, maxiters_lbfgs=maxiters_lbfgs, adam_learning_rate=adam_learning_rate)

    loc_foldername = "synthesised_$(location)"

	# Append a number to the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
	model_iteration = 1
	while isdir(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, "simulation_v$(model_iteration)"))
		model_iteration += 1
	end

    foldername = "simulation_v$(model_iteration)_seed=$(seed)"
	filename = "synthesised_$(location)"

    # Within the plots folder create a folder for each trajectory
    if !isdir(plotsdir("sims", model_name, sim_name, loc_foldername, foldername)) 
        mkpath(plotsdir("sims", model_name, sim_name, loc_foldername, foldername))
    end

    # Plot losses across iterations
    loss_plot = plot(train_losses_final, yscale=:log10, xlabel="Iteration", ylabel="Loss (log scale)", title="Training loss across iterations", label="Train", legend=:topright)
    plot!(loss_plot, val_losses_final, yscale=:log10, label="Val")
    savefig(loss_plot, plotsdir("sims", model_name, sim_name, loc_foldername, foldername, "training_loss_plot.png"))

    # Evaluate final long term results 
    long_term_prob= remake(prob_ude, p = p_trained, tspan = (1.0, 3*365.0), u0 = u0)
    long_term_pred = solve(long_term_prob, Rosenbrock23(), saveat=1, dense = false)

    # Convert to a 1 x N matrix
    x_hat = long_term_pred[3, 1:length(data)]

    # Define the neural network input
    nT = length(x_hat)
    
    # Define input for SR via I_grid
    I_grid = collect(range(0, 1; length=1000))
    nI = length(I_grid)

    nn_input = Float64.(reshape(x_hat ./ p_init.population, 1, nT))
    y_hat_input = Float64.(reshape(I_grid, 1, nI))


    beta_traj = vec(beta_network(nn_input, p_trained.nn_params, st)[1])

    mkpath(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername))

    #========================
    CREATE PLOTS
    ========================#

    plot_dir = plotsdir("sims", model_name, sim_name, loc_foldername, foldername)
    if !isdir(plot_dir) 
        mkpath(plot_dir)
    end

    # Create trajectory plot
    loss_traj = loss_nmse(x_hat, data)

    traj_plot = plot(days[1:length(data)], data[1:length(data)], color=:black, markersize=2, label="Data", 
    xlabel="Day", ylabel="Infectious individuals", title="Infectious trajectory for $(location)", legend=:topright)
    plot!(traj_plot, days[1:length(data)], x_hat, color=:red, linewidth=2, label="Predicted trajectory")
    annotate!(traj_plot, days[round(Int, length(data)/2)], maximum(data), text("NMSE: $(round(loss_traj, sigdigits=3))", :black))

    # Save the plot
    savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

    # Create beta plot

    # Define beta function
    true_beta = beta_function(location, data)
    true_beta_against_xhat = beta_function(location, I_grid*p_init.population)

    loss_beta = loss_nmse(beta_traj, true_beta)

    beta_plot = plot(days[1:length(beta_traj)], true_beta, color=:blue, linewidth=2, label="True beta", 
    xlabel="Day", ylabel="Beta", title="Beta trajectory for $(location)", legend=:topright)
    plot!(beta_plot, days[1:length(beta_traj)], beta_traj, color=:red, linewidth=2, label="Predicted beta")
    annotate!(beta_plot, days[round(Int, length(beta_traj)/2)], maximum(true_beta), text("NMSE: $(round(loss_beta, sigdigits=3))", :black))

    # Save the plot
    savefig(beta_plot, joinpath(plot_dir, "beta_plot.png"))

    # Save the plot
    savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

    # Create beta plot against x_hat
    
    # Evaluate neural network and extract approximation
    y_hat = vec(beta_network(y_hat_input, p_trained.nn_params, st)[1])
    
    loss_I_grid = loss_nmse(y_hat, true_beta_against_xhat)

    beta_against_xhat_plot = plot(I_grid, true_beta_against_xhat, color=:blue, linewidth=2, label="True beta", 
    xlabel="I/N", ylabel="Beta", title="Beta trajectory for $(location)", legend=:topright)
    plot!(beta_against_xhat_plot, I_grid, y_hat, color=:red, linewidth=2, label="Predicted beta")
    annotate!(beta_against_xhat_plot, I_grid[round(Int, length(y_hat)/2)], maximum(true_beta_against_xhat), text("NMSE: $(round(loss_I_grid, sigdigits=3))", :black))

    # Save the plot
    savefig(beta_against_xhat_plot, joinpath(plot_dir, "beta_against_xhat_plot.png"))

    mkpath(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername))
	JLD2.save(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername, "results.jld2"),
		"p", p_trained, "train_losses", train_losses_final, "val_losses", val_losses_final, "prediction", Array(long_term_pred), "beta_prediction", beta_traj,
		"days", days, "seed", seed, "loss_traj", loss_traj, "loss_beta", loss_beta, "loss_I_grid", loss_I_grid)

	println("Finished run: $(location) on thread $(Threads.threadid())")

	return nothing
end

#========================================================
DEFINE HYPERPARAMETERS
=========================================================#

# Number of data points used for training (total number of entries in the dataset)
const train_length = 365
# Define the timespan for the ODE solver
tspan = [1, train_length]

# do 100 simulations 
n_sims = 50
location = "MA"


#========================================================
LOAD DATA
=========================================================#

dataset = JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_beta_exp", "synthesised_$(location).jld2"))

# Extract infectious individuals and days from the dataset
data = dataset["infectious"]
days = dataset["days"]

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

population = POPULATION[location]
prevalence = PREVALENCE[location]
delta = DELTA[location]
R0_reproduction = R0_REPRODUCTION[location]
zeta = ZETA[location]

# Derive other parameters
beta0 = R0_reproduction * (gamma + delta)
I0 = max(1.0, prevalence * population)
S0 = population - E0 - I0 - R0_recovered - D0

# Define initial state
u0 = [S0, E0, I0, R0_recovered, D0]

#========================================================
SET UP MODEL
=========================================================#

hidden_dims = 5
input_size = 1
output_size = 1
activation_function = gelu
final_activation_function = softplus

beta_function = beta_exp
maxiters_adam = 2500
maxiters_lbfgs = 2000
number_of_nn_inputs = 1

adam_learning_rate = 1e-3

# Define strings for file names and directory for results
model_name = "ude_single"
sim_name ="UDE_single_beta=$(beta_function)_adam=$(maxiters_adam)_learning_rate=$(adam_learning_rate)_lbfgs=$(maxiters_lbfgs)_bias=$(learning_bias_bool)_number_of_nn_input=$(number_of_nn_inputs)_finalactivation=$(final_activation_function)_test_val_5_5"
if !isdir(datadir("exp_pro","sims", model_name, sim_name)) 
	mkpath(datadir("exp_pro","sims", model_name, sim_name))
end

seed_grid = rand(1:100000, 100)

for i = 1:length(seed_grid)
    
    rng = Random.seed!(seed_grid[i])

    beta_network, p_nn_temp, st_nn = build_neural_network(rng, hidden_dims, input_size, output_size, 
                                activation_function, final_activation_function)

    seird_nn! = make_seird_nn(beta_network, st_nn, sigma, gamma, input_size)
    prob_ude = ODEProblem(seird_nn!, u0, tspan, p_nn_temp)
    predict_ude = make_predict_ude(prob_ude, train_length)

    run_model(beta_function, data, u0, seed_grid[i], predict_ude, beta_network, prob_ude; maxiters_adam = maxiters_adam, maxiters_lbfgs = maxiters_lbfgs, number_of_nn_inputs=number_of_nn_inputs, adam_learning_rate=adam_learning_rate, learning_bias_bool=learning_bias_bool)
end




