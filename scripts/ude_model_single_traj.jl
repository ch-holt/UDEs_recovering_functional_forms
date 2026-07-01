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
using Random; rng = Random.default_rng()

# Call module
using UDE_FUNCTIONAL_FORMS


#========================================================
MAIN FUNCTION TO TRAIN THE UDE AND SAVE THE RESULTS
=========================================================# 

function run_model(data, u0; maxiters_adam, maxiters_lbfgs)
    println("Starting run: on thread $(Threads.threadid())")

    # Initialise parameters
    p, st = Lux.setup(rng, beta_network)
    p = ComponentArray(p)

    # Combine all parameters into a single object for optimisation
    p_init = ComponentArray(
        nn_params = p,
        gamma = gamma,
        sigma = sigma,
        delta = delta,
        tmax = train_length,
        population = population
    )

    # Make sure to start with a stable parameterization
    l_init = loss_ude(p_init, predict_ude, data, u0)
    println("Initial loss: $l_init")

    p_trained, losses_final = train_ude_single_dataset(p_init, predict_ude, data, u0; maxiters_adam=maxiters_adam, maxiters_lbfgs=maxiters_lbfgs)
    
    # Evaluate final long term results 
    long_term_prob= remake(prob_ude, p = p_trained, tspan = (1.0, 3*365.0), u0 = u0)
    long_term_pred = solve(long_term_prob, Rosenbrock23(), saveat=1, dense = false)

    beta_prediction = [beta_network([long_term_pred[3, i] / population], p_trained.nn_params, st_nn)[1][1] for i in 1:length(long_term_pred[3, :])]

	# Save the result
	fname = "simulation"

	# Append a number to the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
	model_iteration = 1
	while isdir(datadir("exp_pro","sims", model_name, sim_name, "simulation_v$(model_iteration)"))
		model_iteration += 1
	end
    foldername = "simulation_v$(model_iteration)"
	filename = "$(location)_simulation_v$(model_iteration)"

	mkpath(datadir("exp_pro","sims", model_name, sim_name, foldername, filename))


	JLD2.save(datadir("exp_pro","sims", model_name, sim_name, foldername, filename, "results.jld2"),
		"p", p_trained, "losses", losses_final, "prediction", Array(long_term_pred), "beta_prediction", beta_prediction,
		"days", days)

    #========================
    CREATE PLOTS
    ========================#

    plot_dir = plotsdir("sims", model_name, sim_name, foldername, filename)

    # Within this folder create a folder for each trajectory
    if !isdir(plotsdir(plot_dir)) 
        mkpath(plotsdir(plot_dir))
    end

    # Create trajectory plot
    traj_plot = plot(days[1:length(data)], data[1:length(data)], color=:black, markersize=2, label="Data", 
    xlabel="Day", ylabel="Infectious individuals", title="Infectious trajectory for $(location)", legend=:topright)
    plot!(traj_plot, days[1:length(data)], x_hat, color=:red, linewidth=2, label="Predicted trajectory")

    # Save the plot
    savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

    # Create beta plot

    # Define beta function
    
    if beta_function == "exponential"
        true_beta = beta_exp(location, data)
        true_beta_against_xhat = beta_exp(location, I_grid*p_all.population)
    elseif beta_function == "rational"
        true_beta = beta_rational(location, data)
        true_beta_against_xhat = beta_rational(location, I_grid*p_all.population)
    end

    loss = loss_nmse(beta_traj, true_beta, maximum(true_beta)-minimum(true_beta))

    beta_plot = plot(days[1:length(beta_traj)], true_beta, color=:blue, linewidth=2, label="True beta", 
    xlabel="Day", ylabel="Beta", title="Beta trajectory for $(sim_name) $(location)", legend=:topright)
    plot!(beta_plot, days[1:length(beta_traj)], beta_traj, color=:red, linewidth=2, label="Predicted beta")
    annotate!(beta_plot, days[round(Int, length(beta_traj)/2)], maximum(true_beta), text("NMSE: $(round(loss, digits=4))", :black))

    # Save the plot
    savefig(beta_plot, joinpath(plot_dir, "beta_plot.png"))

    # Save the plot
    savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

    # Create beta plot against x_hat
    
    # Evaluate neural network and extract approximation
    y_hat = vec(beta_network(y_hat_input, p_trained, st_nn)[1])
    
    beta_against_xhat_plot = plot(I_grid, true_beta_against_xhat, color=:blue, linewidth=2, label="True beta", 
    xlabel="I/N", ylabel="Beta", title="Beta trajectory for $(sim_name) $(location)", legend=:topright)
    plot!(beta_against_xhat_plot, I_grid, y_hat, color=:red, linewidth=2, label="Predicted beta")

    # Save the plot
    savefig(beta_against_xhat_plot, joinpath(plot_dir, "beta_against_xhat_plot.png"))

    
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

dataset = JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_exponential", "synthesised_$(location).jld2"))

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
final_activation_function = sigmoid

beta_network, p_nn_temp, st_nn = build_neural_network(rng, hidden_dims, input_size, output_size, 
                            activation_function, final_activation_function)

seird_nn! = make_seird_nn(beta_network, st_nn, sigma, gamma, input_size)
prob_ude = ODEProblem(seird_nn!, u0, tspan, p_nn_temp)
predict_ude = make_predict_ude(prob_ude, train_length)

beta_function = "exponential"
maxiters_adam = 1
maxiters_lbfgs = 1
number_of_nn_input = 1

# Define strings for file names and directory for results
model_name = "ude_single"
sim_name ="UDE_single_location=$(location)_beta=$(beta_function)_adam=$(maxiters_adam)_lbfgs=$(maxiters_lbfgs)_number_of_nn_input=$(number_of_nn_input)"
if !isdir(datadir("exp_pro","sims", model_name, sim_name)) 
	mkpath(datadir("exp_pro","sims", model_name, sim_name))
end


for i = 1:1
    run_model(data, u0; maxiters_adam = maxiters_adam, maxiters_lbfgs = maxiters_lbfgs)
end