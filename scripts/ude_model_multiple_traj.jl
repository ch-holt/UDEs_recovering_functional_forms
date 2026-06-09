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
# Call the loss functions
include(joinpath(@__DIR__, "functions.jl"))
using .Functions

#========================================================
DEFINE HYPERPARAMETERS
=========================================================# 

# Number of data points used for training (total number of entries in the dataset)
const train_length = 365
const maxiters = 2500
# set the number of hidden dimensions in the neural network equal to 3
hidden_dims = 5

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

init_state = [0.0, E0, 0.0, R0_recovered, D0]

#========================================================
SET UP MODEL
=========================================================#

# Define the timespan for the ODE solver
tspan = [0, train_length]

# Create neural network to estimate the transmission rate:
# We have two hidden layers with hidden_dims neurons and gelu activation function
# We are taking beta0, zeta, dleta, I(t) and t as inputs and outputting beta(t)
beta_network = Lux.Chain(Lux.Dense(4=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, softplus))

# Initialise placeholder parameters to build the structure for the UDE
p_nn_temp, st_nn = Lux.setup(rng, beta_network)

# Convert to ComponentArray for gradient-based optimisation
# ComponentArray wraps nested parameter structures into flat array keeping named access
p_nn_temp = ComponentArray(p_nn_temp)


# Create placeholder component array to set up neural network
ph_nn = ComponentArray(
                nn_params = p_nn_temp,
                population = 0,
                prevalence = 0,
                beta0 = 0,
                zeta = 0,
                R0_reproduction = 0,
                delta = 0
            )

# Define the model 
function seird_nn!(du, u, p, t)
    S, E, I, R, D = u

    # Define the population size
    N = S + E + I + R

    # If population size less than or equal to zero return zero
    if N <= 0
        du .= 0.0
        return
    end

    # Define normalised inputs for NN
    delta_norm =(log(p.delta) - log(1e-6))/(log(1e-2) - log(1e-6))
    beta0_norm = (p.R0_reproduction - 1.2)/(6.0 - 1.2)
    zeta_norm = p.zeta/0.05
    nn_input = [beta0_norm, zeta_norm, delta_norm, I / p.population]

    # Evaluate neural network and extract scalar
    beta = beta_network(nn_input, p.nn_params, st_nn)[1][1]

    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + p.delta) * I
    du[4] = gamma * I
    du[5] = p.delta * I
end

prob_ude = ODEProblem(seird_nn!, init_state, tspan, p_nn_temp)

#========================================================
PREDICTION AND LOSS FUNCTIONS
=========================================================# 

# Predict number infectious individuals
function predict_ude(p_all, u0)

    prob = remake(prob_ude, p = p_all, u0 = u0)
    sol_ude = solve(prob, Rosenbrock23(), saveat=1.0, dense = false)

    # Return nothing if solve failed or didn't reach full timespan
    if sol_ude.retcode != ReturnCode.Success || length(sol_ude.t) < train_length
        return nothing
    end
    
    I_pred = sol_ude[3, 1:train_length]

    return I_pred
end

# Loss function in functions.jl module

#========================================================
TRAINING
=========================================================# 

function train_ude(nn_params; maxiters_adam, maxiters_lbfgs)

    # Preload all trajectories and store in a vector
    root = datadir("synthetic_trajectories_exponential")
    trajectories = []
    for filename in readdir(root)
        if endswith(filename, ".jld2")
            dataset = JLD2.load(joinpath(root, filename))
            varying_p = ComponentArray(
                population = dataset["varying_p"]["population"],
                prevalence = dataset["varying_p"]["prevalence"],
                delta = dataset["varying_p"]["delta"],
                R0_reproduction = dataset["varying_p"]["R0_reproduction"],
                zeta = dataset["varying_p"]["zeta"]
            )

            push!(trajectories, (
                filename = filename,
                data = dataset["infectious"],
                days = dataset["days"],
                varying_p = varying_p
            ))
        end
    end
    println("Loaded $(length(trajectories)) trajectories into memory")

    # Create 1D vector to track total losses across all trajectories during training
    total_losses = Float64[]
    best_loss = Inf

    # We track the best parameters for all trajectories
    # the relationship between the parameters (e.g. NN weights) is the same across trajectories, but the NN will have different inputs
    best_nn_params = nn_params

    # Set up optimisation
    optimised_state = Optimisers.setup(Optimisers.Adam(1e-3), nn_params)

    for iter in 1:maxiters_adam

        # Print progress so long-running evaluations are visible
        println("Adam iter $iter")

        # Compute the loss, predicted infectious individuals and gradient function for each synthetic trajectory
        total_loss, total_grad = Functions.combined_loss_ude_adam(nn_params, predict_ude, trajectories)

    	# Stop training if 5 consecutive Inf losses
		if total_loss == Inf && length(total_losses) >= 4 && all(isinf, total_losses[end-4:end])
			println("Unstable parameter region. Aborting...")
			break
		end 

		if isnothing(total_grad)
			println("No gradient found. Loss: $total_loss")
			nn_params = best_nn_params
			continue
		end   

        push!(total_losses, total_loss)

        # Store best iteration
        if total_loss < best_loss
            best_loss = total_loss
            best_nn_params = nn_params
        end

        # Update parameters using the gradient
        optimised_state, nn_params = Optimisers.update(optimised_state, nn_params, total_grad)

    end

    # Then do LBFGS optimisation
    adtype = Optimization.AutoZygote()
    optfunc   = Optimization.OptimizationFunction(
                (theta, _) -> Functions.combined_loss_ude_lbfgs(theta, predict_ude, trajectories),
                adtype)
    optprob = Optimization.OptimizationProblem(optfunc, best_nn_params)

    iter_lbfgs = Ref(0)

    res = try
        Optimization.solve(
        optprob,
        Optim.LBFGS(m=10),
        callback = (state, l) -> begin
            iter_lbfgs[] += 1
            push!(total_losses, l)

            if l < best_loss
                best_loss = l
                best_nn_params = state.u
            end

            iter_lbfgs[] % 50 == 0 && println("LBFGS iter $(iter_lbfgs[]): $l")
            return false
        end,
        maxiters = maxiters_lbfgs
        )

    catch e
        println("LBFGS optimisation failed with error: $e")
        return best_nn_params, total_losses
    end

    final_loss = Functions.combined_loss_ude_lbfgs(res.u, predict_ude, trajectories)
    if final_loss < best_loss
        best_nn_params = res.u
    end

    return best_nn_params, total_losses
end

#========================================================
MAIN FUNCTION TO TRAIN THE UDE AND SAVE THE RESULTS
=========================================================# 

function run_model(POPULATION, beta_function, maxiters_adam, maxiters_lbfgs)
    println("Starting run: on thread $(Threads.threadid())")

    # Initialise parameters
    nn_params, st = Lux.setup(rng, beta_network)
    nn_params = ComponentArray(nn_params)

    p_trained, losses_final = train_ude(nn_params; maxiters_adam, maxiters_lbfgs)

    # Save the trained parameters and losses for the combined trajectories

    # Create numbered simulation folders to allow multiple runs of a single set of hyperparameters 
    model_iteration = 1
    while isdir(datadir("sims", model_name, sim_name, "simulation_v$(model_iteration)"))
        model_iteration += 1
    end
    foldername = "simulation_v$model_iteration"

    save(datadir("sims", model_name, sim_name, foldername, "training_results.jld2"), 
    "p_trained", p_trained, "losses_final", losses_final)

    # Evaluate the trained model on each trajectory and save the results
    # Loop through all simulations
    # Define the root file path
    root = datadir("synthetic_trajectories_exponential")
    # Read all files/folders in the root directory
    for location in keys(POPULATION)
        filename = "synthesised_$(location).jld2"
        # Extract trajectory of infectious individuals
        dataset = JLD2.load(datadir("synthetic_trajectories_exponential", filename))
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
        long_term_pred = solve(long_term_prob, Rosenbrock23(), saveat=1, dense = false)
        
        # Convert to a 1 x N matrix
        x_hat = long_term_pred[3, 1:length(data)]

        # Define the neural network input for the current trajectory 
        # Keep beta, zeta, delta constant over time but normalise
        delta_norm =(log(p_all.delta) - log(1e-6))/(log(1e-2) - log(1e-6))
        beta0_norm = (p_all.R0_reproduction - 1.2)/(6.0 - 1.2)
        zeta_norm = p_all.zeta/0.05

        nn_input = vcat(fill(beta0_norm, 1, length(days)), fill(zeta_norm, 1, length(days)), 
            fill(delta_norm, 1, length(days)), reshape(x_hat ./ p_all.population, 1, :))

        # Evaluate neural network and extract approximation
        beta_traj = vec(beta_network(nn_input, p_trained, st_nn)[1])

        # Within this folder create a folder for each trajectory
        if !isdir(datadir("sims", model_name, sim_name, foldername, filename)) 
            mkpath(datadir("sims", model_name, sim_name, foldername, filename))
        end
        root = datadir("sims", model_name, sim_name, foldername, filename, "results.jld2")
        # In this folder save the infectious trajectory results and the beta results for this trajectory
        JLD2.save(root,
            "p", p_trained, "losses", losses_final, "prediction", Array(long_term_pred), "beta_prediction", beta_traj,
            "days", days)
    
        #========================
        CREATE PLOTS
        ========================#

        plot_dir = dirname(root)

        # Create trajectory plot
        traj_plot = plot(days[1:length(data)], data[1:length(data)], color=:black, markersize=2, label="Data", 
        xlabel="Day", ylabel="Infectious individuals", title="Infectious trajectory for $(sim_name) $(location)", legend=:topright)
        plot!(traj_plot, days[1:length(data)], x_hat, color=:red, linewidth=2, label="Predicted trajectory")

        # Save the plot
        savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

        # Create beta plot

        # Define beta function
        I_grid = collect(range(0, 1; length=1000))
        if beta_function == "exponential"
            true_beta = Functions.beta_exp(location, data),:,1
            true_beta_against_xhat = Functions.beta_exp(location, I_grid*p_all.population)
        elseif beta_function == "rational"
            true_beta = Functions.beta_rational(location, data),:,1
            true_beta_against_xhat = Functions.beta_rational(location, I_grid*p_all.population)
        end

        beta_plot = plot(days[1:length(beta_traj)], true_beta, color=:blue, linewidth=2, label="True beta", 
        xlabel="Day", ylabel="Beta", title="Beta trajectory for $(sim_name) $(location)", legend=:topright)
        plot!(beta_plot, days[1:length(beta_traj)], beta_traj, color=:red, linewidth=2, label="Predicted beta")

        # Save the plot
        savefig(beta_plot, joinpath(plot_dir, "beta_plot.png"))

        # Save the plot
        savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

        # Create beta plot against x_hat
        
        # Evaluate neural network and extract approximation
        y_hat_input = vcat(fill(beta0_norm, 1, 1000), fill(zeta_norm, 1, 1000), 
            fill(delta_norm, 1, 1000), reshape(I_grid, 1, :))

        y_hat = vec(beta_network(y_hat_input, p_trained, st_nn)[1])
        
        beta_against_xhat_plot = plot(I_grid, true_beta_against_xhat, color=:blue, linewidth=2, label="True beta", 
        xlabel="Day", ylabel="Beta", title="Beta trajectory for $(sim_name) $(location)", legend=:topright)
        plot!(beta_against_xhat_plot, I_grid, y_hat, color=:red, linewidth=2, label="Predicted beta")

        # Save the plot
        savefig(beta_against_xhat_plot, joinpath(plot_dir, "beta_against_xhat_plot.png"))

    end
    println("Finished run: $(foldername) on thread $(Threads.threadid())")
	return nothing
end

include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION

# Define strings for file names and directory for results
sim_name ="exponential_normalised_inputs_090626"
model_name = "ude_multiple"
if !isdir(datadir("sims", model_name, sim_name)) 
	mkpath(datadir("sims", model_name, sim_name))
end

for i = 1:1
    run_model(POPULATION, "exponential", 2500, 0)
end


