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
# Call the loss functions
include(joinpath(@__DIR__, "functions.jl"))
using .Functions

#========================================================
DEFINE HYPERPARAMETERS
=========================================================#

# Define strings for file names and directory for results
sim_name ="exponential_beta_single_traj"
model_name = "ude_single"
if !isdir(datadir("sims", model_name, sim_name)) 
	mkpath(datadir("sims", model_name, sim_name))
end

# Number of data points used for training (total number of entries in the dataset)
const train_length = 365

# set the number of hidden dimensions in the neural network equal to 3
hidden_dims = 5
# do 100 simulations 
n_sims = 50

#========================================================
LOAD DATA
=========================================================#

dataset = JLD2.load(datadir("synthetic_trajectories_exponential", "synthesised_MA.jld2"))

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

# Extract from estimated ground truths
include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

location = "MA"
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

# Define the timespan for the ODE solver
tspan = [1, train_length]

# Create neural network to estimate the transmission rate:
# We have two hidden layers with hidden_dims neurons and gelu activation function
# We are taking normalised I(t) as an input and outputting beta(t)
beta_network = Lux.Chain(Lux.Dense(1=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, softplus))

# Initialise parameters to build the structure for the UDE
p_nn_temp, st_nn = Lux.setup(rng, beta_network)

# Convert to ComponentArray for gradient-based optimisation
# ComponentArray wraps nested parameter structures into flat array keeping named access
p_nn_temp = ComponentArray(p_nn_temp)

# Create placeholder component array to set up neural network
ph_nn = ComponentArray(
                nn_params = p_nn_temp,
                gamma = gamma,
                sigma = sigma,
                delta = delta,
                tmax = train_length
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
    nn_input = [I / population]

    # Evaluate neural network and extract scalar
    beta = beta_network(nn_input, p.nn_params, st_nn)[1][1]

    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + delta) * I
    du[4] = gamma * I
    du[5] = delta * I
end

prob_ude = ODEProblem(seird_nn!, u0, tspan, ph_nn)

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


function train_ude(p; maxiters_adam, maxiters_lfbgs)

    # Set up optimisation (first Adam then LBFGS)
    optimised_state = Optimisers.setup(Optimisers.Adam(1e-3), p)

    # Create 1D vector to track losses during training
    losses = Float64[]
    best_loss = Inf
    best_p = p

    for iter in 1:maxiters_adam
        # Compute the loss, predicted mortalities and gradient function
        l, back_all = pullback(theta -> Functions.loss_ude(theta, predict_ude, data, u0), p)
        println("Iteration $iter, Loss: $l")
        # Evaluate the gradient of the loss w.r.t p
        grad = back_all((one(l)))[1]

    	# Stop training if 5 consecutive Inf losses
		if l == Inf && length(losses) >= 5 && all(isinf, losses[end-4:end])
			println("Unstable parameter region. Aborting...")
			break
		end 

		if isnothing(grad)
			println("No gradient found. Loss: $l")
			p = best_p
			continue
		end

        push!(losses, l)

        # Store best iteration
        if l < best_loss
            best_loss = l
            best_p = p
        end
        
        # Update parameters using the gradient
        optimised_state, p = Optimisers.update(optimised_state, p, grad)

    end

    # Then do LBFGS optimisation
    adtype = Optimization.AutoZygote()
    optfunc   = Optimization.OptimizationFunction(
                 (theta, _) -> Functions.loss_ude(theta, predict_ude, data, u0),
                 adtype)
    optprob = Optimization.OptimizationProblem(optfunc, best_p)

    iter_lbfgs = Ref(0)
    res = Optimization.solve(
        optprob,
        Optim.LBFGS(m=10),
        callback = (state, l) -> begin
            iter_lbfgs[] += 1
            push!(losses, l)

            if l < best_loss
                best_loss = l
                best_p = state.u
            end

            iter_lbfgs[] % 50 == 0 && println("LBFGS iter $(iter_lbfgs[]): $l")
            return false
        end,
        maxiters = maxiters_lfbgs
    )

    final_loss = Functions.loss_ude(res.u, predict_ude, data, u0)
    if final_loss < best_loss
        best_p = res.u
    end

    return best_p, losses
end

#========================================================
MAIN FUNCTION TO TRAIN THE UDE AND SAVE THE RESULTS
=========================================================# 

function run_model()
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
        tmax = train_length
    )

    # Make sure to start with a stable parameterization
    l_init = Functions.loss_ude(p_init, predict_ude, data, u0)
    println("Initial loss: $l_init")

    p_trained, losses_final = train_ude(p_init, maxiters_adam = 2500, maxiters_lfbgs = 2000)

    # Evaluate final long term results 
    long_term_prob= remake(prob_ude, p = p_trained, tspan = (1.0, 3*365.0), u0 = u0)
    long_term_pred = solve(long_term_prob, Rosenbrock23(), saveat=1, dense = false)

    beta_prediction = [beta_network([long_term_pred[3, i] / population], p_trained.nn_params, st_nn)[1][1] for i in 1:length(long_term_pred[3, :])]

    region = "Massachusetts"
    param_name = hidden_dims

	# Save the result
	fname = "$(region)_$(param_name)_t$(Threads.threadid())"

	# Append a number to the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
	model_iteration = 1
	while isdir(datadir("sims", model_name, sim_name, "$(fname)_v$(model_iteration)"))
		model_iteration += 1
	end
	fname = fname * "_v$model_iteration"

	mkpath(datadir("sims", model_name, sim_name, fname))


	JLD2.save(datadir("sims", model_name, sim_name, fname, "results.jld2"),
		"p", p_trained, "losses", losses_final, "prediction", Array(long_term_pred), "beta_prediction", beta_prediction,
		"days", days)
	println("Finished run: $(region) on thread $(Threads.threadid())")

	return nothing
end

for i = 1:n_sims
    run_model()
end