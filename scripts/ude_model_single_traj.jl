#========================================================
SCRIPT TO TRAIN THE UDE MODEL FOR A SINGLE TRAJECTORY
=========================================================#  

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using Lux
using ComponentArrays
using DataFrames
using Zygote
using Optimisers
using DifferentialEquations
using JLD2
using Optimization
using OptimizationOptimJL
using SciMLSensitivity
using Random

# Call module
using UDE_FUNCTIONAL_FORMS

#========================================================
MAIN FUNCTION TO TRAIN THE UDE AND SAVE THE RESULTS
=========================================================# 

function run_model(sim_name, beta_function, location, data, true_data, train_length, u0, seed, predict_ude, beta_network, prob_ude, noise, r; maxiters_adam, maxiters_lbfgs, adam_learning_rate, number_of_nn_inputs=1, model_name, days, delta, population)
    println("Starting run: on thread $(Threads.threadid())")
    t_start = time()
    rng = Random.seed!(seed)

    # Initialise parameters
    p, st = Lux.setup(rng, beta_network)
    p = ComponentArray(p)
    p = Float64.(p)

    # Combine all parameters into a single object for optimisation
    p_init = ComponentArray(nn_params = p)

    training_data = data[1:train_length]

    p_trained, train_losses_final, val_losses_final = train_ude_single_dataset(p_init, predict_ude, training_data, u0, beta_function, location, noise, r; maxiters_adam=maxiters_adam, maxiters_lbfgs=maxiters_lbfgs, adam_learning_rate=adam_learning_rate)

    loc_foldername = "synthetic_$(location)"

    foldername = "simulation_seed=$(seed)"

    # Evaluate final long term results 
    long_term_prob= remake(prob_ude, p = p_trained, tspan = (1.0, 3*365.0), u0 = u0)
    long_term_pred = solve(long_term_prob, Tsit5(), saveat=1, dense = false)

    # Convert to a 1 x N matrix
    x_hat = long_term_pred[3, 1:length(data)]

    # Define the neural network input
    nT = length(x_hat)
    
    # Define input for SR via I_grid
    I_grid = collect(range(0, 1; length=1000))
    nI = length(I_grid)

    nn_input = Float64.(reshape(x_hat ./ population, 1, nT))
    y_hat_input = Float64.(reshape(I_grid, 1, nI))


    beta_traj = vec(beta_network(nn_input, p_trained.nn_params, st)[1])

    loss_traj_noisy = loss_nmse(x_hat, data)
    loss_traj_true  = loss_nmse(x_hat, true_data)

    true_beta = beta_function(location, true_data)
    loss_beta = loss_nmse(beta_traj, true_beta)

    y_hat = vec(beta_network(y_hat_input, p_trained.nn_params, st)[1])
    true_beta_against_xhat = beta_function(location, I_grid * population)
    loss_I_grid = loss_nmse(y_hat, true_beta_against_xhat)

    train_I = x_hat[1:train_length]
    train_I_over_N = train_I ./ population
    y_hat_train = vec(beta_network(Float64.(reshape(train_I_over_N, 1, train_length)), p_trained.nn_params, st)[1])
    true_beta_train = beta_function(location, train_I)
    loss_I_train = loss_nmse(y_hat_train, true_beta_train)

    mkpath(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername))

    elapsed = time() - t_start

    mkpath(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername))
	JLD2.save(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername, "results.jld2"),
		"location", location, "population", population, "p", p_trained, "train_losses", train_losses_final, "val_losses", val_losses_final, "prediction", Array(long_term_pred), "beta_prediction", beta_traj,
		"days", days, "seed", seed, "noise", noise, "loss_traj_noisy", loss_traj_noisy, "loss_traj_true", loss_traj_true, "loss_beta", loss_beta, "loss_I_grid", loss_I_grid,
        "loss_I_train", loss_I_train, "y_hat", y_hat, "y_hat_train", y_hat_train, "elapsed_seconds", elapsed)

	println("Finished run: $(location) on thread $(Threads.threadid())")

	return nothing
end

#========================================================
DEFINE HYPERPARAMETERS
=========================================================#


# beta functional form - default exp
const BETA_FUNCTIONS = Dict("beta_exp" => beta_exp, "beta_rational" => beta_rational, "beta_mixed" => beta_mixed)
const beta_function= BETA_FUNCTIONS[get(ARGS, 1, "beta_exp")]
# get noise - default 0
const noise = parse(Float64, get(ARGS, 2, "0.0"))
# Number of data points used for training - default 365
const train_length = parse(Int, get(ARGS, 3, "365"))

# print settings
println("Settings: beta=$(beta_function), noise=$(noise), train_length=$(train_length)")

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

#========================================================
DEFINE MODEL SETTINGS
=========================================================#

hidden_dims = 5
input_size = 1
output_size = 1
activation_function = gelu
final_activation_function = softplus
number_of_nn_inputs = 1
adam_learning_rate = 1e-3

maxiters_adam = 2500
maxiters_lbfgs = 2000

const r = noise == 0 ? Inf : 1 / noise^2

model_name = "ude_single"
sim_name = "train_val_midbeta_UDE_single_beta=$(beta_function)_adam=$(maxiters_adam)_lbfgs=$(maxiters_lbfgs)_traindata=$(train_length)_noise=$(noise)"

if !isdir(datadir("exp_pro","sims", model_name, sim_name))
    mkpath(datadir("exp_pro","sims", model_name, sim_name))
end

# do 100 initialisations for each location
# create an array job - each job is given a number so all jobs can run simultaneously
const LOCATIONS = sort(collect(keys(POPULATION)))
location = LOCATIONS[parse(Int, ENV["SLURM_ARRAY_TASK_ID"])]
println("Running simulation for location: $(location)")

#========================================================
LOAD DATA
=========================================================#

dataset = JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_$(beta_function)", "synthetic_$(location)", "noise=$(noise).jld2"))
true_dataset = noise == 0 ? dataset : JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_$(beta_function)", "synthetic_$(location)", "noise=0.0.jld2"))

data = dataset["infectious"]
true_data = true_dataset["infectious"]
days = dataset["days"]

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

# run the model on multiple threads
Threads.@threads for i = 1:100
    # Resume support: skip a seed whose results already exist, so a rerun (e.g. after
    # a timeout or crash) only computes the seeds still missing instead of starting
    # over from seed 1.
    results_path = datadir("exp_pro", "sims", model_name, sim_name, "synthetic_$(location)", "simulation_seed=$(i)", "results.jld2")
    if isfile(results_path)
        println("Seed $(i) already exists for $(location), skipping.")
        continue
    end
    # Catch any errors during the run so that the following seeds still run
    try
        local rng = Random.seed!(i)
        println("Running simulation for seed $(i) on thread $(Threads.threadid())")
        local beta_network, p_nn_temp, st_nn = build_neural_network(rng, hidden_dims, input_size, output_size,
                                    activation_function, final_activation_function)

        local seird_nn! = make_seird_nn(beta_network, st_nn, sigma, gamma, delta, population, input_size)
        local prob_ude = ODEProblem(seird_nn!, u0, tspan, p_nn_temp)
        local predict_ude = make_predict_ude(prob_ude, train_length)

        run_model(sim_name, beta_function, location, data, true_data, train_length, u0, i, predict_ude, beta_network, prob_ude, noise, r;
            maxiters_adam=maxiters_adam, maxiters_lbfgs=maxiters_lbfgs,
            number_of_nn_inputs=number_of_nn_inputs, adam_learning_rate=adam_learning_rate,
            model_name=model_name, days=days, delta=delta, population=population)
    catch e
        println("Error occurred for seed $(i): $e")
    end
end



