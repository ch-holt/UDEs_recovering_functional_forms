#========================================================
SCRIPT TO TRAIN THE UDE MODEL WOITH BEHAVIOURAL DATA
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
using Distributions

# Call module
using UDE_FUNCTIONAL_FORMS

#========================================================
MAIN FUNCTION TO TRAIN THE UDE AND SAVE THE RESULTS
=========================================================# 

function run_model(sim_name, contact_function, beta_function, rho_true, location, data_I, true_data_I, data_c, train_length, u0, seed, predict_ude, contact_network, prob_ude, noise, r, solver; maxiters_adam, maxiters_lbfgs, adam_learning_rate, number_of_nn_inputs=1, model_name, days, delta, population)
    println("Starting run: on thread $(Threads.threadid())")
    t_start = time()
    rng = Random.seed!(seed)

    # Initialise parameters
    p, st = Lux.setup(rng, contact_network)
    p = ComponentArray(p)
    p = Float64.(p)

    # Random starting noise level for each seed (multistart), log-uniform on [1e-3, 1]
    phi_init = exp(rand(rng, Uniform(log(1e-3), log(1.0))))

    # Random starting rho for each seed, 
    # Rho is a probability so it's uniform on [0, 1], stored on logit scale
    rho_init = rand(rng, Uniform(0.0, 1.0))

    # Combine all parameters into a single object for optimisation
    # We use the logit function to put rho on the real line for optimisation
    p_init = noise == 0 ? ComponentArray(nn_params = p, logit_rho = log(rho_init / (1 - rho_init))) :
                          ComponentArray(nn_params = p, log_phi = log(phi_init), logit_rho = log(rho_init / (1 - rho_init)))

    training_data_I = data_I[1:train_length]
    training_data_c = data_c[1:train_length]
    loss_fn = (θ, tpts) -> loss_ude_behaviour(θ, predict_ude, training_data_I, training_data_c, u0, tpts, noise)
    p_trained, train_losses_final, val_losses_final = train_ude_single_dataset(p_init, loss_fn, length(training_data_I); maxiters_adam=maxiters_adam, maxiters_lbfgs=maxiters_lbfgs, adam_learning_rate=adam_learning_rate)

    # Recover the jointly estimated noise level (only estimated when noise>0)
    phi_hat = noise == 0 ? missing : exp(p_trained.log_phi)
    rho_hat = 1 / (1 + exp(-p_trained.logit_rho))

    loc_foldername = "synthetic_$(location)"

    foldername = "simulation_seed=$(seed)"

    # Evaluate final long term results
    long_term_prob= remake(prob_ude, p = p_trained, tspan = (1.0, 3*365.0), u0 = u0)
    long_term_pred = solve(long_term_prob, solver, saveat=1, dense = false)

    # Long-term predictions
    x_hat = long_term_pred[3, 1:length(data_I)]
    c_hat = long_term_pred[6, 1:length(data_I)]

    # Derive the NN inputs
    E_hat  = long_term_pred[2, :]
    I_hat  = long_term_pred[3, :]
    c_hat_full  = long_term_pred[6, :]
    dI_hat = sigma .* E_hat .- (gamma + delta) .* I_hat

    # NN inputs for the contact network
    nn_input_traj = permutedims(hcat(I_hat ./ population, dI_hat, c_hat_full))
    g_traj = vec(contact_network(nn_input_traj, p_trained.nn_params, st)[1])

    # Infectious trajectory fit
    loss_traj_noisy = loss_nmse(x_hat, data_I)
    loss_traj_true  = loss_nmse(x_hat, true_data_I)

    # Contact rate fit
    loss_contacts = loss_nmse(c_hat, data_c)

    # Transmission rate fit: beta(t) = p * c(t)
    beta_hat  = rho_hat .* c_hat
    true_beta = beta_function(location, true_data_I)
    loss_beta = loss_nmse(beta_hat, true_beta)

    # Fit of p
    rho_rel_error = (rho_hat - rho_true) / rho_true

    elapsed = time() - t_start

    mkpath(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername))
	JLD2.save(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername, "results.jld2"),
		"location", location, "population", population, "p", p_trained, "train_losses", train_losses_final, "val_losses", val_losses_final, "prediction", Array(long_term_pred), "beta_prediction", beta_hat,
		"contact_prediction", c_hat, "NN_approx", g_traj, "days", days, "seed", seed, "noise", noise, "loss_traj_noisy", loss_traj_noisy, "loss_traj_true", loss_traj_true, "loss_contacts", loss_contacts, "loss_beta", loss_beta,
        "phi_init", phi_init, "phi_hat", phi_hat, "rho_init", rho_init, "rho_hat", rho_hat, "rho_true", rho_true, "rho_rel_error", rho_rel_error,
        "elapsed_seconds", elapsed)

	println("Finished run: $(location) on thread $(Threads.threadid())")

	return nothing
end


#========================================================
DEFINE HYPERPARAMETERS
=========================================================#


# beta functional form - default exp
const CONTACT_FUNCTIONS = Dict("contacts_exp" => contacts_exp, "contacts_rational" => contacts_rational, "contacts_mixed" => contacts_mixed)
const contact_function = get(ARGS, 6, "contacts_exp")
# get noise - default 0
const noise = parse(Float64, get(ARGS, 2, "0.0"))
# Number of data points used for training - default 365
const train_length = parse(Int, get(ARGS, 3, "365"))

# ODE solver for UDE training/prediction - AutoTsit5(Rosenbrock23()) with default tolerances
const solver = AutoTsit5(Rosenbrock23())

# Number of seeds per location (seeds 1:n_seeds) - default 100; fewer only for timing runs
const n_seeds = parse(Int, get(ARGS, 5, "100"))

# print settings
println("Settings: contact=$(contact_function), noise=$(noise), train_length=$(train_length)")

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
input_size = 3
output_size = 1
activation_function = gelu
# Final activation function. The NN is representing change in contact rate which can be positive or negative -> use the identity function
final_activation_function = identity
number_of_nn_inputs = 1
adam_learning_rate = 1e-3

maxiters_adam = 2500
maxiters_lbfgs = 2000

const r = noise == 0 ? Inf : 1 / noise^2



model_name = "ude_behaviour_model1"
sim_name = "UDE_behaviour_model1_contact=$(contact_function)_traindata=$(train_length)_noise=$(noise)"

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

# Convert contact function to beta function
const BETA_FUNCTIONS = Dict("beta_exp" => beta_exp, "beta_rational" => beta_rational, "beta_mixed" => beta_mixed)
beta_name = replace(contact_function, "contacts" => "beta")
beta_function = BETA_FUNCTIONS[beta_name]

dataset_I = JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_HQ_$(beta_name)", "synthetic_$(location)", "noise=$(noise).jld2"))
dataset_c = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_contact_trajectories_HQ","ground_truth_$(contact_function)", "location_$(location)","synthetic_$(contact_function)_$(location)_over_time.jld2"))
true_dataset_I = noise == 0 ? dataset_I : JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_HQ_$(beta_name)", "synthetic_$(location)", "noise=0.0.jld2"))
# For when noisy behaviour data is introduced: true_dataset_c = noise == 0 ? dataset : JLD2.load(datadir())

data_I = dataset_I["infectious"]
data_c = dataset_c["contact"]
true_data_I = true_dataset_I["infectious"]
# For when noisy behaviour data is introduced: true_data_c = true_dataset_c["contacts"]
days = dataset_I["days"]

population = POPULATION[location]
prevalence = PREVALENCE[location]
delta = DELTA[location]
R0_reproduction = R0_REPRODUCTION[location]
zeta = ZETA[location]

# Derive other parameters
beta0 = R0_reproduction * (gamma + delta)
I0 = max(1.0, prevalence * population)
S0 = population - E0 - I0 - R0_recovered - D0
# Base contact rate for initial conditions - assume known
c0 = C0[location]
# True transmission probability per contact
rho_true = beta0 / c0

# Define initial state
u0 = [S0, E0, I0, R0_recovered, D0, c0]

# run the model on multiple threads
Threads.@threads for i = 1:n_seeds
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
        # Build neural network with three inputs - prevalence, incidence and contacts
        local contact_network, p_nn_temp, st_nn = build_neural_network(rng, hidden_dims, input_size, output_size,
                                    activation_function, final_activation_function)

        local seird_nn! = make_seird_nn_behaviour_model1(contact_network, st_nn, sigma, gamma, delta, population)
        local prob_ude = ODEProblem(seird_nn!, u0, tspan, p_nn_temp)
        # make prediction but return all states
        local predict_ude = make_predict_ude(prob_ude, train_length, solver; states=:)

        run_model(sim_name, contact_function, beta_function, rho_true, location, data_I, true_data_I, data_c,train_length, u0, i, predict_ude, contact_network, prob_ude, noise, r, solver;
            maxiters_adam=maxiters_adam, maxiters_lbfgs=maxiters_lbfgs,
            number_of_nn_inputs=number_of_nn_inputs, adam_learning_rate=adam_learning_rate,
            model_name=model_name, days=days, delta=delta, population=population)
    catch e
        println("Error occurred for seed $(i): $e")
    end
end



