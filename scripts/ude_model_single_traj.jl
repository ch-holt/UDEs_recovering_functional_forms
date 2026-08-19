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
using Zygote
using Optimisers
using DifferentialEquations
using Plots
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
    p_init = ComponentArray(
        nn_params = p,
        gamma = gamma,
        sigma = sigma,
        delta = delta,
        tmax = train_length,
        population = population
    )

    training_data = data[1:train_length]

    p_trained, train_losses_final, val_losses_final = train_ude_single_dataset(p_init, predict_ude, training_data, u0, beta_function, location, noise, r; maxiters_adam=maxiters_adam, maxiters_lbfgs=maxiters_lbfgs, adam_learning_rate=adam_learning_rate)

    loc_foldername = "synthetic_$(location)"

    foldername = "simulation_seed=$(seed)"
	filename = "synthetic_$(location)"

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
    long_term_pred = solve(long_term_prob, Tsit5(), saveat=1, dense = false)

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
    # loss_traj_noisy: fit to the (possibly noisy) data the model actually trained on - diagnostic only
    # loss_traj_true: recovery of the noise-free trajectory - the headline metric for "did we recover the truth"
    loss_traj_noisy = loss_nmse(x_hat, data)
    loss_traj_true = loss_nmse(x_hat, true_data)

    traj_plot = plot(days[1:length(data)], data[1:length(data)], color=:black, markersize=2, label="Noisy data",
    xlabel="Day", ylabel="Infectious individuals", title="Infectious trajectory for $(location)", legend=:topright)
    plot!(traj_plot, days[1:length(true_data)], true_data, color=:gray, linestyle=:dash, linewidth=2, label="True trajectory")
    plot!(traj_plot, days[1:length(data)], x_hat, color=:red, linewidth=2, label="Predicted trajectory")
    annotate!(traj_plot, days[round(Int, length(data)/2)], maximum(data), text("NMSE (vs truth): $(round(loss_traj_true, sigdigits=3))", :black))

    # Save the plot
    savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

    # Create beta plot

    # Define beta function - always computed from the noise-free trajectory, never the noisy training data
    true_beta = beta_function(location, true_data)
    true_beta_against_xhat = beta_function(location, I_grid*p_init.population)

    loss_beta = loss_nmse(beta_traj, true_beta)

    beta_plot = plot(days[1:length(beta_traj)], true_beta, color=:blue, linewidth=2, label="True beta", 
    xlabel="Day", ylabel="Beta", title="Beta trajectory for $(location)", legend=:topright)
    plot!(beta_plot, days[1:length(beta_traj)], beta_traj, color=:red, linewidth=2, label="Predicted beta")
    annotate!(beta_plot, days[round(Int, length(beta_traj)/2)], maximum(true_beta), text("NMSE: $(round(loss_beta, sigdigits=3))", :black))

    # Save the plot
    savefig(beta_plot, joinpath(plot_dir, "beta_plot.png"))

    # Create beta plot against x_hat
    
    # Evaluate neural network and extract approximation
    y_hat = vec(beta_network(y_hat_input, p_trained.nn_params, st)[1])
    
    loss_I_grid = loss_nmse(y_hat, true_beta_against_xhat)

    # Identify I/N positions in the true trajectory where beta is minimum and maximum
    observed_I_over_N = true_data ./ p_init.population
    idx_beta_min = argmin(true_beta)
    idx_beta_max = argmax(true_beta)
    x_at_beta_min = observed_I_over_N[idx_beta_min]
    x_at_beta_max = observed_I_over_N[idx_beta_max]

    beta_against_xhat_plot = plot(I_grid, true_beta_against_xhat, color=:blue, linewidth=2, label="True beta", 
    xlabel="I/N", ylabel="Beta", title="Beta trajectory for $(location)", legend=:topright)
    plot!(beta_against_xhat_plot, I_grid, y_hat, color=:red, linewidth=2, label="Predicted beta")
    vline!(beta_against_xhat_plot, [x_at_beta_min], linestyle=:dot, color=:black, linewidth=2, label=false)
    vline!(beta_against_xhat_plot, [x_at_beta_max], linestyle=:dot, color=:gray40, linewidth=2, label=false)
    annotate!(beta_against_xhat_plot, I_grid[round(Int, length(y_hat)/2)], maximum(true_beta_against_xhat), text("NMSE: $(round(loss_I_grid, sigdigits=3))", :black))

    # Save the plot
    savefig(beta_against_xhat_plot, joinpath(plot_dir, "beta_against_xhat_plot.png"))

    # Create beta vs I/N plot restricted to the training region
    # x-axis is the actual I/N values visited during training (not a uniform grid)
    train_I = x_hat[1:train_length]
    train_I_over_N = train_I ./ p_init.population
    train_I_grid_input = Float64.(reshape(train_I_over_N, 1, train_length))

    y_hat_train = vec(beta_network(train_I_grid_input, p_trained.nn_params, st)[1])
    true_beta_train = beta_function(location, train_I)

    loss_I_train = loss_nmse(y_hat_train, true_beta_train)

    beta_against_train_plot = plot(train_I_over_N, true_beta_train, color=:blue, linewidth=2, label="True beta",
        xlabel="I/N", ylabel="Beta", title="Beta vs I/N (training region, $(location))", legend=:outertopright)
    plot!(beta_against_train_plot, train_I_over_N, y_hat_train, color=:red, linewidth=2, label="Predicted beta")
    annotate!(beta_against_train_plot, train_I_over_N[round(Int, train_length/2)], maximum(true_beta_train),
        text("NMSE: $(round(loss_I_train, sigdigits=3))", :black))

    savefig(beta_against_train_plot, joinpath(plot_dir, "beta_against_xhat_train_plot.png"))

    elapsed = time() - t_start

    mkpath(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername))
	JLD2.save(datadir("exp_pro","sims", model_name, sim_name, loc_foldername, foldername, "results.jld2"),
		"p", p_trained, "train_losses", train_losses_final, "val_losses", val_losses_final, "prediction", Array(long_term_pred), "beta_prediction", beta_traj,
		"days", days, "seed", seed, "noise", noise, "loss_traj_noisy", loss_traj_noisy, "loss_traj_true", loss_traj_true, "loss_beta", loss_beta, "loss_I_grid", loss_I_grid,
        "loss_I_train", loss_I_train, "elapsed_seconds", elapsed)

	println("Finished run: $(location) on thread $(Threads.threadid())")

	return nothing
end

#========================================================
DEFINE HYPERPARAMETERS
=========================================================#

# Number of data points used for training
const train_length = 365
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
DEFINE MODEL SETTINGS (outside location loop to avoid scoping issues)
=========================================================#

hidden_dims = 5
input_size = 1
output_size = 1
activation_function = gelu
final_activation_function = softplus
number_of_nn_inputs = 1
adam_learning_rate = 1e-3

beta_function = beta_rational

maxiters_adam = 2500
maxiters_lbfgs = 2000

noise = 0.1
r = noise == 0 ? Inf : 1 / noise^2

model_name = "ude_single"
sim_name = "UDE_single_beta=$(beta_function)_adam=$(maxiters_adam)_lbfgs=$(maxiters_lbfgs)_traindata=$(train_length)_noise=$(noise)"

if !isdir(datadir("exp_pro","sims", model_name, sim_name))
    mkpath(datadir("exp_pro","sims", model_name, sim_name))
end

# do 100 initialisations
for location in ["WY"]
    println("Running simulation for location: $(location)")

    #========================================================
    LOAD DATA
    =========================================================#

    dataset = JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_beta_exp", "synthetic_$(location)", "noise=$(noise).jld2"))
    true_dataset = noise == 0 ? dataset : JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_beta_exp", "synthetic_$(location)", "noise=0.0.jld2"))

    local data = dataset["infectious"]
    local true_data = true_dataset["infectious"]
    local days = dataset["days"]

    local population = POPULATION[location]
    local prevalence = PREVALENCE[location]
    local delta = DELTA[location]
    local R0_reproduction = R0_REPRODUCTION[location]
    local zeta = ZETA[location]

    # Derive other parameters
    local beta0 = R0_reproduction * (gamma + delta)
    local I0 = max(1.0, prevalence * population)
    local S0 = population - E0 - I0 - R0_recovered - D0

    # Define initial state
    local u0 = [S0, E0, I0, R0_recovered, D0]

    for i = 1:100
        # Catch any errors during the run so that the following seeds still run
        try
            local rng = Random.seed!(i)
            println("Running simulation for seed $(i) on thread $(Threads.threadid())")
            local beta_network, p_nn_temp, st_nn = build_neural_network(rng, hidden_dims, input_size, output_size,
                                        activation_function, final_activation_function)

            local seird_nn! = make_seird_nn(beta_network, st_nn, sigma, gamma, input_size)
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
end


