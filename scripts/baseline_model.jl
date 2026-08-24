#========================================================
SCRIPT TO FIT A CONSTANT BETA TO THE ODE
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
DEFINE HYPERPARAMETERS
=========================================================#

# Number of data points used for training
train_length = 365
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

adam_learning_rate = 1e-3

beta_function = beta_exp

maxiters_adam = 2500
maxiters_lbfgs = 2000

noise = 0.0
r = noise == 0 ? Inf : 1 / noise^2

model_name = "baseline_single"
sim_name = "baseline_single_beta=$(beta_function)_adam=$(maxiters_adam)_lbfgs=$(maxiters_lbfgs)_traindata=$(train_length)_noise=$(noise)"

if !isdir(datadir("exp_pro","sims", model_name, sim_name))
    mkpath(datadir("exp_pro","sims", model_name, sim_name))
end

# do 100 initialisations
for location in ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY",
 "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
 "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]
    println("Running simulation for location: $(location)")

    #========================================================
    LOAD DATA
    =========================================================#

    dataset = JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_$(beta_function)", "synthetic_$(location)", "noise=$(noise).jld2"))
    true_dataset = noise == 0 ? dataset : JLD2.load(datadir("exp_pro", "synthetic_data","synthetic_trajectories_$(beta_function)", "synthetic_$(location)", "noise=0.0.jld2"))

    local data = dataset["infectious"]
    local training_data = data[1:train_length]
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

    # Define parameter to be optimised
    p_init = ComponentArray(beta0 = beta0)

    prob = ODEProblem(make_seird_baseline(sigma, gamma, delta), u0, tspan, p_init)
    predict_ude = make_predict_ude(prob, train_length)

    beta0_hat, train_losses, elapsed = train_baseline_single_dataset(p_init, predict_ude, training_data, u0, noise, r; maxiters_adam=maxiters_adam, maxiters_lbfgs=maxiters_lbfgs, adam_learning_rate=adam_learning_rate)

    #========================================================
    UNCERTAINTY ON beta0 (Laplace / delta-method approximation)
    =========================================================#

    uncertainty = beta0_uncertainty(beta0_hat, predict_ude, training_data, u0, noise, r)
    println("beta0_hat = $(uncertainty.beta0_hat), 95% CI = [$(uncertainty.ci_lower), $(uncertainty.ci_upper)]")

    #========================================================
    SET UP OUTPUT DIRECTORIES
    =========================================================#

    loc_foldername = "synthetic_$(location)"
    plot_dir = plotsdir("sims", model_name, sim_name, loc_foldername)
    if !isdir(plot_dir)
        mkpath(plot_dir)
    end

    #========================================================
    TRAINING LOSS PLOT
    =========================================================#

    loss_plot = plot(train_losses, yscale=:log10, xlabel="Iteration", ylabel="Loss (log scale)",
        title="Training loss across iterations", label="Train", legend=:topright)
    savefig(loss_plot, joinpath(plot_dir, "training_loss_plot.png"))

    #========================================================
    LONG-TERM FORECAST
    =========================================================#

    long_term_prob = remake(prob, p = beta0_hat, tspan = (1.0, 3*365.0), u0 = u0)
    long_term_pred = solve(long_term_prob, Tsit5(), saveat=1, dense=false)

    x_hat = long_term_pred[3, 1:length(data)]
    nT = length(x_hat)

    #========================================================
    TRAJECTORY PLOT
    =========================================================#

    loss_traj_noisy = loss_nmse(x_hat, data)
    loss_traj_true = loss_nmse(x_hat, true_data)

    # Propagate the beta0 CI through the ODE, so the trajectory plot shows what the
    # uncertainty on beta0 implies for the epidemic curve itself, not just the parameter.
    lower_prob = remake(prob, p = ComponentArray(beta0 = uncertainty.ci_lower), tspan = (1.0, 3*365.0), u0 = u0)
    upper_prob = remake(prob, p = ComponentArray(beta0 = uncertainty.ci_upper), tspan = (1.0, 3*365.0), u0 = u0)
    x_lower = solve(lower_prob, Tsit5(), saveat=1, dense=false)[3, 1:length(data)]
    x_upper = solve(upper_prob, Tsit5(), saveat=1, dense=false)[3, 1:length(data)]

    traj_lo = min.(x_lower, x_upper)
    traj_hi = max.(x_lower, x_upper)
    traj_ribbon = (max.(x_hat .- traj_lo, 0.0), max.(traj_hi .- x_hat, 0.0))

    traj_plot = plot(days[1:length(data)], data[1:length(data)], color=:black, markersize=2, label="Noisy data",
        xlabel="Day", ylabel="Infectious individuals", title="Infectious trajectory for $(location)", legend=:topright)
    plot!(traj_plot, days[1:length(true_data)], true_data, color=:gray, linestyle=:dash, linewidth=2, label="True trajectory")
    plot!(traj_plot, days[1:length(data)], x_hat, ribbon=traj_ribbon, fillalpha=0.2, color=:red, linewidth=2, label="Predicted trajectory (95% CI)")
    annotate!(traj_plot, days[round(Int, length(data)/2)], maximum(data), text("NMSE (vs truth): $(round(loss_traj_true, sigdigits=3))", :black))

    savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

    #========================================================
    BETA COMPARISON PLOT (flat line vs true time-varying beta)
    =========================================================#

    beta_traj = fill(beta0_hat.beta0, nT)
    true_beta = beta_function(location, true_data)

    loss_beta = loss_nmse(beta_traj, true_beta)

    beta_plot = plot(days[1:nT], true_beta[1:nT], color=:blue, linewidth=2, label="True beta",
        xlabel="Day", ylabel="Beta", title="Beta trajectory for $(location)", legend=:topright)
    hspan!(beta_plot, [uncertainty.ci_lower, uncertainty.ci_upper], color=:red, alpha=0.15, label="95% CI")
    plot!(beta_plot, days[1:nT], beta_traj, color=:red, linewidth=2, label="Fitted constant beta0")
    annotate!(beta_plot, days[round(Int, nT/2)], maximum(true_beta), text("NMSE: $(round(loss_beta, sigdigits=3))", :black))

    savefig(beta_plot, joinpath(plot_dir, "beta_plot.png"))

    #========================================================
    BETA vs I/N PLOTS (constant beta0 vs true functional beta)
    =========================================================#

    I_grid = collect(range(0, 1; length=1000))
    nI = length(I_grid)

    true_beta_against_xhat = beta_function(location, I_grid .* population)
    y_hat = fill(beta0_hat.beta0, nI)

    loss_I_grid = loss_nmse(y_hat, true_beta_against_xhat)

    # Identify I/N positions in the true trajectory where beta is minimum and maximum
    observed_I_over_N = true_data ./ population
    idx_beta_min = argmin(true_beta)
    idx_beta_max = argmax(true_beta)
    x_at_beta_min = observed_I_over_N[idx_beta_min]
    x_at_beta_max = observed_I_over_N[idx_beta_max]

    beta_against_xhat_plot = plot(I_grid, true_beta_against_xhat, color=:blue, linewidth=2, label="True beta",
        xlabel="I/N", ylabel="Beta", title="Beta vs I/N for $(location)", legend=:topright)
    hspan!(beta_against_xhat_plot, [uncertainty.ci_lower, uncertainty.ci_upper], color=:red, alpha=0.15, label="95% CI")
    plot!(beta_against_xhat_plot, I_grid, y_hat, color=:red, linewidth=2, label="Fitted constant beta0")
    vline!(beta_against_xhat_plot, [x_at_beta_min], linestyle=:dot, color=:black, linewidth=2, label=false)
    vline!(beta_against_xhat_plot, [x_at_beta_max], linestyle=:dot, color=:gray40, linewidth=2, label=false)
    annotate!(beta_against_xhat_plot, I_grid[round(Int, nI/2)], maximum(true_beta_against_xhat), text("NMSE: $(round(loss_I_grid, sigdigits=3))", :black))

    savefig(beta_against_xhat_plot, joinpath(plot_dir, "beta_against_xhat_plot.png"))

    # Restrict to the I/N region actually visited during training
    train_I = x_hat[1:train_length]
    train_I_over_N = train_I ./ population
    y_hat_train = fill(beta0_hat.beta0, train_length)
    true_beta_train = beta_function(location, train_I)

    loss_I_train = loss_nmse(y_hat_train, true_beta_train)

    beta_against_train_plot = plot(train_I_over_N, true_beta_train, color=:blue, linewidth=2, label="True beta",
        xlabel="I/N", ylabel="Beta", title="Beta vs I/N (training region, $(location))", legend=:outertopright)
    hspan!(beta_against_train_plot, [uncertainty.ci_lower, uncertainty.ci_upper], color=:red, alpha=0.15, label="95% CI")
    plot!(beta_against_train_plot, train_I_over_N, y_hat_train, color=:red, linewidth=2, label="Fitted constant beta0")
    annotate!(beta_against_train_plot, train_I_over_N[round(Int, train_length/2)], maximum(true_beta_train),
        text("NMSE: $(round(loss_I_train, sigdigits=3))", :black))

    savefig(beta_against_train_plot, joinpath(plot_dir, "beta_against_xhat_train_plot.png"))

    #========================================================
    SAVE RESULTS
    =========================================================#

    data_dir = datadir("exp_pro", "sims", model_name, sim_name, loc_foldername)
    mkpath(data_dir)

    JLD2.save(joinpath(data_dir, "results.jld2"),
        "beta0_hat", beta0_hat.beta0, "se_beta0", uncertainty.se_beta0,
        "ci_lower", uncertainty.ci_lower, "ci_upper", uncertainty.ci_upper,
        "train_losses", train_losses,
        "prediction", Array(long_term_pred), "beta_prediction", beta_traj, "days", days, "noise", noise,
        "loss_traj_noisy", loss_traj_noisy, "loss_traj_true", loss_traj_true, "loss_beta", loss_beta,"loss_I_grid", loss_I_grid, "loss_I_train", loss_I_train,
        "elapsed_seconds", elapsed)

    println("Finished baseline run: $(location)")
end
