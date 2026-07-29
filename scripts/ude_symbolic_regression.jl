#========================================================
SCRIPT TO USE SYMBOLICREGRESSION.JL FOR SYMBOLIC REGRESSION
=========================================================#  

using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using JLD2
# Import symbolic regression package and MLJ interface
using SymbolicUtils 
using SymbolicRegression
using Symbolics
using DynamicExpressions
using Lux
using MLJ 
using SymbolicRegression
using Plots
using Statistics
using ComponentArrays
using OrdinaryDiffEq
using Random; rng = Random.default_rng()

# Call module
using UDE_FUNCTIONAL_FORMS

function symbolic_regression(sim_name,location, output_dir, obs, input_size)

    #========================================================
    SET UP NEURAL NETWORK
    =========================================================# 

    hidden_dims = 5

    output_size = 1
    activation_function = gelu
    final_activation_function = softplus

    beta_network, p_nn_temp, st_nn = build_neural_network(rng, hidden_dims, input_size, output_size, 
                                activation_function, final_activation_function)

    #=============================================================
    RETRIEVE PREDICTIONS AND PARAMETERS FROM THE BEST SIMULATION
    =============================================================#

    population = POPULATION[location]


    # Retrieve NN parameters that resulted in the lowest error on the training data
    I_nn, best_results = extract_best_ude(sim_name, location, obs, population)

    p_trained = best_results["p"]
    days = best_results["days"]


    #=============================================================
    DEFINE SR and NN INPUTS 
    =============================================================#

    # Extract NN approximation
    norm_i_traj = I_nn ./ population

    SR_input_1000, SR_input_days, nn_input_1000, nn_input_days, I_grid = build_sr_inputs(p_trained, norm_i_traj, input_size)

    # Evaluate neural network and extract approximation
    nn_output_1000 = vec(beta_network(nn_input_1000, p_trained.nn_params, st_nn)[1])
    nn_output_days = vec(beta_network(norm_i_traj, p_trained.nn_params, st_nn)[1])
    
    mach, r, best_equation, simplified_equation = run_symbolic_regression(SR_input_1000, nn_output_1000, output_dir; seed=1234)

    return mach, r, best_equation, simplified_equation, SR_input_1000, SR_input_days, nn_input_days, nn_output_days, nn_input_1000, nn_output_1000, I_grid, I_nn, norm_i_traj
end

#=============================================================
DEFINE PARAMETERS
=============================================================#

sim_name = "UDE_single_beta=beta_exp_adam=2500_learning_rate=0.001_lbfgs=2000_number_of_nn_input=1_finalactivation=softplus_test=first160_val=200"
location = "MA"
beta_function = beta_exp

# Number of NN inputs (single vs multiple datset settings)
input_size = 1

population = POPULATION[location]


# Load the observed data
data = JLD2.load(DrWatson.datadir("exp_pro","synthetic_data","synthetic_trajectories_beta_exp", "synthesised_$(location).jld2"))

# Just use infectious trajectory
obs = data["infectious"]
days = data["days"]

# Create output directory
output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)")
plot_title = "NN approximation no noise"

mach, r, best_equation, simplified_equation, SR_input_1000, SR_input_days, nn_input_days, nn_output_days, nn_input_1000, nn_output_1000, I_grid, I_nn, norm_i_traj = symbolic_regression(sim_name, location, output_dir, obs, input_size)

# Save both report and machine to the same file
JLD2.save(joinpath(output_dir, "SR_report.jld2"), "report", r, "mach", mach)

#=============================================================
MAKE PLOTS
=============================================================#

#=============================================================
PLOT BETA AGAINST DAYS
=============================================================#

# Evaluate the beta function for number of days
true_beta_days = beta_function(location, norm_i_traj.*population)

# Evaluate the SR equation for number of days
SR_beta_days = reshape(predict(mach, (data=SR_input_days, idx=r.best_idx)),:,1)

# Evaluate NN output for number of days
NN_beta_days = reshape(nn_output_days, :, 1)

nmse_SR_true_days = loss_nmse(SR_beta_days, true_beta_days)
nmse_SR_NN_days = loss_nmse(SR_beta_days, NN_beta_days)
nmse_NN_true_days = loss_nmse(NN_beta_days, true_beta_days)

p_comparison = plot(days, true_beta_days[1:length(days)], lw=2.5, label="True β", color=:black)

plot!(p_comparison, days, SR_beta_days[1:length(days)], lw=2.5, ls=:dash, label="SR-recovered β", color=:red, alpha=0.8)
plot!(p_comparison, days, NN_beta_days[1:length(days)], lw=2.5, ls=:dot, label="NN approximation", color=:lightblue, alpha=0.8)
xlabel!(p_comparison, "Day")    
ylabel!(p_comparison, "β")
title!(p_comparison, "Symbolic Regression on $(plot_title) vs true β\nEq: $(simplified_equation)", titlefontsize=8)
plot!(p_comparison, legend=:best, legendfontsize=10)
plot!(p_comparison, grid=true, gridalpha=0.3)

x_ann = days[end] * 0.75
y_ann = maximum(NN_beta_days) * 0.85
dy = maximum(NN_beta_days) * 0.06
annotate!(p_comparison, x_ann, y_ann, text("NMSE (NN vs SR) = $(round(nmse_SR_NN_days, sigdigits=3))", 9))
annotate!(p_comparison, x_ann, y_ann - dy, text("NMSE (SR vs true) = $(round(nmse_SR_true_days, sigdigits=3))", 9))
annotate!(p_comparison, x_ann, y_ann - 2*dy, text("NMSE (NN vs true) = $(round(nmse_NN_true_days, sigdigits=3))", 9))
savefig(p_comparison, joinpath(output_dir, "$(sim_name)_SR_beta_vs_true.png"))

#=============================================================
PLOT BETA BETEEN 0 AND 1
=============================================================#

# Evaluate the true beta function for the grid of 1000 points
true_beta_1000 = beta_function(location, I_grid.*population)

# Evaluate the SR equation for the grid of 1000 points
SR_beta_1000 = reshape(predict(mach, (data=SR_input_1000, idx=r.best_idx)),:,1)

# Evaluate the NN output for the grid of 1000 points
NN_beta_1000 = reshape(nn_output_1000, :, 1)

# Evaluate nmse
nmse_SR_true_1000 = loss_nmse(SR_beta_1000, true_beta_1000)
nmse_SR_NN_1000 = loss_nmse(SR_beta_1000, NN_beta_1000)
nmse_NN_true_1000 = loss_nmse(NN_beta_1000, true_beta_1000)

x_hat_vs_beta = plot(I_grid, true_beta_1000, lw=2.5, label="True β", color=:black)


plot!(x_hat_vs_beta, I_grid, SR_beta_1000, lw=2.5, ls=:dash, label="SR-recovered β", color=:red, alpha=0.8)
plot!(x_hat_vs_beta, I_grid, NN_beta_1000, lw=2.5, ls=:dot, label="NN approximation", color=:lightblue, alpha=0.8)
xlabel!(x_hat_vs_beta, "x̂ = I / N")
ylabel!(x_hat_vs_beta, "β")
title!(x_hat_vs_beta, "Symbolic Regression on $(plot_title) vs true β\nEq: $(simplified_equation)", titlefontsize=8)
plot!(x_hat_vs_beta, legend=:best, legendfontsize=10)
plot!(x_hat_vs_beta, grid=true, gridalpha=0.3)

I_N_obs_min = minimum(SR_input_days)
I_N_obs_max = maximum(SR_input_days)
vline!(x_hat_vs_beta, [I_N_obs_min], linestyle=:dash, color=:gray40, linewidth=1.5, label="Min observed I/N")
vline!(x_hat_vs_beta, [I_N_obs_max], linestyle=:dash, color=:gray70, linewidth=1.5, label="Max observed I/N")

x_ann = maximum(I_grid) * 0.75
y_ann = maximum(SR_beta_1000) * 0.85
dy = maximum(SR_beta_1000) * 0.06
annotate!(x_hat_vs_beta, x_ann, y_ann, text("NMSE (NN vs SR) = $(round(nmse_SR_NN_1000, sigdigits=3))", 9))
annotate!(x_hat_vs_beta, x_ann, y_ann - dy, text("NMSE (SR vs true) = $(round(nmse_SR_true_1000, sigdigits=3))", 9))
annotate!(x_hat_vs_beta, x_ann, y_ann - 2*dy, text("NMSE (NN vs true) = $(round(nmse_NN_true_1000, sigdigits=3))", 9))
savefig(x_hat_vs_beta, joinpath(output_dir, "$(sim_name)_SR_beta_against_x_hat.png"))

#=============================================================
RUN SEIRD + SR MODEL
=============================================================#

# Number of data points used for training (total number of entries in the dataset)
const train_length = 365
# Define the timespan for the ODE solver
tspan = [1, train_length]

# Define initial state

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

# Define ODE problem with recovered SR equation

p_sr = ComponentArray(population=population, delta=delta)
seird_sr! = make_seird_sr(mach, r, sigma, gamma, input_size)
prob_sr = ODEProblem(seird_sr!, u0, tspan, p_sr)
sol_sr = solve(prob_sr, Tsit5(), saveat=1.0)
i_sr = sol_sr[3, 1:length(obs)]

# Plot infection trajectory
i_traj_plot = plot(days, obs, lw=2.5, label="True observations", color=:black)

# Evaluate nmse
nmse_SR_true_traj = loss_nmse(i_sr, obs)
nmse_NN_true_traj = loss_nmse(I_nn, obs)

plot!(i_traj_plot, days, i_sr, lw=2.5, ls=:dash, label="SEIRD_SR model", color=:red, alpha=0.8)
plot!(i_traj_plot, days, I_nn, lw=2.5, ls=:dot, label="UDE model", color=:lightblue, alpha=0.8)
xlabel!(i_traj_plot, "Time")
ylabel!(i_traj_plot, "Infectious Individuals")
title!(i_traj_plot, "Infection Trajectory")
plot!(i_traj_plot, legend=:best, legendfontsize=10)
plot!(i_traj_plot, grid=true, gridalpha=0.3)

x_ann = maximum(days) * 0.75
y_ann = maximum(sol_sr) * 0.85
dy = maximum(sol_sr) * 0.06
annotate!(i_traj_plot, x_ann, y_ann - dy, text("NMSE (SEIRD + SR vs true) = $(round(nmse_SR_true_traj, sigdigits=3))", 9))
annotate!(i_traj_plot, x_ann, y_ann - 2*dy, text("NMSE (UDE vs true) = $(round(nmse_NN_true_traj, sigdigits=3))", 9))
savefig(i_traj_plot, joinpath(output_dir, "$(sim_name)_traj_plot.png"))


#=============================================================
LOSS AND DIFFERENCE PLOTS
=============================================================#

# Plot complexity against NMSE
complexities = r.complexities
losses = r.losses
log_losses = log10.(losses)
p_complexity = scatter(complexities, log_losses, label="SR equations", color=:blue)
xlabel!(p_complexity, "Complexity")
ylabel!(p_complexity, "log10(loss)")
title!(p_complexity, "SR complexity vs log10 loss against $(plot_title)\nEq: $(simplified_equation)", titlefontsize=8)
plot!(p_complexity, legend=:topright, legendfontsize=10)
plot!(p_complexity, grid=true, gridalpha=0.3)
plot!(p_complexity, complexities, log_losses, label=false, color=:blue, lw=2)

# Highlight the complexity chosen by the algorithm (r.best_idx) with a red dot
sel_idx = get(r, :best_idx, nothing)
if sel_idx !== nothing && 1 <= sel_idx <= length(complexities)
    sel_c = complexities[sel_idx]
    sel_l = log_losses[sel_idx]
    scatter!(p_complexity, [sel_c], [sel_l], marker=:circle, markersize=8, color=:red, label="selected equation")
end

savefig(p_complexity, joinpath(output_dir, "$(sim_name)_SR_complexity_vs_loss.png"))


# Plot differences against normalized x̂
NN_minus_true = NN_beta_1000 - true_beta_1000
SR_minus_true = SR_beta_1000 - true_beta_1000


p_diff_xhat = plot(I_grid, NN_minus_true, lw=2.5, ls=:dot, label="y_hat - true β", color=:lightblue, alpha=0.8)
plot!(p_diff_xhat, I_grid, SR_minus_true, lw=2.5, ls=:dot, label="SR result for y_hat - true β", color=:red)
xlabel!(p_diff_xhat, "x̂=I/N")
ylabel!(p_diff_xhat, "Difference in β")
title!(p_diff_xhat, "Difference between true β and y_hat = $(plot_title) and SR approx \nEq: $(simplified_equation)", titlefontsize=8)
plot!(p_diff_xhat, legend=:topright, legendfontsize=10)
plot!(p_diff_xhat, grid=true, gridalpha=0.3)

savefig(p_diff_xhat, joinpath(output_dir, "$(sim_name)_SR_NN_minus_true_beta_xhat.png"))
