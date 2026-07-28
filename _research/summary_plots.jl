
#========================================================
SEED SENSITIVITY — OVERLAY PLOTS ACROSS ALL SEEDS
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using Plots
using Lux
using ComponentArrays
using Random
using ForecastBaselines
using UDE_FUNCTIONAL_FORMS

#========================================================
CONFIGURATION — edit these to match the sim you want
=========================================================#

location    = "MA"
model_name  = "ude_single"
sim_name    = "UDE_single_beta=beta_exp_adam=2500_learning_rate=0.001_lbfgs=2000_number_of_nn_input=1_finalactivation=softplus_test=first160_val=200"

# Must match the architecture used in training
hidden_dims          = 5
input_size           = 1
activation_fn        = gelu
final_activation_fn  = softplus

#========================================================
SET UP NETWORK (needed to evaluate beta vs I/N)
=========================================================#

rng = Random.default_rng()
beta_network, _, st_nn = build_neural_network(rng, hidden_dims, input_size, 1,
                                               activation_fn, final_activation_fn)
population = POPULATION[location]

#========================================================
FIND ALL SEED SIMULATION FOLDERS
=========================================================#

loc_foldername = "synthesised_$(location)"
sim_dir        = datadir("exp_pro", "sims", model_name, sim_name, loc_foldername)
seed_folders   = filter(f -> occursin(r"simulation_v1+_seed=", f), readdir(sim_dir))

println("Found $(length(seed_folders)) seed simulations in $(sim_name)")

#========================================================
LOAD SYNTHETIC DATA AND COMPUTE TRUE BETA
=========================================================#

dataset = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_beta_exp",
                             "synthesised_$(location).jld2"))
data = dataset["infectious"]
days = collect(dataset["days"])

I_grid             = collect(range(0, 1; length=1000))
true_beta_over_time = beta_exp(location, data)
true_beta_01        = beta_exp(location, I_grid .* population)

# I/N range of observed training data (for vertical lines on beta_01_plot)
observed_I_over_N = data ./ population
I_N_min = minimum(observed_I_over_N)
I_N_max = maximum(observed_I_over_N)

#========================================================
INITIALISE PLOTS WITH TRUE BETA AS REFERENCE
=========================================================#

traj_plot = plot(days[1:length(data)], data;
    color=:black, linewidth=2, label="Data",
    xlabel="Day", ylabel="Infectious individuals",
    title="Infectious trajectory — ($(location))",
    legend=:topright)

beta_time_plot = plot(days[1:length(true_beta_over_time)], true_beta_over_time;
    color=:black, linewidth=2, label="True β",
    xlabel="Day", ylabel="β(t)",
    title="Beta over time — ($(location))",
    legend=:topright)

beta_01_plot = plot(I_grid, true_beta_01;
    color=:black, linewidth=2, label="True β",
    xlabel="I/N", ylabel="β(I/N)",
    title="Beta vs I/N — ($(location))",
    legend=:topright)
vline!(beta_01_plot, [I_N_min]; linestyle=:dot, color=:black, linewidth=2, label="Min I/N (observed)")
vline!(beta_01_plot, [I_N_max]; linestyle=:dot, color=:gray40, linewidth=2, label="Max I/N (observed)")

#========================================================
OVERLAY EACH SEED
=========================================================#

inf_trajectories_rows = Vector{Vector{Float64}}()
beta_trajectories_rows = Vector{Vector{Float64}}()
beta_01_trajectories_rows = Vector{Vector{Float64}}()

for folder in seed_folders
    result_path = joinpath(sim_dir, folder, "results.jld2")
    if !isfile(result_path)
        println("Skipping $(folder) — results.jld2 not found")
        continue
    end

    d         = JLD2.load(result_path)
    pred      = d["prediction"]        # (5, 3*365)
    beta_traj = d["beta_prediction"]   # length 365
    p_trained = d["p"]

    # 1. Infectious trajectory
    x_hat = vec(pred[3, 1:length(data)])
    plot!(traj_plot, days[1:length(x_hat)], x_hat;
          color=:red, alpha=0.3, linewidth=1, label="")

    # 2. Beta over time
    plot!(beta_time_plot, days[1:length(beta_traj)], beta_traj;
          color=:red, alpha=0.3, linewidth=1, label="")

    # 3. Beta vs I/N — re-evaluate NN on uniform grid
    nn_input = Float64.(reshape(I_grid, 1, length(I_grid)))
    y_hat    = vec(beta_network(nn_input, p_trained.nn_params, st_nn)[1])
    plot!(beta_01_plot, I_grid, y_hat;
          color=:red, alpha=0.3, linewidth=1, label="")
    
    # Store trajectories for CRPS 
    push!(inf_trajectories_rows, x_hat)
    push!(beta_trajectories_rows, beta_traj)
    push!(beta_01_trajectories_rows, y_hat)
end

#========================================================
CALCULATE CRPS
=========================================================#

# Convert to matrices
inf_trajectories = stack(inf_trajectories_rows; dims=1)
beta_trajectories = stack(beta_trajectories_rows; dims=1)
beta_01_trajectories = stack(beta_01_trajectories_rows; dims=1)

# Add all forecasts
inf_fc = Forecast(horizon = collect(1:length(data)), trajectories = inf_trajectories)
# Add the true trajectory
inf_fc = add_truth(inf_fc, data)
# Calculate CRPS
inf_crps  = score(inf_fc, CRPS_trajectory())

beta_fc = Forecast(horizon = collect(1:length(true_beta_over_time)), trajectories = beta_trajectories)
beta_fc = add_truth(beta_fc, true_beta_over_time)
beta_crps  = score(beta_fc, CRPS_trajectory())

beta_01_fc = Forecast(horizon = collect(1:length(I_grid)), trajectories = beta_01_trajectories)
beta_01_fc = add_truth(beta_01_fc, true_beta_01)
beta_01_crps  = score(beta_01_fc, CRPS_trajectory())

# Add CRPS to plot titles
plot!(traj_plot,title="Infectious trajectory — ($(location))\nCRPS: $(round(inf_crps, sigdigits=3))")
plot!(beta_time_plot,title="Beta over time — ($(location))\nCRPS: $(round(beta_crps, sigdigits=3))")
plot!(beta_01_plot,title="Beta vs I/N — ($(location))\nCRPS: $(round(beta_01_crps, sigdigits=3))")

#========================================================
SAVE
=========================================================#

save_dir = plotsdir("sims", model_name, sim_name, "synthesised_$(location)")
mkpath(save_dir)

savefig(traj_plot,      joinpath(save_dir, "traj_overlay.png"))
savefig(beta_time_plot, joinpath(save_dir, "beta_time_overlay.png"))
savefig(beta_01_plot,   joinpath(save_dir, "beta_01_overlay.png"))

panel = plot(traj_plot, beta_time_plot, beta_01_plot; layout=(1, 3), size=(1800, 500))
savefig(panel, joinpath(save_dir, "panel_overlay.png"))

println("Done — plots saved to:\n  $(save_dir)")
