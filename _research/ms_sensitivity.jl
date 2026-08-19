
#========================================================
MULTISTART SENSITIVITY — PANEL OF OVERLAYS + CRPS TABLE
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using Plots
using Images
using FileIO
using JLD2
using Lux
using ComponentArrays
using Random
using ForecastBaselines
using DataFrames
using CSV
using UDE_FUNCTIONAL_FORMS

#========================================================
CONFIGURATION
=========================================================#

location   = "MA"
model_name = "ude_single"
sim_name   = "UDE_single_tsit5_beta=beta_exp_adam=2500_learning_rate=0.001_lbfgs=2000_number_of_nn_input=1_finalactivation=softplus_traindata=365_mid20beta"
noise = 0

MS_limits  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Must match training architecture
hidden_dims         = 5
input_size          = 1
activation_fn       = gelu
final_activation_fn = softplus

#========================================================
SET UP NETWORK AND TRUE DATA
=========================================================#

rng = Random.default_rng()
beta_network, _, st_nn = build_neural_network(rng, hidden_dims, input_size, 1,
                                               activation_fn, final_activation_fn)
population = POPULATION[location]

# Extract the ground truth (no noise)
dataset = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_beta_exp",
                             "synthetic_$(location)", "noise=0.jld2"))
true_inf = dataset["infectious"]
days     = collect(dataset["days"])

I_grid              = collect(range(0, 1; length=1000))
true_beta_over_time = beta_exp(location, true_inf)
true_beta_01        = beta_exp(location, I_grid .* population)

# Extract the sim with the current noise
sim_dir = datadir("exp_pro", "sims", model_name, sim_name, "synthetic_$(location)", "noise=$(noise).jld2")
src_dir = plotsdir("sims", model_name, sim_name, "synthetic_$(location)", "noise=$(noise).jld2")
mkpath(src_dir)

#========================================================
COMPUTE CRPS AND BUILD TABLE
=========================================================#

table_rows = []

for ms in MS_limits
    seed_folders = get_seed_folders(sim_dir, true, ms)

    inf_rows       = Vector{Vector{Float64}}()
    beta_time_rows = Vector{Vector{Float64}}()
    beta_01_rows   = Vector{Vector{Float64}}()

    for folder in seed_folders
        result_path = joinpath(sim_dir, folder, "results.jld2")
        !isfile(result_path) && continue

        d         = JLD2.load(result_path)
        pred      = d["prediction"]
        beta_traj = d["beta_prediction"]
        p_trained = d["p"]

        i_traj   = vec(pred[3, 1:length(true_inf)])
        nn_input = Float64.(reshape(I_grid, 1, length(I_grid)))
        beta_0_1 = vec(beta_network(nn_input, p_trained.nn_params, st_nn)[1])

        push!(inf_rows,       i_traj)
        push!(beta_time_rows, vec(beta_traj))
        push!(beta_01_rows,   beta_0_1)
    end

    n_seeds = length(inf_rows)

    if n_seeds == 0
        println("MS=$(ms): no seeds found, skipping")
        push!(table_rows, (multistart=ms, n_seeds=0, crps_traj=missing, crps_beta_time=missing, crps_beta_01=missing))
        continue
    end

    inf_fc = Forecast(horizon=collect(1:length(true_inf)), trajectories=stack(inf_rows; dims=1))
    inf_fc = add_truth(inf_fc, true_inf)

    bt_fc  = Forecast(horizon=collect(1:length(true_beta_over_time)), trajectories=stack(beta_time_rows; dims=1))
    bt_fc  = add_truth(bt_fc, true_beta_over_time)

    b01_fc = Forecast(horizon=collect(1:length(I_grid)), trajectories=stack(beta_01_rows; dims=1))
    b01_fc = add_truth(b01_fc, true_beta_01)

    crps_traj      = score(inf_fc,  CRPS_trajectory())
    crps_beta_time = score(bt_fc,   CRPS_trajectory())
    crps_beta_01   = score(b01_fc,  CRPS_trajectory())

    println("MS=$(ms) | n=$(n_seeds) | traj=$(round(crps_traj, sigdigits=3)) | beta_time=$(round(crps_beta_time, sigdigits=3)) | beta_01=$(round(crps_beta_01, sigdigits=3))")

    push!(table_rows, (
        multistart     = ms,
        n_seeds        = n_seeds,
        crps_traj      = crps_traj,
        crps_beta_time = crps_beta_time,
        crps_beta_01   = crps_beta_01,
    ))
end

df = DataFrame(table_rows)
rename!(df, :multistart => "Multistart", :n_seeds => "N seeds",
            :crps_traj => "Trajectory", :crps_beta_time => "Beta in-sample",
            :crps_beta_01 => "Beta out-of-sample")

CSV.write(joinpath(src_dir, "ms_sensitivity_crps_table.csv"), df)
println("\nCRPS table:\n")
println(df)

#========================================================
CRPS VS MS LIMIT PLOTS
=========================================================#

ms_vals    = [r.multistart    for r in table_rows]
crps_trajs = [r.crps_traj     for r in table_rows]
crps_bts   = [r.crps_beta_time for r in table_rows]
crps_b01s  = [r.crps_beta_01  for r in table_rows]

p_traj = plot(ms_vals, crps_trajs;
    marker=:circle, linewidth=2, color=:blue, label="",
    title="CRPS — Trajectory ($(location))",
    left_margin=10Plots.mm, bottom_margin=8Plots.mm)
xlabel!(p_traj, "Multistart limit")
ylabel!(p_traj, "CRPS")

p_bt = plot(ms_vals, crps_bts;
    marker=:square, linewidth=2, color=:red, label="",
    title="CRPS — Beta in-sample ($(location))",
    left_margin=10Plots.mm, bottom_margin=8Plots.mm)
xlabel!(p_bt, "Multistart limit")
ylabel!(p_bt, "CRPS")

p_b01 = plot(ms_vals, crps_b01s;
    marker=:diamond, linewidth=2, color=:green, label="",
    title="CRPS — Beta out-of-sample ($(location))",
    left_margin=10Plots.mm, bottom_margin=8Plots.mm)
xlabel!(p_b01, "Multistart limit")
ylabel!(p_b01, "CRPS")

crps_panel = plot(p_traj, p_bt, p_b01; layout=(1, 3), size=(1800, 500))
savefig(crps_panel, joinpath(src_dir, "ms_sensitivity_crps_panel.png"))
savefig(p_traj,     joinpath(src_dir, "ms_sensitivity_crps_traj.png"))
savefig(p_bt,       joinpath(src_dir, "ms_sensitivity_crps_beta_time.png"))
savefig(p_b01,      joinpath(src_dir, "ms_sensitivity_crps_beta_01.png"))
println("CRPS plots saved to: $(src_dir)")

