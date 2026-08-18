
#========================================================
TRAIN LENGTH COMPARISON — OVERLAY PLOTS ACROSS TRAINING LENGTHS
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
CONFIGURATION
=========================================================#

location    = "MA"
model_name  = "ude_single"
beta_func_name = "beta_exp"
multistart  = true
MS_limit    = 0.3

# Training lengths to compare and their colours
train_lengths = [50, 55, 60, 63, 65, 75, 100, 125, 150, 175, 200]
colours = [:navy, :dodgerblue, :deepskyblue, :cyan, :mediumseagreen,
           :green, :yellow, :orange, :red, :deeppink, :purple]

# Must match training architecture
hidden_dims         = 5
input_size          = 1
activation_fn       = gelu
final_activation_fn = softplus

#========================================================
SET UP NETWORK AND LOAD TRUE DATA
=========================================================#

rng = Random.default_rng()
beta_network, _, st_nn = build_neural_network(rng, hidden_dims, input_size, 1,
                                               activation_fn, final_activation_fn)
population = POPULATION[location]

dataset = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_$(beta_func_name)",
                             "synthesised_$(location).jld2"))
true_inf = dataset["infectious"]
days     = collect(dataset["days"])

I_grid             = collect(range(0, 1; length=1000))
true_beta_over_time = beta_exp(location, true_inf)
true_beta_01        = beta_exp(location, I_grid .* population)

peak_day = days[find_peak(true_inf)]

#========================================================
INITIALISE PLOTS WITH TRUE TRAJECTORY
=========================================================#

traj_plot = plot(days, true_inf;
    color=:black, linewidth=2.5, label="True data",
    xlabel="Day", ylabel="Infectious individuals",
    title="Infectious trajectory by training length — $(location)",
    legend=:topright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
vline!(traj_plot, [peak_day]; linestyle=:dash, color=:black, linewidth=1, label="Peak (day $(Int(peak_day)))")

beta_time_plot = plot(days, true_beta_over_time;
    color=:black, linewidth=2.5, label="True β",
    xlabel="Day", ylabel="β(t)",
    title="Beta over time by training length — $(location)",
    legend=:topright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)

beta_01_plot = plot(I_grid, true_beta_01;
    color=:black, linewidth=2.5, label="True β",
    xlabel="I/N", ylabel="β(I/N)",
    title="Beta vs I/N by training length — $(location)",
    legend=:topright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)

#========================================================
OVERLAY EACH TRAINING LENGTH AND COLLECT CRPS
=========================================================#

crps_inf       = Float64[]
crps_beta_time = Float64[]
crps_beta_01   = Float64[]
valid_lengths  = Int[]

for (train_length, col) in zip(train_lengths, colours)

    sim_name = "UDE_single_beta=$(beta_func_name)_adam=2500_learning_rate=0.001_lbfgs=2000_number_of_nn_input=$(input_size)_finalactivation=softplus_traindata=$(train_length)"
    sim_dir  = datadir("exp_pro", "sims", model_name, sim_name, "synthesised_$(location)")

    if !isdir(sim_dir)
        println("Skipping train_length=$(train_length) — directory not found")
        continue
    end

    seed_folders = get_seed_folders(sim_dir, multistart, MS_limit)
    n_loaded = 0

    inf_rows       = Vector{Vector{Float64}}()
    beta_time_rows = Vector{Vector{Float64}}()
    beta_01_rows   = Vector{Vector{Float64}}()

    for (i, folder) in enumerate(seed_folders)
        result_path = joinpath(sim_dir, folder, "results.jld2")
        if !isfile(result_path)
            continue
        end

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

        # Only add legend label for first seed of each training length
        lbl = (n_loaded == 0) ? "train=$(train_length)" : ""

        plot!(traj_plot,      days[1:length(i_traj)],    i_traj;    color=col, alpha=0.3, linewidth=1, label=lbl)
        plot!(beta_time_plot, days[1:length(beta_traj)], beta_traj; color=col, alpha=0.3, linewidth=1, label=lbl)
        plot!(beta_01_plot,   I_grid, beta_0_1;                     color=col, alpha=0.3, linewidth=1, label=lbl)

        # Add vertical cutoff line on trajectory plot for this training length
        if n_loaded == 0
            vline!(traj_plot, [Float64(train_length)]; linestyle=:dot, color=col, linewidth=1.5, label="")
        end

        n_loaded += 1
    end

    println("train_length=$(train_length): loaded $(n_loaded) seeds")

    if n_loaded == 0
        continue
    end

    # Calculate CRPS for each output
    inf_fc = Forecast(horizon=collect(1:length(true_inf)), trajectories=stack(inf_rows; dims=1))
    inf_fc = add_truth(inf_fc, true_inf)

    bt_fc  = Forecast(horizon=collect(1:length(true_beta_over_time)), trajectories=stack(beta_time_rows; dims=1))
    bt_fc  = add_truth(bt_fc, true_beta_over_time)

    b01_fc = Forecast(horizon=collect(1:length(I_grid)), trajectories=stack(beta_01_rows; dims=1))
    b01_fc = add_truth(b01_fc, true_beta_01)

    push!(crps_inf,       score(inf_fc,  CRPS_trajectory()))
    push!(crps_beta_time, score(bt_fc,   CRPS_trajectory()))
    push!(crps_beta_01,   score(b01_fc,  CRPS_trajectory()))
    push!(valid_lengths,  train_length)
end

# Re-apply axis labels after loop overlay calls
xlabel!(traj_plot, "Day")
ylabel!(traj_plot, "Infectious individuals")
xlabel!(beta_time_plot, "Day")
ylabel!(beta_time_plot, "β(t)")
xlabel!(beta_01_plot, "I/N")
ylabel!(beta_01_plot, "β(I/N)")

#========================================================
CRPS VS TRAINING LENGTH
=========================================================#

crps_inf_plot = plot(valid_lengths, crps_inf;
    marker=:circle, linewidth=2, color=:blue, label="",
    title="CRPS — Infectious trajectory ($(location))",
    left_margin=10Plots.mm, bottom_margin=8Plots.mm)
vline!(crps_inf_plot, [peak_day]; linestyle=:dash, color=:black, linewidth=1, label="Peak (day $(Int(peak_day)))")
xlabel!(crps_inf_plot, "Training length (days)")
ylabel!(crps_inf_plot, "CRPS")

crps_beta_time_plot = plot(valid_lengths, crps_beta_time;
    marker=:square, linewidth=2, color=:red, label="",
    title="CRPS — Beta over time ($(location))",
    left_margin=10Plots.mm, bottom_margin=8Plots.mm)
vline!(crps_beta_time_plot, [peak_day]; linestyle=:dash, color=:black, linewidth=1, label="Peak (day $(Int(peak_day)))")
xlabel!(crps_beta_time_plot, "Training length (days)")
ylabel!(crps_beta_time_plot, "CRPS")

crps_beta_01_plot = plot(valid_lengths, crps_beta_01;
    marker=:diamond, linewidth=2, color=:green, label="",
    title="CRPS — Beta vs I/N ($(location))",
    left_margin=10Plots.mm, bottom_margin=8Plots.mm)
vline!(crps_beta_01_plot, [peak_day]; linestyle=:dash, color=:black, linewidth=1, label="Peak (day $(Int(peak_day)))")
xlabel!(crps_beta_01_plot, "Training length (days)")
ylabel!(crps_beta_01_plot, "CRPS")

#========================================================
SAVE
=========================================================#

save_dir = plotsdir("sims", model_name, "train_length_comparison", "synthesised_$(location)")
mkpath(save_dir)

panel = plot(traj_plot, beta_time_plot, beta_01_plot; layout=(1, 3), size=(1800, 500))
savefig(panel,           joinpath(save_dir, "panel_train_length_comparison.png"))
savefig(traj_plot,       joinpath(save_dir, "traj_train_length_comparison.png"))
savefig(beta_time_plot,  joinpath(save_dir, "beta_time_train_length_comparison.png"))
savefig(beta_01_plot,    joinpath(save_dir, "beta_01_train_length_comparison.png"))
crps_panel = plot(crps_inf_plot, crps_beta_time_plot, crps_beta_01_plot; layout=(1, 3), size=(1800, 500))
savefig(crps_panel,          joinpath(save_dir, "crps_panel_vs_train_length.png"))
savefig(crps_inf_plot,       joinpath(save_dir, "crps_inf_vs_train_length.png"))
savefig(crps_beta_time_plot, joinpath(save_dir, "crps_beta_time_vs_train_length.png"))
savefig(crps_beta_01_plot,   joinpath(save_dir, "crps_beta_01_vs_train_length.png"))

println("Done — plots saved to:\n  $(save_dir)")
