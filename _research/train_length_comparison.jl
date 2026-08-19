
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

location       = "MA"
model_name     = "ude_single"
beta_func_name = "beta_exp"
multistart     = true
noise          = 0.0

# Training lengths to overlay. Each entry specifies the training length to
# include, the MS_limit (multistart validation-loss threshold) used to
# select seeds for that particular training length, and the colour to plot
# it in. Add/remove entries here to control which training lengths appear.
train_length_configs = [
    (train_length=55,  MS_limit=0.6, colour=:navy),
    (train_length=60,  MS_limit=0.7, colour=:dodgerblue),
    (train_length=75, MS_limit=0.5, colour=:orange),
    (train_length=150, MS_limit=0.1, colour=:red),
    (train_length=200, MS_limit=0.3, colour=:purple),
]

# Rolling-forecast horizons — (start, end) in days-after-training-cutoff.
# One plot is produced per horizon below, plus one plot for the full
# 365-day trajectory.
forecast_horizons = [(1, 7), (8, 14), (15, 21), (22, 28)]
forecast_labels   = ["Week 1 forecast (days 1-7 ahead)", "Week 2 forecast (days 8-14 ahead)",
                      "Week 3 forecast (days 15-21 ahead)", "Week 4 forecast (days 22-28 ahead)"]

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
                             "synthetic_$(location)", "noise=$(noise).jld2"))
true_inf = dataset["infectious"]
days     = collect(dataset["days"])

I_grid              = collect(range(0, 1; length=1000))
true_beta_over_time = beta_exp(location, true_inf)
true_beta_01        = beta_exp(location, I_grid .* population)

peak_day = days[find_peak(true_inf)]

#========================================================
LOAD RESULTS FOR EACH TRAINING LENGTH
=========================================================#

config_inf_rows       = Dict{Int, Vector{Vector{Float64}}}()
config_beta_time_rows = Dict{Int, Vector{Vector{Float64}}}()
config_beta_01_rows   = Dict{Int, Vector{Vector{Float64}}}()

crps_inf       = Float64[]
crps_beta_time = Float64[]
crps_beta_01   = Float64[]
valid_configs  = NamedTuple[]

for cfg in train_length_configs
    train_length = cfg.train_length
    MS_limit     = cfg.MS_limit

    sim_name = "UDE_single_beta=$(beta_func_name)_adam=2500_lbfgs=2000_traindata=$(train_length)_noise=$(noise)"
    sim_dir  = datadir("exp_pro", "sims", model_name, sim_name, "synthetic_$(location)")

    if !isdir(sim_dir)
        println("Skipping train_length=$(train_length) — directory not found")
        continue
    end

    seed_folders = get_seed_folders(sim_dir, multistart, MS_limit)

    inf_rows       = Vector{Vector{Float64}}()
    beta_time_rows = Vector{Vector{Float64}}()
    beta_01_rows   = Vector{Vector{Float64}}()

    for folder in seed_folders
        result_path = joinpath(sim_dir, folder, "results.jld2")
        isfile(result_path) || continue

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

    println("train_length=$(train_length), MS_limit=$(MS_limit): loaded $(length(inf_rows)) seeds")

    if isempty(inf_rows)
        continue
    end

    config_inf_rows[train_length]       = inf_rows
    config_beta_time_rows[train_length] = beta_time_rows
    config_beta_01_rows[train_length]   = beta_01_rows

    inf_fc = add_truth(Forecast(horizon=collect(1:length(true_inf)), trajectories=stack(inf_rows; dims=1)), true_inf)
    bt_fc  = add_truth(Forecast(horizon=collect(1:length(true_beta_over_time)), trajectories=stack(beta_time_rows; dims=1)), true_beta_over_time)
    b01_fc = add_truth(Forecast(horizon=collect(1:length(I_grid)), trajectories=stack(beta_01_rows; dims=1)), true_beta_01)

    push!(crps_inf,       score(inf_fc,  CRPS_trajectory()))
    push!(crps_beta_time, score(bt_fc,   CRPS_trajectory()))
    push!(crps_beta_01,   score(b01_fc,  CRPS_trajectory()))
    push!(valid_configs,  cfg)
end

#========================================================
FULL-TRAJECTORY OVERLAY PLOTS (all specified training lengths)
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

for cfg in valid_configs
    train_length = cfg.train_length
    col          = cfg.colour

    for (i, (i_traj, beta_traj, beta_0_1)) in enumerate(zip(config_inf_rows[train_length],
                                                              config_beta_time_rows[train_length],
                                                              config_beta_01_rows[train_length]))
        lbl = (i == 1) ? "train=$(train_length)" : ""
        plot!(traj_plot,      days[1:length(i_traj)],    i_traj;    color=col, alpha=0.3, linewidth=1, label=lbl)
        plot!(beta_time_plot, days[1:length(beta_traj)], beta_traj; color=col, alpha=0.3, linewidth=1, label=lbl)
        plot!(beta_01_plot,   I_grid, beta_0_1;                     color=col, alpha=0.3, linewidth=1, label=lbl)

        if i == 1
            vline!(traj_plot, [Float64(train_length)]; linestyle=:dot, color=col, linewidth=1.5, label="")
        end
    end
end

xlabel!(traj_plot, "Day")
ylabel!(traj_plot, "Infectious individuals")
xlabel!(beta_time_plot, "Day")
ylabel!(beta_time_plot, "β(t)")
xlabel!(beta_01_plot, "I/N")
ylabel!(beta_01_plot, "β(I/N)")

#========================================================
ROLLING FORECAST — ONE PLOT PER FORECAST HORIZON
=========================================================#

horizon_plots = Plots.Plot[]

for ((h_start, h_end), h_label) in zip(forecast_horizons, forecast_labels)

    h_plot = plot(days, true_inf;
        color=:black, linewidth=2.5, label="True data",
        xlabel="Day", ylabel="Infectious individuals",
        title="$(h_label) — $(location)",
        legend=:topright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)

    for cfg in valid_configs
        train_length = cfg.train_length
        col          = cfg.colour

        w_start = train_length + h_start
        w_end   = min(train_length + h_end, length(true_inf))
        if w_start > length(true_inf)
            println("  train_length=$(train_length): $(h_label) is beyond available data — skipped")
            continue
        end

        rows = config_inf_rows[train_length]
        mat  = stack([r[w_start:w_end] for r in rows]; dims=1)
        fc   = add_truth(Forecast(horizon=collect(1:(w_end - w_start + 1)), trajectories=mat), true_inf[w_start:w_end])
        crps = score(fc, CRPS_trajectory())

        for (i, i_traj) in enumerate(rows)
            lbl = (i == 1) ? "train=$(train_length) (CRPS=$(round(crps, sigdigits=3)))" : ""
            plot!(h_plot, days[w_start:w_end], i_traj[w_start:w_end]; color=col, alpha=0.3, linewidth=1.5, label=lbl)
        end

        vline!(h_plot, [Float64(train_length)]; linestyle=:dot, color=col, linewidth=1.5, label="")
    end

    push!(horizon_plots, h_plot)
end

#========================================================
CRPS VS TRAINING LENGTH
=========================================================#

valid_lengths = [cfg.train_length for cfg in valid_configs]

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

save_dir = plotsdir("sims", model_name, "train_length_comparison", "synthetic_$(location)")
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

# Rolling forecast — full trajectory + one plot per horizon (5 plots total)
horizon_filenames = ["rolling_forecast_week1.png", "rolling_forecast_week2.png",
                      "rolling_forecast_week3.png", "rolling_forecast_week4.png"]
for (h_plot, fname) in zip(horizon_plots, horizon_filenames)
    savefig(h_plot, joinpath(save_dir, fname))
end
savefig(traj_plot, joinpath(save_dir, "rolling_forecast_full365.png"))

rolling_panel = plot(traj_plot, horizon_plots...; layout=(1, length(horizon_plots) + 1), size=(500 * (length(horizon_plots) + 1), 500))
savefig(rolling_panel, joinpath(save_dir, "rolling_forecast_panel.png"))

println("Done — plots saved to:\n  $(save_dir)")
