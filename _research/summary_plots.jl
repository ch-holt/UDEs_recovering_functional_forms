
#========================================================
SEED SENSITIVITY — OVERLAY PLOTS ACROSS ALL SEEDS
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using Plots
using Statistics
using Lux
using ComponentArrays
using Random
using ForecastBaselines
using OrdinaryDiffEq
using UDE_FUNCTIONAL_FORMS

multistart = true
noise      = 0.0

# Colours for 1–4 week forecast segments
const FORECAST_COLORS = [:royalblue, :forestgreen, :darkorange, :purple]
const FORECAST_LABELS = ["Week 1", "Week 2", "Week 3", "Week 4"]

#========================================================
CONFIGURATION
=========================================================#

for train_length in [365]
    for location in ["AR"]
        println("Processing location: $(location), train_length: $(train_length)")
        for MS_limit in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            model_name = "ude_single"
            sim_name   = "UDE_single_beta=beta_exp_adam=2500_lbfgs=2000_traindata=$(train_length)_noise=$(noise)"

            # Must match the architecture used in training
            hidden_dims         = 5
            input_size          = 1
            activation_fn       = gelu
            final_activation_fn = softplus

            #========================================================
            SET UP NETWORK
            =========================================================#

            rng = Random.default_rng()
            beta_network, _, st_nn = build_neural_network(rng, hidden_dims, input_size, 1,
                                                          activation_fn, final_activation_fn)
            population = POPULATION[location]

            #========================================================
            FIND ALL SEED SIMULATION FOLDERS
            =========================================================#

            sim_dir      = datadir("exp_pro", "sims", model_name, sim_name, "synthetic_$(location)")
            seed_folders = get_seed_folders(sim_dir, multistart, MS_limit)

            if isempty(seed_folders)
                println("No seeds found for MS=$(MS_limit), skipping.")
                continue
            end

            #========================================================
            LOAD SYNTHETIC DATA AND COMPUTE TRUE BETA
            =========================================================#

            dataset  = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_beta_exp",
                                         "synthetic_$(location)", "noise=$(noise).jld2"))
            true_inf = dataset["infectious"]
            days     = collect(dataset["days"])

            I_grid              = collect(range(0, 1; length=1000))
            true_beta_over_time = beta_exp(location, true_inf)
            true_beta_01        = beta_exp(location, I_grid .* population)

            observed_I_over_N = true_inf ./ population
            I_N_min = minimum(observed_I_over_N)
            I_N_max = maximum(observed_I_over_N)

            # Fine I/N grid spanning only the training region
            train_I_over_N    = true_inf[1:train_length] ./ population
            I_N_train_min     = minimum(train_I_over_N)
            I_N_train_max     = maximum(train_I_over_N)
            I_grid_train      = collect(range(I_N_train_min, I_N_train_max; length=500))
            true_beta_train_region = beta_exp(location, I_grid_train .* population)

            inf_rows        = Vector{Vector{Float64}}()
            beta_time_rows  = Vector{Vector{Float64}}()
            beta_train_rows = Vector{Vector{Float64}}()
            beta_01_rows    = Vector{Vector{Float64}}()

            #========================================================
            RETRIEVE UDE RESULTS
            =========================================================#

            for folder in seed_folders
                result_path = joinpath(sim_dir, folder, "results.jld2")
                if !isfile(result_path)
                    println("Skipping $(folder) — results.jld2 not found")
                    continue
                end

                d         = JLD2.load(result_path)
                pred      = d["prediction"]
                beta_traj = d["beta_prediction"]
                p_trained = d["p"]

                i_traj        = vec(pred[3, 1:length(true_inf)])
                nn_input_01   = Float64.(reshape(I_grid,       1, length(I_grid)))
                nn_input_tr   = Float64.(reshape(I_grid_train, 1, length(I_grid_train)))
                beta_0_1      = vec(beta_network(nn_input_01, p_trained.nn_params, st_nn)[1])
                beta_train_r  = vec(beta_network(nn_input_tr, p_trained.nn_params, st_nn)[1])

                push!(inf_rows,        i_traj)
                push!(beta_time_rows,  beta_traj)
                push!(beta_train_rows, beta_train_r)
                push!(beta_01_rows,    beta_0_1)
            end

            isempty(inf_rows) && continue

            #========================================================
            PLOT 1 — FULL TRAJECTORY (all 365 days)
            =========================================================#

            traj_full, beta_time_full, beta_01_plot =
                plot_ensemble_summary(days, true_inf, I_grid, true_beta_over_time, true_beta_01,
                                      I_N_min, I_N_max, location,
                                      inf_rows, beta_time_rows, beta_01_rows)

            #========================================================
            PLOT 2 — TRAINING WINDOW ONLY (up to train_length)
            =========================================================#

            train_days      = days[1:train_length]
            train_true_inf  = true_inf[1:train_length]

            traj_train = plot(train_days, train_true_inf;
                color=:black, linewidth=2, label="Data",
                legend=:outertopright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
            beta_train_plot = plot(I_grid_train, true_beta_train_region;
                color=:black, linewidth=2, label="True β",
                legend=:outertopright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)

            for i_traj in inf_rows
                plot!(traj_train, train_days, i_traj[1:train_length]; color=:red, alpha=0.3, linewidth=1, label="")
            end
            for beta_r in beta_train_rows
                plot!(beta_train_plot, I_grid_train, beta_r; color=:red, alpha=0.3, linewidth=1, label="")
            end

            # CRPS on training window (infectious)
            inf_mat_train  = stack([r[1:train_length] for r in inf_rows]; dims=1)
            inf_fc_train   = add_truth(Forecast(horizon=collect(1:train_length), trajectories=inf_mat_train), train_true_inf)
            inf_crps_train = score(inf_fc_train, CRPS_trajectory())

            # Mean NMSE across seeds for beta vs I/N (training region)
            beta_nmse_train = mean(loss_nmse(beta_r, true_beta_train_region) for beta_r in beta_train_rows)

            title!(traj_train,     "Infectious (training) — $(location)\nCRPS: $(round(inf_crps_train,  sigdigits=3))")
            xlabel!(traj_train,    "Day")
            ylabel!(traj_train,    "Infectious individuals")
            title!(beta_train_plot, "β vs I/N (training region) — $(location)\nMean NMSE: $(round(beta_nmse_train, sigdigits=3))")
            xlabel!(beta_train_plot, "I/N")
            ylabel!(beta_train_plot, "β")

            #========================================================
            PLOT 3 — FORECAST (training + 4 coloured weeks)
            =========================================================#

            max_day   = min(train_length + 28, length(true_inf))
            plot_days = days[1:max_day]

            traj_forecast      = plot(plot_days, true_inf[1:max_day];
                color=:black, linewidth=2, label="Data",
                legend=:outertopright, left_margin=10Plots.mm, bottom_margin=8Plots.mm,
                titlefontsize=9)
            beta_time_forecast = plot(plot_days, true_beta_over_time[1:max_day];
                color=:black, linewidth=2, label="True β",
                legend=:outertopright, left_margin=10Plots.mm, bottom_margin=8Plots.mm,
                titlefontsize=9)

            first_seed = true
            for (i_traj, beta_traj) in zip(inf_rows, beta_time_rows)
                # Training portion
                plot!(traj_forecast,      days[1:train_length], i_traj[1:train_length];
                    color=:red, alpha=0.3, linewidth=1, label=first_seed ? "UDE fit" : "")
                plot!(beta_time_forecast, days[1:train_length], beta_traj[1:train_length];
                    color=:red, alpha=0.3, linewidth=1, label=first_seed ? "UDE fit" : "")
                # Forecast weeks
                for w in 1:4
                    w_start = train_length + 7*(w-1) + 1
                    w_end_i = min(train_length + 7*w, length(i_traj))
                    w_end_b = min(train_length + 7*w, length(beta_traj))
                    w_start > length(i_traj) && break
                    plot!(traj_forecast, days[w_start:w_end_i], i_traj[w_start:w_end_i];
                        color=FORECAST_COLORS[w], alpha=0.3, linewidth=1, label=first_seed ? FORECAST_LABELS[w] : "")
                    w_start <= length(beta_traj) && plot!(beta_time_forecast, days[w_start:w_end_b], beta_traj[w_start:w_end_b];
                        color=FORECAST_COLORS[w], alpha=0.3, linewidth=1, label=first_seed ? FORECAST_LABELS[w] : "")
                end
                first_seed = false
            end

            vline!(traj_forecast,      [days[train_length]]; linestyle=:dash, color=:black, linewidth=1, label="Train cutoff")
            vline!(beta_time_forecast, [days[train_length]]; linestyle=:dash, color=:black, linewidth=1, label="Train cutoff")

            # Per-week CRPS
            inf_crps_weeks  = Float64[]
            beta_crps_weeks = Float64[]
            for w in 1:4
                w_start = train_length + 7*(w-1) + 1
                w_end_i = min(train_length + 7*w, length(true_inf))
                w_end_b = min(train_length + 7*w, length(true_beta_over_time))
                w_start > length(true_inf) && break
                n_i = w_end_i - w_start + 1
                n_b = w_end_b - w_start + 1
                inf_mat_w  = stack([r[w_start:w_end_i] for r in inf_rows];      dims=1)
                beta_mat_w = stack([r[w_start:w_end_b] for r in beta_time_rows]; dims=1)
                push!(inf_crps_weeks,  score(add_truth(Forecast(horizon=collect(1:n_i), trajectories=inf_mat_w),  true_inf[w_start:w_end_i]),              CRPS_trajectory()))
                push!(beta_crps_weeks, score(add_truth(Forecast(horizon=collect(1:n_b), trajectories=beta_mat_w), true_beta_over_time[w_start:w_end_b]), CRPS_trajectory()))
            end

            crps_str_i = join(["W$(w):$(round(c,sigdigits=2))" for (w,c) in enumerate(inf_crps_weeks)],  " ")
            crps_str_b = join(["W$(w):$(round(c,sigdigits=2))" for (w,c) in enumerate(beta_crps_weeks)], " ")

            title!(traj_forecast,      "Infectious + forecast — $(location)\n$(crps_str_i)")
            xlabel!(traj_forecast,     "Day")
            ylabel!(traj_forecast,     "Infectious individuals")
            title!(beta_time_forecast, "Beta + forecast — $(location)\n$(crps_str_b)")
            xlabel!(beta_time_forecast, "Day")
            ylabel!(beta_time_forecast, "β(t)")

            #========================================================
            SAVE
            =========================================================#

            save_dir = plotsdir("sims", model_name, sim_name, "synthetic_$(location)")
            mkpath(save_dir)

            sfx = multistart ? "_MS=$(MS_limit)" : ""

            # Full trajectory
            savefig(traj_full,      joinpath(save_dir, "traj_full$(sfx).png"))
            savefig(beta_time_full, joinpath(save_dir, "beta_time_full$(sfx).png"))
            savefig(beta_01_plot,   joinpath(save_dir, "beta_01$(sfx).png"))
            # Training only
            savefig(traj_train,      joinpath(save_dir, "traj_train$(sfx).png"))
            savefig(beta_train_plot, joinpath(save_dir, "beta_time_train$(sfx).png"))
            # Forecast
            savefig(traj_forecast,      joinpath(save_dir, "traj_forecast$(sfx).png"))
            savefig(beta_time_forecast, joinpath(save_dir, "beta_time_forecast$(sfx).png"))

            println("Done — plots saved to: $(save_dir)")
        end
    end
end


#========================================================
RETRIEVE SR RESULTS
=========================================================#

SR_inf_rows = Vector{Vector{Float64}}()
SR_beta_time_rows = Vector{Vector{Float64}}()
SR_beta_01_rows = Vector{Vector{Float64}}()

output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)", "synthetic_$(location)")

for folder in seed_folders
    result_path = joinpath(output_dir, folder, "SR_report.jld2")
    if !isfile(result_path)
        println("Skipping $(folder) — SR_report.jld2 not found")
        continue
    end

    SR_beta_days, SR_beta_0_1, SR_inf = JLD2.load(result_path, "SR_beta_days", "SR_beta_0_1", "SR_inf")

    # Store trajectories for CRPS
    push!(SR_inf_rows, SR_inf)
    push!(SR_beta_time_rows, vec(SR_beta_days))
    push!(SR_beta_01_rows, vec(SR_beta_0_1))
end

SR_traj_plot, SR_beta_time_plot, SR_beta_01_plot = plot_ensemble_summary(days, true_inf, I_grid, true_beta_over_time, true_beta_01,
                                                                I_N_min, I_N_max, location,
                                                                SR_inf_rows, SR_beta_time_rows, SR_beta_01_rows)


#========================================================
SAVE
=========================================================#

panel = plot(SR_traj_plot, SR_beta_time_plot, SR_beta_01_plot; layout=(1, 3), size=(1800, 500))
if multistart
    savefig(panel, joinpath(output_dir, "panel_overlay_MS=$(MS_limit).png"))
    savefig(SR_traj_plot,      joinpath(output_dir, "traj_overlay_MS=$(MS_limit).png"))
    savefig(SR_beta_time_plot, joinpath(output_dir, "beta_time_overlay_MS=$(MS_limit).png"))
    savefig(SR_beta_01_plot,   joinpath(output_dir, "beta_01_overlay_MS=$(MS_limit).png"))
else
    savefig(panel, joinpath(output_dir, "panel_overlay.png"))
    savefig(SR_traj_plot,      joinpath(output_dir, "traj_overlay.png"))
    savefig(SR_beta_time_plot, joinpath(output_dir, "beta_time_overlay.png"))
    savefig(SR_beta_01_plot,   joinpath(output_dir, "beta_01_overlay.png"))
end

println("Done — plots saved to:\n  $(output_dir)")

I_nn, best_results, folder = extract_best_ude(sim_dir, true_inf, multistart, MS_limit)
println("Best UDE results found in folder: $(folder)")
