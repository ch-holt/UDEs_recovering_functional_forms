
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

# Colours for 1–4 week forecast segments
const FORECAST_COLORS = [:royalblue, :forestgreen, :darkorange, :purple]
const FORECAST_LABELS = ["Week 1", "Week 2", "Week 3", "Week 4"]

#========================================================
CONFIGURATION
=========================================================#
# Discover every sim folder that actually exists on disk (same rationale as
# scripts/multistart.jl) instead of hardcoding beta/train_length/noise/location.
const BETA_FUNCTIONS = Dict("beta_exp" => beta_exp, "beta_rational" => beta_rational, "beta_mixed" => beta_mixed)
const SIM_NAME_RE = r"^UDE_single_beta=(?<beta>[a-zA-Z_]+)_adam=(?<adam>\d+)_lbfgs=(?<lbfgs>\d+)_traindata=(?<traindata>\d+)_noise=(?<noise>[\d.]+)$"

sims_root = datadir("exp_pro", "sims", "ude_single")

for sim in sort(readdir(sims_root))
    m = match(SIM_NAME_RE, sim)
    isnothing(m) && continue
    !haskey(BETA_FUNCTIONS, m[:beta]) && continue

    beta_function = BETA_FUNCTIONS[m[:beta]]
    train_length  = parse(Int, m[:traindata])
    noise         = parse(Float64, m[:noise])
    sim_root_dir  = joinpath(sims_root, sim)

    for loc_folder in sort(readdir(sim_root_dir))
        !isdir(joinpath(sim_root_dir, loc_folder)) && continue
        !startswith(loc_folder, "synthetic_") && continue
        location = replace(loc_folder, "synthetic_" => "")
        !haskey(POPULATION, location) && continue

        println("Processing beta: $(m[:beta]), location: $(location), train_length: $(train_length), noise: $(noise)")
            for MS_limit in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                model_name = "ude_single"
                sim_name   = sim

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

                sim_dir = joinpath(sim_root_dir, loc_folder)

                seeds_to_keep_path = joinpath(sim_dir, "seeds_to_keep_MS=$(MS_limit).jld2")
                if !isfile(seeds_to_keep_path)
                    println("No seeds_to_keep file for MS=$(MS_limit) in $(sim_dir) (run multistart.jl first), skipping.")
                    continue
                end
                seed_folders = get_seed_folders(sim_dir, multistart, MS_limit)

                if isempty(seed_folders)
                    println("No seeds found for MS=$(MS_limit), skipping.")
                    continue
                end

                #========================================================
                LOAD SYNTHETIC DATA AND COMPUTE TRUE BETA
                =========================================================#

                dataset_path      = datadir("exp_pro", "synthetic_data", "synthetic_trajectories_$(beta_function)",
                                             "synthetic_$(location)", "noise=$(noise).jld2")
                true_dataset_path = noise == 0 ? dataset_path : datadir("exp_pro", "synthetic_data", "synthetic_trajectories_$(beta_function)",
                                             "synthetic_$(location)", "noise=0.0.jld2")
                if !isfile(dataset_path) || !isfile(true_dataset_path)
                    println("Missing synthetic data for $(sim_name)/$(location), skipping.")
                    continue
                end
                dataset      = JLD2.load(dataset_path)
                true_dataset = JLD2.load(true_dataset_path)
                true_inf  = true_dataset["infectious"]
                noisy_inf = dataset["infectious"]
                days      = collect(dataset["days"])

                I_grid              = collect(range(0, 1; length=1000))
                true_beta_over_time = beta_function(location, true_inf)
                true_beta_01        = beta_function(location, I_grid .* population)

                observed_I_over_N = true_inf ./ population
                I_N_min = minimum(observed_I_over_N)
                I_N_max = maximum(observed_I_over_N)

                # Fine I/N grid spanning only the training region
                train_I_over_N    = true_inf[1:train_length] ./ population
                I_N_train_min     = minimum(train_I_over_N)
                I_N_train_max     = maximum(train_I_over_N)
                I_grid_train      = collect(range(I_N_train_min, I_N_train_max; length=500))
                true_beta_train_region = beta_function(location, I_grid_train .* population)

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
                train_noisy_inf = noisy_inf[1:train_length]

                train_true_beta = true_beta_over_time[1:train_length]

                traj_train = plot(train_days, train_noisy_inf;
                    color=:black, linewidth=2, label="Observed data",
                    legend=:outertopright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
                beta_time_train = plot(train_days, train_true_beta;
                    color=:black, linewidth=2, label="True β",
                    legend=:outertopright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
                beta_train_plot = plot(I_grid_train, true_beta_train_region;
                    color=:black, linewidth=2, label="True β",
                    legend=:outertopright, left_margin=10Plots.mm, bottom_margin=8Plots.mm)

                for i_traj in inf_rows
                    plot!(traj_train,      train_days, i_traj[1:train_length];   color=:red, alpha=0.3, linewidth=1, label="")
                end
                for beta_traj in beta_time_rows
                    plot!(beta_time_train, train_days, beta_traj[1:train_length]; color=:red, alpha=0.3, linewidth=1, label="")
                end
                for beta_r in beta_train_rows
                    plot!(beta_train_plot, I_grid_train, beta_r;                  color=:red, alpha=0.3, linewidth=1, label="")
                end

                # CRPS on training window (infectious)
                inf_mat_train  = stack([r[1:train_length] for r in inf_rows]; dims=1)
                inf_fc_train   = add_truth(Forecast(horizon=collect(1:train_length), trajectories=inf_mat_train), train_noisy_inf)
                inf_crps_train = score(inf_fc_train, CRPS_trajectory())

                # CRPS for beta over time (training window)
                beta_mat_time_train  = stack([r[1:train_length] for r in beta_time_rows]; dims=1)
                beta_fc_time_train   = add_truth(Forecast(horizon=collect(1:train_length), trajectories=beta_mat_time_train), train_true_beta)
                beta_crps_time_train = score(beta_fc_time_train, CRPS_trajectory())

                # CRPS for beta vs I/N (training region)
                beta_mat_train  = stack(beta_train_rows; dims=1)
                beta_fc_train   = add_truth(Forecast(horizon=collect(1:length(I_grid_train)), trajectories=beta_mat_train), true_beta_train_region)
                beta_crps_train = score(beta_fc_train, CRPS_trajectory())

                title!(traj_train,      "Infectious (training) — $(location)\nCRPS: $(round(inf_crps_train,       sigdigits=3))")
                xlabel!(traj_train,     "Day")
                ylabel!(traj_train,     "Infectious individuals")
                title!(beta_time_train, "Beta over time (training) — $(location)\nCRPS: $(round(beta_crps_time_train, sigdigits=3))")
                xlabel!(beta_time_train, "Day")
                ylabel!(beta_time_train, "β(t)")
                title!(beta_train_plot, "β vs I/N (training region) — $(location)\nCRPS: $(round(beta_crps_train,      sigdigits=3))")
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
                savefig(beta_time_train, joinpath(save_dir, "beta_time_train$(sfx).png"))
                savefig(beta_train_plot, joinpath(save_dir, "beta_train_region$(sfx).png"))
                # Forecast
                savefig(traj_forecast,      joinpath(save_dir, "traj_forecast$(sfx).png"))
                savefig(beta_time_forecast, joinpath(save_dir, "beta_time_forecast$(sfx).png"))

                println("Done — plots saved to: $(save_dir)")

                #========================================================
                BEST UDE (SR retrieval skipped — no SR_report.jld2 outputs
                exist locally yet; re-add that block once the SR pipeline
                has been run for these sims)
                =========================================================#

                try
                    I_nn, best_results, best_folder = extract_best_ude(sim_dir, true_inf, multistart, MS_limit)
                    println("Best UDE results found in folder: $(best_folder)")
                catch e
                    println("Could not determine best UDE for $(sim_name)/$(location) MS=$(MS_limit): $(e)")
                end
            end
        end
    end
end
