
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
using SymbolicRegression
using MLJ
using SymbolicUtils
using OrdinaryDiffEq
using UDE_FUNCTIONAL_FORMS

#========================================================
CONFIGURATION
=========================================================#
const train_length = 365

for MS_limit in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    location = "MA"
    model_name = "ude_single"
    sim_name = "UDE_single_tsit5_beta=beta_exp_adam=2500_learning_rate=0.001_lbfgs=2000_number_of_nn_input=1_finalactivation=softplus_traindata=$(train_length)"
    multistart = true

    # Must match the architecture used in training
    hidden_dims          = 5
    input_size           = 1
    activation_fn        = gelu
    final_activation_fn  = softplus

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

    # If multistart then only overlay seeds_to_keep
    sim_dir = datadir("exp_pro", "sims", model_name, sim_name, "synthetic_$(location)")
    seed_folders = get_seed_folders(sim_dir, multistart, MS_limit)

    #========================================================
    LOAD SYNTHETIC DATA AND COMPUTE TRUE BETA
    =========================================================#

    dataset = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_beta_exp",
                                "synthetic_$(location).jld2"))
    true_inf = dataset["infectious"]
    days = collect(dataset["days"])

    I_grid             = collect(range(0, 1; length=1000))
    true_beta_over_time = beta_exp(location, true_inf)
    true_beta_01        = beta_exp(location, I_grid .* population)

    # I/N range of observed training data (for vertical lines on beta_01_plot)
    observed_I_over_N = true_inf ./ population
    I_N_min = minimum(observed_I_over_N)
    I_N_max = maximum(observed_I_over_N)

    inf_rows = Vector{Vector{Float64}}()
    beta_time_rows = Vector{Vector{Float64}}()
    beta_01_rows = Vector{Vector{Float64}}()

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

        # Infectious trajectory
        i_traj = vec(pred[3, 1:length(true_inf)])

        # Beta vs I/N — re-evaluate NN on uniform grid
        nn_input = Float64.(reshape(I_grid, 1, length(I_grid)))
        beta_0_1    = vec(beta_network(nn_input, p_trained.nn_params, st_nn)[1])
        
        # Store trajectories for CRPS 
        push!(inf_rows, i_traj)
        push!(beta_time_rows, beta_traj)
        push!(beta_01_rows, beta_0_1)
    end

    traj_plot, beta_time_plot, beta_01_plot = plot_ensemble_summary(days, true_inf, I_grid, true_beta_over_time, true_beta_01,
                                                                    I_N_min, I_N_max, location,
                                                                    inf_rows, beta_time_rows, beta_01_rows)

    #========================================================
    SAVE
    =========================================================#

    save_dir = plotsdir("sims", model_name, sim_name, "synthetic_$(location)")
    mkpath(save_dir)

    panel = plot(traj_plot, beta_time_plot, beta_01_plot; layout=(1, 3), size=(1800, 500))
    if multistart
        savefig(panel, joinpath(save_dir, "panel_overlay_MS=$(MS_limit).png"))
        savefig(traj_plot,      joinpath(save_dir, "traj_overlay_MS=$(MS_limit).png"))
        savefig(beta_time_plot, joinpath(save_dir, "beta_time_overlay_MS=$(MS_limit).png"))
        savefig(beta_01_plot,   joinpath(save_dir, "beta_01_overlay_MS=$(MS_limit).png"))
    else
        savefig(panel, joinpath(save_dir, "panel_overlay.png"))
        savefig(traj_plot,      joinpath(save_dir, "traj_overlay.png"))
        savefig(beta_time_plot, joinpath(save_dir, "beta_time_overlay.png"))
        savefig(beta_01_plot,   joinpath(save_dir, "beta_01_overlay.png"))
    end

    println("Done — plots saved to:\n  $(save_dir)")
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