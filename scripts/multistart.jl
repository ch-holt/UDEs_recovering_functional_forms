#========================================================
SCRIPT TO DO MULTISTART AS A POST-PROCESSING STEP
=========================================================#  

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using DataFrames
using CSV

# Call module
using UDE_FUNCTIONAL_FORMS

function summarise_results(sim_name_dir::String, population::Real)

    rows = []
    # Each seed run is stored in simulation_seed=* subfolders
    for sim_folder in readdir(sim_name_dir; join=true)
        !isdir(sim_folder) && continue
        !occursin("simulation_seed=", basename(sim_folder)) && continue

        results_file = joinpath(sim_folder, "results.jld2")
        !isfile(results_file) && continue

        data = JLD2.load(results_file)

        seed            = get(data, "seed",            missing)
        loss_traj_noisy = get(data, "loss_traj_noisy", missing)
        loss_traj_true  = get(data, "loss_traj_true",  missing)
        loss_beta       = get(data, "loss_beta",       missing)
        loss_I_grid     = get(data, "loss_I_grid",     missing)
        val_losses      = get(data, "val_losses",      missing)
        best_val_loss = ismissing(val_losses) ? missing : minimum(val_losses)
        # Jointly estimated noise level (missing for noise=0)
        phi_hat         = get(data, "phi_hat",         missing)
        phi_init        = get(data, "phi_init",        missing)

        # "prediction" is the long-term solve (3 years, row 3 = infectious compartment);
        # check feasibility over the full saved horizon.
        prediction   = get(data, "prediction", missing)
        max_forecast = ismissing(prediction) ? missing : maximum(prediction[3, :])
        feasible     = ismissing(max_forecast) ? missing : (max_forecast <= population)

        push!(rows, (
            sim_folder      = basename(sim_folder),
            seed            = seed,
            nmse_traj_noisy = loss_traj_noisy,
            nmse_traj_true  = loss_traj_true,
            nmse_beta       = loss_beta,
            nmse_I_grid     = loss_I_grid,
            best_val_loss   = best_val_loss,
            phi_hat         = phi_hat,
            phi_init        = phi_init,
            max_forecast    = max_forecast,
            feasible        = feasible,
        ))
    end

    df = DataFrame(rows)
    sort!(df, [:seed])


    return df
end


# Discover every sim folder that actually exists on disk, rather than hardcoding
# beta/train_length/noise/location combinations
const BETA_FUNCTIONS = Dict("beta_exp" => beta_exp, "beta_rational" => beta_rational, "beta_mixed" => beta_mixed)
const SIM_NAME_RE = r"^(?<prefix>.*?)UDE_single_beta=(?<beta>[a-zA-Z_]+)_adam=(?<adam>\d+)_lbfgs=(?<lbfgs>\d+)_traindata=(?<traindata>\d+)_noise=(?<noise>[\d.]+)(?<estphi>_estphi(?:_[a-z0-9_]+)?)?$"

sims_root = datadir("exp_pro", "sims", "ude_single")
feasibility_rows = []

for sim in sort(readdir(sims_root))
    m = match(SIM_NAME_RE, sim)
    isnothing(m) && continue
    !haskey(BETA_FUNCTIONS, m[:beta]) && continue

    beta_function = m[:beta]
    train_length  = parse(Int, m[:traindata])
    noise         = parse(Float64, m[:noise])
    sim_root_dir  = joinpath(sims_root, sim)

    for loc_folder in sort(readdir(sim_root_dir))
        !isdir(joinpath(sim_root_dir, loc_folder)) && continue
        !startswith(loc_folder, "synthetic_") && continue
        location = replace(loc_folder, "synthetic_" => "")
        !haskey(POPULATION, location) && continue

        println("Processing beta: $(beta_function), location: $(location), train_length: $(train_length), noise: $(noise)")

        sim_name_dir = joinpath(sim_root_dir, loc_folder)
        population   = POPULATION[location]

        df       = summarise_results(sim_name_dir, population)
        if nrow(df) == 0
            println("No seed results found in $(sim_name_dir), skipping.")
            continue
        end
        out_path = joinpath(sim_name_dir, "results_summary.csv")
        CSV.write(out_path, df)
        println("Saved summary to: $out_path")

        # First check forecasts over the full 3-year horizon are reasonable (i.e. less than the population size).
        # Seeds with missing predictions are treated as infeasible (can't verify them), and are
        # dropped here so they never enter the val-loss-based multistart cut below.
        n_rows_before    = nrow(df)
        feasible_mask    = coalesce.(df.feasible, false)
        infeasible_seeds = df[.!feasible_mask, :seed]
        df               = df[feasible_mask, :]
        n_infeasible     = n_rows_before - nrow(df)
        pct_infeasible   = 100 * n_infeasible / n_rows_before
        println("Removed $(n_infeasible) infeasible seed(s) " *
                "(max_forecast > population=$(population)): $(infeasible_seeds)")

        push!(feasibility_rows, (
            sim_name       = sim,
            beta           = beta_function,
            train_length   = train_length,
            noise          = noise,
            location       = location,
            n_seeds        = n_rows_before,
            n_infeasible   = n_infeasible,
            pct_infeasible = pct_infeasible,
        ))

        # Sort by validation loss (ascending) once
        sort!(df, :best_val_loss, rev=false)
        n_rows = size(df, 1)

        if n_rows == 0
            println("No feasible seeds remain in $(sim_name_dir), skipping multistart cuts.")
            continue
        end

        for MS_limit in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            seeds_to_keep_path = joinpath(sim_name_dir, "seeds_to_keep_MS=$(MS_limit).jld2")
            if isfile(seeds_to_keep_path)
                println("Already have $seeds_to_keep_path, skipping.")
                continue
            end

            n_rows_MS     = Int(floor(n_rows * MS_limit))
            seeds_to_keep = df[1:end-n_rows_MS, :seed]

            @save seeds_to_keep_path seeds_to_keep
            println("Saved usable seeds to: $seeds_to_keep_path")
        end
    end
end

# Only write the ude_single summary if any ude_single folders were processed
if !isempty(feasibility_rows)
    feasibility_df = DataFrame(feasibility_rows)
    sort!(feasibility_df, :pct_infeasible, rev=true)
    feasibility_out_path = datadir("exp_pro", "feasibility_summary.csv")
    CSV.write(feasibility_out_path, feasibility_df)
    println("\nSaved feasibility summary ($(nrow(feasibility_df)) combos) to: $(feasibility_out_path)")
end


#========================================================
MULTISTART FOR THE BEHAVIOURAL UDE (ude_behaviour_model1)
=========================================================#

# Behaviour sim folders, e.g. UDE_behaviour_model1_contact=contacts_exp_traindata=365_noise=0.0_dI=N0_c_loss_norm
const BEHAVIOUR_SIM_NAME_RE = r"^UDE_behaviour_model1_contact=(?<contact>contacts_(exp|rational|mixed))_traindata=(?<traindata>\d+)_noise=(?<noise>[\d.]+)_dI=(?<dI>none|N0)(?<suffix>.*)$"

function summarise_behaviour_results(sim_name_dir::String, location::String, contact::String, train_length::Int)

    # Noise-free ground truth for this location
    beta_name = replace(contact, "contacts" => "beta")
    I_true = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_HQ_$(beta_name)",
                               "synthetic_$(location)", "noise=0.0.jld2"))["infectious"]
    c_true = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_contact_trajectories_HQ", "ground_truth_$(contact)",
                               "location_$(location)", "synthetic_$(contact)_$(location)_over_time.jld2"))["contact"]

    # Same validation days as train_ude_single_dataset
    val_tpts   = 5:5:train_length
    population = POPULATION[location]

    rows = []
    for sim_folder in readdir(sim_name_dir; join=true)
        !isdir(sim_folder) && continue
        !occursin("simulation_seed=", basename(sim_folder)) && continue

        results_file = joinpath(sim_folder, "results.jld2")
        !isfile(results_file) && continue

        # Only read the keys we need (much faster than loading every seed in full)
        seed, prediction, solve_failed = jldopen(results_file, "r") do file
            file["seed"], file["prediction"], haskey(file, "solve_failed") ? file["solve_failed"] : false
        end

        # Feasible if the long-term solve finished and the 3-year forecast stays below the population
        # (NaN padding from a failed solve makes the maximum NaN, which also counts as infeasible)
        max_forecast = maximum(prediction[3, :])
        feasible     = !solve_failed && max_forecast <= population

        # Validation NMSEs recomputed from the saved prediction, each normalised by its own data range
        val_I = feasible ? loss_nmse(prediction[3, val_tpts], I_true[val_tpts]) : missing
        val_c = feasible ? loss_nmse(prediction[6, val_tpts], c_true[val_tpts]) : missing

        push!(rows, (
            sim_folder   = basename(sim_folder),
            seed         = seed,
            solve_failed = solve_failed,
            max_forecast = max_forecast,
            feasible     = feasible,
            val_I        = val_I,
            val_c        = val_c,
        ))
    end

    df = DataFrame(rows)
    nrow(df) > 0 && sort!(df, [:seed])

    return df
end

behaviour_root = datadir("exp_pro", "sims", "ude_behaviour_model1")
behaviour_feasibility_rows = []

for sim in (isdir(behaviour_root) ? sort(readdir(behaviour_root)) : String[])
    m = match(BEHAVIOUR_SIM_NAME_RE, sim)
    isnothing(m) && continue

    noise = parse(Float64, m[:noise])
    noise == 0 || error("Behaviour multistart only supports noise = 0 so far ($(sim))")
    contact      = String(m[:contact])
    train_length = parse(Int, m[:traindata])
    sim_root_dir = joinpath(behaviour_root, sim)

    for loc_folder in sort(readdir(sim_root_dir))
        !isdir(joinpath(sim_root_dir, loc_folder)) && continue
        !startswith(loc_folder, "synthetic_") && continue
        location = replace(loc_folder, "synthetic_" => "")
        !haskey(POPULATION, location) && continue

        println("Processing $(sim), location: $(location)")

        sim_name_dir = joinpath(sim_root_dir, loc_folder)
        df = summarise_behaviour_results(sim_name_dir, location, contact, train_length)
        if nrow(df) == 0
            println("No seed results found in $(sim_name_dir), skipping.")
            continue
        end
        CSV.write(joinpath(sim_name_dir, "results_summary.csv"), df)

        n_saved      = nrow(df)
        n_failed     = count(df.solve_failed)
        n_infeasible = count(.!df.feasible)
        df_feasible  = df[df.feasible, :]
        println("Saved $(n_saved) seed(s): $(n_failed) solve failed, $(n_infeasible) infeasible in total")

        push!(behaviour_feasibility_rows, (
            sim_name       = sim,
            location       = location,
            n_saved        = n_saved,
            n_solve_failed = n_failed,
            n_infeasible   = n_infeasible,
            pct_infeasible = 100 * n_infeasible / n_saved,
        ))

        nrow(df_feasible) == 0 && (println("No feasible seeds remain, skipping multistart cuts."); continue)

        # Rank on both data streams: sum of the infection and contact validation NMSEs (ascending)
        ranked = df_feasible[sortperm(df_feasible.val_I .+ df_feasible.val_c), :]
        n_rows = nrow(ranked)

        for MS_limit in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            seeds_to_keep_path = joinpath(sim_name_dir, "seeds_to_keep_MS=$(MS_limit).jld2")
            if isfile(seeds_to_keep_path)
                println("Already have $seeds_to_keep_path, skipping.")
                continue
            end

            n_rows_MS     = Int(floor(n_rows * MS_limit))
            seeds_to_keep = ranked[1:end-n_rows_MS, :seed]

            @save seeds_to_keep_path seeds_to_keep
            println("Saved usable seeds to: $seeds_to_keep_path")
        end
    end
end

if !isempty(behaviour_feasibility_rows)
    behaviour_feasibility_df = DataFrame(behaviour_feasibility_rows)
    behaviour_feasibility_out_path = datadir("exp_pro", "feasibility_summary_behaviour.csv")
    CSV.write(behaviour_feasibility_out_path, behaviour_feasibility_df)
    println("\nSaved behaviour feasibility summary ($(nrow(behaviour_feasibility_df)) combos) to: $(behaviour_feasibility_out_path)")
end

