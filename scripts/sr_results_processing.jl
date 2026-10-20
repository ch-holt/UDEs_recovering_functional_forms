#========================================================
SCRIPT TO PROCESS THE SR RESULTS - 
    1) Extract the Pareto fronts for each SR run (i.e. each seed)
    2) Extract the validation set losses for each UDE run (i.e. each seed) and the equivalent MS limit it gets removed
=========================================================#  

using DrWatson

@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using SymbolicRegression
using DataFrames
using CSV
using Statistics

# Call module
using UDE_FUNCTIONAL_FORMS


const BETA_FUNCTIONS = Dict("beta_exp" => beta_exp, "beta_rational" => beta_rational, "beta_mixed" => beta_mixed)
const beta_function = BETA_FUNCTIONS[get(ARGS, 1, "beta_exp")]
const noise = parse(Float64, get(ARGS, 2, "0.0"))
const train_length = parse(Int, get(ARGS, 3, "365"))
const solver_name = get(ARGS, 4, "autotsit5")
# Collect all seeds (i.e. no multistart yet)
const MS_limit = parse(Float64, get(ARGS, 5, "0.0"))
const multistart = true

# Files are saved in a different folder due to file size and OneDrive
const SR_RESULTS_ROOT = expanduser("~/UDE_results")


#==============================================================
# Extract the Pareto fronts and validation losses for each seed
==============================================================#

# Extract and save a summary of each seed's hall of fame (i.e. equation, complexity, loss, ..)
function extract_fronts(sr_root, sim_name)
    # make tuple to save results
    rows = NamedTuple[]
    for location in sort(collect(keys(POPULATION)))
        for setting in filter(startswith("SR_input="), readdir(sr_root))
            SR_settings_run = joinpath(sr_root, setting)
            # MS saved in the UDE root -> just processing all seeds for now (i.e. MS=0.0)
            root = datadir("exp_pro","sims",  "ude_single", sim_name, "synthetic_$(location)")
            seed_folders = get_seed_folders(root, multistart, MS_limit)
            for seed_folder in seed_folders
                output_dir = joinpath(SR_settings_run, "synthetic_$(location)", seed_folder)
                report_path = joinpath(output_dir, "SR_report.jld2")
                if !isfile(report_path)
                    println("No SR results for $(location) $(seed_folder), skipping.")
                    continue
                end
                # Open the report for each SR run
                r = jldopen(report_path, "r") do f
                    f["report"]
                end
                # get the seed number
                seed = parse(Int, split(seed_folder, "=")[2])
                # extract the results for each equation in the hall of fame
                for i in eachindex(r.equations)
                    push!(rows, (; setting, location, seed,
                                complexity  = r.complexities[i],
                                loss        = r.losses[i],
                                score       = r.scores[i],
                                equation    = r.equation_strings[i],
                                is_selected = i == r.best_idx))
                end

            end
        end
    end
    fronts = DataFrame(rows)

    return fronts
end

# Extract the validation set losses for each UDE run -> this helps us rank and save when each seed is removed by MS
function build_seed_table(sim_name)
    seed_tables = DataFrame[]
    for location in sort(collect(keys(POPULATION)))
        root = datadir("exp_pro","sims",  "ude_single", sim_name, "synthetic_$(location)")
        # Read the summary created in summarise_results.jl and multistart.jl
        df = CSV.read(joinpath(root, "results_summary.csv"), DataFrame)
        # Select the columns to keep
        df = select(df, :seed, :best_val_loss, :feasible)
        # Add location
        df.location .= location
        # Select best seed (i.e. seed with lowest val loss and feasible)
        best = minimum(df.best_val_loss[df.feasible])
        df.is_best_seed = df.feasible .& (df.best_val_loss .== best)

        # work out what seeds are removed when
        # set all to 0
        df.max_ms_kept = zeros(nrow(df))
        # then start from 0.1
        for limit in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
            seeds_to_keep = JLD2.load(joinpath(root, "seeds_to_keep_MS=$(limit).jld2"), "seeds_to_keep")
            df.max_ms_kept[in(seeds_to_keep).(df.seed)] .= limit
        end
        # Put dataframe into seed_tables
        push!(seed_tables, df)
    end
    # Put each locations' table on top of each other
    seed_table = vcat(seed_tables...)

    println("Seed table: $(nrow(seed_table)) rows, $(count(seed_table.is_best_seed)) best seeds")
    println(combine(groupby(seed_table, :max_ms_kept), nrow => :n_seeds))

    return seed_table
end

#======================================================================
# Create an ensemble with the seeds at a specified multistart threshold
======================================================================#

function ensemble_seeds(seed_table, location; ms)
    # Select the seeds that are kept at the specified multistart threshold
    selected_seeds = seed_table[(seed_table.location .== location) .& (seed_table.max_ms_kept .>= ms), :seed]
    return selected_seeds
end

#======================================================================
# Load the SR results
======================================================================#

function load_sr_ensemble(sr_root, setting, location, seeds)

    # Load the SR report for each seed and extract the infectious and beta trajectories

    sr_inf_seeds = Vector{Vector{Float64}}()
    sr_beta_0_1_seeds = Vector{Vector{Float64}}()

    for seed in seeds
        output_dir = joinpath(sr_root, setting, "synthetic_$(location)", "simulation_seed=$(seed)")
        report_path = joinpath(output_dir, "SR_report.jld2")
        if !isfile(report_path)
            println("No SR results for $(location) $(seed), skipping.")
            continue
        end
        sr_inf, sr_beta_0_1 = jldopen(report_path, "r") do f
            f["SR_inf"], vec(f["SR_beta_0_1"])
        end

        push!(sr_inf_seeds, sr_inf)
        push!(sr_beta_0_1_seeds, sr_beta_0_1)
    end

    return (; setting, location, seeds, sr_inf_seeds, sr_beta_0_1_seeds)

end

#======================================================================
# Load the truth
======================================================================#

function compute_ground_truth(beta_function, location, train_length)

    data_suffix = "HQ_"
    dataset = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_$(data_suffix)$(beta_function)",
                                 "synthetic_$(location)", "noise=0.0.jld2"))
    true_inf = dataset["infectious"][1:train_length]
    I_grid = collect(range(0, 1; length=1000))

    # Split inputs into in-sample and out-of-sample
    I_N_max  = maximum(true_inf ./ POPULATION[location])
    in_sample_bool  = I_grid .<= I_N_max
    out_of_sample_bool = I_grid .>  I_N_max
    true_beta_I = beta_function(location, I_grid .* POPULATION[location])
    return (; true_inf, I_grid, true_beta_I, in_sample_bool, out_of_sample_bool)
end


#====================
# Calculate the CRPS
=====================#

# Evaluates the CRPS for one time point
# CRPS(F, y) = E|X − y| − ½ E|X − X′|
# x is the vector of predictions, y is the true value
function crps_sample(x::Vector{Float64}, y::Float64)
    n = length(x)
    # Evaluate average distance to the true value (accuracy)
    # Outputs a vector of MAEs for every individual seed
    accuracy = mean(abs.(x .- y))
    # Then we evaluate the spread (i.e. average distance between all pairs of predictions)
    s = 0.0
    @inbounds for i in 1:n, j in 1:n
        s += abs(x[i] - x[j])
    end
    # Divide by n^2 to get average distance (and the 2 comes from the half in the equation)
    spread = s / (2 * n^2)
    return accuracy - spread
end


# Combines CRPS for all time points in a trajectory (mean of the CRPS for each time point)
function crps_of(rows, truth; log1p_transform::Bool=false, floor::Union{Nothing,Float64}=nothing)
    preds = stack(rows; dims=1)
    # Floor the predictions (e.g. at zero for infectious counts) -> also prevents numerical errors when taking log1p
    if !isnothing(floor)
        preds   = max.(preds, floor)
        truth = max.(truth, floor)
    end
    # Optional to score on the log scale
    # Log the predictions if they vary a lot in magnitude (e.g. for infectious counts)
    if log1p_transform
        preds   = log1p.(preds)
        truth = log1p.(truth)
    end
    n_horizon = length(truth)
    errs = [crps_sample(preds[:, h], truth[h]) for h in 1:n_horizon]
    return mean(errs)
end

#============================================================
# Function to calculate the CRPS for each setting and location
============================================================#

function crps_of_ensemble(settings, locations, seed_table, ms, sr_root, ground_truth)

    rows = NamedTuple[]
    # Loop over settings and locations
    for setting in settings, location in locations
        location == first(locations) && println("Scoring $(setting)")
        # Extract the seeds that are kept at the specified multistart threshold
        seeds = ensemble_seeds(seed_table, location; ms)
        # Create an ensemble of the SR results for these seeds
        ensemble = load_sr_ensemble(sr_root, setting, location, seeds)

        # Calculate the CRPS for each individual setting and location
        crps_inf      = crps_of(ensemble.sr_inf_seeds, ground_truth[location].true_inf; log1p_transform = true, floor = 0.0)
        crps_beta_in  = crps_of([r[ground_truth[location].in_sample_bool]  for r in ensemble.sr_beta_0_1_seeds], ground_truth[location].true_beta_I[ground_truth[location].in_sample_bool])
        crps_beta_out = crps_of([r[ground_truth[location].out_of_sample_bool] for r in ensemble.sr_beta_0_1_seeds], ground_truth[location].true_beta_I[ground_truth[location].out_of_sample_bool])

        push!(rows, (; setting, location, crps_inf, crps_beta_in, crps_beta_out))
    
    end

    df = DataFrame(rows)

    # Extract the baseline CRPS for the relative CRPS
    baseline = CSV.read(projectdir("_research", "crps_tables", "crps_ideal_autotsit5_.csv"), DataFrame)
    baseline = filter(:beta_function => ==(string(beta_function)), baseline)
    baseline = select(baseline, :location, :log_crps_baseline_inf, :crps_baseline_beta_insample, :crps_baseline_beta_outsample)
    df = leftjoin(df, baseline, on = :location)
    df.rcrps_inf      = df.crps_inf      ./ df.log_crps_baseline_inf
    df.rcrps_beta_in  = df.crps_beta_in  ./ df.crps_baseline_beta_insample
    df.rcrps_beta_out = df.crps_beta_out ./ df.crps_baseline_beta_outsample


    return df
end

sim_name = "$(solver_name)_UDE_single_beta=$(beta_function)_adam=2500_lbfgs=2000_traindata=$(train_length)_noise=$(noise)"
sr_root = joinpath(SR_RESULTS_ROOT, sim_name)   

# Only rebuild if the file doesn't exist yet
save_dir = datadir("exp_pro", "sr_comparison", sim_name)
mkpath(save_dir)
fronts_path = joinpath(save_dir, "sr_pareto_fronts.jld2")
if isfile(fronts_path)
    fronts = JLD2.load(fronts_path, "fronts")
else
    fronts = extract_fronts(sr_root, sim_name)
    jldsave(fronts_path; fronts)
end

# Only rebuild if the file doesn't exist yet
seed_path = joinpath(save_dir, "ude_seed_table.jld2")
if isfile(seed_path)
    seed_table = JLD2.load(seed_path, "seed_table")
else
    seed_table = build_seed_table(sim_name)
    jldsave(seed_path; seed_table)
end

# Create dataframe of the CRPS for each setting and location
crps_path = joinpath(save_dir, "crps_summary_ms=$(MS_limit).jld2")
if isfile(crps_path)
    crps_summary = JLD2.load(crps_path, "crps_summary")
else
    settings = filter(startswith("SR_input="), readdir(sr_root))
    locations = sort(collect(keys(POPULATION)))
    ground_truth = Dict{String, Any}()
    for location in locations
        ground_truth[location] = compute_ground_truth(beta_function, location, train_length)
    end
    crps_summary = crps_of_ensemble(settings, locations, seed_table, MS_limit, sr_root, ground_truth)
    jldsave(crps_path; crps_summary)
end