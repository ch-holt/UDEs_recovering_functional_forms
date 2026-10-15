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

# Save in the exp_pro folder
jldsave(joinpath(save_dir, "sr_pareto_fronts.jld2"); fronts)

# Rebuild if doesn't exist
seed_path = joinpath(save_dir, "ude_seed_table.jld2")
if isfile(seed_path)
    seed_table = JLD2.load(seed_path, "seed_table")
else
    seed_table = build_seed_table(sim_name)
    jldsave(seed_path; seed_table)
end
