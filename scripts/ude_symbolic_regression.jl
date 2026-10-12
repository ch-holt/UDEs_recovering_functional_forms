#========================================================
SCRIPT TO USE SYMBOLICREGRESSION.JL FOR SYMBOLIC REGRESSION
=========================================================#  

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using JLD2
# Import symbolic regression package and MLJ interface
using SymbolicUtils 
using SymbolicRegression
using Symbolics
using DynamicExpressions
using Lux
using MLJ 
using SymbolicRegression
using Statistics
using ComponentArrays
using OrdinaryDiffEq
using Random; rng = Random.default_rng()

# Call module
using UDE_FUNCTIONAL_FORMS

function symbolic_regression(I_nn, best_results, output_dir, input_size, location, seed, setting)

    #========================================================
    SET UP NEURAL NETWORK
    =========================================================# 

    hidden_dims = 5

    output_size = 1
    activation_function = gelu
    final_activation_function = softplus

    beta_network, p_nn_temp, st_nn = build_neural_network(rng, hidden_dims, input_size, output_size, 
                                activation_function, final_activation_function)

    #=============================================================
    RETRIEVE PREDICTIONS AND PARAMETERS
    =============================================================#

    population = POPULATION[location]


    p_trained = best_results["p"]
    days = best_results["days"]


    #=============================================================
    DEFINE SR and NN INPUTS 
    =============================================================#

    # Extract NN approximation
    norm_i_traj = I_nn ./ population

    # Evaluate neural network and extract approximation
    nn_output_days = vec(beta_network(norm_i_traj, p_trained.nn_params, st_nn)[1])

    max_i_norm = maximum(norm_i_traj)
    min_i_norm = minimum(norm_i_traj)

    SR_input_1000, SR_input_0_1, SR_input_days, nn_input_1000, nn_input_0_1, nn_input_days, I_grid, range_0_1 = build_sr_inputs(p_trained, population, norm_i_traj, min_i_norm, max_i_norm, input_size)
    
    nn_output_1000 = vec(beta_network(nn_input_1000, p_trained.nn_params, st_nn)[1])
    nn_output_0_1 = vec(beta_network(nn_input_0_1, p_trained.nn_params, st_nn)[1])

    # Choose the grid SR is fitted on (results are always evaluated on the [0, 1] grid and the days)
    if setting.input == "unit"
        # 1000 points on [0, 1]
        SR_input_fit, nn_output_fit = SR_input_0_1, nn_output_0_1
    elseif setting.input == "train"
        # 1000 points across the training range
        SR_input_fit, nn_output_fit = SR_input_1000, nn_output_1000
    elseif setting.input == "train_pm10"
        # 100 points across the training range, extended by 10% of its width at each end (clipped at 0)
        width = max_i_norm - min_i_norm
        I_grid_pm10 = collect(range(max(0.0, min_i_norm - 0.1 * width), max_i_norm + 0.1 * width; length=100))
        SR_input_fit = reshape(I_grid_pm10, :, 1)
        nn_output_fit = vec(beta_network(reshape(I_grid_pm10, 1, :), p_trained.nn_params, st_nn)[1])
    else
        error("Unknown SR input $(setting.input)")
    end

    mach, r, best_equation, simplified_equation = run_symbolic_regression(SR_input_fit, nn_output_fit, output_dir; seed=seed,
        parsimony=setting.parsimony, maxsize=setting.maxsize, complexity_of_constants=setting.complexity_of_constants)

    return mach, r, best_equation, simplified_equation, SR_input_1000, SR_input_0_1, SR_input_days, nn_input_days, nn_output_days, nn_input_1000, nn_output_1000, nn_input_0_1, nn_output_0_1, I_grid, range_0_1, I_nn, norm_i_traj, SR_input_fit, nn_output_fit
end

#=============================================================
DEFINE PARAMETERS
=============================================================#

# beta functional form - default exp
const BETA_FUNCTIONS = Dict("beta_exp" => beta_exp, "beta_rational" => beta_rational, "beta_mixed" => beta_mixed)
const beta_function = BETA_FUNCTIONS[get(ARGS, 1, "beta_exp")]
# noise level of the UDE runs to read - default 0
const noise = parse(Float64, get(ARGS, 2, "0.0"))

# Number of data points the UDE was trained on - default 365
const train_length = parse(Int, get(ARGS, 3, "365"))
# Solver the UDE was trained with - default autotsit5 (AutoTsit5(Rosenbrock23()), default reltol)
const solver_name = get(ARGS, 4, "autotsit5")
# Multistart cut-off used to choose which seeds to keep - default 0.3
const MS_limit = parse(Float64, get(ARGS, 5, "0.3"))
const multistart = true

println("Settings: beta=$(beta_function), noise=$(noise), train_length=$(train_length), solver=$(solver_name), MS_limit=$(MS_limit)")

# SR experiments, each run on every seed. maxsize=30 is SymbolicRegression.jl's default
# 1. Which NN input/output pairs SR is fitted on (parsimony=3, maxsize=30, complexity_of_constants=2)
const SR_INPUT_SETTINGS = [(input=input, parsimony=3, maxsize=30, complexity_of_constants=2) for input in ["unit", "train", "train_pm10"]]
# 2. SR settings, fitted on the NN on 1000 points on [0, 1]
const SR_OPTION_SETTINGS = [(input="unit", parsimony=parsimony, maxsize=maxsize, complexity_of_constants=coc)
                            for parsimony in [2, 3, 4] for maxsize in [10, 14] for coc in [1, 2, 3]]
const SR_SETTINGS = vcat(SR_INPUT_SETTINGS, SR_OPTION_SETTINGS)

sr_name(setting) = "SR_input=$(setting.input)_parsimony=$(setting.parsimony)_maxsize=$(setting.maxsize)_coc=$(setting.complexity_of_constants)"




# create an array job - each job is given a number so all locations run simultaneously
const LOCATIONS = sort(collect(keys(POPULATION)))
location = LOCATIONS[parse(Int, ENV["SLURM_ARRAY_TASK_ID"])]
println("Running symbolic regression for location: $(location)")

sim_name = "$(solver_name)_UDE_single_beta=$(beta_function)_adam=2500_lbfgs=2000_traindata=$(train_length)_noise=$(noise)"
root = datadir("exp_pro","sims",  "ude_single", sim_name, "synthetic_$(location)")

# Number of NN inputs (single vs multiple datset settings)
input_size = 1
population = POPULATION[location]

# Load the observed data
data = JLD2.load(DrWatson.datadir("exp_pro","synthetic_data","synthetic_trajectories_HQ_$(beta_function)", "synthetic_$(location)", "noise=$(noise).jld2"))

# Just use infectious trajectory
obs = data["infectious"]
days = data["days"]

# Define the timespan for the ODE solver
tspan = [1, 365]

# Define initial state

# Latent period of 3 days represented by incubation rate sigma
sigma = 1/3 
# Infectious period of 10 days represented by recovery rate gamma
gamma = 1/10

# Define initial state same as the generated data
# Retrieve fixed parameters
E0 = 1.0
R0_recovered = 0.0
D0 = 0.0

population = POPULATION[location]
prevalence = PREVALENCE[location]
delta = DELTA[location]
R0_reproduction = R0_REPRODUCTION[location]
zeta = ZETA[location]

# Derive other parameters
beta0 = R0_reproduction * (gamma + delta)
I0 = max(1.0, prevalence * population)
S0 = population - E0 - I0 - R0_recovered - D0

# Define initial state
u0 = [S0, E0, I0, R0_recovered, D0]

# Define ODE problem with recovered SR equation

seed_folders = get_seed_folders(root, multistart, MS_limit)

# Count failed runs so the job is marked FAILED in SLURM if any seed errors
const n_failed = Ref(0)

# Seeds run one after another on one thread: SR with deterministic=true uses a global
# counter that is not thread safe, so threading over seeds makes results irreproducible
for setting in SR_SETTINGS, folder in seed_folders

    # Create output directory
    output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)", sr_name(setting), "synthetic_$(location)", folder)

    # Resume support: skip a seed whose SR results already exist
    if isfile(joinpath(output_dir, "SR_report.jld2"))
        println("$(sr_name(setting)) for $(folder) already exists for $(location), skipping.")
        continue
    end
    # Catch any errors so that the other seeds still run
    try
        println("Running $(sr_name(setting)) for $(folder)")

        # Find UDE results for this seed
        I_nn, results = extract_ude(root, obs, folder)

        # Use the same seed as the UDE initialisation (encoded in the folder name)
        seed = parse(Int, split(folder, "=")[2])

        t_start_sr = time()
        mach, r, best_equation, simplified_equation, SR_input_1000, SR_input_0_1, SR_input_days, nn_input_days, nn_output_days, nn_input_1000, nn_output_1000, nn_input_0_1, nn_output_0_1, I_grid, range_0_1, I_nn, norm_i_traj, SR_input_fit, nn_output_fit = symbolic_regression(I_nn, results, output_dir, input_size, location, seed, setting)
        elapsed_sr = time() - t_start_sr 

        #=============================================================
        EVALUATE BETA AGAINST DAYS
        =============================================================#

        # Evaluate the beta function for number of days
        true_beta_days = beta_function(location, norm_i_traj.*population)

        # Evaluate the SR equation for number of days
        SR_beta_days = reshape(predict(mach, (data=SR_input_days, idx=r.best_idx)),:,1)

        # Evaluate NN output for number of days
        NN_beta_days = reshape(nn_output_days, :, 1)

        #=============================================================
        EVALUATE BETA BETEEN 0 AND 1
        =============================================================#

        # Evaluate the true beta function for the grid of 1000 points
        true_beta_0_1 = beta_function(location, range_0_1.*population)

        # Evaluate the SR equation for the grid of 1000 points
        SR_beta_0_1 = reshape(predict(mach, (data=SR_input_0_1, idx=r.best_idx)),:,1)

        # Evaluate the NN output for the grid of 1000 points
        NN_beta_0_1 = reshape(nn_output_0_1, :, 1)

        #=============================================================
        RUN SEIRD + SR MODEL
        =============================================================#


        seird_sr! = make_seird_sr(mach, r, sigma, gamma, delta, population, input_size)
        prob_sr = ODEProblem(seird_sr!, u0, tspan)
        sol_sr = solve(prob_sr, AutoTsit5(Rosenbrock23()), saveat=1.0)
        i_sr = sol_sr[3, 1:length(obs)]

        # Save both report and machine to the same file and all trajectories
        JLD2.save(joinpath(output_dir, "SR_report.jld2"), "report", r, "mach", mach,
            "SR_beta_days", SR_beta_days, "SR_beta_0_1", SR_beta_0_1, "SR_inf", i_sr,
            "true_beta_days", true_beta_days, "NN_beta_days", NN_beta_days,
            "true_beta_0_1", true_beta_0_1, "NN_beta_0_1", NN_beta_0_1,
            "range_0_1", range_0_1, "SR_input_days", SR_input_days, "I_nn", I_nn,
            "obs", obs, "days", days, "simplified_equation", simplified_equation,
            "sr_setting", setting, "SR_input_fit", SR_input_fit, "NN_output_fit", nn_output_fit,
            "elapsed_seconds", elapsed_sr)
    catch e
        println("Error occurred for $(sr_name(setting)), $(folder): $e")
        n_failed[] += 1
    end
end

if n_failed[] > 0
    println("$(n_failed[]) SR run(s) failed for $(location)")
    exit(1)
end   