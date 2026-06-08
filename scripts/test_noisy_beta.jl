#=====================================================================
SCRIPT TO TEST HOW WELL SYMBOLIC REGRESSION WORKS WITH NOISY FUNCTIONS
======================================================================#  

using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using Random
using LinearAlgebra
using Plots
using MLJ
using SymbolicRegression
using CSV
using DataFrames

include(joinpath(@__DIR__, "functions.jl"))
using .Functions
include(joinpath(@__DIR__, "symbolic_regression_module.jl"))
using .symbolic_regression_module

include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

#========================================================
LOAD DATA
=========================================================#

# First extract raw synthesised data
dataset = JLD2.load(datadir("synthesised_trajectories_single", "synthesised_MA.jld2"))

# Extract infectious individuals and days from the dataset
data = dataset["infectious"]
days = dataset["days"]

# Evaluate the true beta function for the noiseless data
location = "MA"
true_beta_noiseless = Functions.beta_exp(location, data)


function SR_on_noisy_beta(noise_SD,location, data, true_beta_noiseless)
    # Add noise
    # Set seed for reproducibility
    rng = Random.seed!(1234)

    noisy_beta = Functions.add_gaussian_noise(noise_SD, true_beta_noiseless, rng)

    #========================================================
    UNDERTAKE SYMBOLIC REGRESSION
    =========================================================#

    # Define beta input (normalised in SR module)
    x_hat = reshape(data, :, 1)

    # Define beta output
    y_hat = noisy_beta 

    # Create output directory
    sim_name = "noisy_beta_noise_SD_$(noise_SD)"
    plot_title = "noisy beta with SD = $(noise_SD)"
    mkpath(joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)_outputs"))
    output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)_outputs")

    symbolic_regression_module.symbolic_regression(x_hat, y_hat, sim_name, location, output_dir, plot_title, 1234)
end

#noise_SD_grid = [0.01, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.04, 0.05]


noise_SD_grid = [0]

results_list = []

for noise_SD in noise_SD_grid


    mach, r = SR_on_noisy_beta(noise_SD, location, data, true_beta_noiseless)

    # Store mse

    # Recover SR prediction
    population = POPULATION[location]
    SR_beta = predict(mach, x_hat/population)
    # Recover true beta result
    true_beta_SR = Functions.beta_exp(location, x_hat)
    mse_SR_true = Functions.loss_mse(SR_beta, true_beta_SR)

    # Store equation
    best_equation = r.equation_strings[r.best_idx]

    push!(results_list, (noise_SD=noise_SD, mse=mse_SR_true, best_equation=best_equation))

end

    # Create grid to store results
    results = DataFrame(results_list)
    results_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "SR_noisy_beta_results")
    mkpath(results_dir)
    JLD2.save(joinpath(results_dir, "SR_noisy_beta_results.jld2"), "results", results)

    # Convert to DataFrame and save as CSV
    CSV.write(joinpath(results_dir, "SR_noisy_beta_results.csv"), results)