using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using SymbolicUtils 
using SymbolicRegression
using Symbolics
using DynamicExpressions
using JLD2
using Random
using Statistics
using Plots
using MLJ
using CSV
using DataFrames

include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA
include(joinpath(@__DIR__, "functions.jl"))
using .Functions
include(joinpath(@__DIR__, "symbolic_regression_module.jl"))
using .symbolic_regression_module
include(joinpath(@__DIR__, "classify_sr_equations.jl"))
using .SR_equation_classification

# SR settings
SR_settings = "niterations=100_populations=15_populationsize=33_parsimony=0.0032_constantcomplexity=2_v10.5"


# First extract raw synthesised data
dataset = JLD2.load(datadir("synthesised_trajectories_single", "synthesised_MA.jld2"))


# Extract infectious individuals and days from the dataset
data = dataset["infectious"]
days = dataset["days"]


# Evaluate the true beta function for the noiseless data
location = "MA"

# x_hat normalised in SR module
x_hat = reshape(data, :, 1)
y_hat = reshape(Functions.beta_exp(location, data), :, 1)
#y_hat = true_beta = 0.5 .* exp.(-5 .* data)

seed_grid = rand(1:100000, 5)
#seed_grid = 86180

results_list = []


for seed in seed_grid

    # Create output directory
    sim_name = "noiseless_beta_seed_$(seed)"
    plot_title = "noiseless beta with seed = $(seed)"
    mkpath(joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)_outputs"))
    output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)_outputs")

    mach, r = symbolic_regression_module.symbolic_regression(x_hat, y_hat, sim_name, location, output_dir, plot_title, seed)

    # Store nmse
    # Recover SR prediction
    population = POPULATION[location]
    SR_beta = MLJ.predict(mach, x_hat ./ population)

    # Recover true beta result
    true_beta_SR = Functions.beta_exp(location, x_hat)
    nmse_SR_true = Functions.loss_nmse(SR_beta, true_beta_SR, maximum(true_beta_SR) - minimum(true_beta_SR))

    # Store equation
    best_tree = r.equations[r.best_idx].tree 
    best_equation = r.equations[r.best_idx]
    println(best_equation)
    #best_symbolic = SymbolicRegression.node_to_symbolic(best_tree, mach.model)
    

    # simplified_equation = SymbolicUtils.simplify(SymbolicUtils.expand(best_symbolic))
    
    best_symbolic = SymbolicRegression.node_to_symbolic(best_equation)
    println(best_symbolic)
    simplified_equation = SymbolicUtils.simplify(best_symbolic, expand=true)

    # Round to 3 significant figures for easier classification
    best_equation = Functions._format_equation_sigfigs(string(best_equation))
    simplified_equation = Functions._format_equation_sigfigs(string(simplified_equation))

    println("Simplified equation: ", simplified_equation)

    # Store complexity
    best_complexity = r.complexities[r.best_idx]

    # Store equation classification
    equation_classification, classification_description = SR_equation_classification.classify_eqn(string(simplified_equation))
    
    push!(results_list, (seed=seed, mse=mse_SR_true, best_equation=best_equation, simplified_equation=simplified_equation, best_complexity=best_complexity, equation_classification=equation_classification, classification_description=classification_description))

end

# Create grid to store results
results = DataFrame(results_list)
results_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "SR_noiseless_beta_results")
mkpath(results_dir)
JLD2.save(joinpath(results_dir, "SR_noiseless_beta_results.jld2"), "results", results)

# Convert to DataFrame and save as CSV
CSV.write(joinpath(results_dir, "SR_noiseless_beta_results_$(SR_settings).csv"), results)