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
using .Functions

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

# Add noise
# Set seed for reproducibility
rng = Random.seed!(1234)
noise_SD = 0.05

noisy_beta = Functions.add_gaussian_noise(noise_SD, true_beta_noiseless, rng)

#========================================================
UNDERTAKE SYMBOLIC REGRESSION
=========================================================#

# Define beta input

include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

population = POPULATION[location]
x_hat = reshape(data./population, :, 1)

# Define beta output
y_hat = noisy_beta 

# Create output directory
sim_name = "noisy_beta_noise_SD_$(noise_SD)"
plot_title = "noisy beta with SD = $(noise_SD)"
mkpath(joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)_outputs"))
output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)_outputs")

symbolic_regression_module.symbolic_regression(x_hat, y_hat, sim_name, location, output_dir, plot_title, 1234)
