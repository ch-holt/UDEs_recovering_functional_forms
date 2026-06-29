#===================================================================================
SCRIPT TO USE SYMBOLICREGRESSION.JL FOR SYMBOLIC REGRESSION ON MULTIPLE TRAJECTORIES
====================================================================================#  

using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
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
using Plots
using Statistics
using ComponentArrays
using DataFrames
using Random; rng = Random.default_rng()
# Call the loss functions
include(joinpath(@__DIR__, "functions.jl"))
using .Functions
include(joinpath(@__DIR__, "symbolic_regression_module.jl"))
using .symbolic_regression_module
include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

#========================================================
SET UP NEURAL NETWORK
=========================================================# 

# Define the NN architecture
hidden_dims = 5

const gamma = 1/10

# Retrieve nn architecture
beta_network = Lux.Chain(Lux.Dense(4=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, sigmoid))

# Initialise parameters
p_nn_temp, st_nn = Lux.setup(rng, beta_network)

#=============================================================
RETRIEVE PREDICTIONS AND PARAMETERS FROM THE BEST SIMULATION
=============================================================#

sim_settings = "exponential_testing_one_input_100626"
sim_name = "simulation_v1"
location = "MA"
filename = "synthesised_$(location).jld2"
population = POPULATION[location]

dataset = JLD2.load(datadir("synthetic_trajectories_exponential", filename))

# Just use infectious trajectory
obs = dataset["infectious"]
days = dataset["days"]

# Retrieve NN parameters
training_results = JLD2.load(DrWatson.datadir("sims", "ude_multiple", sim_settings, sim_name, "training_results.jld2"))
results = JLD2.load(DrWatson.datadir("sims", "ude_multiple", sim_settings, sim_name, filename, "results.jld2"))
p_trained = training_results["p_trained"]
days = results["days"]

# Retrieve normalised infectious trajectory
norm_i_traj = results["prediction"][3, 1:length(obs)]./ population

# Normalise the parameters for the symbolic regression
delta_norm =(log(varying_p.delta) - log(1e-6))/(log(1e-2) - log(1e-6))
beta0_norm = (varying_p.R0_reproduction- 1.2)/(6.0 - 1.2)
zeta_norm = varying_p.zeta/0.05

varying_p = ComponentArray(
    population = dataset["varying_p"]["population"],
    prevalence = dataset["varying_p"]["prevalence"],
    delta = dataset["varying_p"]["delta"],
    R0_reproduction = dataset["varying_p"]["R0_reproduction"],
    zeta = dataset["varying_p"]["zeta"]
)

# Derive beta0 specific to current trajectory
beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)

# Define the grid of I values for the symbolic regression
I_grid = collect(range(0, 1; length=1000))

# Define the SR inputs (1000 inputs)
SR_input_pair_1000 = DataFrame(
    I_norm = vec(I_grid),
    beta0_norm = fill(beta0_norm, 1000),
    zeta_norm = fill(zeta_norm, 1000),
    delta_norm = fill(delta_norm, 1000)
)

nn_input_1000 = vcat(fill(beta0_norm, 1, 1000), fill(zeta_norm, 1, 1000), 
                fill(delta_norm, 1, 1000), reshape(I_grid, 1, :))

# Evaluate neural network and extract approximation (1000 inputs)
nn_output_1000 = vec(beta_network(nn_input_1000, p_trained, st_nn)[1])

nn_input_days = vcat(fill(beta0_norm, 1, length(norm_i_traj)), fill(zeta_norm, 1, length(norm_i_traj)), 
                fill(delta_norm, 1, length(norm_i_traj)), reshape(norm_i_traj, 1, :))

nn_output_days = vec(beta_network(nn_input_days, p_trained, st_nn)[1])


# Create output directory
plot_title = "NN approximation no noise"
sim_name_SR = "UDE_multiple_exponential_normalised_$(sim_name)_$(filename)_v3"
output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name_SR)")

symbolic_regression_module.symbolic_regression(SR_input_pair_1000, nn_output_1000, sim_name_SR, location, output_dir, plot_title, 1234, "multi", norm_i_traj, nn_output_days, "exponential")