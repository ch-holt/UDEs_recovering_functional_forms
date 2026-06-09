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

# Retrieve nn architecture
beta_network = Lux.Chain(Lux.Dense(4=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, softplus))

# Initialise parameters
p_nn_temp, st_nn = Lux.setup(rng, beta_network)

#=============================================================
RETRIEVE PREDICTIONS AND PARAMETERS FROM THE BEST SIMULATION
=============================================================#

sim_settings = "exponential_080626"
sim_name = "simulation_v1"
location = "NE"
filename = "synthesised_$(location).jld2"
population = POPULATION[location]

dataset = JLD2.load(datadir("synthetic_trajectories_exponential", filename))
# Just use infectious trajectory
obs = dataset["infectious"]
days = dataset["days"]

# Retrieve NN parameters that resulted in the lowest error on the training data
training_results = JLD2.load(DrWatson.datadir("sims", "ude_multiple", sim_settings, sim_name, "training_results.jld2"))
results = JLD2.load(DrWatson.datadir("sims", "ude_multiple", sim_settings, sim_name, filename, "results.jld2"))
p_trained = training_results["p_trained"]
days = results["days"]
norm_i_traj = results["prediction"][3, 1:length(obs)]./ population

varying_p = ComponentArray(
    population = dataset["varying_p"]["population"],
    prevalence = dataset["varying_p"]["prevalence"],
    delta = dataset["varying_p"]["delta"],
    R0_reproduction = dataset["varying_p"]["R0_reproduction"],
    zeta = dataset["varying_p"]["zeta"]
)

# Derive beta0 specific to current trajectory
beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)

I_grid = collect(range(0, 1; length=1000))

nn_input = vcat(fill(beta0, 1, 1000), fill(varying_p.zeta, 1, 1000), 
    fill(varying_p.delta, 1, 1000), reshape(I_grid, 1, :))

x_hat = DataFrame(
    I = vec(I_grid),
    beta0 = fill(beta0, 1000),
    zeta = fill(varying_p.zeta, 1000),
    delta = fill(varying_p.delta, 1000)
)

# Evaluate neural network and extract approximation
y_hat = vec(beta_network(nn_input, p_trained, st_nn)[1])
nn_output = vec(beta_network(vcat(fill(beta0, 1, length(norm_i_traj)), fill(varying_p.zeta, 1, length(norm_i_traj)), 
                fill(varying_p.delta, 1, length(norm_i_traj)), reshape(norm_i_traj, 1, :)), p_trained, st_nn)[1])
# Create output directory
plot_title = "NN approximation no noise"
sim_name_SR = "UDE_multiple_exponential_$(sim_name)_$(filename)"
output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name_SR)")

symbolic_regression_module.symbolic_regression(x_hat, y_hat, sim_name_SR, location, output_dir, plot_title, 1234, "multi", norm_i_traj, nn_output, "exponential")