#========================================================
SCRIPT TO USE SYMBOLICREGRESSION.JL FOR SYMBOLIC REGRESSION
=========================================================#  

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
using Random; rng = Random.default_rng()

# Call module
using UDE_FUNCTIONAL_FORMS

include(joinpath(@__DIR__, "symbolic_regression_module.jl"))
using .symbolic_regression_module

#========================================================
SET UP NEURAL NETWORK
=========================================================# 

hidden_dims = 5
input_size = 1
output_size = 1
activation_function = gelu
final_activation_function = sigmoid

beta_network, p_nn_temp, st_nn = build_neural_network(rng, hidden_dims, input_size, output_size, 
                            activation_function, final_activation_function)

#=============================================================
RETRIEVE PREDICTIONS AND PARAMETERS FROM THE BEST SIMULATION
=============================================================#

sim_settings = "UDE_multiple_exponential_one_input_nmse"
sim_name = "simulation_v1"
location = "MA"
filename = "synthesised_$(location).jld2"
population = POPULATION[location]

dataset = JLD2.load(datadir("synthetic_trajectories_beta_exp", filename))

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

x_grid = collect(range(0, 1; length=1000))
nn_input = reshape(x_grid, 1, :)
x_hat = reshape(x_grid, :, 1)

# Evaluate neural network and extract approximation
y_hat = vec(beta_network(nn_input, p_trained, st_nn)[1])
nn_output = vec(beta_network(reshape(norm_i_traj, 1, :), p_trained, st_nn)[1])

# Create output directory
plot_title = "NN approximation no noise"
sim_name_SR = "$(sim_settings)_$(sim_name)_$(location)_inputs=1_it=100_constcomplex=5"

output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name_SR)")

symbolic_regression_module.symbolic_regression(x_hat, y_hat, sim_name_SR, location, output_dir, plot_title, 1234, "single", norm_i_traj, nn_output, "exponential")
