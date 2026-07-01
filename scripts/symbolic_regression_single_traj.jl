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

sim_name = "exponential_beta_single_traj"
location = "MA"
population = POPULATION[location]

# Load the observed data
dataset = JLD2.load(DrWatson.datadir("exp_pro","synthetic_data","synthetic_trajectories_exponential", "synthesised_MA.jld2"))
   
# Just use infectious trajectory
obs = dataset["infectious"]
days = dataset["days"]

# Retrieve NN parameters that resulted in the lowest error on the training data
I_nn, best_results = Functions.extract_best_ude(sim_name, obs, population)

p_trained = best_results["p"]
days = best_results["days"]

# Extract NN approximation (x_hat normalised in SR module)
norm_i_traj = I_nn ./ population

x_grid = collect(range(0, 1; length=1000))
nn_input = reshape(x_grid, 1, :)
x_hat = reshape(x_grid, :, 1)

# Evaluate neural network and extract approximation
y_hat = vec(beta_network(nn_input, p_trained.nn_params, st_nn)[1])
nn_output = vec(beta_network(norm_i_traj, p_trained.nn_params, st_nn)[1])

# Create output directory
plot_title = "NN approximation no noise"
sim_name_SR = "UDE_$(sim_name)_softplus_v4"
output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name_SR)")

symbolic_regression_module.symbolic_regression(x_hat, y_hat, sim_name_SR, location, output_dir, plot_title, 1234, "single", norm_i_traj, nn_output, "exponential")


