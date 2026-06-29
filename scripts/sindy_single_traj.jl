#=================================================================================================
SCRIPT TO RECOVER THE FUNCTIONAL FORM OF THE TRANSMISSION RATE FOR A SINGLE TRAJECTORY USING SINDY
==================================================================================================#

using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using LinearAlgebra
using Lux
using JLD2
using ComponentArrays
using Plots
include(joinpath(@__DIR__, "functions.jl"))
using .Functions
using Statistics
using Random; rng = Random.default_rng()

#=============================================================
SET UP
=============================================================#

# Define simulation name -- name of the directory containing the 100 sims of UDE training
sim_name = "270426_post_nina_comments"

# Define the NN architecture
hidden_dims = 5

beta_network = Lux.Chain(Lux.Dense(1=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, softplus))

# Initialise parameters
p_nn_temp, st_nn = Lux.setup(rng, beta_network)


#=============================================================
DEFINE INITAL STATE AND PARAMETERS
=============================================================#

# Define training length
const train_length = 365

# Latent period of 3 days represented by incubation rate sigma
const sigma = 1/3 
# Infectious period of 10 days represented by recovery rate gamma
const gamma = 1/10

# Define initial state same as the generated data
# Retrieve fixed parameters
const E0 = 1.0
const R0_recovered = 0.0
const D0 = 0.0

# Extract from estimated ground truths
include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

location = "MA"
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
init_state = [S0, E0, I0, R0_recovered, D0]


#=============================================================
CREATE BASIS
=============================================================#

# Generate library of candidate functions
# We have one state variable u[1](t)
@variables u[1:1]
# Normalise the number of infectious individuals so that the exponential term doesn't overflow
u_scaled = u[1] / population
k = zeta * delta * population
exp_terms = [   
    exp(-zeta*u_scaled),           
    exp(-delta*u_scaled),                            
    exp(-zeta*delta*u_scaled),            
    exp(-zeta*population*u_scaled),            
    exp(-delta*population*u_scaled),            
    exp(-k*u_scaled)        
]
#poly_terms = DataDrivenDiffEq.polynomial_basis([u_scaled], 3)

poly_terms = [
    1,    
    u[1],
    u[1]^2,
    u[1]^3
]

h = Num[vcat(exp_terms,  poly_terms)...]

# Define basis
basis = DataDrivenDiffEq.Basis(h, u)


#=============================================================
RETRIEVE PREDICTIONS AND PARAMETERS FROM THE BEST SIMULATION
=============================================================#

# Load the observed data
dataset = JLD2.load(datadir("synthesised_trajectories_single", "synthesised_MA.jld2"))
   
# Just use infectious trajectory
obs = dataset["infectious"]
days = dataset["days"]

# Retrieve NN parameters that resulted in the lowest error on the training data

# Define the root file path
root = datadir("sims", "ude_single", sim_name)

# Collect results from all simulations
results_list = []
for filename in readdir(root)
    # Only include directories
    if isdir(joinpath(root, filename))
        # Extract results, predictions and losses
        SR_results = JLD2.load(datadir("sims", "ude_single", sim_name, filename, "results.jld2"))
        pred = SR_results["prediction"]
        # Extract the predicted infectious trajectory for the training data
        i_traj = pred[3, 1:length(obs)]
        nmse = Functions.loss_nmse(i_traj, obs, population)
        push!(results_list, (nmse=nmse, fname=filename, i_traj=i_traj))
    end
end

# Find the simulation with the lowest MSE
best_idx = argmin(r.mse for r in results_list)
best_mse = results_list[best_idx].mse
best_fname = results_list[best_idx].fname

# Extract the data but convert to a 1 x N matrix
x_hat = reshape(results_list[best_idx].i_traj, 1, :)

# Extract the NN parameters from the best simulation
best_results = JLD2.load(datadir("sims", "ude_single", sim_name, best_fname, "results.jld2"))
p_trained = best_results["p"]
days = best_results["days"]

# Extract NN approximation (normalised)
nn_input = x_hat ./ population


# Evaluate neural network and extract approximation
y_hat = beta_network(nn_input, p_trained.nn_params, st_nn)[1]

#=============================================================
DEFINE AND SOLVE THE SPARSE REGRESSION PROBLEM
=============================================================#

nn_problem = DataDrivenDiffEq.DirectDataDrivenProblem(x_hat, y_hat)

# Use STLSQ to solve the sparse regression problem
# Define the shrinking cut off
lambda = 1e-2
opt = DataDrivenSparse.STLSQ(lambda)

# Solve the sparse regression problem
options = DataDrivenDiffEq.DataDrivenCommonOptions(maxiters = 10_000,
    normalize = DataDrivenDiffEq.DataNormalization(DataDrivenDiffEq.ZScoreTransform),
    selector = DataDrivenDiffEq.bic, digits = 6)
nn_res = DataDrivenDiffEq.solve(nn_problem, basis, opt; options=options)
# nn_res = DataDrivenDiffEq.solve(nn_problem, basis, opt, progress=true, normalize=true, denoise=false)

nn_eqs = DataDrivenDiffEq.get_basis(nn_res)
nn_params = DataDrivenDiffEq.get_parameter_values(nn_eqs)
println(nn_res)
println(nn_eqs)
println(nn_params)


#=============================================================
DEFINE ODE SYSTEM USING THE BETA THAT HAS BEEN RECOVERED FROM THE NN BY SYMBOLIC REGRESSION
=============================================================#

function recovered_dynamics!(du, u, p, t)
    S, E, I, R, D = u
    # Define the population size
    N = S + E + I + R
    # If population size less than or equal to zero return zero
    if N <= 0
        du .= 0.0
        return
    end
    # Evaluate symbolic-regression approximation
    beta = nn_res([I], p, t)[1]
    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + delta) * I
    du[4] = gamma * I
    du[5] = delta * I     
end

# Define ODE problem with recovered parameters form symbolic regression and solve
sindy_prob = ODEProblem(recovered_dynamics!, init_state, (1,train_length), nn_params)
pred_sindy = solve(sindy_prob, Tsit5(), u0=init_state, tspan=(1,train_length), saveat=1)

#=============================================================
DEFINE ODE SYSTEM USING THE KNOWN FUNCTIONAL FORM FOR THE TIME-VARYING BETA
=============================================================#

# Define known beta function
function true_beta(I)
    arg_exp = zeta * delta * I
    beta = beta0 * exp(-arg_exp)
    return beta
end


#=============================================================
EVALUATE BETA TRAJECTORIES
=============================================================#

# Do all relative to the same 'true' trajectory for comparison

# True beta from the true ODE trajectory (no SINDY dynamics)
beta_true = true_beta.(obs)

# Beta learned by the trained NN on the same true-system I(t)
nn_input_true = reshape(obs, 1, :)./population
beta_nn_on_true = vec(beta_network(nn_input_true, p_trained.nn_params, st_nn)[1])

# Recovered beta evaluated along the same I(t) for comparison
beta_recovered_on_true = [nn_res([I], nn_params, t)[1] for (I, t) in zip(obs, days)]
# Do relative to their own trajectories for comparison

# Beta learned by the trained NN on the NN approximation
beta_nn = vec(y_hat) 

# Recovered beta evaluated along the SINDy approximation
beta_recovered = [nn_res([I], nn_params, t)[1] for (I, t) in zip(pred_sindy[3, :], pred_sindy.t)]



#=============================================================
PLOT RESULTS
=============================================================#

# Plot predicted trajectories (UDE and SINDy model) against the observed data
p1 = plot(days, obs, label="Observed data", lw=2, legend=:topright)
plot!(p1, days, vec(x_hat), label="NN trajectory", lw=2, ls=:dot)
plot!(p1, pred_sindy.t, pred_sindy[3, :], label="SINDY prediction", lw=2, ls=:dash)
xlabel!(p1, "Day")
ylabel!(p1, "Daily deaths")
title!(p1, "SINDY approximation of daily deaths")
nmse_sindy = Functions.loss_nmse(pred_sindy[3, :], obs, population)
nmse_nn = Functions.loss_nmse(vec(x_hat), obs, population)
x_ann = days[end]
y_top = maximum(obs)
annotate!(p1, x_ann, y_top * 0.70, text("NMSE SINDY: $(round(nmse_sindy, digits=4))", 9, :right))
annotate!(p1, x_ann, y_top * 0.62, text("NMSE NN: $(round(nmse_nn, digits=4))", 9, :right))

# Plot the beta trajectories evaluated against their respective predicted/observed trajectories
p2 = plot(days, beta_true, label="True β from true ODE I(t)", lw=2)
plot!(p2, days, beta_nn, label="NN-learned β on I(t) learned by NN", lw=2, ls=:dot)
plot!(p2, days, beta_recovered, label="Recovered β on recovered ODE I(t)", lw=2, ls=:dash)
xlabel!(p2, "Day")
ylabel!(p2, "β")
title!(p2, "Transmission rate using synthesised/learned/recovered I(t) respective to their own trajectories")

# Plot the beta trajectories evaluated against the observed trajectory
p3 = plot(days, beta_true, label="True β from true ODE I(t)", lw=2)
plot!(p3, days, beta_nn_on_true, label="NN-learned β on true ODE I(t)", lw=2, ls=:dot)
plot!(p3, days, beta_recovered_on_true, label="Recovered β on true ODE I(t)", lw=2, ls=:dash)
xlabel!(p3, "Day")
ylabel!(p3, "β")
title!(p3, "Transmission rate using true-system I(t)")

pl = plot(p1, p2, p3, layout=(3, 1), size=(900, 1050))

# Save the plot
savefig(pl, joinpath(@__DIR__, "figures", "$(sim_name)_sindy_trajectory.png"))           

display(pl)

# Evaluate NMSE using the median
nmse = Functions.loss_nmse(vec(y_hat), beta_true, maximum(beta_true)-minimum(beta_true))
println(Functions.loss_nmse(vec(y_hat), beta_true, maximum(beta_true)-minimum(beta_true)))

p_beta = plot(days, beta_true, label = "True β", lw = 2)
plot!(p_beta, days, vec(y_hat), label = "Learned β", lw = 2, ls = :dash)



xlabel!(p_beta, "Day")
ylabel!(p_beta, "β")
title!(p_beta, "Learned vs true transmission rate")

x_ann = days[end] * 0.75
y_ann = maximum(beta_true) * 0.85
dy = maximum(beta_true) * 0.06
annotate!(p_beta, x_ann, y_ann, text("MSE = $(round(mse, sigdigits=3))", 9))

display(p_beta)

savefig(p_beta, joinpath(@__DIR__, "figures", "$(sim_name)_sindy_beta_comparison.png"))

