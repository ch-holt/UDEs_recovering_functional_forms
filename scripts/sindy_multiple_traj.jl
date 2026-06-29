#===================================================================================================
SCRIPT TO RECOVER THE FUNCTIONAL FORM OF THE TRANSMISSION RATE FOR MULTIPLE TRAJECTORIES USING SINDY
===================================================================================================#
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

# Define simulation name and number
sim_name = "synthesised_use_5_inputs_optimal_250326"
sim_num = "simulation_v2"

# Define the NN architecture
hidden_dims = 5

beta_network = Lux.Chain(Lux.Dense(5=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
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

#=============================================================
RETRIEVE PREDICTIONS AND PARAMETERS
=============================================================#

# Retrieve NN parameters and trajectory that resulted in the lowest error on the training data

# Define the root file path
root = datadir("sims", "ude_multiple", sim_name, sim_num)

# include ground truth parameters for all locations
include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

# Accumulate per-trajectory blocks
X_blocks = Matrix{Float64}[]
Y_blocks = Matrix{Float64}[]

# Collect results from all trajectories
results_list = []
for filename in readdir(root)
    # Only include directories
    if isdir(joinpath(root, filename))

        # Extract estimated ground truths
        stem = splitext(basename(filename))[1]
        location = split(stem, "_")[end]        
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

        # Extract prediction
        results_k = load(datadir("sims", "ude_multiple", sim_name, sim_num, filename, "results.jld2"))
        I_k = results_k["infectious_traj_prediction"][3,2:end]
        days_k = vec(Float64.(results_k["days"][1:end])) 

        # Extract beta trajectory for the current trajectory
        beta_k = results_k["beta_traj"]
        y_hat_k = reshape(beta_k, 1, :)

        # Define length
        T = length(I_k)


        # Create matrix with 5 rows and length(pred) column
        # Normalise I and time
        x_hat_k = vcat(
            fill(beta0,1,T), 
            fill(zeta,1,T), 
            fill(delta,1,T), 
            reshape(I_k./population, 1, :),
            reshape(days_k./T, 1, :)
        )

        # Store a block for each trajectory
        push!(X_blocks, x_hat_k)
        push!(Y_blocks, y_hat_k)

    end
end

# Concatonate all trajectories into a single matrix for evaluation
X_hat = isempty(X_blocks) ? Matrix{Float64}(undef, 5, 0) : hcat(X_blocks...)
# Concatonate all beta trajectories into a single matrix for evaluation
Y_hat = isempty(Y_blocks) ? Matrix{Float64}(undef, 1, 0) : hcat(Y_blocks...)

@show size(X_hat) size(Y_hat)
@assert size(X_hat, 2) == size(Y_hat, 2)

#=============================================================
CREATE BASIS
=============================================================#

# Generate library of candidate functions
# We have four neural network inputs and time
@variables t u[1:4]
# Normalise the number of infectious individuals so that the exponential term doesn't overflow
# u[4] = u[4] / population
u_vec = collect(u)
poly_terms = DataDrivenDiffEq.polynomial_basis(u_vec, 3)
exp_terms = [
    exp(-u[1]*u[4]),                
    exp(-u[2]*u[4]),               
    exp(-u[3]*u[4]),                 
    exp(-u[1]*u[2]*u[4]),            
    exp(-u[1]*u[3]*u[4]),            
    exp(-u[2]*u[3]*u[4]),            
    exp(-u[1]*u[2]*u[3]*u[4]),           
    u[1]*exp(-u[2]*u[3]*u[4]),                  
]

h = Num[vcat(poly_terms, exp_terms)...]
# Define basis
basis = DataDrivenDiffEq.Basis(h, u_vec, iv=t)

#=============================================================
DEFINE AND SOLVE THE SPARSE REGRESSION PROBLEM
=============================================================#

nn_problem = DataDrivenDiffEq.DirectDataDrivenProblem(X_hat, Y_hat)

# Use STLSQ to solve the sparse regression problem
# Define the shrinking cut off
lambda = 1e-1
opt = DataDrivenSparse.STLSQ(lambda)

# Solve the sparse regression problem
options = DataDrivenDiffEq.DataDrivenCommonOptions(maxiters = 10_000,
          normalize = DataDrivenDiffEq.DataNormalization(DataDrivenDiffEq.ZScoreTransform),
          selector = DataDrivenDiffEq.bic, digits = 1,
          data_processing = DataDrivenDiffEq.DataProcessing(split = 0.9,
          batchsize = 30,
          shuffle = true,
          rng = rng))

nn_res = DataDrivenDiffEq.solve(nn_problem, basis, opt, options=options)
# nn_res = DataDrivenDiffEq.solve(nn_problem, basis, opt, progress=true, normalize=true, denoise=false)

nn_eqs = DataDrivenDiffEq.get_basis(nn_res)
nn_params = DataDrivenDiffEq.get_parameter_values(nn_eqs)
println(nn_res)
println(nn_eqs)
println(nn_params)



#=============================================================
DEFINE ODE SYSTEM USING THE BETA THAT HAS BEEN RECOVERED FROM THE NN BY SYMBOLIC REGRESSION
=============================================================#

function recovered_dynamics!(du, u, p, t, beta0, zeta, delta)
    S, E, I, R, D = u
    # Define the population size
    N = S + E + I + R
    # If population size less than or equal to zero return zero
    if N <= 0
        du .= 0.0
        return
    end
    # Evaluate symbolic-regression approximation
    beta = nn_res([beta0, zeta, delta, I, t], p, t)[1]
    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + delta) * I
    du[4] = gamma * I
    du[5] = delta * I     
end

#=============================================================
DEFINE ODE SYSTEM USING THE KNOWN FUNCTIONAL FORM FOR THE TIME-VARYING BETA
=============================================================#

# Define known beta function
function true_beta(beta0, zeta, delta, I)
    arg_exp = clamp(zeta * delta * I, -100.0, 100.0)
    beta = beta0 * exp(-arg_exp)
    return beta
end

# Define dynamics using the known beta function
function true_dynamics!(du, u, p, t, beta0, zeta, delta)
    S, E, I, R, D = u
    # Define the population size
    N = S + E + I + R
    # If population size less than or equal to zero return zero
    if N <= 0
        du .= 0.0
        return
    end
    # Evaluate true transmission function
    beta = true_beta(beta0, zeta, delta, I)
    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + delta) * I
    du[4] = gamma * I
    du[5] = delta * I
end

#=============================================================
CREATE LOOP TO MAKE PLOTS FOR EACH LOCATION
=============================================================#

function plot_results(sim_num, sim_name, synthesised_data)

    # Load the observed data and varying parameters
    dataset = load(datadir("synthesised_trajectories", synthesised_data))
    varying_p = ComponentArray(
        population = dataset["varying_p"]["population"],
        prevalence = dataset["varying_p"]["prevalence"],
        delta = dataset["varying_p"]["delta"],
        R0_reproduction = dataset["varying_p"]["R0_reproduction"],
        zeta = dataset["varying_p"]["zeta"]
    )

    # Load trained parameters for this simulation
    trainfile = load(datadir("sims", "ude_multiple", sim_name, sim_num, "training_results.jld2"))
    p_nn = trainfile["p_trained"]

    # Derive parameters specific to current trajectory
    beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)
    I0 = max(1.0, varying_p.prevalence * varying_p.population)
    S0 = varying_p.population - E0 - I0 - R0_recovered - D0

    # Define initial state
    init_state = [S0, E0, I0, R0_recovered, D0]


    # Define ODE problem with recovered symbolic regression beta
    sindy_f = (du, u, p, t) -> recovered_dynamics!(du, u, p, t, beta0, varying_p.zeta, varying_p.delta)
    sindy_prob = ODEProblem(sindy_f, init_state, (1.0, train_length), nn_params)
    pred_sindy = solve(sindy_prob, Tsit5(), saveat=1.0)

    # Define ODE problem using the known beta function
    true_f = (du, u, p, t) -> true_dynamics!(du, u, p, t, beta0, varying_p.zeta, varying_p.delta)
    true_prob = ODEProblem(true_f, init_state, (1.0, train_length), nothing)
    pred_true = solve(true_prob, Tsit5(), saveat=1.0)


    # Extract true trajectory 
    I_true = dataset["infectious"]
    days = dataset["days"]

    # Define the root file path
    root = datadir("sims", "ude_multiple", sim_name, sim_num, synthesised_data, "results.jld2")
    plot_dir = dirname(root)

    # Extract UDE predictions for epidemic trajectory
    results = load(root)
    pred = results["infectious_traj_prediction"]

    # Extract the predicted infectious trajectory for the training data
    I_nn = pred[3, 1:length(I_true)]

    # Extract UDE beta trajectory
    beta_nn = results["beta_traj"]

    # Do all relative to the same 'true' trajectory for comparison


    # True beta on the true trajectory
    beta_true_on_true = true_beta.(Ref(beta0), Ref(varying_p.zeta), Ref(varying_p.delta), I_true)

    # True beta on the NN / SINDy trajectories
    beta_true_on_nn = true_beta.(Ref(beta0), Ref(varying_p.zeta), Ref(varying_p.delta), I_nn)
    beta_true_on_sindy = true_beta.(Ref(beta0), Ref(varying_p.zeta), Ref(varying_p.delta), pred_sindy[3, :])

    I_scaled = I_true ./ varying_p.population
    t_scaled = days ./ train_length

    nn_input_true = vcat(
        fill(beta0, 1, length(days)),
        fill(varying_p.zeta, 1, length(days)),
        fill(varying_p.delta, 1, length(days)),
        reshape(I_scaled, 1, :),
        reshape(t_scaled, 1, :)
    )

    beta_nn_on_true = vec(beta_network(nn_input_true, p_nn, st_nn)[1])

    # Recovered beta evaluated along the true trajectory
    beta_recovered_on_true = [nn_res([beta0, varying_p.zeta, varying_p.delta, I, t], nn_params)[1] for (I, t) in zip(I_true, pred_true.t)]

    # Learned beta on the NN trajectory
    beta_nn = vec(results["beta_traj"])

    # Recovered beta along the SINDy approximation
    beta_recovered = [nn_res([beta0, varying_p.zeta, varying_p.delta, I, t], nn_params)[1] for (I, t) in zip(pred_sindy[3, :], pred_sindy.t)]


    # Generate true beta values
    beta_true = true_beta.(beta0, varying_p.zeta, varying_p.delta, I_true)
    
    # Ensure both are vectors
    if ndims(beta_true) > 1
        beta_true = vec(beta_true)
    end
    if ndims(beta_nn) > 1
        beta_nn = vec(beta_nn)
    end
        
    # Plot predicted trajectories (UDE and SINDy model) against the observed data
    p1 = plot(days, I_true, label="Observed data", lw=2)
    plot!(p1, days, I_nn, label="NN trajectory", lw=2, ls=:dot)
    plot!(p1, pred_sindy.t, pred_sindy[3, :], label="SINDY prediction", lw=2, ls=:dash)
    xlabel!(p1, "Day")
    ylabel!(p1, "Infectious individuals")
    title!(p1, "SINDY approximation of infectious individuals")
    nmse_sindy = Functions.loss_nmse(pred_sindy[3, :], I_true, varying_p.population)
    nmse_nn = Functions.loss_nmse(I_nn, I_true, varying_p.population)
    annotate!(p1, days[end], maximum(I_true), text("NMSE SINDY: $(round(nmse_sindy, digits=4))", 9, :right))
    annotate!(p1, days[end], maximum(I_true) * 0.9, text("NMSE NN: $(round(nmse_nn, digits=4))", 9, :right))


    p2 = plot(days, beta_true_on_nn, label="True β on NN I(t)", lw=2)
    plot!(p2, days, beta_nn, label="NN-learned β on NN I(t)", lw=2, ls=:dot)
    plot!(p2, days, beta_recovered, label="Recovered β on SINDy I(t)", lw=2, ls=:dash)

    p3 = plot(days, beta_true_on_true, label="True β on true I(t)", lw=2)
    plot!(p3, days, beta_nn_on_true, label="NN-learned β on true I(t)", lw=2, ls=:dot)
    plot!(p3, days, beta_recovered_on_true, label="Recovered β on true I(t)", lw=2, ls=:dash)

    p_beta = plot(days, beta_true_on_true, label="True β", lw=2)
    plot!(p_beta, days, beta_nn_on_true, label="Learned β", lw=2, ls=:dash)

    xlabel!(p_beta, "Day")
    ylabel!(p_beta, "β")
    title!(p_beta, "Learned vs true transmission rate")

    display(p_beta)
    # Save the plot
    savefig(p_beta, joinpath(plot_dir, "$(sim_name)_sindy_beta_comparison.png"))

end

#========================================================
PRODUCE PLOTS OF SIMULATIONS
=========================================================# 


sim_num = "simulation_v2"
sim_name = "synthesised_use_5_inputs_optimal_250326"
for filename in readdir(datadir("sims", "ude_multiple", sim_name, sim_num))
    if endswith(filename, ".jld2") && isdir(datadir("sims", "ude_multiple", sim_name, sim_num, filename))
        plot_results(sim_num, sim_name, filename)
    end
end

