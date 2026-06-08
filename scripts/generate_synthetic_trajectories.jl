
#========================================================
SCRIPT TO GENERATE SYNTHETIC TRAJECTORIES
=========================================================# 

using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)

using DrWatson

# Packages required
using CSV
using JLD2
using DataFrames
using DifferentialEquations
using ComponentArrays
using DSP
using Plots 

include(joinpath(@__DIR__, "functions.jl"))
using .Functions

#========================================================
DEFINE THE MODEL
=========================================================#

# Define the model to generate data with a functional form for beta
# p must be of the form (beta0, delta, sigma, gamma, zeta)
function seird_functional!(du, u, p, t)
    S, E, I, R, D = u

    # Define the population size
    N = S + E + I + R

    # If population size less than or equal to zero return zero
    if N <= 0
        du .= 0.0
        return
    end

    # Define the functional form of beta 
    if p.beta == "exponential"
        beta_function = Functions.beta_exp
    elseif p.beta == "rational"   
        beta_function = Functions.beta_rational
    end
    beta = beta_function(p.location, I)

    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - p.sigma * E
    du[3] = p.sigma * E - (p.gamma + p.delta) * I
    du[4] = p.gamma * I
    du[5] = p.delta * I
end

#========================================================
FUNCTION TO RUN THE MODEL
=========================================================#

# Define function to run the SEIRD model with a functional form for beta
# p must be of the form (beta0, delta, sigma, gamma, zeta)
# fixed_p must be of the form (sigma, gamma, zeta, population, E0, R0_recovered, D0)
# varying_p must be of the form (population,prevalence, delta, R0_reproduction)
function run_seird_functional_form(fixed_p, varying_p, obs_length, location, beta)

    # Retrieve fixed parameters
    sigma = fixed_p.sigma
    gamma = fixed_p.gamma
    E0 = fixed_p.E0
    R0_recovered = fixed_p.R0_recovered 
    D0 = fixed_p.D0

    # Retrieve estimated parameters
    population = varying_p.population
    prevalence = varying_p.prevalence
    delta = varying_p.delta
    R0_reproduction = varying_p.R0_reproduction
    zeta = varying_p.zeta

    # Derive other parameters
    beta0 = R0_reproduction * (gamma + delta)
    I0 = max(1.0, prevalence * population)
    S0 = population - E0 - I0 - R0_recovered - D0

    # Define initial state
    init_state = [S0, E0, I0, R0_recovered, D0]

    # Define parameters for the ODE solver
    p = ComponentArray(
        beta0 = beta0,
        delta = delta,
        sigma = sigma,
        gamma = gamma,
        zeta = zeta,
        beta = beta,
        location = location
    )

    # Align solver times with saved labels days = 1:obs_length
    tspan = [1, obs_length]

    # Define ODE problem 
    prob = ODEProblem(seird_functional!, init_state, tspan, p)
    # Solve ODE problem saving every day
    sol = solve(prob, Tsit5(), saveat=1.0, dense = false)
    
    return sol
end

function generate_synthetic_data(fixed_p, varying_p, obs_length, location, beta)
    # Run model
    sim = run_seird_functional_form(fixed_p, varying_p, obs_length, location, beta)

    # Extract raw states 
    s_traj = sim[1, :]
    e_traj = sim[2, :]
    i_traj = sim[3, :]
    r_traj = sim[4, :]
    d_traj = sim[5, :]



    # Save the result
    fname = "synthesised_$(location)" * ".jld2"
	mkpath(datadir("synthetic_trajectories_$(beta)"))


	save(datadir("synthetic_trajectories_$(beta)", fname),
		"fixed_p", fixed_p, "varying_p", varying_p, "days", 1:obs_length, 
        "susceptible", s_traj, "exposed", e_traj, "infectious", i_traj, "recovered", r_traj, "deaths", d_traj)

    println("Finished generating synthetic data for $(fname)")

    return s_traj, e_traj, i_traj, r_traj, d_traj

end


#========================================================
DEFINE PARAMETERS AND GENERATE SYNTHETIC DATA
=========================================================#

# Define fixed parameters that remain constant across all simulations
# They do not influence the transmission rate that we are trying to learn

# Initial conditions
const E0 = 1.0
const R0_recovered = 0.0
const D0 = 0.0

# Latent period of 3 days represented by incubation rate sigma
const sigma = 1/3 
# Infectious period of 10 days represented by recovery rate gamma
const gamma = 1/10

# Extract varying parameters that form the functional form of beta that we are trying to learn
# This will create different trajectories that we will learn using the neural network

include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA


# Loop through each combination of parameters for each state and generate synthetic data
#for location in keys(POPULATION)
# Just generating a single trajectory
for location in keys(POPULATION)
    beta = "rational"


    local population = POPULATION[location]
    local prevalence = PREVALENCE[location]
    local delta = DELTA[location]
    local R0_reproduction = R0_REPRODUCTION[location]
    local zeta = ZETA[location]

    # Create component arrays
    fixed_p = ComponentArray(sigma = sigma,     
                            gamma = gamma, 
                            E0 = E0, 
                            R0_recovered = R0_recovered, 
                            D0 = D0)

    varying_p = ComponentArray(population = population,
                                prevalence = prevalence, 
                                delta = delta, 
                                R0_reproduction = R0_reproduction,
                                zeta = zeta)

    # Generate data and save to JLD2 file
    generate_synthetic_data(fixed_p, varying_p, 365, location, beta)
end
