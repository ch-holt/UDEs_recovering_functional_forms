
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
using Distributions

# Call module
using UDE_FUNCTIONAL_FORMS

#========================================================
FUNCTION TO GENERATE BETA TRAJECTORIES
=========================================================#

function generate_ground_truth_beta(beta_functions)
    for beta_function in beta_functions
        println("Generating synthetic trajectories for $(beta_function)")
        for location in keys(POPULATION)
            println("Generating synthetic trajectories for $(location)")
            beta_name = string(beta_function)
            dataset = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_$(beta_name)", "synthetic_$(location).jld2"))

            # Extract infectious individuals and days from the dataset
            obs = dataset["infectious"]

            generated_beta_over_time = beta_function(location, obs)

            x_grid = collect(range(0, 1; length=1000))
            x_hat = reshape(x_grid, :, 1)
            
            generated_beta_between_0_1 = beta_function(location, x_hat*POPULATION[location])

            # Create string of beta function to display on plot
            beta0 = R0_REPRODUCTION[location] * (gamma + DELTA[location])
            exponent_coeff = ZETA[location]*DELTA[location]
            if beta_function == beta_exp
                beta_eqn = "β(x) = $beta0 * exp(-$exponent_coeff"
            elseif beta_function == beta_rational
                beta_eqn = "β(x) = $beta0 / (1 + $exponent_coeff"
            end
            beta_eqn = format_equation_sigfigs(beta_eqn)


            # Save the generated beta trajectory to a JLD2 file
            beta_name = string(beta_function)

            mkpath(datadir("exp_pro", "synthetic_data","synthetic_beta_trajectories", "ground_truth_$(beta_name)", "location_$(location)"))
            save(datadir("exp_pro", "synthetic_data","synthetic_beta_trajectories", "ground_truth_$(beta_name)", "location_$(location)","synthetic_$(beta_name)_$(location)_over_time.jld2"), "beta", generated_beta_over_time, "days", dataset["days"])
            save(datadir("exp_pro", "synthetic_data","synthetic_beta_trajectories", "ground_truth_$(beta_name)", "location_$(location)","synthetic_$(beta_name)_$(location)_between_0_1.jld2"), "beta", generated_beta_between_0_1, "x_hat", x_hat)


            # Plot the generated beta trajectory over time
            plot_dir_over_time = plotsdir("synthetic_data","synthetic_beta_trajectories", "ground_truth_$(beta_name)", "location_$(location)")
            mkpath(plot_dir_over_time)
            plot_filename = joinpath(plot_dir_over_time, "synthetic_$(beta_name)_$(location)_over_time.png")
            plot(dataset["days"], generated_beta_over_time, label="Generated Beta", xlabel="Days", ylabel="Beta", title="Generated Beta for $(location)\n$(beta_eqn) I)", legend=:topright)

            savefig(plot_filename)

            # Plot the generated beta trajectory against x_hat
            plot_dir_between_0_1 = plotsdir("synthetic_data","synthetic_beta_trajectories", "ground_truth_$(beta_name)", "location_$(location)")
            mkpath(plot_dir_between_0_1)
            plot_filename_xhat = joinpath(plot_dir_between_0_1, "synthetic_$(beta_name)_$(location)_between_0_1.png")
            plot(x_hat, generated_beta_between_0_1, label="Generated Beta", xlabel="I/N", ylabel="Beta", title="Generated Beta Trajectory for $(location) between 0 and 1 \n$(beta_eqn) I/N)", legend=:topright)
            savefig(plot_filename_xhat)
        end

    end
end

#========================================================
FUNCTION TO RUN THE MODEL
=========================================================#

function generate_synthetic_data(fixed_p, varying_p, obs_length, location, beta_function)
    # Run model
    sim = run_seird_functional_form(beta_function, location, fixed_p, varying_p, obs_length)

    # Extract raw states 
    s_traj = sim[1, :]
    e_traj = sim[2, :]
    i_traj = sim[3, :]
    r_traj = sim[4, :]
    d_traj = sim[5, :]

    # Save the result
    fname = "synthetic_$(location).jld2"
    beta_name = string(beta_function)
	mkpath(datadir("exp_pro","synthetic_data","synthetic_trajectories_$(beta_name)"))

	save(datadir("exp_pro","synthetic_data","synthetic_trajectories_$(beta_name)", fname),
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


# Loop through each combination of parameters for each state and generate synthetic data
#for location in keys(POPULATION)
# Just generating a single trajectory
for location in keys(POPULATION)
    beta_function = beta_exp

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
    generate_synthetic_data(fixed_p, varying_p, 365, location, beta_function)
end

#generate_ground_truth_beta([beta_exp, beta_rational])