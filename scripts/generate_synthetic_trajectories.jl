
#========================================================
SCRIPT TO GENERATE SYNTHETIC TRAJECTORIES
=========================================================# 


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
using Random

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
            dataset = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_HQ_$(beta_name)", "synthetic_$(location)", "noise=0.0.jld2"))

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
                beta_eqn = "β(I) = $beta0 * exp(-$exponent_coeff I)"
            elseif beta_function == beta_rational
                beta_eqn = "β(I) = $beta0 / (1 + $exponent_coeff I)"
            elseif beta_function == beta_mixed
                beta_eqn = "β(I) = $beta0 * exp(-$exponent_coeff I) / (1 + $exponent_coeff I)"
            end
            beta_eqn = format_equation_sigfigs(beta_eqn)


            # Save the generated beta trajectory to a JLD2 file
            beta_name = string(beta_function)

            mkpath(datadir("exp_pro", "synthetic_data","synthetic_beta_trajectories_HQ", "ground_truth_$(beta_name)", "location_$(location)"))
            save(datadir("exp_pro", "synthetic_data","synthetic_beta_trajectories_HQ", "ground_truth_$(beta_name)", "location_$(location)","synthetic_$(beta_name)_$(location)_over_time.jld2"), "beta", generated_beta_over_time, "days", dataset["days"])
            save(datadir("exp_pro", "synthetic_data","synthetic_beta_trajectories_HQ", "ground_truth_$(beta_name)", "location_$(location)","synthetic_$(beta_name)_$(location)_between_0_1.jld2"), "beta", generated_beta_between_0_1, "x_hat", x_hat)


            # Plot the generated beta trajectory over time
            plot_dir_over_time = plotsdir("synthetic_data","synthetic_beta_trajectories_HQ", "ground_truth_$(beta_name)", "location_$(location)")
            mkpath(plot_dir_over_time)
            plot_filename = joinpath(plot_dir_over_time, "synthetic_$(beta_name)_$(location)_over_time.png")
            plot(dataset["days"], generated_beta_over_time, label="Generated Beta", xlabel="Days", ylabel="Beta", title="Generated Beta for $(location)\n$(beta_eqn)", legend=:topright)

            savefig(plot_filename)

            # Plot the generated beta trajectory against x_hat
            plot_dir_between_0_1 = plotsdir("synthetic_data","synthetic_beta_trajectories_HQ", "ground_truth_$(beta_name)", "location_$(location)")
            mkpath(plot_dir_between_0_1)
            plot_filename_xhat = joinpath(plot_dir_between_0_1, "synthetic_$(beta_name)_$(location)_between_0_1.png")
            plot(x_hat, generated_beta_between_0_1, label="Generated Beta", xlabel="I/N", ylabel="Beta", title="Generated Beta Trajectory for $(location) between 0 and 1 \n$(beta_eqn)", legend=:topright)
            savefig(plot_filename_xhat)
        end

    end
end

#========================================================
FUNCTION TO GENERATE GROUND TRUTH TRAJECTORIES
=========================================================#

function generate_synthetic_data(fixed_p, varying_p, obs_length, location, beta_function, noise_levels)
    # Run model
    sim = run_seird_functional_form(beta_function, location, fixed_p, varying_p, obs_length)

    # Extract raw states 
    s_traj = sim[1, :]
    e_traj = sim[2, :]
    i_traj = sim[3, :]
    r_traj = sim[4, :]
    d_traj = sim[5, :]

    # Save the result
    beta_name = string(beta_function)
	mkpath(datadir("exp_pro","synthetic_data","synthetic_trajectories_HQ_$(beta_name)", "synthetic_$(location)"))

    
    println("Adding noise to the infectious trajectory for $(location)")

    for noise in noise_levels
        noisy_i_traj, r = add_neg_bin_noise(i_traj, noise)
        fname_noisy = "noise=$(noise).jld2"
        save(datadir("exp_pro","synthetic_data","synthetic_trajectories_HQ_$(beta_name)", "synthetic_$(location)", fname_noisy),
            "fixed_p", fixed_p, "varying_p", varying_p, "days", 1:obs_length,
            "susceptible", s_traj, "exposed", e_traj, "infectious", noisy_i_traj, "r", r, "recovered", r_traj, "deaths", d_traj)
    println("Finished generating synthetic data for $(fname_noisy)")
    end

    return

end

#========================================================
FUNCTION TO GENERATE BASELINE CONTACT DATA
=========================================================#

# Function to calculate the wieghted IQR
function weighted_quantile(values, weights, q)
    # sort values and weights by values
    sorted_idx = sortperm(values)
    vals = values[sorted_idx]
    # make a cumulative sum of the weights
    cumwts = cumsum(weights[sorted_idx])
    # find the value corresponding to the desired quantile
    target = q * cumwts[end]
    # returns the value that gives this cumulative weight
    return vals[findfirst(cumwts .>= target)]
end

# Function to find the mean and IQR for each division in the EpiStorm data and save to a JLD2 file
function generate_division_contact()
    # Load EpiStorm data
    epistorm_data = CSV.read(datadir("exp_raw", "imputed-resp2.csv"), DataFrame)
    # group by division and compute weighted mean and IQR
    # take each region
    results = combine(groupby(epistorm_data, :hh_loc_reg9)) do df
        contacts = df[!, Symbol("total.contacts")]
        weights = df[!, :weight_ipsos]
        # evaluate the weighted mean for that region 
        weighted_mean = sum(contacts .* weights) / sum(weights)
        # Do IQR
        q25 = weighted_quantile(contacts, weights, 0.25)
        q75 = weighted_quantile(contacts, weights, 0.75)
        # output a row
        (mean = weighted_mean, q25 = q25, q75 = q75, n = nrow(df))
    end

    JLD2.save(datadir("exp_pro", "synthetic_data", "baseline_contacts_division.jld2"), "results", results)

    return results

end

# map the states to the respective divisions
const STATE_TO_DIVISION = Dict(
    # New England
    "CT" => "New England", "ME" => "New England", "MA" => "New England",
    "NH" => "New England", "RI" => "New England", "VT" => "New England",

    # Mid-Atlantic
    "NJ" => "Mid-Atlantic", "NY" => "Mid-Atlantic", "PA" => "Mid-Atlantic",

    # East North Central
    "IL" => "East-North Central", "IN" => "East-North Central",
    "MI" => "East-North Central", "OH" => "East-North Central",
    "WI" => "East-North Central",

    # West North Central
    "IA" => "West-North Central", "KS" => "West-North Central",
    "MN" => "West-North Central", "MO" => "West-North Central",
    "NE" => "West-North Central", "ND" => "West-North Central",
    "SD" => "West-North Central",

    # South Atlantic
    "DE" => "South Atlantic", "FL" => "South Atlantic", "GA" => "South Atlantic",
    "MD" => "South Atlantic", "NC" => "South Atlantic", "SC" => "South Atlantic",
    "VA" => "South Atlantic", "WV" => "South Atlantic", "DC" => "South Atlantic",

    # East South Central
    "AL" => "East-South Central", "KY" => "East-South Central",
    "MS" => "East-South Central", "TN" => "East-South Central",

    # West South Central
    "AR" => "West-South Central", "LA" => "West-South Central",
    "OK" => "West-South Central", "TX" => "West-South Central",

    # Mountain
    "AZ" => "Mountain", "CO" => "Mountain", "ID" => "Mountain",
    "MT" => "Mountain", "NV" => "Mountain", "NM" => "Mountain",
    "UT" => "Mountain", "WY" => "Mountain",

    # Pacific
    "AK" => "Pacific", "CA" => "Pacific", "HI" => "Pacific",
    "OR" => "Pacific", "WA" => "Pacific"
)

# Function to sample the baseline mean contacts for a state
function baseline_contacts(locations, seed)
    # Retrieve JLD2 to save the baseline contacts for each state
    path = datadir("exp_raw", "estimated_ground_truth_parameters.jld2")
    ground_truth = JLD2.load(path)

    # Set seed for reproducibility
    rng = Xoshiro(seed)

    # Load the baseline contacts for each division
    division_contacts = JLD2.load(datadir("exp_pro", "synthetic_data", "baseline_contacts_division.jld2"))["results"]
    
    C0 = Dict{String, Float32}()

    # Loop through the locations in sorted order for reproducibility
    for location in sort(collect(locations))
        # Map the state to the respective division
        division = STATE_TO_DIVISION[location]
        # Extract the IQR bounds for each division
        division_row = filter(row -> row.hh_loc_reg9 == division, division_contacts)
        isempty(division_row) && error("No baseline contacts row for division $(division) (state $(location))")
        Lbound=Float32(division_row.q25[1])
        Ubound=Float32(division_row.q75[1])
        # Uniform distribution for c0 values between LBound and UBound
        C0[location] = rand(rng, Uniform(Lbound, Ubound))
    end
    ground_truth["C0"] = C0
    JLD2.save(path, ground_truth)

end

#========================================================
FUNCTION TO GENERATE GROUND TRUTH BEHAVIOURAL DATA
=========================================================#

function generate_ground_truth_contacts(contact_functions)
    for contact_function in contact_functions
        println("Generating synthetic trajectories for $(contact_function)")
        for location in keys(POPULATION)
            println("Generating synthetic trajectories for $(location)")
            contact_name = string(contact_function)
            beta_name = replace(string(contact_function), "contacts" => "beta")
            dataset = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_HQ_$(beta_name)", "synthetic_$(location)", "noise=0.0.jld2"))

            # Extract infectious individuals and days from the dataset
            obs = dataset["infectious"]

            generated_contact_over_time = contact_function(location, obs)

            x_grid = collect(range(0, 1; length=1000))
            x_hat = reshape(x_grid, :, 1)
            
            generated_contact_between_0_1 = contact_function(location, x_hat*POPULATION[location])

            # Create string of contact function to display on plot
            c0 = C0[location]
            exponent_coeff = ZETA[location]*DELTA[location]
            if contact_function == contacts_exp
                contact_eqn = "c(I) = $c0 * exp(-$exponent_coeff I)"
            elseif contact_function == contacts_rational
                contact_eqn = "c(I) = $c0 / (1 + $exponent_coeff I)"
            elseif contact_function == contacts_mixed
                contact_eqn = "c(I) = $c0 * exp(-$exponent_coeff I) / (1 + $exponent_coeff I)"
            end
            contact_eqn = format_equation_sigfigs(contact_eqn)



            # Save the generated contact trajectory to a JLD2 file


            mkpath(datadir("exp_pro", "synthetic_data","synthetic_contact_trajectories_HQ", "ground_truth_$(contact_name)", "location_$(location)"))
            save(datadir("exp_pro", "synthetic_data","synthetic_contact_trajectories_HQ", "ground_truth_$(contact_name)", "location_$(location)","synthetic_$(contact_name)_$(location)_over_time.jld2"), "contact", generated_contact_over_time, "days", dataset["days"])
            save(datadir("exp_pro", "synthetic_data","synthetic_contact_trajectories_HQ", "ground_truth_$(contact_name)", "location_$(location)","synthetic_$(contact_name)_$(location)_between_0_1.jld2"), "contact", generated_contact_between_0_1, "x_hat", x_hat)


            # Plot the generated contact trajectory over time
            plot_dir_over_time = plotsdir("synthetic_data","synthetic_contact_trajectories_HQ", "ground_truth_$(contact_name)", "location_$(location)")
            mkpath(plot_dir_over_time)
            plot_filename = joinpath(plot_dir_over_time, "synthetic_$(contact_name)_$(location)_over_time.png")
            plot(dataset["days"], generated_contact_over_time, label="Generated Contact", xlabel="Days", ylabel="Contact", title="Generated Contact for $(location)\n$(contact_eqn)", legend=:topright)
            savefig(plot_filename)

            # Plot the generated contact trajectory against x_hat
            plot_dir_between_0_1 = plotsdir("synthetic_data","synthetic_contact_trajectories_HQ", "ground_truth_$(contact_name)", "location_$(location)")
            mkpath(plot_dir_between_0_1)
            plot_filename_xhat = joinpath(plot_dir_between_0_1, "synthetic_$(contact_name)_$(location)_between_0_1.png")
            plot(x_hat, generated_contact_between_0_1, label="Generated Contact", xlabel="I/N", ylabel="Contact", title="Generated Contact Trajectory for $(location) between 0 and 1 \n$(contact_eqn)", legend=:topright)
            savefig(plot_filename_xhat)
        end

    end
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

noise_levels = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
const beta_function = [beta_exp, beta_rational, beta_mixed]
contact_function = [contacts_exp, contacts_rational, contacts_mixed]

# Loop through each combination of parameters for each state and generate synthetic data
#for location in keys(POPULATION)
# Just generating a single trajectory
for location in keys(POPULATION)
    for beta_function in beta_function
        local population = POPULATION[location]
        local prevalence = PREVALENCE[location]
        local c0 = C0[location]
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
        #generate_synthetic_data(fixed_p, varying_p, 365, location, beta_function, noise_levels)

    end
end
#generate_division_contact()
baseline_contacts(collect(keys(POPULATION)), 1234)
#generate_ground_truth_beta(beta_function)
generate_ground_truth_contacts(contact_function)
