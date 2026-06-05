#=============================================================
MODULE CONTAINING FUNCTIONS USED MULTIPLE TIMES IN THE PROJECT
==============================================================# 

module Functions

export loss_ude
export loss_mse
export beta_exp
export beta_rational
export extract_best_ude
export add_gaussian_noise

include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA
using DrWatson
using JLD2

# Loss function using MSE
function loss_ude(p_all, predict_ude, data)
    pred = predict_ude(p_all)

    # Align lengths
    n = min(length(pred), length(data))
    pred = pred[1:n]
    data = data[1:n]

    # Mean squared error
    mse = sum((pred .- data).^2)/length(data)

    # L2 penalty on NN weights (regularisation)
    l2_penalty = 1e-4 * sum(abs2, p_all.nn_params)

    return mse + l2_penalty, pred
end

# Loss function using MSE for evaluation of performance
function loss_mse(pred, data)

    # Mean squared error
    mse = sum((pred .- data).^2)/length(data)

    return mse
end

# Evaluate the exponential beta function
function beta_exp(location, obs)
    
    delta = DELTA[location]
    R0_reproduction = R0_REPRODUCTION[location]
    zeta = ZETA[location]

    # Derive other parameters
    # Infectious period of 10 days represented by recovery rate gamma
    gamma = 1/10
    beta0 = R0_reproduction * (gamma + delta)

    true_beta = beta0 .* exp.(-zeta .* delta .* obs)

    return true_beta

end 

function beta_rational(location, obs)
    
    delta = DELTA[location]
    R0_reproduction = R0_REPRODUCTION[location]
    zeta = ZETA[location]

    # Derive other parameters
    # Infectious period of 10 days represented by recovery rate gamma
    gamma = 1/10
    beta0 = R0_reproduction * (gamma + delta)

    true_beta = beta0 ./ (1 .+ zeta .* delta .* obs)

    return true_beta

end

function extract_best_ude(sim_name, obs)
    # Define the root file path
    root = DrWatson.datadir("sims", "ude_single", sim_name)

    # Collect results from all simulations
    results_list = []
    for filename in readdir(root)
        # Only include directories
        if isdir(joinpath(root, filename))
            # Extract results, predictions and losses
            SR_results = JLD2.load(DrWatson.datadir("sims", "ude_single", sim_name, filename, "results.jld2"))
            pred = SR_results["prediction"]
            # Extract the predicted infectious trajectory for the training data
            i_traj = pred[3, 1:length(obs)]
            mse = Functions.loss_mse(i_traj, obs)
            push!(results_list, (mse=mse, fname=filename, i_traj=i_traj))
        end
    end

    # Find the simulation with the lowest MSE
    best_idx = argmin(r.mse for r in results_list)
    best_mse = results_list[best_idx].mse
    best_fname = results_list[best_idx].fname

    # Extract the data but convert to a 1 x N matrix
    I_nn = reshape(results_list[best_idx].i_traj, 1, :)

    # Extract the NN parameters from the best simulation
    best_results = JLD2.load(DrWatson.datadir("sims", "ude_single", sim_name, best_fname, "results.jld2"))

    return I_nn, best_results
end

function add_gaussian_noise(noise_SD, data, rng)

    # add Gaussian noise
    sd = noise_SD * max(0.5, maximum(data))
    noise = randn(rng, length(data)) .* sd
    noisy_beta = data .+ noise

    # Remove negative values
    noisy_beta = max.(noisy_beta, 0.0)

    return noisy_beta
end

function _format_equation_sigfigs(equation_text::AbstractString)
    number_pattern = r"(?<![A-Za-z_])[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
    return replace(equation_text, number_pattern => match -> begin
        number = parse(Float64, match)
        string(round(number, sigdigits=3))
    end)
end

end

