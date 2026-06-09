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
using ComponentArrays
using Zygote
using Optimization

# Loss function using MSE
function loss_ude(p_all, predict_ude, data, u0)
    pred = predict_ude(p_all, u0)

    if isnothing(pred)
        println("ODE solve failed")
        return 1e20
    end

    # Align lengths
    n = min(length(pred), length(data))
    pred = pred[1:n]
    data = data[1:n]

    # Mean squared error
    mse = sum((pred .- data).^2)/length(data)

    # L2 penalty on NN weights (regularisation)
    l2_penalty = 1e-4 * sum(abs2, p_all.nn_params)

    return mse + l2_penalty
end

function combined_loss_ude_adam(nn_params, predict_ude, trajectories)
    
    gamma = 1/10
    println("Evaluating combined loss for current parameters across $(length(trajectories)) trajectories...")

    # Loop through all simulations
    individual_losses = Float64[]
    total_grad = zero(nn_params)
    total_loss = 0.0

    for (i, traj) in enumerate(trajectories)
        data = traj.data
        varying_p = traj.varying_p

        # Derive beta0 specific to current trajectory
        beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)

        # Update the parameters for the current trajectory to include the varying parameters
        p_all = ComponentArray(
            nn_params = nn_params,
            population = varying_p.population,
            prevalence = varying_p.prevalence,
            beta0 = beta0,
            zeta = varying_p.zeta,
            R0_reproduction = varying_p.R0_reproduction,
            delta = varying_p.delta
        )

        # Define initial state for the current trajectory
        E0 = 1.0
        R0_recovered = 0.0
        D0 = 0.0
        I0 = max(1.0, varying_p.prevalence * varying_p.population)
        S0 = varying_p.population - E0 - I0 - R0_recovered - D0
        u0 = [S0, E0, I0, R0_recovered, D0]

        println("Evaluating trajectory $i")

        # Compute the loss and gradient for the current trajectory
        l, back_all = pullback(theta -> Functions.loss_ude(theta, predict_ude, data, u0), p_all)

        if !isfinite(l)
            return 1e20, nothing
        end

        # Evaluate the gradient of the loss for the current trajectory w.r.t p_all
        grad = back_all((one(l)))[1]

        if isnothing(grad) || isnothing(grad.nn_params)
            return total_loss + l, nothing
        end

        println("Loss for trajectory $i: $l")
        push!(individual_losses, l)

        total_loss += l
        total_grad .+= grad.nn_params

    end
    return total_loss, total_grad
end

function combined_loss_ude_lbfgs(nn_params, predict_ude, trajectories)
    
    gamma = 1/10
    println("Evaluating combined loss for current parameters across $(length(trajectories)) trajectories...")

    total_loss = 0.0

    for (i, traj) in enumerate(trajectories)
        println(i)
        data = traj.data
        varying_p = traj.varying_p

        # Derive beta0 specific to current trajectory
        beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)

        # Update the parameters for the current trajectory to include the varying parameters
        p_all = ComponentArray(
            nn_params = nn_params,
            population = varying_p.population,
            prevalence = varying_p.prevalence,
            beta0 = beta0,
            zeta = varying_p.zeta,
            R0_reproduction = varying_p.R0_reproduction,
            delta = varying_p.delta
        )

        # Define initial state for the current trajectory
        E0 = 1.0
        R0_recovered = 0.0
        D0 = 0.0
        I0 = max(1.0, varying_p.prevalence * varying_p.population)
        S0 = varying_p.population - E0 - I0 - R0_recovered - D0
        u0 = [S0, E0, I0, R0_recovered, D0]

        println("Evaluating trajectory $i")

        # Check the ODE is solvable and ignore if not
        pred_check = Zygote.ignore() do
            predict_ude(p_all, u0)
        end


        if isnothing(pred_check)
            total_loss += 1e20
            continue
        end


        # Compute the loss for the current trajectory
        pred = predict_ude(p_all, u0)

        # Align lengths
        n = min(length(pred), length(data))
        pred = pred[1:n]
        data = data[1:n]

        # Mean squared error
        mse = sum((pred .- data).^2)/length(data)

        # L2 penalty on NN weights (regularisation)
        l2_penalty = 1e-4 * sum(abs2, p_all.nn_params)

        l = mse + l2_penalty

        println("Loss for trajectory $i: $l")

        total_loss += ifelse(isfinite(l), l, 1e20)

    end

    return total_loss
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

