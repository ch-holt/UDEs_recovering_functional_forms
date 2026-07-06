#=============================================================
NMSE LOSS FUNCTION
==============================================================# 

function loss_nmse(pred, data, normalising_factor)

    # Mean squared error
    nmse = sum((pred ./ normalising_factor .- data ./ normalising_factor).^2)/length(data)

    return nmse
end

#=============================================================
LOSS FUNCTION FOR SINGLE DATASET USING NMSE
==============================================================# 

function loss_ude(p_all, predict_ude, data, u0, beta_network, st_nn, valn, learning_bias_bool)
    pred = predict_ude(p_all, u0)

    if isnothing(pred)
        println("ODE solve failed")
        return Inf
    end

    # Align lengths
    n = min(length(pred), length(data))
    pred_correct_length = pred[1:n]
    data_correct_length = data[1:n]

    # Mean squared error
    nmse = loss_nmse(pred_correct_length, data_correct_length, p_all.population)

    # Add learning bias to soft constrain beta(1)=0
    if valn == 1
        learning_bias = 1e-6 * relu((beta_network([1.0], p_all.nn_params, st_nn)[1][1]))
    elseif valn == 4
        learning_bias = 1e-6 * relu((beta_network([p_all.beta0, p_all.zeta, p_all.delta, 1.0], p_all.nn_params, st_nn)[1][1]))
    end

    if learning_bias_bool
        return nmse + learning_bias
    else
        return nmse
    end
end

function regularisation(nn_params)
    # L2 penalty on NN weights (regularisation)
    l2_penalty = 1e-6 * sum(abs2, nn_params)

    return l2_penalty
end

#=============================================================
COMBINED LOSS FOR MULTIPLE DATASETS USING ADAM OPTIMISER
==============================================================# 

function combined_loss_ude_adam(beta_network, st_nn, nn_params, number_of_nn_inputs, predict_ude, trajectories)
    
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
        l, back_all = pullback(theta -> loss_ude(theta, predict_ude, data, u0, beta_network, st_nn, number_of_nn_inputs), p_all)

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

    # Average loss and gradient across all trajectories and datapoints
    total_loss /= length(trajectories)
    total_grad ./= length(trajectories)

    # add regularisation
    reg_loss, reg_back = pullback(theta -> regularisation(theta), nn_params)
    reg_grad = reg_back(one(reg_loss))[1]

    total_loss += reg_loss
    total_grad .+= reg_grad



    return total_loss, total_grad
end

#=============================================================
COMBINED LOSS FOR MULTIPLE DATASETS USING LBFGS
==============================================================# 

function combined_loss_ude_lbfgs(beta_network, st_nn, nn_params, number_of_nn_inputs, predict_ude, trajectories)
    
    gamma = 1/10
    println("Evaluating combined LBFGS loss for current parameters across $(length(trajectories)) trajectories...")
    total_grad = zero(nn_params)
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

        l, back_all = pullback(theta -> loss_ude(theta, predict_ude, data, u0, beta_network, st_nn, number_of_nn_inputs), p_all)
        if !isfinite(l)
           error("ODE solve failed for trajectory $i. Loss: $l")
        end
        grad = back_all((one(l)))[1]
        println("Loss for trajectory $i: $l")

        total_loss += l
        total_grad .+= grad.nn_params
    end

    total_loss /= length(trajectories)
    total_grad ./= length(trajectories)

    reg_loss, reg_back = pullback(theta -> regularisation(theta), nn_params)
    reg_grad = reg_back(one(reg_loss))[1]
    total_loss += reg_loss
    total_grad .+= reg_grad

    return total_loss, total_grad
end

