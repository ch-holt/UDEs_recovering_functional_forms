#=============================================================
NMSE LOSS FUNCTION
==============================================================# 

function loss_nmse(pred, data)

    # Mean squared error
    #nmse = sum((pred ./ normalising_factor .- data ./ normalising_factor).^2)/length(data)
    normalising_factor = maximum(data) - minimum(data)
    nmse = sum((pred ./ normalising_factor .- data ./ normalising_factor).^2)/length(data)
    return nmse
end

# Negative binomial loss
# Assuming pred is the mean of the negative binomial distribution, r is the dispersion parameter, and data is the observations
function loss_negbin(pred, data, r)
    # ensure prediction is positive
    pred = max.(pred, eps())
    nll = sum((data .+ r) .* log.(r .+ pred) .- data .* log.(pred) .+ loggamma.(r) .- loggamma.(data .+ r) .- r.*log.(r))
    return nll
end

# Poisson negative log-likelihood, dropping the constant sum(log k!) as loss_negbin does,
# so it is the r -> infinity limit of loss_negbin
function loss_poisson(pred, data)
    pred = max.(pred, eps())
    return sum(pred .- data .* log.(pred))
end

# loss_negbin evaluated in BigFloat (default 256-bit precision) with r = exp(-2 log_phi), no bounds.
# The value and gradients are returned as Float64, so the optimiser and the ODE stay in Float64.
function loss_negbin_bigfloat(pred, data, log_phi)
    return Float64(loss_negbin(BigFloat.(pred), data, exp(-2 * BigFloat(log_phi))))
end

Zygote.@adjoint function loss_negbin_bigfloat(pred, data, log_phi)
    val, back = Zygote.pullback((p, l) -> loss_negbin(p, data, exp(-2 * l)), BigFloat.(pred), BigFloat(log_phi))
    function loss_negbin_bigfloat_pullback(ȳ)
        g_pred, g_log_phi = back(BigFloat(ȳ))
        return (Float64.(g_pred), nothing, Float64(g_log_phi))
    end
    return Float64(val), loss_negbin_bigfloat_pullback
end

# The poisson_switch loss variant uses the Poisson NLL once r reaches this value
const POISSON_SWITCH_R = 1e5

# How the noise level phi is turned into the NB dispersion r in loss_ude:
#   "bounded"        phi = exp(log_phi) + 1e-3, r = 1/phi^2 + 1e-2   (r in ~[1e-2, 1e6])
#   "bigfloat"       r = 1/phi^2 with no bounds, NLL evaluated in BigFloat (tests whether Float64 rounding is the problem)
#   "poisson_switch" r = 1/phi^2 with no bounds, Poisson NLL once r >= POISSON_SWITCH_R
const LOSS_VARIANTS = ("bounded", "bigfloat", "poisson_switch")

# The noise level the likelihood actually uses for a given log_phi
phi_effective(log_phi, loss_variant) = loss_variant == "bounded" ? exp(log_phi) + 1e-3 : exp(log_phi)

#=============================================================
LOSS FUNCTION FOR SINGLE DATASET USING NMSE
==============================================================# 

function loss_ude(p_all, predict_ude, data, u0, tpts, noise, r; loss_variant="bounded")
    pred = predict_ude(p_all, u0)

    if isnothing(pred)
        println("ODE solve failed")
        return Inf
    end
    if noise == 0
        # Mean squared error on the requested time points
        nmse = loss_nmse(pred[tpts], data[tpts])
    else
        # Negative binomial loss on the requested time points (see LOSS_VARIANTS)
        if loss_variant == "bounded"
            # we enforce a lower bound on r via addition of small amount to r
            # and an upper bound via adding a small amount to phi
            phi_used = exp(p_all.log_phi) + 1e-3
            r_used = 1/phi_used^2 + 1e-2
            nmse = loss_negbin(pred[tpts], data[tpts], r_used)
        elseif loss_variant == "bigfloat"
            # r is formed in BigFloat as well, so it cannot overflow when log_phi is very negative
            nmse = loss_negbin_bigfloat(pred[tpts], data[tpts], p_all.log_phi)
        elseif loss_variant == "poisson_switch"
            r_used = exp(-2 * p_all.log_phi)
            nmse = r_used >= POISSON_SWITCH_R ? loss_poisson(pred[tpts], data[tpts]) :
                                                loss_negbin(pred[tpts], data[tpts], r_used)
        else
            error("Unknown loss_variant $(loss_variant); expected one of $(LOSS_VARIANTS)")
        end
    end
    return nmse

end

function regularisation(nn_params)
    # L2 penalty on NN weights (regularisation)
    l2_penalty = 1e-6 * sum(abs2, nn_params)

    return l2_penalty
end

#=============================================================
COMBINED LOSS FOR MULTIPLE DATASETS USING ADAM OPTIMISER
==============================================================# 

function combined_loss_ude_adam(beta_network, st_nn, nn_params, number_of_nn_inputs, predict_ude, trajectories, noise, r)
    
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
        tpts = eachindex(data)
        l, back_all = pullback(theta -> loss_ude(theta, predict_ude, data, u0, tpts, noise, r), p_all)

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

function combined_loss_ude_lbfgs(beta_network, st_nn, nn_params, number_of_nn_inputs, predict_ude, trajectories, noise, r)
    
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

        tpts = eachindex(data)
        l, back_all = pullback(theta -> loss_ude(theta, predict_ude, data, u0, tpts, noise, r), p_all)
        if !isfinite(l)
           error("ODE solve failed for trajectory $i. Loss: $l")
        end
        grad = back_all((one(l)))[1]
        if isnothing(grad) || isnothing(grad.nn_params)
            error("No gradient available for trajectory $i")
        end
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

