#========================================================
FUNCTION TO TRAIN BASELINE MODEL ON SINGLE DATASET
=========================================================#

function train_baseline_single_dataset(p, predict_ude, training_data, u0, noise, r; maxiters_adam, maxiters_lbfgs, adam_learning_rate)

    t_start = time()

    # Set up optimisation (first Adam then LBFGS)
    optimised_state = Optimisers.setup(Optimisers.Adam(adam_learning_rate), p)

    # Fit to every observed time point - no held-out split, since a single scalar
    # parameter has no meaningful overfitting risk to guard against.
    tpts = eachindex(training_data)

    train_losses = Float64[]

    best_loss = Inf
    best_p = p

    for iter in 1:maxiters_adam

        train_l, back_all = pullback(theta -> loss_ude(theta, predict_ude, training_data, u0, tpts, noise, r), p)
        println("Iteration $iter, Loss: $train_l")
        grad = back_all((one(train_l)))[1]

        # Stop training if 5 consecutive Inf losses
        if train_l == Inf && length(train_losses) >= 5 && all(isinf, train_losses[end-4:end])
            println("Unstable parameter region. Aborting...")
            break
        end

        if isnothing(grad)
            println("No gradient found. Loss: $train_l")
            p = best_p
            continue
        end

        push!(train_losses, train_l)

        if train_l < best_loss
            best_loss = train_l
            best_p = p
        end

        optimised_state, p = Optimisers.update(optimised_state, p, grad)
    end

    # Then LBFGS
    adtype = Optimization.AutoZygote()
    optfunc = Optimization.OptimizationFunction(
        (theta, _) -> loss_ude(theta, predict_ude, training_data, u0, tpts, noise, r),
        adtype)
    optprob = Optimization.OptimizationProblem(optfunc, best_p)

    iter_lbfgs = Ref(0)
    res = Optimization.solve(
        optprob,
        Optim.LBFGS(m=10),
        callback = (state, train_l) -> begin
            iter_lbfgs[] += 1
            push!(train_losses, train_l)
            iter_lbfgs[] % 50 == 0 && println("LBFGS iter $(iter_lbfgs[]): $train_l")
            return false
        end,
        maxiters = maxiters_lbfgs
    )

    elapsed = time() - t_start

    # Return the actual LBFGS-converged point, not the best-training-loss checkpoint
    # (best_p, which only exists as a divergence-recovery fallback / LBFGS warm start).
    # The Hessian/uncertainty step needs a genuine stationary point of the loss - res.u
    # is that; best_p has no such guarantee.
    return res.u, train_losses, elapsed
end

#========================================================
FUNCTION TO ESTIMATE UNCERTAINTY ON THE FITTED beta0
VIA A LAPLACE / DELTA-METHOD APPROXIMATION AT THE OPTIMUM
=========================================================#

function beta0_uncertainty(best_p, predict_ude, training_data, u0, noise, r)

    tpts = eachindex(training_data)

    if noise == 0
        # loss_ude is NMSE here, not a Gaussian NLL (it's scaled by an arbitrary
        # normalising factor), so go via the standard nonlinear least squares
        # (delta-method) formula instead of differentiating it directly.
        J = ForwardDiff.jacobian(p -> predict_ude(p, u0)[tpts], best_p)
        resid = predict_ude(best_p, u0)[tpts] .- training_data[tpts]
        sigma2_hat = sum(resid .^ 2) / (length(tpts) - 1)
        var_beta0 = sigma2_hat / (J' * J)[1, 1]
    else
        # loss_ude is already the negative binomial NLL here (up to beta0-independent
        # constants), so its Hessian directly gives the Fisher information.
        nll(p) = loss_ude(p, predict_ude, training_data, u0, tpts, noise, r)
        H = ForwardDiff.hessian(nll, best_p)
        var_beta0 = 1 / H[1, 1]
    end

    se_beta0 = sqrt(var_beta0)
    beta0_hat = best_p.beta0

    return (beta0_hat = beta0_hat, se_beta0 = se_beta0,
            ci_lower = beta0_hat - 1.96 * se_beta0, ci_upper = beta0_hat + 1.96 * se_beta0)
end

#========================================================
FUNCTION TO TRAIN UDE MODEL ON SINGLE DATASET
=========================================================#

function train_ude_single_dataset(p, predict_ude, training_data, u0, beta_function, location, noise, r; maxiters_adam, maxiters_lbfgs, adam_learning_rate)

    # Set up optimisation (first Adam then LBFGS)
    optimised_state = Optimisers.setup(Optimisers.Adam(adam_learning_rate), p)

    # Temporal split 


    # define training and validation time points
    # Find when the trajectory goes flat
    #flat_start = find_flat_start(training_data; window=14, rel_threshold=0.01)
    #println("Flat start found at time point: $(flat_start)")
    #train_cutoff = round(Int, 0.8 * flat_start)
    #train_tpts = collect(1:train_cutoff)
    #val_tpts   = setdiff(1:length(training_data), train_tpts)

    # Random split

    val_tpts = 5:5:length(training_data)
    train_tpts = setdiff(1:length(training_data), val_tpts)

    # Middle 20% of beta
    #beta_traj = beta_function(location, training_data)

    #q40 = quantile(beta_traj, 0.4)
    #q60 = quantile(beta_traj, 0.6)

    #val_tpts   = findall(b -> q40 <= b <= q60, beta_traj)
    #train_tpts = setdiff(1:length(training_data), val_tpts)

    # Create 1D vector to track losses during training
    train_losses = Float64[]
    val_losses = Float64[]

    best_loss = Inf
    best_p = p

    for iter in 1:maxiters_adam

        # Compute the loss, predicted mortalities and gradient function
        train_l, back_all = pullback(theta -> loss_ude(theta, predict_ude, training_data, u0, train_tpts, noise, r) + regularisation(theta.nn_params), p)
        println("Iteration $iter, Loss: $train_l")
        # Evaluate the gradient of the loss w.r.t p
        grad = back_all((one(train_l)))[1]

    	# Stop training if 5 consecutive Inf losses
		if train_l == Inf && length(train_losses) >= 5 && all(isinf, train_losses[end-4:end])
			println("Unstable parameter region. Aborting...")
			break
		end

		if isnothing(grad)
			println("No gradient found. Loss: $train_l")
			p = best_p
			continue
		end

        push!(train_losses, train_l)

        # Compute validation loss
        val_l = loss_ude(p, predict_ude, training_data, u0, val_tpts, noise, r) + regularisation(p.nn_params)
        push!(val_losses, val_l)

        # Store best iteration (minimising validation loss)
        if val_l < best_loss
            best_loss = val_l
            best_p = p
        end
        
        # Update parameters using the gradient
        optimised_state, p = Optimisers.update(optimised_state, p, grad)

    end

    # Then do LBFGS optimisation
    adtype = Optimization.AutoZygote()
    optfunc   = Optimization.OptimizationFunction(
                 (theta, _) -> loss_ude(theta, predict_ude, training_data, u0, train_tpts, noise, r) + regularisation(theta.nn_params),
                 adtype)
    optprob = Optimization.OptimizationProblem(optfunc, best_p)

    iter_lbfgs = Ref(0)
    res = Optimization.solve(
        optprob,
        Optim.LBFGS(m=10),
        callback = (state, train_l) -> begin
            iter_lbfgs[] += 1
            push!(train_losses, train_l)

            # Compute validation loss
            val_l = loss_ude(state.u, predict_ude, training_data, u0, val_tpts, noise, r) + regularisation(state.u.nn_params)
            push!(val_losses, val_l)

            if val_l < best_loss
                best_loss = val_l
                best_p = state.u
            end

            iter_lbfgs[] % 50 == 0 && println("LBFGS iter $(iter_lbfgs[]): $train_l")
            return false
        end,
        maxiters = maxiters_lbfgs
    )

    return best_p, train_losses, val_losses
end

#========================================================
FUNCTION TO TRAIN UDE MODEL ON MULTIPLE DATASETS
=========================================================# 

function train_ude_multiple_datasets(nn_params, predict_ude, trajectories, beta_network, st_nn, number_of_nn_inputs; maxiters_adam, maxiters_lbfgs)


    # Create 1D vector to track total losses across all trajectories during training
    total_losses = Float64[]
    best_loss = Inf

    # We track the best parameters for all trajectories
    # the relationship between the parameters (e.g. NN weights) is the same across trajectories, but the NN will have different inputs
    best_nn_params = nn_params

    # Set up optimisation
    optimised_state = Optimisers.setup(Optimisers.Adam(1e-3), nn_params)

    for iter in 1:maxiters_adam

        # Print progress so long-running evaluations are visible
        println("Adam iter $iter")

        # Compute the loss, predicted infectious individuals and gradient function for each synthetic trajectory
        total_loss, total_grad = combined_loss_ude_adam(beta_network, st_nn, nn_params, number_of_nn_inputs, predict_ude, trajectories, noise, r)

    	# Stop training if 5 consecutive Inf losses
		if total_loss == Inf && length(total_losses) >= 5 && all(isinf, total_losses[end-4:end])
			println("Unstable parameter region. Aborting...")
			break
		end 

		if isnothing(total_grad)
			println("No gradient found. Loss: $total_loss")
			nn_params = best_nn_params
			continue
		end   

        push!(total_losses, total_loss)

        # Store best iteration
        if total_loss < best_loss
            best_loss = total_loss
            best_nn_params = nn_params
        end

        # Update parameters using the gradient
        optimised_state, nn_params = Optimisers.update(optimised_state, nn_params, total_grad)

    end

    # Then do LBFGS optimisation
    optfunc   = Optimization.OptimizationFunction(
                (theta, _) -> combined_loss_ude_lbfgs(beta_network, st_nn, theta, number_of_nn_inputs, predict_ude, trajectories, noise, r)[1],
                grad = (G, theta, _)-> (G .= combined_loss_ude_lbfgs(beta_network, st_nn, theta, number_of_nn_inputs, predict_ude, trajectories, noise, r)[2])
                )
    optprob = Optimization.OptimizationProblem(optfunc, best_nn_params)

    iter_lbfgs = Ref(0)

    res = try
        Optimization.solve(
        optprob,
        Optim.LBFGS(m=10),
        callback = (state, l) -> begin
            iter_lbfgs[] += 1
            push!(total_losses, l)

            if l < best_loss
                best_loss = l
                best_nn_params = state.u
            end

            iter_lbfgs[] % 50 == 0 && println("LBFGS iter $(iter_lbfgs[]): $l")
            return false
        end,
        maxiters = maxiters_lbfgs
        )

    catch e
        println("LBFGS optimisation failed with error: $e")
        return best_nn_params, total_losses
    end

    println("LBFGS finished with retcode: $(res.retcode)") 

    final_loss, _ = combined_loss_ude_lbfgs(beta_network, st_nn, res.u, number_of_nn_inputs, predict_ude, trajectories, noise, r)

    if final_loss < best_loss
        best_nn_params = res.u
    end

    return best_nn_params, total_losses
end