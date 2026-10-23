#========================================================
FUNCTION TO TRAIN BASELINE MODEL ON SINGLE DATASET
=========================================================#

function train_baseline_single_dataset(p, predict_ude, training_data, u0, noise; maxiters_adam, maxiters_lbfgs, adam_learning_rate)

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

        train_l, back_all = pullback(theta -> loss_ude(theta, predict_ude, training_data, u0, tpts, noise), p)
        iter % 50 == 0 && println("Iteration $iter, Loss: $train_l")
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
        (theta, _) -> loss_ude(theta, predict_ude, training_data, u0, tpts, noise),
        adtype)
    optprob = Optimization.OptimizationProblem(optfunc, best_p)

    iter_lbfgs = Ref(0)
    final_p = try
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
        res.u
    catch e
        println("LBFGS failed ($e), returning best Adam parameters.")
        best_p
    end

    elapsed = time() - t_start

    return final_p, train_losses, elapsed
end

#========================================================
FUNCTION TO ESTIMATE UNCERTAINTY ON THE FITTED beta0
VIA A LAPLACE / DELTA-METHOD APPROXIMATION AT THE OPTIMUM
=========================================================#

function beta0_uncertainty(best_p, predict_ude, training_data, u0, noise)

    tpts = eachindex(training_data)

    if noise == 0
        # loss_ude is NMSE - use standard nonlinear least squares
        # (delta-method) formula
        J = ForwardDiff.jacobian(p -> predict_ude(p, u0)[tpts], best_p)
        resid = predict_ude(best_p, u0)[tpts] .- training_data[tpts]
        sigma2_hat = sum(resid .^ 2) / (length(tpts) - 1)
        var_beta0 = sigma2_hat / (J' * J)[1, 1]
    else
        # loss_ude is already the negative binomial NLL here (up to beta0-independent
        # constants), so its Hessian directly gives the Fisher information.
        nll(p) = loss_ude(p, predict_ude, training_data, u0, tpts, noise)
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

function train_ude_single_dataset(p, loss_fn, n_obs; maxiters_adam, maxiters_lbfgs, adam_learning_rate)

    # Set up optimisation (first Adam then LBFGS)
    optimised_state = Optimisers.setup(Optimisers.Adam(adam_learning_rate), p)

    # Random split

    val_tpts = 5:5:n_obs
    train_tpts = setdiff(1:n_obs, val_tpts)

    # Create 1D vector to track losses during training
    train_losses = Float64[]
    val_losses = Float64[]

    best_loss = Inf
    best_p = p

    for iter in 1:maxiters_adam

        # Compute the loss, predicted infections and gradient function
        train_l, back_all = pullback(theta -> loss_fn(theta, train_tpts) + regularisation(theta.nn_params), p)
        iter % 50 == 0 && println("Iteration $iter, Loss: $train_l")
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
        val_l = loss_fn(p, val_tpts) + regularisation(p.nn_params)
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
                 (theta, _) -> loss_fn(theta, train_tpts) + regularisation(theta.nn_params),
                 adtype)
    optprob = Optimization.OptimizationProblem(optfunc, best_p)

    iter_lbfgs = Ref(0)
    try
        res = Optimization.solve(
            optprob,
            Optim.LBFGS(m=10),
            callback = (state, train_l) -> begin
                iter_lbfgs[] += 1
                push!(train_losses, train_l)

                # Compute validation loss
                val_l = loss_fn(state.u, val_tpts) + regularisation(state.u.nn_params)
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
    catch e
        println("LBFGS failed ($e), returning best Adam parameters.")
    end


    return best_p, train_losses, val_losses
end
