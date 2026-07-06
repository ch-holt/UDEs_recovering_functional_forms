#========================================================
FUNCTION TO TRAIN UDE MODEL ON SINGLE DATASET
=========================================================# 

function train_ude_single_dataset(p, predict_ude, data, u0, beta_network, st_nn, number_of_nn_inputs, learning_bias_bool; maxiters_adam, maxiters_lbfgs, adam_learning_rate)

    # Set up optimisation (first Adam then LBFGS)
    optimised_state = Optimisers.setup(Optimisers.Adam(adam_learning_rate), p)

    # Create 1D vector to track losses during training
    losses = Float64[]
    best_loss = Inf
    best_p = p

    for iter in 1:maxiters_adam

        # Compute the loss, predicted mortalities and gradient function
        l, back_all = pullback(theta -> loss_ude(theta, predict_ude, data, u0, beta_network, st_nn, number_of_nn_inputs, learning_bias_bool) + regularisation(theta.nn_params), p)
        println("Iteration $iter, Loss: $l")
        # Evaluate the gradient of the loss w.r.t p
        grad = back_all((one(l)))[1]

    	# Stop training if 5 consecutive Inf losses
		if l == Inf && length(losses) >= 5 && all(isinf, losses[end-4:end])
			println("Unstable parameter region. Aborting...")
			break
		end 

		if isnothing(grad)
			println("No gradient found. Loss: $l")
			p = best_p
			continue
		end

        push!(losses, l)

        # Store best iteration
        if l < best_loss
            best_loss = l
            best_p = p
        end
        
        # Update parameters using the gradient
        optimised_state, p = Optimisers.update(optimised_state, p, grad)

    end

    # Then do LBFGS optimisation
    adtype = Optimization.AutoZygote()
    optfunc   = Optimization.OptimizationFunction(
                 (theta, _) -> loss_ude(theta, predict_ude, data, u0, beta_network, st_nn, number_of_nn_inputs, learning_bias_bool) + regularisation(theta.nn_params),
                 adtype)
    optprob = Optimization.OptimizationProblem(optfunc, best_p)

    iter_lbfgs = Ref(0)
    res = Optimization.solve(
        optprob,
        Optim.LBFGS(m=10),
        callback = (state, l) -> begin
            iter_lbfgs[] += 1
            push!(losses, l)

            if l < best_loss
                best_loss = l
                best_p = state.u
            end

            iter_lbfgs[] % 50 == 0 && println("LBFGS iter $(iter_lbfgs[]): $l")
            return false
        end,
        maxiters = maxiters_lbfgs
    )

    final_loss = loss_ude(res.u, predict_ude, data, u0, beta_network, st_nn, number_of_nn_inputs, learning_bias_bool) + regularisation(res.u.nn_params)
    if final_loss < best_loss
        best_p = res.u
    end

    return best_p, losses
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
        total_loss, total_grad = combined_loss_ude_adam(beta_network, st_nn, nn_params, number_of_nn_inputs, predict_ude, trajectories)

    	# Stop training if 5 consecutive Inf losses
		if total_loss == Inf && length(total_losses) >= 4 && all(isinf, total_losses[end-4:end])
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
                (theta, _) -> combined_loss_ude_lbfgs(beta_network, st_nn, theta, number_of_nn_inputs, predict_ude, trajectories)[1],
                grad = (G, theta, _)-> (G .= combined_loss_ude_lbfgs(beta_network, st_nn, theta, number_of_nn_inputs, predict_ude, trajectories)[2])
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

    final_loss, _ = combined_loss_ude_lbfgs(beta_network, st_nn, res.u, number_of_nn_inputs, predict_ude, trajectories)

    if final_loss < best_loss
        best_nn_params = res.u
    end

    return best_nn_params, total_losses
end