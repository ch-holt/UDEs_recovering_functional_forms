#======================================
MODULE TO UNDERTAKE SYMBOLIC REGRESSION
=======================================#  

include("functions.jl")

module symbolic_regression_module

export symbolic_regression

include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA
import Main: Functions
using SymbolicUtils 
using SymbolicRegression
using Symbolics
using DynamicExpressions
using MLJ
using Plots
using Statistics
using JLD2
using DataFrames




function symbolic_regression(SR_input_pair, SR_output_pair, sim_name, location, output_dir, plot_title, seed, sing_or_multi, norm_i_traj, nn_output_days, beta_function)
    
    # Recover true beta result
    population = POPULATION[location]
    r0_reproduction = R0_REPRODUCTION[location]
    delta = DELTA[location]
    zeta = ZETA[location]
    gamma =1/10

    model = SRRegressor(
        niterations=800,
        binary_operators=[+, -, *, /],
        unary_operators=[exp],
        maxsize = 20,
        output_directory = output_dir,
        # Make results reproducible by disabling multithreading
        #populations = 15,
        population_size = 50,
        parsimony = 0.01,
        complexity_of_constants = 2,
        parallelism=:serial,
        seed = seed,
        deterministic = true,
        batching = false
    )

    # Create and train model on this data
    mach = machine(model, SR_input_pair, SR_output_pair)
    fit!(mach)

    r =report(mach)

    # Save both report and machine to the same file
    JLD2.save(joinpath(output_dir, "SR_report.jld2"), "report", r, "mach", mach)

    best_equation=r.equations[r.best_idx]
    best_symbolic = SymbolicRegression.node_to_symbolic(best_equation)
    simplified_equation = SymbolicUtils.simplify(best_symbolic, expand=true)

    # Round to 3 significant figures
    simplified_equation = Functions._format_equation_sigfigs(string(simplified_equation))

    # Plot SR prediction against true beta

    # Recover SR prediction for the best equation (1000 points)
    SR_evaluated_beta_1000 = predict(mach, (data=SR_input_pair, idx=r.best_idx))

    # Make the input for SR a vector (1000 points)
    SR_target_beta_1000 = vec(SR_output_pair)

    days = 1:length(norm_i_traj)

    #=============================================================
    MAKE PLOTS
    =============================================================#

    # Plot true beta against the input for the SR (y_hat) against the SR evaluation over time

    # First retrieve the SR prediction for the true infection trajectory
    if sing_or_multi == "single"
        SR_input_days = reshape(norm_i_traj, :, 1)
    elseif sing_or_multi == "multi"
        # Normalise the inputs for the NN and SR
        beta0 = r0_reproduction * (gamma + delta)
        delta_norm =(log(delta) - log(1e-6))/(log(1e-2) - log(1e-6))
        beta0_norm = (r0_reproduction- 1.2)/(6.0 - 1.2)
        zeta_norm = zeta/0.05
        SR_input_days = DataFrame(
                            I     = vec(norm_i_traj),
                            beta0 = fill(beta0_norm, length(norm_i_traj)),
                            zeta  = fill(zeta_norm,  length(norm_i_traj)),
                            delta = fill(delta_norm, length(norm_i_traj))
    )
    end

    # Evaluate the beta function for normalised inputs (number of days)
    if beta_function == "exponential"
        true_beta_days_traj = reshape(Functions.beta_exp(location, norm_i_traj.*population),:,1)
    elseif beta_function == "rational"
        true_beta_days_traj = reshape(Functions.beta_rational(location, norm_i_traj.*population),:,1)
    end

    # Evaluate the SR equation for number of days
    SR_evaluated_beta_days = reshape(predict(mach, (data=SR_input_days, idx=r.best_idx)),:,1)
    
    # Evaluate the SR output pair for number of days (e.g. the nn output for the true trajectory)
    target_beta_days = reshape(nn_output_days, :, 1)

    mse_SR_true_days = Functions.loss_mse(SR_evaluated_beta_days, true_beta_days_traj)
    mse_SR_input_days = Functions.loss_mse(SR_evaluated_beta_days, target_beta_days)
    mse_input_true_days = Functions.loss_mse(target_beta_days, true_beta_days_traj)

    p_comparison = plot(days, true_beta_days_traj[1:length(days)], lw=2.5, label="True β", color=:black)
    
    plot!(p_comparison, days, SR_evaluated_beta_days[1:length(days)], lw=2.5, ls=:dash, label="SR-recovered β", color=:red, alpha=0.8)
    plot!(p_comparison, days, target_beta_days[1:length(days)], lw=2.5, ls=:dot, label="SR input", color=:lightblue, alpha=0.8)
    xlabel!(p_comparison, "Day")    
    ylabel!(p_comparison, "β")
    title!(p_comparison, "Symbolic Regression on $(plot_title) vs true β\nEq: $(simplified_equation)", titlefontsize=8)
    plot!(p_comparison, legend=:best, legendfontsize=10)
    plot!(p_comparison, grid=true, gridalpha=0.3)

    x_ann = days[end] * 0.75
    y_ann = maximum(target_beta_days) * 0.85
    dy = maximum(target_beta_days) * 0.06
    annotate!(p_comparison, x_ann, y_ann, text("MSE (input vs SR) = $(round(mse_SR_input_days, sigdigits=3))", 9))
    annotate!(p_comparison, x_ann, y_ann - dy, text("MSE (SR vs true) = $(round(mse_SR_true_days, sigdigits=3))", 9))
    annotate!(p_comparison, x_ann, y_ann - 2*dy, text("MSE (input vs true) = $(round(mse_input_true_days, sigdigits=3))", 9))

    savefig(p_comparison, joinpath(output_dir, "$(sim_name)_SR_beta_vs_true.png"))

    # Plot complexity against MSE
    complexities = r.complexities
    losses = r.losses
    log_losses = log10.(losses)
    p_complexity = scatter(complexities, log_losses, label="SR equations", color=:blue)
    xlabel!(p_complexity, "Complexity")
    ylabel!(p_complexity, "log10(loss)")
    title!(p_complexity, "SR complexity vs log10 loss against $(plot_title)\nEq: $(simplified_equation)", titlefontsize=8)
    plot!(p_complexity, legend=:topright, legendfontsize=10)
    plot!(p_complexity, grid=true, gridalpha=0.3)
    plot!(p_complexity, complexities, log_losses, label=false, color=:blue, lw=2)

    # Highlight the complexity chosen by the algorithm (r.best_idx) with a red dot
    sel_idx = get(r, :best_idx, nothing)
    if sel_idx !== nothing && 1 <= sel_idx <= length(complexities)
        sel_c = complexities[sel_idx]
        sel_l = log_losses[sel_idx]
        scatter!(p_complexity, [sel_c], [sel_l], marker=:circle, markersize=8, color=:red, label="selected equation")
    end

    savefig(p_complexity, joinpath(output_dir, "$(sim_name)_SR_complexity_vs_loss.png"))


    # Plot beta vs x_hat

    # Evaluate the true beta function for the grid of 1000 points
    if sing_or_multi == "single"
        SR_input_pair_infectious_1000 = SR_input_pair
        if beta_function == "exponential"
            true_beta_1000 = Functions.beta_exp(location, SR_input_pair.*population)
        elseif beta_function == "rational"
            true_beta_1000 = Functions.beta_rational(location, SR_input_pair.*population)
        end
    elseif sing_or_multi == "multi"
        SR_input_pair_infectious_1000 = SR_input_pair.I
        if beta_function == "exponential"
            true_beta_1000 = Functions.beta_exp(location, SR_input_pair.I.*population)
        elseif beta_function == "rational"
            true_beta_1000 = Functions.beta_rational(location, SR_input_pair.I.*population)
        end
    end

    mse_SR_true = Functions.loss_mse(SR_evaluated_beta_1000, true_beta_1000)
    mse_SR_input = Functions.loss_mse(SR_evaluated_beta_1000, SR_target_beta_1000)
    mse_input_true = Functions.loss_mse(SR_target_beta_1000, true_beta_1000)
    
    x_hat_vs_beta = plot(SR_input_pair_infectious_1000, true_beta_1000, lw=2.5, label="True β", color=:black)


    plot!(x_hat_vs_beta, SR_input_pair_infectious_1000, SR_evaluated_beta_1000, lw=2.5, ls=:dash, label="SR-recovered β", color=:red, alpha=0.8)
    plot!(x_hat_vs_beta, SR_input_pair_infectious_1000, SR_target_beta_1000, lw=2.5, ls=:dot, label="SR input", color=:lightblue, alpha=0.8)
    xlabel!(x_hat_vs_beta, "x̂ = I / N")
    ylabel!(x_hat_vs_beta, "β")
    title!(x_hat_vs_beta, "Symbolic Regression on $(plot_title) vs true β\nEq: $(simplified_equation)", titlefontsize=8)
    plot!(x_hat_vs_beta, legend=:best, legendfontsize=10)
    plot!(x_hat_vs_beta, grid=true, gridalpha=0.3)

    x_ann = maximum(SR_input_pair_infectious_1000) * 0.75
    y_ann = maximum(SR_target_beta_1000) * 0.85
    dy = maximum(SR_target_beta_1000) * 0.06
    annotate!(x_hat_vs_beta, x_ann, y_ann, text("MSE (input vs SR) = $(round(mse_SR_input, sigdigits=3))", 9))
    annotate!(x_hat_vs_beta, x_ann, y_ann - dy, text("MSE (SR vs true) = $(round(mse_SR_true, sigdigits=3))", 9))
    annotate!(x_hat_vs_beta, x_ann, y_ann - 2*dy, text("MSE (input vs true) = $(round(mse_input_true, sigdigits=3))", 9))
    savefig(x_hat_vs_beta, joinpath(output_dir, "$(sim_name)_SR_beta_against_x_hat.png"))

    # Plot differences against days
    NN_minus_true = target_beta_days - true_beta_days_traj
    SR_minus_true = SR_evaluated_beta_days - true_beta_days_traj
    p_diff = plot(days, NN_minus_true[1:length(days)], lw=2.5, ls=:dot, label="y_hat - true β", color=:lightblue, alpha=0.8)
    plot!(p_diff, days, SR_minus_true[1:length(days)], lw=2.5, ls=:dot, label="SR result for y_hat - true β", color=:red)
    xlabel!(p_diff, "Days")
    ylabel!(p_diff, "Difference in β")
    title!(p_diff, "Difference between true β and y_hat = $(plot_title) and SR approx \nEq: $(simplified_equation)", titlefontsize=8)
    plot!(p_diff, legend=:best, legendfontsize=10)
    plot!(p_diff, grid=true, gridalpha=0.3)

    savefig(p_diff, joinpath(output_dir, "$(sim_name)_SR_NN_minus_true_beta.png"))

    # Plot differences against normalized x̂
    NN_minus_true = SR_target_beta_1000 - true_beta_1000
    SR_minus_true = SR_evaluated_beta_1000 - true_beta_1000


    p_diff_xhat = plot(SR_input_pair_infectious_1000, NN_minus_true, lw=2.5, ls=:dot, label="y_hat - true β", color=:lightblue, alpha=0.8)
    plot!(p_diff_xhat, SR_input_pair_infectious_1000, SR_minus_true, lw=2.5, ls=:dot, label="SR result for y_hat - true β", color=:red)
    xlabel!(p_diff_xhat, "x̂=I/N")
    ylabel!(p_diff_xhat, "Difference in β")
    title!(p_diff_xhat, "Difference between true β and y_hat = $(plot_title) and SR approx \nEq: $(simplified_equation)", titlefontsize=8)
    plot!(p_diff_xhat, legend=:topright, legendfontsize=10)
    plot!(p_diff_xhat, grid=true, gridalpha=0.3)

    savefig(p_diff_xhat, joinpath(output_dir, "$(sim_name)_SR_NN_minus_true_beta_xhat.png"))

    return mach, r
end


end