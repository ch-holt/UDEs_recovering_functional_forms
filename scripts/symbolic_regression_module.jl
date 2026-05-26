include("functions.jl")
using .Functions

module symbolic_regression_module

export symbolic_regression





include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

using SymbolicRegression
using MLJ
using Plots
using Statistics
using JLD2

function symbolic_regression(x_hat, y_hat, sim_name, location, output_dir, plot_title, seed)

    model = SRRegressor(
        niterations=100,
        binary_operators=[+, -, *, /],
        unary_operators=[exp],
        maxsize = 20,
        output_directory = output_dir,
        # Make results reproducible by disabling multithreading
        parallelism=:serial,
        seed = seed,
        deterministic = true,
        batching = false
    )

    # Create and train model on this data
    mach = machine(model, x_hat, y_hat)
    fit!(mach)

    r =report(mach)

    JLD2.save(joinpath(output_dir, "SR_report.jld2"), "report", r)
    save(joinpath(output_dir, "SR_report.jld2"), "report", r)

    # Select and print the 'best' equation selected by the algorithm
    println(r.best_idx)


    # The equation for that index:
    println(r.equation_strings[r.best_idx])

    # Plot SR prediction against true beta

    # Recover SR prediction
    SR_beta = predict(mach, x_hat)

    # Recover true beta result
    population = POPULATION[location]
    true_beta_SR = Main.Functions.beta_exp(location, x_hat*population)

    # Make the input for SR a vector
    target_beta = vec(y_hat)

    mse_SR_true = Main.Functions.loss_mse(SR_beta, true_beta_SR)
    mse_SR_input = Main.Functions.loss_mse(SR_beta, target_beta)

    
    days = 1:length(target_beta)

    # Plot true beta against the input for the SR against the SR
    p_comparison = plot(days, true_beta_SR, lw=2.5, label="True β", color=:black)
    
    plot!(p_comparison, days, SR_beta, lw=2.5, ls=:dash, label="SR-recovered β", color=:red, alpha=0.8)
    plot!(p_comparison, days, target_beta, lw=2.5, ls=:dot, label="SR input", color=:lightblue, alpha=0.8)
    xlabel!(p_comparison, "Day")
    title!(p_comparison, "Symbolic Regression on $(plot_title) vs true β")
    plot!(p_comparison, legend=:best, legendfontsize=10)
    plot!(p_comparison, grid=true, gridalpha=0.3)

    x_ann = days[end] * 0.75
    y_ann = maximum(target_beta) * 0.85
    dy = maximum(target_beta) * 0.06
    annotate!(p_comparison, x_ann, y_ann, text("MSE (input vs SR) = $(round(mse_SR_input, sigdigits=3))", 9))
    annotate!(p_comparison, x_ann, y_ann - dy, text("MSE (SR vs true) = $(round(mse_SR_true, sigdigits=3))", 9))

    display(p_comparison)
    savefig(p_comparison, joinpath(output_dir, "$(sim_name)_SR_beta_vs_true.png"))

    # Plot complexity against MSE
    complexities = r.complexities
    losses = r.losses
    log_losses = log10.(losses)
    p_complexity = scatter(complexities, log_losses, label="SR equations", color=:blue)
    xlabel!(p_complexity, "Complexity")
    ylabel!(p_complexity, plot_title)
    title!(p_complexity, "SR complexity vs log10 loss against" * plot_title)
    plot!(p_complexity, grid=true, gridalpha=0.3)
    plot!(p_complexity, complexities, log_losses, label=false, color=:blue, lw=2)

    # Highlight the complexity chosen by the algorithm (r.best_idx) with a red dot
    sel_idx = get(r, :best_idx, nothing)
    if sel_idx !== nothing && 1 <= sel_idx <= length(complexities)
        sel_c = complexities[sel_idx]
        sel_l = log_losses[sel_idx]
        scatter!(p_complexity, [sel_c], [sel_l], marker=:circle, markersize=8, color=:red, label="selected equation")
    end

    display(p_complexity)
    savefig(p_complexity, joinpath(output_dir, "$(sim_name)_SR_complexity_vs_loss.png"))

    return mach, r
end


end