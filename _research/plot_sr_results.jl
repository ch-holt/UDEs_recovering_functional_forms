#========================================================
SCRIPT TO PLOT THE SYMBOLIC REGRESSION RESULTS
(saved by scripts/ude_symbolic_regression.jl)
=========================================================#

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using JLD2
using SymbolicRegression
using MLJ
using Plots

# Call module
using UDE_FUNCTIONAL_FORMS

#=============================================================
DEFINE PARAMETERS
=============================================================#

# Same arguments as scripts/ude_symbolic_regression.jl
const BETA_FUNCTIONS = Dict("beta_exp" => beta_exp, "beta_rational" => beta_rational, "beta_mixed" => beta_mixed)
const beta_function = BETA_FUNCTIONS[get(ARGS, 1, "beta_exp")]
const noise = parse(Float64, get(ARGS, 2, "0.0"))
const train_length = parse(Int, get(ARGS, 3, "365"))
const solver_name = get(ARGS, 4, "autotsit5")
const MS_limit = parse(Float64, get(ARGS, 5, "0.3"))
const multistart = true

sim_name = "$(solver_name)_UDE_single_beta=$(beta_function)_adam=2500_lbfgs=2000_traindata=$(train_length)_noise=$(noise)"

for location in sort(collect(keys(POPULATION)))

    root = datadir("exp_pro","sims",  "ude_single", sim_name, "synthetic_$(location)")
    seed_folders = get_seed_folders(root, multistart, MS_limit)

    for folder in seed_folders

        output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "$(sim_name)", "synthetic_$(location)", folder)
        report_path = joinpath(output_dir, "SR_report.jld2")
        if !isfile(report_path)
            println("No SR results for $(location) $(folder), skipping.")
            continue
        end
        plot_title = "SR on $(location), train_length=$(train_length), noise=$(noise)"

        sr = JLD2.load(report_path)
        r = sr["report"]
        SR_beta_days = sr["SR_beta_days"]
        SR_beta_0_1 = sr["SR_beta_0_1"]
        i_sr = sr["SR_inf"]
        true_beta_days = sr["true_beta_days"]
        NN_beta_days = sr["NN_beta_days"]
        true_beta_0_1 = sr["true_beta_0_1"]
        NN_beta_0_1 = sr["NN_beta_0_1"]
        range_0_1 = sr["range_0_1"]
        SR_input_days = sr["SR_input_days"]
        I_nn = sr["I_nn"]
        obs = sr["obs"]
        days = sr["days"]
        simplified_equation = sr["simplified_equation"]

        #=============================================================
        PLOT BETA AGAINST DAYS
        =============================================================#

        nmse_SR_true_days = loss_nmse(vec(SR_beta_days), vec(true_beta_days))
        nmse_SR_NN_days   = loss_nmse(vec(SR_beta_days), vec(NN_beta_days))
        nmse_NN_true_days = loss_nmse(vec(NN_beta_days), vec(true_beta_days))

        p_comparison = plot(days, true_beta_days[1:length(days)], lw=2.5, label="True β", color=:black)

        plot!(p_comparison, days, SR_beta_days[1:length(days)], lw=2.5, ls=:dash, label="SR-recovered β", color=:red, alpha=0.8)
        plot!(p_comparison, days, NN_beta_days[1:length(days)], lw=2.5, ls=:dot, label="NN approximation", color=:lightblue, alpha=0.8)
        xlabel!(p_comparison, "Day")
        ylabel!(p_comparison, "β")
        title!(p_comparison, "Symbolic Regression on $(plot_title) vs true β\nEq: $(simplified_equation)", titlefontsize=8)
        plot!(p_comparison, legend=:best, legendfontsize=10)
        plot!(p_comparison, grid=true, gridalpha=0.3)

        x_ann = days[end] * 0.75
        y_ann = maximum(NN_beta_days) * 0.85
        dy = maximum(NN_beta_days) * 0.06
        annotate!(p_comparison, x_ann, y_ann, text("NMSE (NN vs SR) = $(round(nmse_SR_NN_days, sigdigits=3))", 9))
        annotate!(p_comparison, x_ann, y_ann - dy, text("NMSE (SR vs true) = $(round(nmse_SR_true_days, sigdigits=3))", 9))
        annotate!(p_comparison, x_ann, y_ann - 2*dy, text("NMSE (NN vs true) = $(round(nmse_NN_true_days, sigdigits=3))", 9))
        savefig(p_comparison, joinpath(output_dir, "$(sim_name)_SR_beta_vs_true.png"))

        #=============================================================
        PLOT BETA BETEEN 0 AND 1
        =============================================================#

        # Evaluate nmse
        nmse_SR_true_0_1 = loss_nmse(vec(SR_beta_0_1), vec(true_beta_0_1))
        nmse_SR_NN_0_1   = loss_nmse(vec(SR_beta_0_1), vec(NN_beta_0_1))
        nmse_NN_true_0_1 = loss_nmse(vec(NN_beta_0_1), vec(true_beta_0_1))

        x_hat_vs_beta = plot(range_0_1, true_beta_0_1, lw=2.5, label="True β", color=:black)


        plot!(x_hat_vs_beta, range_0_1, SR_beta_0_1, lw=2.5, ls=:dash, label="SR-recovered β", color=:red, alpha=0.8)
        plot!(x_hat_vs_beta, range_0_1, NN_beta_0_1, lw=2.5, ls=:dot, label="NN approximation", color=:lightblue, alpha=0.8)
        xlabel!(x_hat_vs_beta, "x̂ = I / N")
        ylabel!(x_hat_vs_beta, "β")
        title!(x_hat_vs_beta, "Symbolic Regression on $(plot_title) vs true β\nEq: $(simplified_equation)", titlefontsize=8)
        plot!(x_hat_vs_beta, legend=:topleft, legendfontsize=8)
        plot!(x_hat_vs_beta, grid=true, gridalpha=0.3)

        I_N_obs_min = minimum(SR_input_days)
        I_N_obs_max = maximum(SR_input_days)
        vline!(x_hat_vs_beta, [I_N_obs_min], linestyle=:dash, color=:gray40, linewidth=1.5, label="Min observed I/N")
        vline!(x_hat_vs_beta, [I_N_obs_max], linestyle=:dash, color=:gray70, linewidth=1.5, label="Max observed I/N")

        y_range = maximum(SR_beta_0_1) - minimum(SR_beta_0_1)
        x_ann = maximum(range_0_1) * 0.60
        y_ann = minimum(SR_beta_0_1) + y_range * 0.35
        dy = y_range * 0.12
        annotate!(x_hat_vs_beta, x_ann, y_ann,        text("NMSE (NN vs SR) = $(round(nmse_SR_NN_0_1, sigdigits=3))", :left, 9))
        annotate!(x_hat_vs_beta, x_ann, y_ann - dy,   text("NMSE (SR vs true) = $(round(nmse_SR_true_0_1, sigdigits=3))", :left, 9))
        annotate!(x_hat_vs_beta, x_ann, y_ann - 2*dy, text("NMSE (NN vs true) = $(round(nmse_NN_true_0_1, sigdigits=3))", :left, 9))
        savefig(x_hat_vs_beta, joinpath(output_dir, "$(sim_name)_SR_beta_against_x_hat.png"))

        #=============================================================
        PLOT SEIRD + SR MODEL
        =============================================================#

        # Plot infection trajectory
        i_traj_plot = plot(days, obs, lw=2.5, label="True observations", color=:black)

        # Evaluate nmse
        nmse_SR_true_traj = loss_nmse(i_sr, obs)
        nmse_NN_true_traj = loss_nmse(vec(I_nn), obs)
        plot!(i_traj_plot, days, i_sr, lw=2.5, ls=:dash, label="SEIRD_SR model", color=:red, alpha=0.8)
        plot!(i_traj_plot, days, vec(I_nn), lw=2.5, ls=:dot, label="UDE model", color=:lightblue, alpha=0.8)
        xlabel!(i_traj_plot, "Time")
        ylabel!(i_traj_plot, "Infectious Individuals")
        title!(i_traj_plot, "Infection Trajectory")
        plot!(i_traj_plot, legend=:best, legendfontsize=10)
        plot!(i_traj_plot, grid=true, gridalpha=0.3)

        y_max = maximum(vcat(obs, i_sr, vec(I_nn)))
        x_ann = maximum(days) * 0.95
        y_ann = y_max * 0.25
        dy    = y_max * 0.10
        annotate!(i_traj_plot, x_ann, y_ann,      text("NMSE (SEIRD+SR vs true) = $(round(nmse_SR_true_traj, sigdigits=3))", :right, 9))
        annotate!(i_traj_plot, x_ann, y_ann - dy, text("NMSE (UDE vs true) = $(round(nmse_NN_true_traj, sigdigits=3))", :right, 9))
        savefig(i_traj_plot, joinpath(output_dir, "$(sim_name)_traj_plot.png"))


        #=============================================================
        LOSS AND DIFFERENCE PLOTS
        =============================================================#

        # Plot complexity against NMSE
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


        # Plot differences against normalized x̂
        NN_minus_true = vec(NN_beta_0_1) - vec(true_beta_0_1)
        SR_minus_true = vec(SR_beta_0_1) - vec(true_beta_0_1)


        p_diff_xhat = plot(range_0_1, NN_minus_true, lw=2.5, ls=:dot, label="y_hat - true β", color=:lightblue, alpha=0.8)
        plot!(p_diff_xhat, range_0_1, SR_minus_true, lw=2.5, ls=:dot, label="SR result for y_hat - true β", color=:red)
        xlabel!(p_diff_xhat, "x̂=I/N")
        ylabel!(p_diff_xhat, "Difference in β")
        title!(p_diff_xhat, "Difference between true β and y_hat = $(plot_title) and SR approx \nEq: $(simplified_equation)", titlefontsize=8)
        plot!(p_diff_xhat, legend=:topright, legendfontsize=10)
        plot!(p_diff_xhat, grid=true, gridalpha=0.3)

        savefig(p_diff_xhat, joinpath(output_dir, "$(sim_name)_SR_NN_minus_true_beta_xhat.png"))

        # Save panel with the three main plots
        panel = plot(i_traj_plot, p_comparison, x_hat_vs_beta; layout=(1, 3), size=(1800, 500))
        savefig(panel, joinpath(output_dir, "$(sim_name)_panel.png"))
    end
end
