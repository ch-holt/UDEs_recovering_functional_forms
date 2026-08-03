function plot_ensemble_summary(days, true_inf, I_grid, true_beta_over_time, true_beta_01,
                                I_N_min, I_N_max, location,
                                inf_rows, beta_time_rows, beta_01_rows)


    #========================================================
    INITIALISE PLOTS WITH TRUE BETA AS REFERENCE
    =========================================================#

    traj_plot = plot(days[1:length(true_inf)], true_inf;
        color=:black, linewidth=2, label="Data",
        xlabel="Day", ylabel="Infectious individuals",
        title="Infectious trajectory — ($(location))",
        legend=:topright, left_margin=10Plots.mm)

    beta_time_plot = plot(days[1:length(true_beta_over_time)], true_beta_over_time;
        color=:black, linewidth=2, label="True β",
        xlabel="Day", ylabel="β(t)",
        title="Beta over time — ($(location))",
        legend=:topright)

    beta_01_plot = plot(I_grid, true_beta_01;
        color=:black, linewidth=2, label="True β",
        xlabel="I/N", ylabel="β(I/N)",
        title="Beta vs I/N — ($(location))",
        legend=:topright)
    vline!(beta_01_plot, [I_N_min]; linestyle=:dot, color=:black, linewidth=2, label="Min I/N (observed)")
    vline!(beta_01_plot, [I_N_max]; linestyle=:dot, color=:gray40, linewidth=2, label="Max I/N (observed)")

    #========================================================
    OVERLAY EACH SEED
    =========================================================#


    for i_traj in inf_rows
        plot!(traj_plot, days[1:length(i_traj)], i_traj; color=:red, alpha=0.3, linewidth=1, label="")
    end
    for beta_traj in beta_time_rows
        plot!(beta_time_plot, days[1:length(beta_traj)], beta_traj; color=:red, alpha=0.3, linewidth=1, label="")
    end
    for beta_0_1 in beta_01_rows
        plot!(beta_01_plot, I_grid, beta_0_1; color=:red, alpha=0.3, linewidth=1, label="")
    end
        

    #========================================================
    CALCULATE CRPS
    =========================================================#

    # Convert to matrices
    inf_trajectories = stack(inf_rows; dims=1)
    beta_trajectories = stack(beta_time_rows; dims=1)
    beta_01_trajectories = stack(beta_01_rows; dims=1)

    # Add all forecasts
    inf_fc = Forecast(horizon = collect(1:length(true_inf)), trajectories = inf_trajectories)
    # Add the true trajectory
    inf_fc = add_truth(inf_fc, true_inf)
    # Calculate CRPS
    inf_crps  = score(inf_fc, CRPS_trajectory())

    beta_fc = Forecast(horizon = collect(1:length(true_beta_over_time)), trajectories = beta_trajectories)
    beta_fc = add_truth(beta_fc, true_beta_over_time)
    beta_crps  = score(beta_fc, CRPS_trajectory())

    beta_01_fc = Forecast(horizon = collect(1:length(I_grid)), trajectories = beta_01_trajectories)
    beta_01_fc = add_truth(beta_01_fc, true_beta_01)
    beta_01_crps  = score(beta_01_fc, CRPS_trajectory())

    # Add CRPS to plot titles
    title!(traj_plot, "Infectious trajectory — ($(location))\nCRPS: $(round(inf_crps, sigdigits=3))")
    xlabel!(traj_plot, "Day")
    ylabel!(traj_plot, "Infectious individuals")

    title!(beta_time_plot, "Beta over time — ($(location))\nCRPS: $(round(beta_crps, sigdigits=3))")
    xlabel!(beta_time_plot, "Day")
    ylabel!(beta_time_plot, "β(t)")

    title!(beta_01_plot, "Beta vs I/N — ($(location))\nCRPS: $(round(beta_01_crps, sigdigits=3))")
    xlabel!(beta_01_plot, "I/N")
    ylabel!(beta_01_plot, "β(I/N)")

    return traj_plot, beta_time_plot, beta_01_plot
end