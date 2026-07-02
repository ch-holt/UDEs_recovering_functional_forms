
#========================================================
PARAMETER SENSITIVITY PLOTS FOR BETA EXPONENTIAL
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DrWatson
using JLD2
using Plots
using UDE_FUNCTIONAL_FORMS

# Recovery rate (infectious period of 10 days)
const gamma = 1/10

locations = collect(keys(POPULATION))

#========================================================
LOAD BETA TRAJECTORIES
=========================================================#

betas_over_time   = Dict{String, Vector{Float64}}()
days_dict         = Dict{String, Vector{Float64}}()
betas_between_0_1 = Dict{String, Vector{Float64}}()
x_hat_dict        = Dict{String, Vector{Float64}}()

for location in locations
    path_ot = datadir("exp_pro", "synthetic_data", "synthetic_beta_trajectories",
                      "ground_truth_beta_exp", "location_$(location)",
                      "synthetic_beta_exp_$(location)_over_time.jld2")
    path_01 = datadir("exp_pro", "synthetic_data", "synthetic_beta_trajectories",
                      "ground_truth_beta_exp", "location_$(location)",
                      "synthetic_beta_exp_$(location)_between_0_1.jld2")

    d_ot = JLD2.load(path_ot)
    betas_over_time[location] = vec(Float64.(d_ot["beta"]))
    days_dict[location]       = Float64.(collect(d_ot["days"]))

    d_01 = JLD2.load(path_01)
    betas_between_0_1[location] = vec(Float64.(d_01["beta"]))
    x_hat_dict[location]        = vec(Float64.(d_01["x_hat"]))
end

#========================================================
EXTRACT PARAMETERS
=========================================================#

beta0_dict    = Dict(loc => R0_REPRODUCTION[loc] * (gamma + DELTA[loc]) for loc in locations)
delta_dict    = Dict(loc => DELTA[loc] for loc in locations)
zeta_dict     = Dict(loc => ZETA[loc] for loc in locations)
zeta_delta_dict = Dict(loc => ZETA[loc] * DELTA[loc] for loc in locations)

#========================================================
HELPER: BUILD OVERLAY PLOT COLOURED BY PARAMETER
=========================================================#

function make_sensitivity_plot(locations, x_dict, y_dict, param_dict, xlabel, ylabel, param_label)
    param_vals = [param_dict[loc] for loc in locations]
    param_min, param_max = minimum(param_vals), maximum(param_vals)
    span = param_max - param_min

    norm_val(v) = span > 0 ? (v - param_min) / span : 0.5

    cmap = cgrad(:blues)

    p = plot(; xlabel=xlabel, ylabel=ylabel, legend=false,
               colorbar=true, colorbar_title=param_label,
               clims=(param_min, param_max), color=cmap)

    for loc in locations
        c = cmap[norm_val(param_dict[loc])]
        plot!(p, x_dict[loc], y_dict[loc]; color=c, alpha=0.5, linewidth=0.8)
    end

    # Invisible scatter with zcolor drives the colorbar rendering
    scatter!(p, [NaN], [NaN];
             zcolor=[param_min], clims=(param_min, param_max),
             color=cmap, colorbar=true, colorbar_title=param_label,
             label="", markersize=0)

    return p
end

#========================================================
GENERATE AND SAVE PLOTS
=========================================================#

save_dir = plotsdir("synthetic_data", "synthetic_beta_trajectories",
                    "ground_truth_beta_exp", "parameter_sensitivity")
mkpath(save_dir)

params = [
    ("beta0",       beta0_dict,       "β₀"),
    ("delta",       delta_dict,       "δ"),
    ("zeta",        zeta_dict,        "ζ"),
    ("zeta_delta",  zeta_delta_dict,  "ζ·δ"),
]

for (param_name, param_dict, param_label) in params

    # --- over time ---
    p_ot = make_sensitivity_plot(
        locations, days_dict, betas_over_time, param_dict,
        "Days", "β(t)", param_label
    )
    title!(p_ot, "β exp over time  |  colour = $(param_label)")
    savefig(p_ot, joinpath(save_dir, "beta_exp_sensitivity_$(param_name)_over_time.png"))

    # --- between 0 and 1 ---
    p_01 = make_sensitivity_plot(
        locations, x_hat_dict, betas_between_0_1, param_dict,
        "I/N", "β(I/N)", param_label
    )
    title!(p_01, "β exp vs I/N  |  colour = $(param_label)")
    savefig(p_01, joinpath(save_dir, "beta_exp_sensitivity_$(param_name)_between_0_1.png"))

    println("Saved plots for $(param_name)")
end

println("Done — 8 plots saved to:\n  $(save_dir)")
