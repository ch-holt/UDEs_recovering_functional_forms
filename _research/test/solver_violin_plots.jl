#========================================================
SOLVER COMPARISON VIOLINS — same design as the split-strategy comparison
in state_distribution_plots.jl's "by_form" section (split_comparison_panel),
just with solver as the x-axis/grouping variable instead of split. Kept as
its own script rather than folded into state_distribution_plots.jl so the
existing split-strategy plots there are never at risk of being touched.

One figure per (score kind, metric) — rCRPS/CRPS x infectious/beta
in-sample/beta out-of-sample, 6 total — each with one panel per beta form,
solver on the x-axis within each panel. Reads crps_ideal_<prefix>.csv
(built by crps_log_summary_tables.jl, currently pointed at the solver
prefixes — see its SIM_NAME_PREFIXES).
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using CSV
using DataFrames
using Statistics
using Plots
using StatsPlots

const BETA_FORMS = ["beta_exp", "beta_rational", "beta_mixed"]
const FORM_LABEL = Dict("beta_exp" => "exponential", "beta_rational" => "rational", "beta_mixed" => "mixed")

const SOLVER_PREFIXES = ["rosenbrock23_", "vern7_", "autotsit5_", "delta_pop_"]
# "delta_pop_" is the random-split comparison from earlier -- its UDE runs
# used Tsit5, so it stands in as the Tsit5 arm here rather than rerunning
# an identical job. Its crps_ideal_delta_pop_.csv was already built against
# the correct (non-HQ) ground truth it was actually trained on.
const SOLVER_DISPLAY_NAME = Dict("rosenbrock23_" => "Rosenbrock23", "vern7_" => "Vern7",
                                  "autotsit5_" => "AutoTsit5", "delta_pop_" => "Tsit5")
const SOLVER_COLOR = Dict("rosenbrock23_" => :steelblue, "vern7_" => :darkorange,
                           "autotsit5_" => :seagreen, "delta_pop_" => :firebrick)

save_dir = projectdir("_research", "state_distribution_plots", "by_solver")
mkpath(save_dir)

solver_crps = Dict(prefix => CSV.read(projectdir("_research", "crps_tables", "crps_ideal_$(prefix).csv"), DataFrame)
                    for prefix in SOLVER_PREFIXES)

# One panel: all three solvers on the x-axis, for one (metric, beta_form)
# combo, titled with the beta form. show_parity_line draws the rCRPS=1
# reference (only meaningful for a ratio metric, so raw-CRPS calls pass
# false) -- same convention as split_comparison_panel in
# state_distribution_plots.jl.
function solver_comparison_panel(metric::Symbol, beta_form::String, ylab; show_parity_line::Bool)
    parts = DataFrame[]
    for prefix in SOLVER_PREFIXES
        sub = filter(:beta_function => ==(beta_form), solver_crps[prefix])
        sub = dropmissing(sub, metric)
        sub = sub[sub[!, metric] .> 0, :]   # log10 undefined at/below 0
        append!(parts, [DataFrame(value = log10.(sub[!, metric]),
                                   solver_label = fill(SOLVER_DISPLAY_NAME[prefix], nrow(sub)),
                                   solver_color = fill(SOLVER_COLOR[prefix], nrow(sub)))])
    end
    combined = vcat(parts...)
    p = @df combined violin(:solver_label, :value, group=:solver_label,
                             color=:solver_color, alpha=0.4, legend=false, linewidth=0)
    @df combined boxplot!(p, :solver_label, :value, fillalpha=0, linewidth=1.5, color=:black, outliers=false)
    @df combined dotplot!(p, :solver_label, :value, color=:black, markersize=2.5, markeralpha=0.5)
    show_parity_line && hline!(p, [0.0]; linestyle=:dash, color=:red, linewidth=1, label="")
    xlabel!(p, "Solver")
    ylabel!(p, ylab)
    title!(p, FORM_LABEL[beta_form])
    plot!(p, left_margin=12Plots.mm, bottom_margin=6Plots.mm, titlefontsize=10)
    return p
end

const SCORE_KIND_TITLE = Dict("rcrps" => "rCRPS", "crps" => "CRPS")

for (score_kind, metrics, ylab_prefix, show_parity_line) in [
        ("rcrps", [(:rcrps_inf, "infectious", "infectious (scored on the log scale)"),
                   (:rcrps_beta_insample, "beta_insample", "beta in-sample"),
                   (:rcrps_beta_outsample, "beta_outsample", "beta out-of-sample")],
         "log10(rCRPS)", true),
        ("crps",  [(:log_crps_ude_inf, "infectious", "infectious (scored on the log scale)"),
                   (:crps_nn_beta_insample, "beta_insample", "beta in-sample"),
                   (:crps_nn_beta_outsample, "beta_outsample", "beta out-of-sample")],
         "log10(CRPS)", false)]
    for (metric, slug, display) in metrics
        panels = [solver_comparison_panel(metric, bf, "$(ylab_prefix), $(display)"; show_parity_line)
                  for bf in BETA_FORMS]
        p = plot(panels...; layout=(1, length(BETA_FORMS)), size=(500 * length(BETA_FORMS), 500),
                 plot_title="$(SCORE_KIND_TITLE[score_kind]) by solver, grouped by beta form ($(display))")
        out_path = joinpath(save_dir, "$(score_kind)_violin_by_form_$(slug).png")
        savefig(p, out_path)
        println("Saved $(score_kind) by-form violin ($(slug)) to: $(out_path)")
    end
end
