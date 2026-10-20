#========================================================
RELTOL SENSITIVITY VIOLINS — same design as solver_violin_plots.jl, with
8 arms on the x-axis: Tsit5 and AutoTsit5 at reltol 1e-3 (the default),
1e-5, 1e-6, 1e-10, ordered by reltol (Tsit5 then AutoTsit5 within each).

The default-tolerance arms are the existing runs: Tsit5 = delta_pop_
(trained on the original, non-HQ data) and AutoTsit5 = autotsit5_.
Reads crps_ideal_<prefix>.csv (built by crps_log_summary_tables.jl).

One figure per (score kind, metric) — rCRPS/CRPS x infectious/beta
in-sample/beta out-of-sample, 6 total — each with one panel per beta form.
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using CSV
using DataFrames
using Statistics
using Plots
using StatsPlots

const BETA_FORMS = ["beta_exp", "beta_rational", "beta_mixed"]
const FORM_LABEL = Dict("beta_exp" => "exponential", "beta_rational" => "rational", "beta_mixed" => "mixed")

# (csv prefix, solver, reltol label)
const ARMS = [("delta_pop_", "Tsit5", "1e-3"), ("autotsit5_", "AutoTsit5", "1e-3"),
              ("tsit5_rtol1e-5_", "Tsit5", "1e-5"), ("autotsit5_rtol1e-5_", "AutoTsit5", "1e-5"),
              ("tsit5_rtol1e-6_", "Tsit5", "1e-6"), ("autotsit5_rtol1e-6_", "AutoTsit5", "1e-6"),
              ("tsit5_rtol1e-10_", "Tsit5", "1e-10"), ("autotsit5_rtol1e-10_", "AutoTsit5", "1e-10")]
const SOLVER_COLOR = Dict("Tsit5" => :firebrick, "AutoTsit5" => :seagreen)
arm_label(solver, reltol) = "$(solver)\n$(reltol)"

save_dir = projectdir("_research", "state_distribution_plots", "by_reltol")
mkpath(save_dir)

arm_crps = Dict(prefix => CSV.read(projectdir("_research", "crps_tables", "crps_ideal_$(prefix).csv"), DataFrame)
                for (prefix, _, _) in ARMS)

function reltol_panel(metric::Symbol, beta_form::String, ylab; show_parity_line::Bool)
    parts = DataFrame[]
    for (i, (prefix, solver, reltol)) in enumerate(ARMS)
        sub = filter(:beta_function => ==(beta_form), arm_crps[prefix])
        sub = dropmissing(sub, metric)
        sub = sub[sub[!, metric] .> 0, :]   # log10 undefined at/below 0
        push!(parts, DataFrame(value = log10.(sub[!, metric]),
                               arm_idx = fill(i, nrow(sub)),
                               arm_color = fill(SOLVER_COLOR[solver], nrow(sub))))
    end
    combined = vcat(parts...)
    p = @df combined violin(:arm_idx, :value, group=:arm_idx,
                             color=:arm_color, alpha=0.4, legend=false, linewidth=0,
                             xticks=(1:length(ARMS), [arm_label(s, r) for (_, s, r) in ARMS]))
    @df combined boxplot!(p, :arm_idx, :value, fillalpha=0, linewidth=1.5, color=:black, outliers=false)
    @df combined dotplot!(p, :arm_idx, :value, color=:black, markersize=2.5, markeralpha=0.5)
    show_parity_line && hline!(p, [0.0]; linestyle=:dash, color=:red, linewidth=1, label="")
    xlabel!(p, "Solver and reltol")
    ylabel!(p, ylab)
    title!(p, FORM_LABEL[beta_form])
    plot!(p, left_margin=12Plots.mm, bottom_margin=14Plots.mm, titlefontsize=10, xtickfontsize=7)
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
        panels = [reltol_panel(metric, bf, "$(ylab_prefix), $(display)"; show_parity_line)
                  for bf in BETA_FORMS]
        p = plot(panels...; layout=(1, length(BETA_FORMS)), size=(750 * length(BETA_FORMS), 620),
                 plot_title="$(SCORE_KIND_TITLE[score_kind]) by solver and reltol, grouped by beta form ($(display))")
        out_path = joinpath(save_dir, "$(score_kind)_violin_by_form_$(slug).png")
        savefig(p, out_path)
        println("Saved $(score_kind) by-form violin ($(slug)) to: $(out_path)")
    end
end
