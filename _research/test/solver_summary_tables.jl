#========================================================
SOLVER COMPARISON SUMMARY TABLE — same design as crps_summary_tables.jl
(one row per solver x beta form, raw-CRPS n/mean/Q1/median/Q3/Tukey
whiskers, plus a count of how many states the baseline beats the UDE on
via the rcrps_* columns), kept separate so the split-strategy summary
table is never at risk of being overwritten.

Mean is arithmetic (not geometric) for the same reason as
crps_summary_tables.jl: raw CRPS isn't a ratio, so there's no near-zero-
denominator instability to guard against, and arithmetic mean is the
standard aggregate for a proper scoring rule across forecast instances.
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using CSV
using DataFrames
using Statistics

const SOLVER_PREFIXES = ["rosenbrock23_", "vern7_", "autotsit5_", "delta_pop_"]
# "delta_pop_" stands in as the Tsit5 arm -- see solver_violin_plots.jl header.
const SOLVER_DISPLAY_NAME = Dict("rosenbrock23_" => "Rosenbrock23", "vern7_" => "Vern7",
                                  "autotsit5_" => "AutoTsit5", "delta_pop_" => "Tsit5")
const BETA_FORMS = ["beta_exp", "beta_rational", "beta_mixed"]

function tukey_whiskers(x::Vector{Float64}, q1, q3)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    within = filter(v -> lower_fence <= v <= upper_fence, x)
    return minimum(within), maximum(within)
end

function metric_stats(x::Vector{Float64})
    n = length(x)
    m = mean(x)
    q1, med, q3 = quantile(x, (0.25, 0.5, 0.75))
    whisker_low, whisker_high = tukey_whiskers(x, q1, q3)
    return (; n, mean=m, q1, median=med, q3, whisker_low, whisker_high)
end

baseline_wins(x) = count(>(1), skipmissing(x))

rows = NamedTuple[]
for prefix in SOLVER_PREFIXES
    df = CSV.read(projectdir("_research", "crps_tables", "crps_ideal_$(prefix).csv"), DataFrame)
    for bf in BETA_FORMS
        sub = filter(:beta_function => ==(bf), df)

        inf = metric_stats(collect(skipmissing(sub.log_crps_ude_inf)))
        ins = metric_stats(collect(skipmissing(sub.crps_nn_beta_insample)))
        out = metric_stats(collect(skipmissing(sub.crps_nn_beta_outsample)))

        push!(rows, (;
            solver = SOLVER_DISPLAY_NAME[prefix], beta_form = bf,
            infectious_n = inf.n, infectious_mean = inf.mean,
            infectious_q1 = inf.q1, infectious_median = inf.median, infectious_q3 = inf.q3,
            infectious_whisker_low = inf.whisker_low, infectious_whisker_high = inf.whisker_high,
            infectious_baseline_wins = baseline_wins(sub.rcrps_inf),
            beta_insample_n = ins.n, beta_insample_mean = ins.mean,
            beta_insample_q1 = ins.q1, beta_insample_median = ins.median, beta_insample_q3 = ins.q3,
            beta_insample_whisker_low = ins.whisker_low, beta_insample_whisker_high = ins.whisker_high,
            beta_insample_baseline_wins = baseline_wins(sub.rcrps_beta_insample),
            beta_outsample_n = out.n, beta_outsample_mean = out.mean,
            beta_outsample_q1 = out.q1, beta_outsample_median = out.median, beta_outsample_q3 = out.q3,
            beta_outsample_whisker_low = out.whisker_low, beta_outsample_whisker_high = out.whisker_high,
            beta_outsample_baseline_wins = baseline_wins(sub.rcrps_beta_outsample),
        ))
    end
end

result = DataFrame(rows)
out_path = projectdir("_research", "crps_tables", "solver_summary_raw.csv")
CSV.write(out_path, result)
println("Saved to: $(out_path)")
show(result, allrows=true, allcols=true)
