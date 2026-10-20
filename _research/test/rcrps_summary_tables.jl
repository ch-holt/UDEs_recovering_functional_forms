#========================================================
rCRPS SUMMARY TABLES for the split-strategy and solver comparisons.

rCRPS = CRPS_ude / CRPS_baseline per state, so < 1 means the UDE beats the
constant-beta baseline and lower is better. Aggregated per (group, beta
form, metric) over the 51 states with the GEOMETRIC mean (not arithmetic):
a ratio blows up when the baseline CRPS is near zero, and the geometric
mean is the standard way to average CRPS ratios (same convention as
crps_log_summary_tables.jl's summarise_rcrps). Median/Q1/Q3 are
scale-invariant. Tukey whiskers (1.5*IQR) are computed on the log scale to
match the log10 axis the violin plots are drawn on, then mapped back.

Also computes, from the same rcrps_* columns:
  - baseline wins: number of states with rCRPS > 1
  - head-to-head win tallies: for each (beta form, metric), which group has
    the lowest geometric mean / lowest median, counted across the 9
    (3 beta forms x 3 metrics) comparisons

Reads crps_ideal_<prefix>.csv (train_length=365, noise=0.0, all 51 states
complete). Writes rcrps_summary_splits.csv / rcrps_summary_solvers.csv and
prints markdown tables (3 s.f.) ready to paste.
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using CSV
using DataFrames
using Statistics
using Printf

const BETA_FORMS = ["beta_exp", "beta_rational", "beta_mixed"]
const FORM_LABEL = Dict("beta_exp" => "exponential", "beta_rational" => "rational", "beta_mixed" => "mixed")
const METRICS = [(:rcrps_inf, "infectious", "Infectious"),
                 (:rcrps_beta_insample, "beta_insample", "Beta in-sample"),
                 (:rcrps_beta_outsample, "beta_outsample", "Beta out-of-sample")]

# (comparison name, [(csv prefix, display label), ...]). "delta_pop_" is the
# Tsit5 arm in the solver comparison (see solver_violin_plots.jl).
const COMPARISONS = [
    ("splits",  [("delta_pop_", "Random"), ("train_val_temporal_", "Temporal"), ("train_val_midbeta_", "Mid-beta")]),
    ("solvers", [("rosenbrock23_", "Rosenbrock23"), ("vern7_", "Vern7"), ("autotsit5_", "AutoTsit5"), ("delta_pop_", "Tsit5")]),
]

gmean(x) = exp(mean(log.(x)))

function log_whiskers(x::Vector{Float64})
    l = log.(x)
    q1, q3 = quantile(l, (0.25, 0.75))
    iqr = q3 - q1
    within = filter(v -> q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr, l)
    return exp(minimum(within)), exp(maximum(within))
end

function metric_stats(raw)
    x = Float64[v for v in skipmissing(raw) if isfinite(v) && v > 0]
    q1, med, q3 = quantile(x, (0.25, 0.5, 0.75))
    wl, wh = log_whiskers(x)
    return (; n=length(x), gmean=gmean(x), q1, median=med, q3, whisker_low=wl, whisker_high=wh,
              baseline_wins=count(>(1), skipmissing(raw)))
end

# 3 significant figures without scientific notation or dropped trailing zeros
function fmt3(x)
    x == 0 && return "0"
    d = max(0, 2 - floor(Int, log10(abs(x))))
    return Printf.format(Printf.Format("%." * string(d) * "f"), x)
end

function md_table(header, rows)
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "| ", join(fill("---", length(header)), " | "), " |")
    for r in rows
        println(io, "| ", join(r, " | "), " |")
    end
    return String(take!(io))
end

for (comp_name, group) in COMPARISONS
    labels = [lab for (_, lab) in group]
    stats = Dict{Tuple{String,String,String}, Any}()   # (label, beta_form, slug) -> stats
    for (prefix, lab) in group
        df = CSV.read(projectdir("_research", "crps_tables", "crps_ideal_$(prefix).csv"), DataFrame)
        for bf in BETA_FORMS
            sub = filter(:beta_function => ==(bf), df)
            for (col, slug, _) in METRICS
                stats[(lab, bf, slug)] = metric_stats(sub[!, col])
            end
        end
    end

    # CSV: one row per (group, beta form), all stats for all three metrics
    csv_rows = NamedTuple[]
    for bf in BETA_FORMS, lab in labels
        fields = Pair{Symbol,Any}[:group => lab, :beta_form => bf]
        for (_, slug, _) in METRICS
            s = stats[(lab, bf, slug)]
            for k in (:n, :gmean, :q1, :median, :q3, :whisker_low, :whisker_high, :baseline_wins)
                push!(fields, Symbol("$(slug)_$(k)") => getproperty(s, k))
            end
        end
        push!(csv_rows, NamedTuple(fields))
    end
    CSV.write(projectdir("_research", "crps_tables", "rcrps_summary_$(comp_name).csv"), DataFrame(csv_rows))

    # winners = lowest rCRPS within each (beta form, metric) for each statistic
    win_gmean  = Dict{Tuple{String,String}, String}()
    win_median = Dict{Tuple{String,String}, String}()
    for bf in BETA_FORMS, (_, slug, _) in METRICS
        win_gmean[(bf, slug)]  = labels[argmin([stats[(l, bf, slug)].gmean  for l in labels])]
        win_median[(bf, slug)] = labels[argmin([stats[(l, bf, slug)].median for l in labels])]
    end

    println("\n", "="^70, "\n  rCRPS summary — ", comp_name, " (geometric mean; lowest per column in bold)\n", "="^70)

    header = ["Group", "Beta form"]
    for (_, _, disp) in METRICS
        push!(header, "$(disp) geo. mean", "$(disp) median")
    end
    rows = Vector{Vector{String}}()
    for bf in BETA_FORMS, lab in labels
        r = [lab, FORM_LABEL[bf]]
        for (_, slug, _) in METRICS
            s = stats[(lab, bf, slug)]
            g = fmt3(s.gmean);  win_gmean[(bf, slug)]  == lab && (g = "***$(g)***")
            m = fmt3(s.median); win_median[(bf, slug)] == lab && (m = "***$(m)***")
            push!(r, g, m)
        end
        push!(rows, r)
    end
    println(md_table(header, rows))

    println("Win tally (lowest rCRPS wins; 9 comparisons each for median and geo. mean)\n")
    tally_rows = Vector{Vector{String}}()
    for lab in labels
        nmed  = count(k -> win_median[k] == lab, keys(win_median))
        nmean = count(k -> win_gmean[k]  == lab, keys(win_gmean))
        push!(tally_rows, [lab, string(nmed), string(nmean), string(nmed + nmean)])
    end
    println(md_table(["Group", "Median", "Geo. mean", "All"], tally_rows))

    println("Baseline wins (states out of 51 with rCRPS > 1), by beta form\n")
    bw_rows = Vector{Vector{String}}()
    for bf in BETA_FORMS, lab in labels
        push!(bw_rows, [lab, FORM_LABEL[bf], [string(stats[(lab, bf, slug)].baseline_wins) for (_, slug, _) in METRICS]...])
    end
    println(md_table(["Group", "Beta form", "Infectious baseline wins", "Beta in-sample baseline wins", "Beta out-of-sample baseline wins"], bw_rows))

    println("Baseline wins, totals across beta forms\n")
    bt_rows = Vector{Vector{String}}()
    for lab in labels
        per_metric = [sum(stats[(lab, bf, slug)].baseline_wins for bf in BETA_FORMS) for (_, slug, _) in METRICS]
        push!(bt_rows, [lab, string.(per_metric)..., string(sum(per_metric))])
    end
    println(md_table(["Group", "Infectious baseline wins", "Beta in-sample baseline wins", "Beta out-of-sample baseline wins", "Baseline wins"], bt_rows))
end

println("\nSaved: rcrps_summary_splits.csv, rcrps_summary_solvers.csv in ", projectdir("_research", "crps_tables"))
