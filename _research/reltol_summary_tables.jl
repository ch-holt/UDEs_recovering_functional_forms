#========================================================
SUMMARY TABLES for the reltol sensitivity sweep (8 arms: Tsit5 and
AutoTsit5 at reltol 1e-3 [default], 1e-5, 1e-6, 1e-10). Analogues of the
solver comparison Tables 6-11:

  6. raw CRPS summary (arithmetic mean / median)
  7. raw CRPS win counts (lowest wins; Median / Mean / All)
  8. rCRPS summary (geometric mean / median)
  9. rCRPS win counts (Median / Geo. mean / All)
 10. baseline wins (states out of 51 with rCRPS > 1)
 11. computation time (reltol_timing_overall.csv)

Raw CRPS columns: log_crps_ude_inf, crps_nn_beta_insample,
crps_nn_beta_outsample; rCRPS columns: rcrps_inf, rcrps_beta_insample,
rcrps_beta_outsample. Lowest value wins, winners in ***bold-italic***, 3 s.f.
Default arms: Tsit5 = delta_pop_, AutoTsit5 = autotsit5_.
=========================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using CSV
using DataFrames
using Statistics
using Printf

const BETA_FORMS = ["beta_exp", "beta_rational", "beta_mixed"]
const FORM_LABEL = Dict("beta_exp" => "exponential", "beta_rational" => "rational", "beta_mixed" => "mixed")
const METRIC_SLUGS = [("infectious", "Infectious"), ("beta_insample", "Beta in-sample"), ("beta_outsample", "Beta out-of-sample")]
const RAW_COLS  = Dict("infectious" => :log_crps_ude_inf, "beta_insample" => :crps_nn_beta_insample, "beta_outsample" => :crps_nn_beta_outsample)
const RCRPS_COLS = Dict("infectious" => :rcrps_inf, "beta_insample" => :rcrps_beta_insample, "beta_outsample" => :rcrps_beta_outsample)

# (csv prefix, display label)
const ARMS = [("delta_pop_", "Tsit5 (1e-3)"), ("autotsit5_", "AutoTsit5 (1e-3)"),
              ("tsit5_rtol1e-5_", "Tsit5 (1e-5)"), ("autotsit5_rtol1e-5_", "AutoTsit5 (1e-5)"),
              ("tsit5_rtol1e-6_", "Tsit5 (1e-6)"), ("autotsit5_rtol1e-6_", "AutoTsit5 (1e-6)"),
              ("tsit5_rtol1e-10_", "Tsit5 (1e-10)"), ("autotsit5_rtol1e-10_", "AutoTsit5 (1e-10)")]
const LABELS = [l for (_, l) in ARMS]

gmean(x) = exp(mean(log.(x)))

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

commas(x) = replace(@sprintf("%.0f", x), r"(?<=\d)(?=(\d{3})+$)" => ",")

data = Dict(lab => CSV.read(projectdir("_research", "crps_tables", "crps_ideal_$(prefix).csv"), DataFrame)
            for (prefix, lab) in ARMS)

vals(lab, bf, col) = Float64[v for v in skipmissing(filter(:beta_function => ==(bf), data[lab])[!, col]) if isfinite(v)]

# Generic summary table + win tally for one score kind. `centre` is the
# "mean" statistic (arithmetic for raw CRPS, geometric for rCRPS).
function summary_and_wins(title, cols, centre, centre_name, csv_name)
    stats = Dict{Tuple{String,String,String}, NamedTuple}()
    for lab in LABELS, bf in BETA_FORMS, (slug, _) in METRIC_SLUGS
        x = vals(lab, bf, cols[slug])
        x = centre === gmean ? filter(>(0), x) : x
        stats[(lab, bf, slug)] = (; centre=centre(x), median=median(x))
    end
    win_c = Dict{Tuple{String,String}, String}()
    win_m = Dict{Tuple{String,String}, String}()
    for bf in BETA_FORMS, (slug, _) in METRIC_SLUGS
        win_c[(bf, slug)] = LABELS[argmin([stats[(l, bf, slug)].centre for l in LABELS])]
        win_m[(bf, slug)] = LABELS[argmin([stats[(l, bf, slug)].median for l in LABELS])]
    end

    println("\n", "="^70, "\n  ", title, "\n", "="^70)
    header = ["Solver (reltol)", "Beta form"]
    for (_, disp) in METRIC_SLUGS
        push!(header, "$(disp) $(centre_name)", "$(disp) median")
    end
    rows = Vector{Vector{String}}()
    csv_rows = NamedTuple[]
    for bf in BETA_FORMS, lab in LABELS
        r = [lab, FORM_LABEL[bf]]
        fields = Pair{Symbol,Any}[:arm => lab, :beta_form => bf]
        for (slug, _) in METRIC_SLUGS
            s = stats[(lab, bf, slug)]
            c = fmt3(s.centre); win_c[(bf, slug)] == lab && (c = "***$(c)***")
            m = fmt3(s.median); win_m[(bf, slug)] == lab && (m = "***$(m)***")
            push!(r, c, m)
            push!(fields, Symbol("$(slug)_$(centre_name)") => s.centre, Symbol("$(slug)_median") => s.median)
        end
        push!(rows, r)
        push!(csv_rows, NamedTuple(fields))
    end
    println(md_table(header, rows))
    CSV.write(projectdir("_research", "crps_tables", csv_name), DataFrame(csv_rows))

    println("Win counts (lowest wins; 9 comparisons each for median and $(centre_name))\n")
    trows = Vector{Vector{String}}()
    for lab in LABELS
        nm = count(k -> win_m[k] == lab, keys(win_m))
        nc = count(k -> win_c[k] == lab, keys(win_c))
        push!(trows, [lab, string(nm), string(nc), string(nm + nc)])
    end
    println(md_table(["Solver (reltol)", "Median wins", "$(uppercasefirst(centre_name)) wins", "All wins"], trows))
end

summary_and_wins("Raw CRPS summary (arithmetic mean; lowest per column in bold)", RAW_COLS, mean, "mean", "reltol_summary_raw.csv")
summary_and_wins("rCRPS summary (geometric mean; lowest per column in bold)", RCRPS_COLS, gmean, "geo. mean", "reltol_summary_rcrps.csv")

# Table 10: baseline wins (rCRPS > 1), totals across beta forms (3 x 51 states per metric)
println("\n", "="^70, "\n  Baseline wins (state x beta-form cases with rCRPS > 1)\n", "="^70)
bt_rows = Vector{Vector{String}}()
bw_csv = NamedTuple[]
for lab in LABELS
    per_metric = [sum(count(>(1), vals(lab, bf, RCRPS_COLS[slug])) for bf in BETA_FORMS) for (slug, _) in METRIC_SLUGS]
    push!(bt_rows, [lab, string.(per_metric)..., string(sum(per_metric))])
    push!(bw_csv, (; arm=lab, infectious=per_metric[1], beta_insample=per_metric[2], beta_outsample=per_metric[3], baseline_wins=sum(per_metric)))
end
println(md_table(["Solver (reltol)", "Infectious", "Beta in-sample", "Beta out-of-sample", "Baseline wins"], bt_rows))
CSV.write(projectdir("_research", "crps_tables", "reltol_baseline_wins.csv"), DataFrame(bw_csv))

# Table 11: computation time
timing_path = projectdir("_research", "crps_tables", "reltol_timing_overall.csv")
if isfile(timing_path)
    println("\n", "="^70, "\n  Computation time (per seed; train_length=365)\n", "="^70)
    tdf = CSV.read(timing_path, DataFrame; types=Dict(:reltol => String))   # else "1e-3" parses as Float64
    trows = Vector{Vector{String}}()
    for lab in LABELS
        m = match(r"^(\w+) \((.+)\)$", lab)
        r = tdf[(tdf.solver .== m[1]) .& (string.(tdf.reltol) .== m[2]), :]
        isempty(r) && continue
        r = r[1, :]
        push!(trows, [lab, commas(r.n), commas(r.mean_seconds), commas(r.median_seconds), commas(r.total_compute_hours)])
    end
    println(md_table(["Solver (reltol)", "n seeds", "Mean (s)", "Median (s)", "Total compute-hours (3 beta forms, 51 locations, 100 seeds)"], trows))
else
    println("\n(no reltol_timing_overall.csv yet — run reltol_timing_summary.jl for Table 11)")
end
