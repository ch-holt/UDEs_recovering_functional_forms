#========================================================
SCRIPT TO PLOT THE SR SETTINGS COMPARISON
(tables made by scripts/sr_results_processing.jl)
    1) rCRPS violins: one violin per SR setting (each point = one location),
       plus the UDE ensemble for reference. Three panels: infectious,
       beta in-sample, beta out-of-sample. rCRPS = CRPS / baseline CRPS,
       so < 1 beats the constant-beta baseline (red dashed line at 1).
    2) Complexity histograms: complexity of the equation SR selected, one
       panel per SR setting (all locations x seeds), each bar stacked by the
       multistart level at which that seed's UDE gets removed. Shows whether
       multistart removes the more complex equations.
=========================================================#

using DrWatson

@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using DataFrames
using CSV
using Statistics
using Plots
using StatsPlots

# Call module
using UDE_FUNCTIONAL_FORMS

const beta_function = beta_exp
const noise = 0.0
const train_length = 365
const solver_name = "autotsit5"
# Multistart level the SR ensemble was built with (must match the saved crps_summary file) - default 0.0
# e.g. julia --project=. _research/sr_comparison_plots.jl 0.5
const MS_limit = parse(Float64, get(ARGS, 1, "0.0"))

sim_name = "$(solver_name)_UDE_single_beta=$(beta_function)_adam=2500_lbfgs=2000_traindata=$(train_length)_noise=$(noise)"
save_dir = datadir("exp_pro", "sr_comparison", sim_name)
plot_dir = plotsdir("sims", "test", "ude_single", "sr_comparison_$(beta_function)")
mkpath(plot_dir)

crps_summary = JLD2.load(joinpath(save_dir, "crps_summary_ms=$(MS_limit).jld2"), "crps_summary")

#==============================================================
# Setting names, order and colours
==============================================================#

# "SR_input=unit_parsimony=3_maxsize=30_coc=2" -> "unit p3 m30 c2"
function short_name(setting)
    m = match(r"SR_input=(.+)_parsimony=(\d+)_maxsize=(\d+)_coc=(\d+)", setting)
    return "$(m[1]) p$(m[2]) m$(m[3]) c$(m[4])"
end

# Parse the setting values so the settings can be put in a fixed order
function setting_values(setting)
    m = match(r"SR_input=(.+)_parsimony=(\d+)_maxsize=(\d+)_coc=(\d+)", setting)
    return (; input = m[1], parsimony = parse(Int, m[2]), maxsize = parse(Int, m[3]), coc = parse(Int, m[4]))
end

# The experiments in scripts/ude_symbolic_regression.jl:
#   input grid (p3 m30 c2) = unit / train / train_pm10 / ons10 at parsimony=3, maxsize=30, coc=2
#   input grid (p2 m10 c3) = train / train_pm10 / ons10 at parsimony=2, maxsize=10, coc=3
#                            (unit at p2 m10 c3 is in the SR options experiment)
#   SR options             = parsimony x maxsize x coc, fitted on the unit grid
const EXPERIMENTS = ["Input grid (p3 m30 c2)", "Input grid (p2 m10 c3)", "SR options"]
function experiment(setting)
    v = setting_values(setting)
    v.maxsize == 30 && return "Input grid (p3 m30 c2)"
    v.input != "unit" && return "Input grid (p2 m10 c3)"
    return "SR options"
end

const INPUT_ORDER = Dict("unit" => 1, "train" => 2, "train_pm10" => 3, "ons10" => 4)

# Same order in every panel (so a setting is always in the same place):
# experiments in EXPERIMENTS order, then by input grid, parsimony, maxsize, coc
settings = sort(unique(crps_summary.setting); by = s -> begin
    v = setting_values(s)
    (findfirst(==(experiment(s)), EXPERIMENTS), INPUT_ORDER[v.input], v.parsimony, v.maxsize, v.coc)
end)

# One colour per experiment, UDE in neutral grey
const GROUP_COLOR = Dict("Input grid (p3 m30 c2)" => "#2a78d6", "Input grid (p2 m10 c3)" => "#1f9e89",
                         "SR options" => "#eb6834", "UDE" => "#8c8c8c")

#==============================================================
# UDE reference (same baseline file the SR rCRPS was divided by)
# Note: the UDE ensemble here used the old ms_limit_for seed set, not MS_limit above
==============================================================#

baseline = CSV.read(projectdir("_research", "crps_tables", "crps_ideal_autotsit5_.csv"), DataFrame)
baseline = filter(:beta_function => ==(string(beta_function)), baseline)

# metric => (SR column, UDE column, panel title)
const METRICS = [
    (sr = :rcrps_inf,      ude = :rcrps_inf,            title = "Infectious (log scale)"),
    (sr = :rcrps_beta_in,  ude = :rcrps_beta_insample,  title = "β in-sample (I/N reached in training)"),
    (sr = :rcrps_beta_out, ude = :rcrps_beta_outsample, title = "β out-of-sample (extrapolation)"),
]

#==============================================================
# Violin panel
==============================================================#

# rCRPS can be enormous when an SR equation explodes outside the training range
# (e.g. 1e39), which would squash every other violin flat. Values above Y_CAP are
# drawn as triangles at the cap instead, and counted in the panel title.
const Y_CAP = 3.0   # log10(rCRPS), i.e. 1000x worse than the baseline

function violin_panel(metric; show_legend = false)
    labels = vcat(short_name.(settings), "UDE")
    groups = vcat(experiment.(settings), "UDE")
    values = vcat([crps_summary[crps_summary.setting .== s, metric.sr] for s in settings],
                  [baseline[!, metric.ude]])

    p = plot(; legend = false, xticks = (1:length(labels), labels), xrotation = 45,
             ylabel = "log10(rCRPS)", title = metric.title, titlefontsize = 11,
             left_margin = 10Plots.mm, bottom_margin = 14Plots.mm, grid = :y, gridalpha = 0.3)

    n_capped = 0
    y_max = -Inf
    for (x, (vals, group)) in enumerate(zip(values, groups))
        log_vals = log10.(filter(v -> isfinite(v) && v > 0, vals))
        isempty(log_vals) && continue
        shown  = filter(<=(Y_CAP), log_vals)
        capped = count(>(Y_CAP), log_vals)
        n_capped += capped
        isempty(shown) || (y_max = max(y_max, maximum(shown)))

        if length(shown) > 1
            violin!(p, fill(x, length(shown)), shown; color = GROUP_COLOR[group], alpha = 0.35, linewidth = 0, label = "")
            boxplot!(p, fill(x, length(shown)), shown; fillalpha = 0, linewidth = 1.2, color = :black, outliers = false, bar_width = 0.3, label = "")
        end
        dotplot!(p, fill(x, length(shown)), shown; color = GROUP_COLOR[group], markersize = 2.5, markerstrokewidth = 0, alpha = 0.7, label = "")
        capped > 0 && scatter!(p, [x], [Y_CAP]; marker = :utriangle, markersize = 6, color = :black, label = "")
    end

    # rCRPS = 1: same as the constant-beta baseline
    hline!(p, [0.0]; linestyle = :dash, color = :red, linewidth = 1.2, label = "")
    n_capped > 0 && title!(p, "$(metric.title)  [▲ $(n_capped) location(s) with rCRPS > 10^$(Int(Y_CAP))]")

    # Legend for the experiment colours (empty series, so nothing is drawn), in one row
    # across the top, with headroom added so it sits above the data
    if show_legend
        ylims!(p, (Plots.ylims(p)[1], max(y_max, 0.0) + 0.8))
        for group in vcat(EXPERIMENTS, "UDE")
            scatter!(p, [NaN], [NaN]; marker = :square, markersize = 6, markerstrokewidth = 0,
                     color = GROUP_COLOR[group], label = group)
        end
        plot!(p; legend = :top, legend_columns = length(EXPERIMENTS) + 1, foreground_color_legend = nothing)
    end
    return p
end

panels = [violin_panel(m; show_legend = (i == 1)) for (i, m) in enumerate(METRICS)]
fig = plot(panels...; layout = (3, 1), size = (2000, 1800),
           plot_title = "SR settings vs UDE — rCRPS against the constant-β baseline ($(beta_function), MS=$(MS_limit))",
           plot_titlefontsize = 13)

out_path = joinpath(plot_dir, "sr_rcrps_violin_ms=$(MS_limit).png")
savefig(fig, out_path)
println("Saved: $(out_path)")

#==============================================================
# Summary table: median rCRPS per setting (what the violins show)
==============================================================#

summary = combine(groupby(crps_summary, :setting),
                  :rcrps_inf      => median => :median_rcrps_inf,
                  :rcrps_beta_in  => median => :median_rcrps_beta_in,
                  :rcrps_beta_out => median => :median_rcrps_beta_out,
                  [:rcrps_inf, :rcrps_beta_in, :rcrps_beta_out] => ((a, b, c) -> count(!isfinite, vcat(a, b, c))) => :n_nonfinite)
summary.setting = short_name.(summary.setting)
sort!(summary, :median_rcrps_beta_in)
show(summary, allrows = true, allcols = true)
println()

#==============================================================
# 2) Complexity of the selected equation, stacked by multistart removal
==============================================================#

fronts     = JLD2.load(joinpath(save_dir, "sr_pareto_fronts.jld2"), "fronts")
seed_table = JLD2.load(joinpath(save_dir, "ude_seed_table.jld2"), "seed_table")

# One row per SR run (setting, location, seed): the equation SR selected,
# with the multistart level its UDE seed survives up to
selected = filter(:is_selected => identity, fronts)
selected = leftjoin(selected, select(seed_table, :location, :seed, :max_ms_kept), on = [:location, :seed])
@assert nrow(selected) == length(settings) * 51 * 100 "Expected one selected equation per setting x location x seed"
@assert !any(ismissing, selected.max_ms_kept) "Some SR runs have no matching UDE seed"

# max_ms_kept = largest MS limit the seed survives, so it is removed at the next one up
# (0.0 = removed at MS=0.1, the worst 10%; 0.9 = survives every limit, the best 10%)
const MS_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ms_label(ms) = ms == 0.9 ? "survives all (best 10%)" : "removed at MS=$(round(ms + 0.1, digits = 1))"

# Ordered light -> dark blue (one hue): darker = better UDE fit (survives more multistart)
const MS_COLORS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                   "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# Ground-truth form a / exp(x1 / b) costs coc per constant + 1 each for /, exp, /, x1
# (checked against the fronts: GT-form equations sit at 6, 8, 10 for coc = 1, 2, 3)
gt_complexity(setting) = 4 + 2 * setting_values(setting).coc

# Shared x range so every panel is on the same axis
const COMPLEXITY_RANGE = minimum(selected.complexity):maximum(selected.complexity)

function complexity_panel(setting; show_legend = false)
    sub = filter(:setting => ==(setting), selected)
    counts = combine(groupby(sub, [:complexity, :max_ms_kept]), nrow => :n)

    # Matrix of counts: rows = complexity, columns = MS level (in MS_LEVELS order, so colours
    # mean the same thing in every panel), zero where no seed selected that complexity
    Y = zeros(Int, length(COMPLEXITY_RANGE), length(MS_LEVELS))
    for r in eachrow(counts)
        Y[r.complexity - first(COMPLEXITY_RANGE) + 1, findfirst(==(r.max_ms_kept), MS_LEVELS)] = r.n
    end

    p = groupedbar(collect(COMPLEXITY_RANGE), Y; bar_position = :stack, bar_width = 0.8,
                   color = permutedims(MS_COLORS), linewidth = 0,
                   label = show_legend ? permutedims(ms_label.(MS_LEVELS)) : "",
                   legend = show_legend ? :outerright : false,
                   title = short_name(setting), titlefontsize = 10, titlefontcolor = GROUP_COLOR[experiment(setting)],
                                      xlims = (first(COMPLEXITY_RANGE) - 0.5, last(COMPLEXITY_RANGE) + 0.5), grid = :y, gridalpha = 0.3)
    vline!(p, [gt_complexity(setting)]; linestyle = :dash, color = :red, linewidth = 1.2, label = show_legend ? "ground-truth complexity" : "")
    return p
end

hist_panels = [complexity_panel(s) for s in settings]
# Legend as its own panel (the slot after the last histogram), so it doesn't cover any histogram:
# empty series (NaN), so only the legend entries are drawn
legend_panel = plot(; framestyle = :none, legend = :left, legendfontsize = 9, xlims = (0, 1), ylims = (0, 1))
for (ms, color) in zip(reverse(MS_LEVELS), reverse(MS_COLORS))
    scatter!(legend_panel, [NaN], [NaN]; marker = :square, markersize = 7, markerstrokewidth = 0, color = color, label = ms_label(ms))
end
plot!(legend_panel, [NaN], [NaN]; linestyle = :dash, color = :red, linewidth = 1.2, label = "ground-truth complexity")

# 3 panels per row, enough rows for every setting plus the legend panel
n_rows = cld(length(hist_panels) + 1, 3)
hist_fig = plot(hist_panels..., legend_panel; layout = (n_rows, 3), size = (1500, 300 * n_rows),
                plot_title = "Complexity of the SR-selected equation (x) vs number of SR runs (y), coloured by multistart removal\n($(beta_function), 51 locations × 100 seeds per panel)",
                plot_titlefontsize = 12, left_margin = 6Plots.mm, bottom_margin = 4Plots.mm)
hist_path = joinpath(plot_dir, "sr_complexity_by_ms.png")
savefig(hist_fig, hist_path)
println("Saved: $(hist_path)")

# Numbers behind the question "does multistart remove the more complex equations?":
# median complexity of the selected equation among seeds removed at MS=0.1 vs the best 10%
ms_summary = combine(groupby(selected, :setting),
    [:complexity, :max_ms_kept] => ((c, m) -> median(c[m .== 0.0])) => :median_complexity_worst10,
    [:complexity, :max_ms_kept] => ((c, m) -> median(c[m .== 0.9])) => :median_complexity_best10,
    :setting => (s -> gt_complexity(first(s))) => :gt_complexity)
ms_summary.setting = short_name.(ms_summary.setting)
show(ms_summary, allrows = true, allcols = true)
println()
