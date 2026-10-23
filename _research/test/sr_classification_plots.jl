#========================================================
SCRIPT TO CLASSIFY THE SR EQUATIONS AND PLOT WHAT SR RECOVERED
(uses the Pareto fronts + UDE seed table from scripts/sr_results_processing.jl
 and the classifier in _research/sr_equation_classifier.jl)

    0) Classify every equation on every Pareto front (exact ground-truth
       form / ground truth + correction / perturbed exponent / other, plus
       whether its a and b are within 20% of the truth). Saved, so this
       only runs once.
    1) Search vs selection: for each SR run, did SR select the ground truth
       (a), find it but select something else (b, selection failure), or
       never find it (c, search failure)? Done twice:
         strict  = exact ground-truth form
         lenient = exact form or ground truth + a correction
       Both require the parameters to be within 20% (REQUIRE_PARAMS).
       Bars are stacked by the multistart level the UDE seed is removed at.
    2) Mechanism shares: what kind of equation SR selected, per setting.
    3) Best seed: the same for the single best UDE seed per location
       (lowest validation loss, i.e. MS ~ 0.99), compared with all seeds,
       plus a table of the best seed's selected equation for every
       setting x location.
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

include(joinpath(@__DIR__, "..", "sr_equation_classifier.jl"))
using .SREquationClassifier

const beta_function = beta_exp
const noise = 0.0
const train_length = 365
const solver_name = "autotsit5"

# Classifier settings
const PARAM_RTOL = 0.2           # a and b within 20% of the truth
const REQUIRE_PARAMS = true      # ground truth on the front must also have the right a and b
const REBUILD_CLASSIFIED = true # set true once after changing the classifier

sim_name = "$(solver_name)_UDE_single_beta=$(beta_function)_adam=2500_lbfgs=2000_traindata=$(train_length)_noise=$(noise)"
save_dir = datadir("exp_pro", "sr_comparison", sim_name)
plot_dir = plotsdir("sims", "test", "ude_single", "sr_comparison_$(beta_function)")
mkpath(plot_dir)

# Check the classifier before using it
run_classifier_tests()

#==============================================================
# Setting names, order and colours (same as sr_comparison_plots.jl)
==============================================================#

function setting_values(setting)
    m = match(r"SR_input=(.+)_parsimony=(\d+)_maxsize=(\d+)_coc=(\d+)", setting)
    return (; input = m[1], parsimony = parse(Int, m[2]), maxsize = parse(Int, m[3]), coc = parse(Int, m[4]))
end
short_name(setting) = (v = setting_values(setting); "$(v.input) p$(v.parsimony) m$(v.maxsize) c$(v.coc)")
const EXPERIMENTS = ["Input grid (p3 m30 c2)", "Input grid (p2 m10 c3)", "SR options"]
function experiment(setting)
    v = setting_values(setting)
    v.maxsize == 30 && return "Input grid (p3 m30 c2)"
    v.input != "unit" && return "Input grid (p2 m10 c3)"
    return "SR options"
end
const INPUT_ORDER = Dict("unit" => 1, "train" => 2, "train_pm10" => 3, "ons10" => 4)
setting_order(s) = (v = setting_values(s); (findfirst(==(experiment(s)), EXPERIMENTS), INPUT_ORDER[v.input], v.parsimony, v.maxsize, v.coc))

# Multistart colours (light = removed first, dark = survives all), as in the complexity histogram
const MS_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ms_label(ms) = ms == 0.9 ? "survives all (best 10%)" : "removed at MS=$(round(ms + 0.1, digits = 1))"
const MS_COLORS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                   "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

#==============================================================
# 0) Classify every equation on every front
==============================================================#

fronts     = JLD2.load(joinpath(save_dir, "sr_pareto_fronts.jld2"), "fronts")
seed_table = JLD2.load(joinpath(save_dir, "ude_seed_table.jld2"), "seed_table")
settings   = sort(unique(fronts.setting); by = setting_order)
locations  = sort(unique(fronts.location))

# True a, b per location: beta(x) = a * exp(-b x) with x = I/N, read off beta_function
# the same way the classifier reads them off an equation (a = beta(0), b = -log(beta(1)/a))
function true_params(location)
    a = beta_function(location, 0.0)
    b = -log(beta_function(location, Float64(POPULATION[location])) / a)
    return (; a, b)
end
truth = Dict(location => true_params(location) for location in locations)

classified_path = joinpath(save_dir, "sr_pareto_fronts_classified.jld2")
# Also rebuild if the saved file was made from different fronts (e.g. before SR settings were added)
saved = isfile(classified_path) && !REBUILD_CLASSIFIED ? JLD2.load(classified_path, "fronts") : nothing
if !isnothing(saved) && nrow(saved) == nrow(fronts)
    fronts = saved
else
    # Structure and a, b depend only on the equation, so classify each distinct string once
    # (truth only enters params_ok, which needs the location)
    eq_class = Dict{String, Any}()
    for eq in unique(fronts.equation)
        structure, gt_term = classify_structure(parse_equation(eq))
        a_hat, b_hat = isnothing(gt_term) ? (NaN, NaN) : exp_params(gt_term)
        eq_class[eq] = (; structure, a_hat, b_hat)
    end
    fronts.structure = [String(eq_class[eq].structure) for eq in fronts.equation]
    fronts.a_hat     = [eq_class[eq].a_hat for eq in fronts.equation]
    fronts.b_hat     = [eq_class[eq].b_hat for eq in fronts.equation]
    fronts.params_ok = [isapprox(r.a_hat, truth[r.location].a; rtol = PARAM_RTOL) &&
                        isapprox(r.b_hat, truth[r.location].b; rtol = PARAM_RTOL) for r in eachrow(fronts)]
    jldsave(classified_path; fronts)
end

println("\n=== Structure of every equation on every front ===")
println(combine(groupby(fronts, :structure), nrow => :n, :params_ok => mean => :frac_params_ok))

# A few examples per category, to check the classifier by eye
println("\n=== Examples (unit p3 m30 c2, AL) ===")
for structure in ["exact", "gt_plus_correction", "perturbed_exponent", "other"]
    ex = filter(r -> r.structure == structure && r.location == "AL" &&
                     r.setting == "SR_input=unit_parsimony=3_maxsize=30_coc=2", fronts)
    println("[$(structure)]")
    for r in eachrow(first(unique(ex, :equation), 3))
        println("    c$(r.complexity)  params_ok=$(r.params_ok)  $(format_equation_sigfigs(r.equation))")
    end
end

#==============================================================
# 1) Search vs selection (strict and lenient ground truth)
==============================================================#

const DEFINITIONS = [
    (name = "strict",  title = "Strict: exact a·exp(-b·x)",               categories = ("exact",)),
    (name = "lenient", title = "Lenient: exact or ground truth + correction", categories = ("exact", "gt_plus_correction")),
]
const OUTCOMES = [
    (code = "a", title = "(a) selected the ground truth"),
    (code = "b", title = "(b) on the front, not selected (selection failure)"),
    (code = "c", title = "(c) not on the front (search failure)"),
]

is_gt(structure, params_ok, categories) = structure in categories && (!REQUIRE_PARAMS || params_ok)

# One row per SR run: the outcome under each definition
runs = combine(groupby(fronts, [:setting, :location, :seed])) do g
    out = Dict{Symbol, String}()
    for d in DEFINITIONS
        gt = is_gt.(g.structure, g.params_ok, Ref(d.categories))
        out[Symbol("outcome_", d.name)] = any(gt .& g.is_selected) ? "a" : any(gt) ? "b" : "c"
    end
    (; out...)
end
runs = leftjoin(runs, select(seed_table, :location, :seed, :max_ms_kept, :is_best_seed), on = [:location, :seed])

function outcome_panel(definition, outcome)
    sub = filter(Symbol("outcome_", definition.name) => ==(outcome.code), runs)
    counts = combine(groupby(sub, [:setting, :max_ms_kept]), nrow => :n)
    # rows = settings (in order), columns = MS levels
    Y = zeros(Int, length(settings), length(MS_LEVELS))
    for r in eachrow(counts)
        Y[findfirst(==(r.setting), settings), findfirst(==(r.max_ms_kept), MS_LEVELS)] = r.n
    end
    return groupedbar(1:length(settings), Y; bar_position = :stack, bar_width = 0.8, linewidth = 0,
                      color = permutedims(MS_COLORS), label = "", legend = false,
                      xticks = (1:length(settings), short_name.(settings)), xlabel = "SR setting",
                      xrotation = 45, ylims = (0, 51 * 100), ylabel = "SR runs (of 5100)",
                      title = "$(definition.title)\n$(outcome.title)", titlefontsize = 9, grid = :y, gridalpha = 0.3)
end

# Legend panel for the MS colours (empty series, only the legend entries are drawn)
function ms_legend_panel()
    p = plot(; framestyle = :none, legend = :left, legendfontsize = 9, xlims = (0, 1), ylims = (0, 1))
    for (ms, color) in zip(reverse(MS_LEVELS), reverse(MS_COLORS))
        scatter!(p, [NaN], [NaN]; marker = :square, markersize = 7, markerstrokewidth = 0, color = color, label = ms_label(ms))
    end
    return p
end

outcome_panels = [outcome_panel(d, o) for o in OUTCOMES for d in DEFINITIONS]
search_fig = plot(outcome_panels..., ms_legend_panel();
                  layout = @layout([grid(3, 2) a{0.13w}]), size = (2000, 2100),
                  plot_title = "Was the ground truth on the Pareto front? ($(beta_function), each bar = 51 locations × 100 seeds, parameters within $(Int(100PARAM_RTOL))%)",
                  plot_titlefontsize = 12, left_margin = 8Plots.mm, bottom_margin = 12Plots.mm)
search_path = joinpath(plot_dir, "sr_search_vs_selection.png")
savefig(search_fig, search_path)
println("\nSaved: $(search_path)")

# Counts per setting, and the gap between the two definitions:
# runs where only the lenient definition finds the ground truth (strict (c), lenient (a) or (b))
outcome_summary = combine(groupby(runs, :setting),
    :outcome_strict  => (o -> count(==("a"), o)) => :strict_a,
    :outcome_strict  => (o -> count(==("b"), o)) => :strict_b,
    :outcome_strict  => (o -> count(==("c"), o)) => :strict_c,
    :outcome_lenient => (o -> count(==("a"), o)) => :lenient_a,
    :outcome_lenient => (o -> count(==("b"), o)) => :lenient_b,
    :outcome_lenient => (o -> count(==("c"), o)) => :lenient_c,
    [:outcome_strict, :outcome_lenient] => ((s, l) -> count((s .== "c") .& (l .!= "c"))) => :only_with_correction)
sort!(outcome_summary, :setting; by = setting_order)
outcome_summary.setting = short_name.(outcome_summary.setting)
println("\n=== Search vs selection: SR runs per outcome (of 5100 per setting) ===")
show(outcome_summary, allrows = true, allcols = true)
println()
CSV.write(joinpath(save_dir, "sr_search_vs_selection.csv"), outcome_summary)

#==============================================================
# 2) Mechanism shares of the selected equation
==============================================================#

# Category of the selected equation, with exact / +correction split by whether a, b are right
function mechanism(structure, params_ok)
    structure == "exact"              && return params_ok ? "exact, a & b within 20%" : "exact form, a or b off"
    structure == "gt_plus_correction" && return params_ok ? "GT + correction, a & b within 20%" : "GT + correction, a or b off"
    structure == "perturbed_exponent" && return "perturbed exponent"
    return "other"
end
const MECHANISMS = ["exact, a & b within 20%", "exact form, a or b off",
                    "GT + correction, a & b within 20%", "GT + correction, a or b off",
                    "perturbed exponent", "other"]
# Blue = exact, aqua = + correction (dark = right parameters, light = wrong), orange, grey
const MECHANISM_COLORS = ["#2a78d6", "#9ec5f4", "#1baf7a", "#a3e3c8", "#eb6834", "#c8c8c8"]

selected = filter(:is_selected => identity, fronts)
selected.mechanism = mechanism.(selected.structure, selected.params_ok)
selected = leftjoin(selected, select(seed_table, :location, :seed, :is_best_seed), on = [:location, :seed])

function mechanism_shares(df)
    Y = zeros(length(settings), length(MECHANISMS))
    for (i, s) in enumerate(settings)
        sub = df.mechanism[df.setting .== s]
        Y[i, :] = [count(==(m), sub) / length(sub) for m in MECHANISMS]
    end
    return Y
end

function mechanism_panel(df, title; show_legend)
    groupedbar(1:length(settings), mechanism_shares(df); bar_position = :stack, bar_width = 0.8, linewidth = 0,
               color = permutedims(MECHANISM_COLORS), label = show_legend ? permutedims(MECHANISMS) : "",
               legend = show_legend ? :outerright : false,
               xticks = (1:length(settings), short_name.(settings)), xrotation = 45,
               ylims = (0, 1), ylabel = "share of selected equations", title = title, titlefontsize = 11,
               grid = :y, gridalpha = 0.3)
end

mech_fig = plot(mechanism_panel(selected, "All seeds (51 locations × 100 seeds per setting)"; show_legend = true),
                mechanism_panel(filter(:is_best_seed => identity, selected),
                                "Best UDE seed only (lowest validation loss, 1 per location, 51 per setting)"; show_legend = true);
                layout = (2, 1), size = (1700, 1300), left_margin = 10Plots.mm, bottom_margin = 14Plots.mm,
                plot_title = "What kind of equation SR selected ($(beta_function))", plot_titlefontsize = 13)
mech_path = joinpath(plot_dir, "sr_selected_mechanism.png")
savefig(mech_fig, mech_path)
println("\nSaved: $(mech_path)")

#==============================================================
# 3) Best seed per location
==============================================================#

best = filter(:is_best_seed => identity, selected)
@assert nrow(best) == length(settings) * length(locations) "Expected one best seed per setting x location"

best_table = select(best, :setting, :location, :seed, :complexity, :mechanism,
                    :a_hat, :b_hat, :equation => ByRow(format_equation_sigfigs) => :equation)
best_table.a_true = [truth[l].a for l in best_table.location]
best_table.b_true = [truth[l].b for l in best_table.location]
sort!(best_table, [order(:setting; by = setting_order), :location])
best_table.setting = short_name.(best_table.setting)
CSV.write(joinpath(save_dir, "sr_best_seed_equations.csv"), best_table)
println("\nSaved: $(joinpath(save_dir, "sr_best_seed_equations.csv"))")

# Share of selected equations that are the ground truth (right parameters), best seed vs all seeds
gt_share(df, categories) = [mean(is_gt.(df.structure[df.setting .== s], df.params_ok[df.setting .== s], Ref(categories))) for s in settings]

function best_vs_all_panel(definition)
    p = plot(; xticks = (1:length(settings), short_name.(settings)), xrotation = 45, ylims = (0, 1),
             ylabel = "share selecting the ground truth", title = definition.title, titlefontsize = 11,
             legend = :topright, grid = :y, gridalpha = 0.3)
    all_share  = gt_share(selected, definition.categories)
    best_share = gt_share(best, definition.categories)
    # Thin line joining each pair so the change is easy to read
    for i in eachindex(settings)
        plot!(p, [i, i], [all_share[i], best_share[i]]; color = "#c8c8c8", linewidth = 2, label = "")
    end
    scatter!(p, 1:length(settings), all_share;  color = "#8c8c8c", markersize = 7, markerstrokewidth = 0, label = "all seeds")
    scatter!(p, 1:length(settings), best_share; color = "#2a78d6", markersize = 7, markerstrokewidth = 0, label = "best UDE seed per location")
    return p
end

best_fig = plot([best_vs_all_panel(d) for d in DEFINITIONS]...; layout = (2, 1), size = (1500, 1200),
                left_margin = 10Plots.mm, bottom_margin = 14Plots.mm,
                plot_title = "Does the best UDE seed give a cleaner SR recovery? ($(beta_function), a & b within $(Int(100PARAM_RTOL))%)",
                plot_titlefontsize = 13)
best_path = joinpath(plot_dir, "sr_best_seed_vs_all.png")
savefig(best_fig, best_path)
println("Saved: $(best_path)")

# One example Pareto front, like the screenshot table: best seed, AL, unit p3 m30 c2
example_setting = "SR_input=unit_parsimony=3_maxsize=30_coc=2"
example_seed = only(seed_table.seed[(seed_table.location .== "AL") .& seed_table.is_best_seed])
example = filter(r -> r.setting == example_setting && r.location == "AL" && r.seed == example_seed, fronts)
println("\n=== Pareto front: AL, best seed ($(example_seed)), $(short_name(example_setting)); truth a=$(round(truth["AL"].a, sigdigits = 3)), b=$(round(truth["AL"].b, sigdigits = 3)) ===")
for r in eachrow(example)
    println(rpad("c$(r.complexity)$(r.is_selected ? " *" : "")", 6), rpad(mechanism(r.structure, r.params_ok), 36), format_equation_sigfigs(r.equation))
end
