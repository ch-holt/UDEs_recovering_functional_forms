#========================================================
COMPARE TRAINING SETTINGS — BEHAVIOURAL UDE (exponential contacts)

Justifies each training decision by comparing the setting before and after it
was introduced, in the order the decisions were made:

    1. ΔI raw                    →  ΔI ÷ N0
    2. ΔI ÷ N0                   →  ÷ N0 + range-normalised contact loss
    3. contact input raw         →  contact input ÷ 100  /  ÷ sum(c)
       (all with ÷ N0, normalised contact loss and re-initialisation)

For each comparison:
    - rCRPS violins: per location, log10(CRPS_after / CRPS_before) for every
      evaluation quantity (below 0 = the change helped that location)
    - for three example locations (25th percentile, median and 75th percentile of the
      ranking), one figure per evaluation graph
      (infections, contacts, g, β, ρ̂, ĉ(I), Q3b slices): one row per location,
      one column per setting in the order introduced, y axis shared along each row

CRPS tables come from present_results_behaviour.jl (crps_summary_MS=<x>.csv in
each sim's plot folder); any missing table is computed here and saved there.
Outputs go to plots/.../present_results_behaviour/diagnostics/settings_comparison_exp_MS=<x>/
=========================================================#

include(joinpath(@__DIR__, "present_results_behaviour.jl"))   # functions only; its run is skipped

#========================================================
CONFIGURATION
=========================================================#

# Multistart limit used for every setting (first command-line argument, default 0.8),
# e.g. julia --project=. _research/compare_behaviour_settings.jl 0.5
const COMPARE_MS = isempty(ARGS) ? 0.8 : parse(Float64, ARGS[1])

const SIM_BASE = "UDE_behaviour_model1_contact=contacts_exp_traindata=365_noise=0.0_dI="

# Every setting in the comparisons: sim folder and the short label used in titles
const SETTINGS = Dict(
    :raw    => (sim = SIM_BASE * "none",                        label = "ΔI raw"),
    :N0     => (sim = SIM_BASE * "N0",                          label = "ΔI ÷N0"),
    :loss   => (sim = SIM_BASE * "N0_c_loss_norm",              label = "÷N0, c-loss norm"),
    :reinit => (sim = SIM_BASE * "N0_c_loss_norm_reinit",       label = "c raw"),
    :c100   => (sim = SIM_BASE * "N0_c=100_c_loss_norm_reinit", label = "c ÷100"),
    :csum   => (sim = SIM_BASE * "N0_c=sum_c_loss_norm_reinit", label = "c ÷sum"),
)

# Comparison 3 uses the re-initialised run as its reference, because the c ÷100 and
# c ÷sum runs also have re-initialisation; against :loss the difference would mix
# re-initialisation with the input scaling.
const COMPARISONS = [
    (name = "1_dI_scaling",    title = "ΔI scaling",              reference = :raw,    alternatives = [:N0]),
    (name = "2_contact_loss",  title = "Contact loss weighting",  reference = :N0,     alternatives = [:loss]),
    (name = "3_contact_input", title = "Contact input scaling",   reference = :reinit, alternatives = [:c100, :csum]),
]

# Column order of the example-location figures: every setting once, in the order the
# decisions were introduced, with the label used in those figures
const LADDER = [:raw, :N0, :loss, :reinit, :c100, :csum]
const LADDER_LABELS = Dict(:raw => "1. ΔI raw", :N0 => "2. ΔI ÷N0", :loss => "3. + c-loss norm",
                           :reinit => "4. + re-init", :c100 => "5a. re-init, c ÷100",
                           :csum => "5b. re-init, c ÷sum")

# The setting each one was compared against when introduced (for the ρ̂ titles)
const LADDER_PREVIOUS = Dict(:N0 => :raw, :loss => :N0, :reinit => :loss, :c100 => :reinit, :csum => :reinit)

# Signed relative error of the median ρ̂, in percent
rho_median_error(r) = 100 * (median(r.rho_hat) / r.truth.rho_true - 1)
fmt_pct(x) = "$(round(x, sigdigits = 2))%"

# Example locations: best, median and worst by mean rank of these CRPS columns in this setting
# (the in-sample fit to the two observed streams, which the loss weights equally).
# The ranking always uses SELECTION_MS, so every MS cut shows the same locations.
# Set MANUAL_LOCATIONS to override.
const SELECTION_SETTING = :reinit
const SELECTION_MS      = 0.8
# Positions in the ranking (0 = best, 1 = worst) of the example locations
const EXAMPLE_QUANTILES = (0.25, 0.5, 0.75)
const SELECTION_COLUMNS = ["crps_I", "crps_c"]
const MANUAL_LOCATIONS  = String[]

const OUT_DIR = plotsdir("sims", MODEL_NAME, "present_results_behaviour", "diagnostics",
                         "settings_comparison_exp_MS=$(COMPARE_MS)")

#========================================================
CRPS TABLES
=========================================================#

crps_path(sim, ms) = joinpath(plotsdir("sims", MODEL_NAME, "present_results_behaviour", sim.sim_name),
                              "crps_summary_MS=$(ms).csv")

# Load the CRPS table for a setting, computing (and saving) it if missing
function crps_table(key; ms = COMPARE_MS)
    sim = parse_sim(SETTINGS[key].sim)
    path = crps_path(sim, ms)
    isfile(path) && return CSV.read(path, DataFrame)
    println("No CRPS table for $(sim.label) at MS=$(ms); computing it")
    rows = []
    for loc in find_locations(sim)
        r = location_results(sim, loc; ms_limit = ms)
        isnothing(r) || push!(rows, crps_row(r))
    end
    df = DataFrame(rows)
    df = select(df, vcat(["location", "n_retained"], sort(filter(c -> c ∉ ("location", "n_retained"), names(df)))))
    mkpath(dirname(path))
    CSV.write(path, df)
    println("Saved: $(path)")
    return df
end

#========================================================
LOCATION SELECTION
=========================================================#

function select_locations(df)
    !isempty(MANUAL_LOCATIONS) && return MANUAL_LOCATIONS
    sub = dropmissing(df[:, vcat(["location"], SELECTION_COLUMNS)])
    ranks = [invperm(sortperm(sub[!, c])) for c in SELECTION_COLUMNS]
    sub.mean_rank = mean(stack(ranks; dims = 2); dims = 2)[:, 1]
    sort!(sub, :mean_rank)
    CSV.write(joinpath(OUT_DIR, "location_selection.csv"), sub)
    positions = unique([1 + round(Int, q * (nrow(sub) - 1)) for q in EXAMPLE_QUANTILES])
    picked = sub.location[positions]
    println("Example locations (quantiles $(EXAMPLE_QUANTILES), ranks $(positions), by mean rank of $(join(SELECTION_COLUMNS, ", ")) ",
            "in '$(SETTINGS[SELECTION_SETTING].label)' at MS=$(SELECTION_MS)): $(join(picked, ", "))")
    return picked
end

#========================================================
rCRPS VIOLINS
=========================================================#

# Q5 has no CRPS column: compare the relative error of the median ρ̂ instead,
# |median ρ̂ − ρ_true| / ρ_true, as the same after/before ratio (below 0 = closer to ρ_true)
const RHO_PANEL = ("rho_median_rel_error", "Q5 ρ̂: |median ρ̂ − ρ_true| / ρ_true")
add_rho_error!(df) = (df[!, first(RHO_PANEL)] = abs.(df.rho_median .- df.rho_true) ./ df.rho_true; df)

# One panel per evaluation quantity; one violin per alternative setting, showing
# log10(CRPS_alternative / CRPS_reference) across locations
function build_rcrps_violins(comp, tables)
    ref = tables[comp.reference]
    colors = palette(:tab10, max(length(comp.alternatives), 2))
    panels = Plots.Plot[]
    summary_rows = []
    for (col, title) in vcat(VIOLIN_PANELS, [RHO_PANEL])
        hasproperty(ref, col) || continue
        p = plot(; title = title, ylabel = col == first(RHO_PANEL) ? "log10 error ratio" : "log10 rCRPS",
            legend = false, xrotation = 15)
        hline!(p, [0]; color = :black, linestyle = :dash)
        ticks = String[]
        for (i, key) in enumerate(comp.alternatives)
            alt = tables[key]
            joined = innerjoin(ref[:, ["location", col]], alt[:, ["location", col]]; on = :location,
                               makeunique = true)
            vals = [log10(b / a) for (a, b) in zip(joined[!, col], joined[!, col * "_1"])
                    if !ismissing(a) && !ismissing(b) && a > 0 && b > 0]
            n_better = count(<(0), vals)
            push!(ticks, "$(SETTINGS[key].label)\n$(n_better)/$(length(vals)) better")
            push!(summary_rows, (comparison = comp.name, quantity = col, setting = SETTINGS[key].label,
                                 n_locations = length(vals), n_better = n_better,
                                 median_log10_rcrps = isempty(vals) ? missing : median(vals)))
            isempty(vals) && continue
            length(vals) > 1 && violin!(p, fill(i, length(vals)), vals; color = colors[i], alpha = 0.4, linewidth = 0)
            dotplot!(p, fill(i, length(vals)), vals; color = colors[i], markersize = 2.5, markerstrokewidth = 0)
        end
        xticks!(p, (1:length(comp.alternatives), ticks))
        xlims!(p, (0.4, length(comp.alternatives) + 0.6))
        push!(panels, p)
    end
    ncols = 4
    nrows = cld(length(panels), ncols)
    append!(panels, blank_plots(nrows * ncols - length(panels)))
    fig = plot(panels...; layout = (nrows, ncols), size = (600 * ncols, 480 * nrows),
        plot_title = "$(comp.title): rCRPS against '$(SETTINGS[comp.reference].label)' " *
                     "(below 0 = better), exp, MS=$(COMPARE_MS)",
        plot_titlefontsize = 18, titlefontsize = 12, guidefontsize = 10, tickfontsize = 9,
        left_margin = 12Plots.mm, bottom_margin = 14Plots.mm, top_margin = 4Plots.mm)
    out_path = joinpath(OUT_DIR, "rcrps_$(comp.name).png")
    savefig(fig, out_path)
    println("Saved: $(out_path)")
    return DataFrame(summary_rows)
end

#========================================================
EXAMPLE-LOCATION PANELS
=========================================================#

# The panels of in_sample_plots and out_of_sample_plots, in order: (file name, figure title)
const GRAPHS = [
    (in_sample_plots, [("Q1_infectious", "Infectious I(t)"), ("Q2a_contacts", "Contacts c(t)"),
                       ("Q3a_g_true_trajectory", "g(t) on the true trajectory"), ("Q4_beta", "β(t) = ρ̂·ĉ(t)"),
                       ("Q5_rho", "ρ̂ against ρ_true")]),
    (out_of_sample_plots, [("Q2b_c_vs_I", "ĉ against I/N"), ("Q3b_slice1", "ĝ/ΔI_ref against I/N"),
                           ("Q3b_slice2_0.5peak", "ĝ at I/N = 0.5×peak"), ("Q3b_slice2_peak", "ĝ at I/N = peak"),
                           ("Q3b_slice2_2peak", "ĝ at I/N = 2×peak")]),
]

# Give every panel in a row the same y range, so the settings can be compared by eye
function share_y!(row)
    lims = [ylims(p) for p in row]
    lo, hi = minimum(first.(lims)), maximum(last.(lims))
    foreach(p -> ylims!(p, (lo, hi)), row)
end

# One figure per graph: one row per example location, one column per setting (LADDER order)
function build_location_panels(results, locations)
    keys_in_order = LADDER
    for (plots_fn, graphs) in GRAPHS
        # panels[loc][setting] = that setting's list of panels for the location
        panels = Dict(loc => Dict(k => (isnothing(results[(k, loc)]) ? nothing : plots_fn(results[(k, loc)]))
                                  for k in keys_in_order) for loc in locations)
        for (j, (file_name, graph_title)) in enumerate(graphs)
            grid = Plots.Plot[]
            for loc in locations
                row = Plots.Plot[]
                for k in keys_in_order
                    ps = panels[loc][k]
                    if isnothing(ps)
                        push!(row, plot(; legend = false, grid = false, showaxis = false,
                                        title = "$(loc) — $(LADDER_LABELS[k]): no retained seeds"))
                        continue
                    end
                    p = ps[j]
                    title = "$(loc) — $(LADDER_LABELS[k])"
                    # ρ̂ panels also give the median error, and what it was in the previous setting
                    if file_name == "Q5_rho"
                        title *= ": " * fmt_pct(rho_median_error(results[(k, loc)]))
                        prev = get(LADDER_PREVIOUS, k, nothing)
                        if !isnothing(prev) && !isnothing(results[(prev, loc)])
                            title *= " (prev " * fmt_pct(rho_median_error(results[(prev, loc)])) * ")"
                        end
                    end
                    title!(p, title)
                    push!(row, p)
                end
                # ρ̂ histograms have their own x ranges and counts, so are left unlinked. The ΔI raw
                # column keeps its own axis: its ensemble is so wide it would flatten the other settings.
                shared = [p for (k, p) in zip(keys_in_order, row) if k != :raw && !isempty(p.series_list)]
                file_name == "Q5_rho" || share_y!(shared)
                append!(grid, row)
            end
            ncols = length(keys_in_order)
            fig = plot(grid...; layout = (length(locations), ncols), size = (560 * ncols, 560 * length(locations)),
                plot_title = "$(graph_title) by training setting — exp, MS=$(COMPARE_MS)", plot_titlefontsize = 18,
                titlefontsize = 12, guidefontsize = 10, tickfontsize = 9, legendfontsize = 8,
                left_margin = 16Plots.mm, right_margin = 3Plots.mm, bottom_margin = 7Plots.mm, top_margin = 8Plots.mm)
            out_path = joinpath(OUT_DIR, "settings_$(file_name).png")
            savefig(fig, out_path)
            println("Saved: $(out_path)")
        end
    end
end

#========================================================
CRPS BY SETTING (one line per location)
=========================================================#

# Lines through the settings: 1 → 2 → 3 → 4 → 5a, and 4 → 5b (5a and 5b are alternatives)
const LADDER_PATHS = [[:raw, :N0, :loss, :reinit, :c100], [:reinit, :csum]]
const EXAMPLE_COLORS = [:darkorange, :crimson, :royalblue]

# One panel per quantity: x = setting (in the order introduced, left to right), y = log10 CRPS,
# one faint line per location, the example locations in colour and the median across locations in black
function build_crps_by_setting(tables, locations)
    x_of = Dict(k => i for (i, k) in enumerate(LADDER))
    log_value(k, loc, col) = begin
        row = tables[k][tables[k].location .== loc, col]
        (isempty(row) || ismissing(row[1]) || row[1] <= 0) ? NaN : log10(row[1])
    end
    all_locations = sort(unique(tables[:reinit].location))
    panels = Plots.Plot[]
    for (col, title) in vcat(VIOLIN_PANELS, [RHO_PANEL])
        all(k -> hasproperty(tables[k], col), LADDER) || continue
        p = plot(; title = title, ylabel = col == first(RHO_PANEL) ? "log10 error" : "log10 CRPS",
            xticks = (1:length(LADDER), [LADDER_LABELS[k] for k in LADDER]), xrotation = 30,
            xlims = (0.6, length(LADDER) + 0.4), legend = false)
        draw!(locs, vals_of; kwargs...) = for path in LADDER_PATHS
            plot!(p, [x_of[k] for k in path], [vals_of(k) for k in path]; kwargs...)
        end
        for loc in all_locations
            draw!([loc], k -> log_value(k, loc, col); color = :gray70, alpha = 0.5, linewidth = 1)
        end
        for (loc, c) in zip(locations, EXAMPLE_COLORS)
            draw!([loc], k -> log_value(k, loc, col); color = c, linewidth = 2, marker = :circle,
                markersize = 3, markerstrokewidth = 0)
        end
        median_of(k) = (v = filter(isfinite, [log_value(k, loc, col) for loc in all_locations]);
                        isempty(v) ? NaN : median(v))
        draw!(all_locations, median_of; color = :black, linewidth = 3, marker = :diamond, markersize = 5)
        push!(panels, p)
    end
    # Key panel naming the coloured lines
    key = plot(; grid = false, showaxis = false, legend = :left, legendfontsize = 12)
    for (loc, c) in zip(locations, EXAMPLE_COLORS)
        plot!(key, [NaN], [NaN]; color = c, linewidth = 2, label = loc)
    end
    plot!(key, [NaN], [NaN]; color = :black, linewidth = 3, label = "Median across locations")
    plot!(key, [NaN], [NaN]; color = :gray70, linewidth = 1, label = "Each location")
    push!(panels, key)

    ncols = 4
    nrows = cld(length(panels), ncols)
    append!(panels, blank_plots(nrows * ncols - length(panels)))
    fig = plot(panels...; layout = (nrows, ncols), size = (650 * ncols, 480 * nrows),
        plot_title = "CRPS by training setting, one line per location — exp, MS=$(COMPARE_MS)",
        plot_titlefontsize = 18, titlefontsize = 12, guidefontsize = 10, tickfontsize = 9,
        left_margin = 12Plots.mm, bottom_margin = 22Plots.mm, top_margin = 4Plots.mm)
    out_path = joinpath(OUT_DIR, "crps_by_setting.png")
    savefig(fig, out_path)
    println("Saved: $(out_path)")
end

#========================================================
RUN
=========================================================#

mkpath(OUT_DIR)
used = unique(vcat(LADDER, [[c.reference; c.alternatives] for c in COMPARISONS]...))
tables = Dict(k => add_rho_error!(crps_table(k)) for k in used)

locations = select_locations(crps_table(SELECTION_SETTING; ms = SELECTION_MS))

# Full per-location results only for the example locations
results = Dict((k, loc) => location_results(parse_sim(SETTINGS[k].sim), loc; ms_limit = COMPARE_MS)
               for k in used for loc in locations)

summaries = DataFrame[]
for comp in COMPARISONS
    push!(summaries, build_rcrps_violins(comp, tables))
end
build_location_panels(results, locations)
build_crps_by_setting(tables, locations)
summary_path = joinpath(OUT_DIR, "rcrps_summary.csv")
CSV.write(summary_path, vcat(summaries...))
println("Saved: $(summary_path)")
