#========================================================
PRESENT RESULTS — JOINTLY ESTIMATED DISPERSION WITHOUT A CLAMP

Runs: beta_exp, train_length=365, AutoTsit5(Rosenbrock23()) with default
tolerances, random (every-5th-point) validation split, noise level phi
estimated jointly with the NN as log_phi (init phi=0.17), r = exp(-2 log_phi)
with NO bounds (commit d83fe5e). Folders:
  autotsit5_UDE_single_beta=beta_exp_..._noise=<n>_estphi_noclamp

A seed is "blown up" if phi_hat lies outside the proposed clamp range
[1e-3, 10] (r in [1e-2, 1e6]): "phi -> 0" below it (r -> infinity) and
"phi -> infinity" above it (r -> 0). In practice there is a wide gap: no seed
has phi_hat in [1e-4, 1e-3), so the exact threshold does not matter.

Figures (plots/sims/test/ude_single/estphi_noclamp/):
  01 phi_hat vs true noise, every seed, pooled over all 51 locations
  02 blow-up rate by noise level, split by direction
  03 validation loss of blown-up vs converged seeds (within-location rank,
     plus raw values for the example location)
  04 what multistart selects: phi_hat of the lowest-val-loss seed per
     location, and % blown-up seeds among the seeds kept at each MS limit
  05 numerical error of loss_negbin vs r (Float64 vs 512-bit BigFloat)
  06 training-loss traces for example seeds (full run and zoom on the
     Adam -> LBFGS switch)
  07 infectious / beta recovery (NMSE) of blown-up vs converged seeds
  08 phi_hat / true noise for converged seeds
  09 infectious, beta(t) and beta vs I/N for blown-up vs converged seeds,
     example location, a few noise levels
  10 comparison with the old fixed-r noise runs (selected seed per location)
Plus summary_by_noise.csv with the headline numbers.

Comparison with the old runs (UDE_single_beta=beta_exp_..._noise=<n>):
those used the TRUE r (fixed), the old pred-only NLL, an older solver set-up
and older code, and only cover 21 locations. So figure 10 is restricted to
the locations both sweeps share, and differences are not attributable to the
dispersion estimation alone. Their validation losses are on a different
scale, but NMSE metrics are computed the same way.
=========================================================#

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using JLD2
using CSV
using DataFrames
using Statistics
using Random
using Printf
using StatsPlots
using SpecialFunctions
using StatsBase: ordinalrank
using UDE_FUNCTIONAL_FORMS

#========================================================
CONFIGURATION
=========================================================#

const NOISE_LEVELS     = [0.01, 0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
const EXAMPLE_LOCATION = "MA"
const EXAMPLE_NOISES   = [0.01, 0.1, 0.4]
const TRACE_NOISE      = 0.1
const PHI_LO, PHI_HI   = 1e-3, 10.0      # proposed clamp bounds on phi
const PHI_DISPLAY_MIN  = 1e-160          # display floor for log axes only
const MS_LIMITS        = [0.0, 0.5, 0.8, 0.9]
const N_EXAMPLE_LINES  = 30              # max seeds drawn per group in figure 09
const TRAIN_LENGTH     = 365
const N_ADAM           = 2500

const SIMS_ROOT = datadir("exp_pro", "sims", "test", "ude_single")
const SAVE_DIR  = plotsdir("sims", "test", "ude_single", "estphi_noclamp")
mkpath(SAVE_DIR)

new_sim(n) = "autotsit5_UDE_single_beta=beta_exp_adam=2500_lbfgs=2000_traindata=$(TRAIN_LENGTH)_noise=$(n)_estphi_noclamp"
old_sim(n) = "UDE_single_beta=beta_exp_adam=2500_lbfgs=2000_traindata=$(TRAIN_LENGTH)_noise=$(n)"

# Colours: one per entity, used consistently across every figure
const C_CONV  = "#2a78d6"   # converged seeds (new runs)
const C_LOW   = "#eb6834"   # blown up, phi -> 0 (r -> infinity)
const C_HIGH  = "#4a3aa7"   # blown up, phi -> infinity (r -> 0)
const C_OLD   = "#1baf7a"   # old fixed-r runs
const C_TRUTH = "#0b0b0b"
const C_MUTED = "#52514e"
const C_DATA  = "#a9a8a3"
const MS_BLUES = ["#a9c9f0", "#6aa2e6", "#2a78d6", "#18508f"]   # sequential, one hue

# (status, legend label, colour) in a fixed order
const STATUSES = [("converged", "Converged",          C_CONV),
                  ("low",       "Blown up, φ̂ → 0",   C_LOW),
                  ("high",      "Blown up, φ̂ → ∞",   C_HIGH)]

default(dpi=200, titlefontsize=11, guidefontsize=10, tickfontsize=8, legendfontsize=8,
        gridalpha=0.12, foreground_color_legend=nothing, background_color_legend=nothing,
        left_margin=4Plots.mm, bottom_margin=4Plots.mm)

const NX   = length(NOISE_LEVELS)
const XCAT = (1:NX, string.(NOISE_LEVELS))
noise_index(n) = findfirst(==(n), NOISE_LEVELS)
status_of(phi) = phi < PHI_LO ? "low" : phi > PHI_HI ? "high" : "converged"
save(p, name) = (savefig(p, joinpath(SAVE_DIR, name)); println("Saved $(name)"))

#========================================================
LOAD PER-SEED SUMMARIES (written by scripts/multistart.jl)
=========================================================#

function load_summaries(sim_for_noise)
    parts = DataFrame[]
    for n in NOISE_LEVELS
        dir = joinpath(SIMS_ROOT, sim_for_noise(n))
        isdir(dir) || continue
        for loc_folder in readdir(dir)
            f = joinpath(dir, loc_folder, "results_summary.csv")
            isfile(f) || continue
            df = CSV.read(f, DataFrame)
            nrow(df) == 0 && continue
            df[!, :noise]    = fill(n, nrow(df))
            df[!, :location] = fill(replace(loc_folder, "synthetic_" => ""), nrow(df))
            push!(parts, select(df, :noise, :location, :seed, :nmse_traj_true, :nmse_beta,
                                :nmse_I_grid, :best_val_loss, :phi_hat, :feasible))
        end
    end
    return vcat(parts...; cols=:union)
end

new = load_summaries(new_sim)
new.status = status_of.(new.phi_hat)
new.blown  = new.status .!= "converged"
old = load_summaries(old_sim)
println("New runs: $(nrow(new)) seeds over $(length(unique(new.location))) locations; ",
        "old runs: $(nrow(old)) seeds over $(length(unique(old.location))) locations")

# Within-location rank of each seed's validation loss (0 = lowest, 1 = highest)
transform!(groupby(new, [:noise, :location]),
           :best_val_loss => (v -> length(v) == 1 ? zeros(1) : (ordinalrank(v) .- 1) ./ (length(v) - 1)) => :val_rank)

# The seed scripts/multistart.jl ranks first: lowest validation loss among feasible seeds
function best_by_val(df; filter_fn = _ -> true)
    out = DataFrame()
    for g in groupby(df, [:noise, :location])
        sub = filter(r -> coalesce(r.feasible, false) && filter_fn(r), g)
        nrow(sub) == 0 && continue
        push!(out, sub[argmin(sub.best_val_loss), :]; cols=:union)
    end
    return out
end

best_new      = best_by_val(new)
best_new_conv = best_by_val(new; filter_fn = r -> r.status == "converged")
best_old      = best_by_val(old)

# Categorical x position with horizontal jitter, so neighbouring noise levels never overlap
jitter(rng, noises; w=0.28) = noise_index.(noises) .+ w .* (2 .* rand(rng, length(noises)) .- 1)

# Side-by-side boxes, one per status, at every noise level
function status_boxes!(sp, df, col; statuses=STATUSES, width=0.26)
    k = length(statuses)
    for (i, n) in enumerate(NOISE_LEVELS), (j, (st, lab, c)) in enumerate(statuses)
        v = df[(df.noise .== n) .& (df.status .== st), col]
        isempty(v) && continue
        boxplot!(sp, fill(i + (j - (k + 1) / 2) * width, length(v)), v, bar_width=0.9width, color=c,
                 fillalpha=0.75, linecolor=C_MUTED, ms=1, msw=0, label=i == 1 ? lab : "")
    end
    return sp
end

#========================================================
SUMMARY TABLE
=========================================================#

summary_df = DataFrame(noise=Float64[], seeds=Int[], pct_blown=Float64[], pct_phi_to_0=Float64[],
                       pct_phi_to_inf=Float64[], best_val_seed_blown=String[],
                       median_ratio_converged=Float64[])
for n in NOISE_LEVELS
    s = new[new.noise .== n, :]
    b = best_new[best_new.noise .== n, :]
    c = s[s.status .== "converged", :]
    push!(summary_df, (n, nrow(s), 100mean(s.blown), 100mean(s.status .== "low"),
                       100mean(s.status .== "high"), "$(count(b.blown))/$(nrow(b))",
                       median(c.phi_hat ./ n)))
end
CSV.write(joinpath(SAVE_DIR, "summary_by_noise.csv"), summary_df)
show(summary_df; allrows=true); println()

#========================================================
01 PHI_HAT VS TRUE NOISE — EVERY SEED
=========================================================#

rng = MersenneTwister(1)
phi_axis = (yscale=:log10, ylims=(PHI_DISPLAY_MIN / 10, 1e22), yticks=10.0 .^ (-160:20:0))

p01 = plot(; xticks=XCAT, xlims=(0.4, NX + 0.6), xlabel="True noise level", ylabel="Estimated noise level φ̂",
           title="φ̂ for every seed (51 locations × 100 seeds per noise level)",
           legend=:outerright, size=(1150, 620), phi_axis...)
hline!(p01, [PHI_LO, PHI_HI], color=C_MUTED, ls=:dot, lw=1, label="Proposed clamp bounds\n(φ ∈ [10⁻³, 10])")
for (st, lab, c) in STATUSES
    s = new[new.status .== st, :]
    scatter!(p01, jitter(rng, s.noise), max.(s.phi_hat, PHI_DISPLAY_MIN), ms=1.5, msw=0,
             alpha=st == "converged" ? 0.25 : 0.35, color=c, label="$(lab) ($(nrow(s)) seeds)")
end
for (i, n) in enumerate(NOISE_LEVELS)
    plot!(p01, [i - 0.38, i + 0.38], [n, n], color=C_TRUTH, lw=2, label=i == 1 ? "True noise level" : "")
end
save(p01, "01_phi_hat_vs_true_noise.png")

#========================================================
02 BLOW-UP RATE BY NOISE LEVEL
=========================================================#

p02 = groupedbar(string.(summary_df.noise), [summary_df.pct_phi_to_0 summary_df.pct_phi_to_inf],
                 bar_position=:stack, color=[C_LOW C_HIGH], linecolor=:white, lw=1,
                 label=["φ̂ → 0  (r → ∞)" "φ̂ → ∞  (r → 0)"],
                 xlabel="True noise level", ylabel="% of seeds blown up",
                 title="Seeds whose φ̂ left the range [10⁻³, 10]", legend=:topright,
                 ylims=(0, 1.2maximum(summary_df.pct_blown)), size=(800, 500))
for (i, v) in enumerate(summary_df.pct_blown)
    annotate!(p02, i - 0.5, v + 0.8, text(@sprintf("%.1f%%", v), 8, C_MUTED))
end
save(p02, "02_blowup_rate_by_noise.png")

#========================================================
03 VALIDATION LOSS — BLOWN-UP VS CONVERGED
=========================================================#

p03a = plot(xticks=XCAT, xlabel="True noise level", ylabel="Rank of validation loss within location\n(0 = lowest, 1 = highest)",
            title="Rank of each seed's validation loss among its location's 100 seeds", legend=:top,
            legend_columns=3, ylims=(-0.02, 1.15))
status_boxes!(p03a, new, :val_rank)

ex = sort(new[(new.noise .== TRACE_NOISE) .& (new.location .== EXAMPLE_LOCATION), :], :best_val_loss)
negval = -ex.best_val_loss
p03b = plot(yscale=:log10, xlabel="Seed, ordered by validation loss (lowest first)",
            ylabel="−(best validation loss)", legend=:topright,
            title="$(EXAMPLE_LOCATION), noise = $(TRACE_NOISE): raw validation losses")
for (st, lab, c) in STATUSES
    idx = findall((ex.status .== st) .& (negval .> 0))   # log axis needs a positive value
    isempty(idx) || scatter!(p03b, idx, negval[idx], ms=4, msw=0, color=c, label=lab)
end
p03 = plot(p03a, p03b, layout=(1, 2), size=(1500, 580), left_margin=8Plots.mm, bottom_margin=10Plots.mm)
save(p03, "03_val_loss_blown_vs_converged.png")

#========================================================
04 WHAT MULTISTART SELECTS
=========================================================#

p04a = plot(; xticks=XCAT, xlims=(0.4, NX + 0.6), xlabel="True noise level", ylabel="φ̂ of the lowest-validation-loss seed",
            title="Seed multistart ranks first, one point per location\n(numbers: locations where that seed is blown up)",
            legend=:outerbottom, legend_columns=4, phi_axis...)
hline!(p04a, [PHI_LO, PHI_HI], color=C_MUTED, ls=:dot, label="Proposed clamp bounds")
for (st, lab, c) in STATUSES
    s = best_new[best_new.status .== st, :]
    nrow(s) == 0 && continue
    scatter!(p04a, jitter(rng, s.noise), max.(s.phi_hat, PHI_DISPLAY_MIN), ms=3.5, msw=0, alpha=0.75, color=c, label=lab)
end
for (i, n) in enumerate(NOISE_LEVELS)
    plot!(p04a, [i - 0.38, i + 0.38], [n, n], color=C_TRUTH, lw=2, label=i == 1 ? "True noise level" : "")
    annotate!(p04a, i, 1e15, text(summary_df.best_val_seed_blown[i], 8, C_MUTED))
end

# Replicates scripts/multistart.jl: feasible seeds sorted by val loss, drop the worst floor(n*MS)
ms_pct = zeros(length(MS_LIMITS), NX)
for (j, n) in enumerate(NOISE_LEVELS), (k, ms) in enumerate(MS_LIMITS)
    pct = Float64[]
    for g in groupby(new[new.noise .== n, :], :location)
        sub = sort(filter(r -> coalesce(r.feasible, false), g), :best_val_loss)
        nr = nrow(sub); nr == 0 && continue
        kept = sub[1:(nr - floor(Int, nr * ms)), :]
        push!(pct, 100mean(kept.blown))
    end
    ms_pct[k, j] = mean(pct)
end
p04b = plot(xticks=XCAT, xlabel="True noise level", ylabel="% of kept seeds that are blown up\n(mean over locations)",
            title="Share of blown-up seeds in the multistart-kept set", legend=:topright, ylims=(0, 105))
for (k, ms) in enumerate(MS_LIMITS)
    plot!(p04b, 1:NX, ms_pct[k, :], color=MS_BLUES[k], lw=2, marker=:circle, ms=4, msw=0,
          label="MS = $(ms) (keep best $(round(Int, 100(1 - ms)))%)")
end
p04 = plot(p04a, p04b, layout=(1, 2), size=(1600, 680), left_margin=8Plots.mm, bottom_margin=8Plots.mm)
save(p04, "04_multistart_selection.png")

#========================================================
05 NUMERICAL ERROR OF THE NEGATIVE BINOMIAL LOSS VS r
=========================================================#

# Reference: the same expression evaluated in 512-bit BigFloat, using the
# noise-free trajectory as the mean and the noisy data at TRACE_NOISE
setprecision(BigFloat, 512)
synth(n) = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_HQ_beta_exp",
                             "synthetic_$(EXAMPLE_LOCATION)", "noise=$(n).jld2"))
mu_true = Float64.(synth(0.0)["infectious"][1:TRAIN_LENGTH])
y_obs   = synth(TRACE_NOISE)["infectious"][1:TRAIN_LENGTH]
r_grid  = 10.0 .^ range(-2, 40; length=169)
loss_f(r)   = UDE_FUNCTIONAL_FORMS.loss_negbin(mu_true, y_obs, r)
loss_big(r) = UDE_FUNCTIONAL_FORMS.loss_negbin(BigFloat.(mu_true), BigFloat.(y_obs), BigFloat(r))
abs_err = [begin
               a = loss_f(r); b = loss_big(r)
               isfinite(a) ? max(Float64(abs(a - b)), 1e-12) : NaN
           end for r in r_grid]

p05 = plot(r_grid, abs_err, xscale=:log10, yscale=:log10, color=C_TRUTH, lw=2, label="|Float64 − exact|",
           xlabel="Dispersion r  (= 1/φ²)", ylabel="Absolute error in the loss",
           title="Floating-point error in loss_negbin ($(EXAMPLE_LOCATION), noise = $(TRACE_NOISE) data, true mean)",
           legend=:topleft, size=(950, 560), xticks=10.0 .^ (-2:4:40))
vspan!(p05, [1 / 0.4^2, 1 / 0.01^2], color=C_CONV, alpha=0.12, label="r for your noise sweep (0.01–0.4)")
vline!(p05, [1e-2, 1e6], color=C_MUTED, ls=:dot, label="Proposed clamp bounds (r ∈ [10⁻², 10⁶])")
save(p05, "05_loss_precision_vs_r.png")

#========================================================
06 TRAINING-LOSS TRACES — EXAMPLE SEEDS
=========================================================#

# The losses are negative here (the constant sum log k! is dropped from the NLL), so plot
# -loss on a log axis: higher = lower loss. Any non-negative value cannot be shown and is reported.
neg_loss(tl) = [x < 0 ? -x : NaN for x in tl]
seed_file(sim, n, loc, seed) = joinpath(SIMS_ROOT, sim(n), "synthetic_$(loc)", "simulation_seed=$(seed)", "results.jld2")

trace_pool = sort(new[(new.noise .== TRACE_NOISE) .& (new.location .== EXAMPLE_LOCATION), :], :best_val_loss)
p06a = plot(xlabel="Iteration", ylabel="−(training loss)  (log scale; higher = lower loss)", yscale=:log10,
            title="Whole run", legend=:topleft)
p06b = plot(xlabel="Iteration", ylabel="−(training loss)  (log scale)", yscale=:log10, legend=false,
            title="Zoom: iterations $(N_ADAM - 10)–$(N_ADAM + 30) (markers = individual iterations)",
            xlims=(N_ADAM - 10, N_ADAM + 30))
trace_values = Float64[]
for (st, lab, c) in STATUSES
    k = findfirst(==(st), trace_pool.status)
    isnothing(k) && continue
    row = trace_pool[k, :]
    tl  = JLD2.load(seed_file(new_sim, TRACE_NOISE, EXAMPLE_LOCATION, row.seed), "train_losses")
    l   = @sprintf("%s (seed %d, φ̂ = %.2g, %d LBFGS iterations)", lab, row.seed, row.phi_hat, length(tl) - N_ADAM - 1)
    n_nonneg = count(>=(0), tl)
    n_nonneg > 0 && println("Figure 06: seed $(row.seed) has $(n_nonneg) non-negative training losses, not shown on the log axis")
    plot!(p06a, neg_loss(tl), color=c, lw=1.5, label=l)
    plot!(p06b, neg_loss(tl), color=c, lw=1.5, marker=:circle, ms=2.5, msw=0)
    append!(trace_values, filter(isfinite, neg_loss(tl)))
end
# vline! spans ±Inf, which a log axis cannot draw, so give the switch line explicit finite ends
y_switch = (minimum(trace_values) / 2, maximum(trace_values) * 2)
for sp in (p06a, p06b)
    plot!(sp, [N_ADAM, N_ADAM], collect(y_switch), color=C_MUTED, ls=:dash, lw=1.5, label="Adam → LBFGS")
end
p06 = plot(p06a, p06b, layout=(1, 2), size=(1500, 560), left_margin=8Plots.mm, bottom_margin=10Plots.mm,
           plot_title="$(EXAMPLE_LOCATION), true noise = $(TRACE_NOISE): the best converged seed vs the seed multistart ranks first",
           plot_titlefontsize=11)
save(p06, "06_training_traces_example.png")

#========================================================
07 RECOVERY OF INFECTIOUS AND BETA — BLOWN-UP VS CONVERGED
=========================================================#

metrics = [(:nmse_traj_true, "NMSE infectious vs truth (days 1–365)"),
           (:nmse_beta,      "NMSE β(t) vs truth"),
           (:nmse_I_grid,    "NMSE β vs I/N (full 0–1 grid)")]
panels07 = [status_boxes!(plot(xticks=XCAT, yscale=:log10, xlabel="True noise level", ylabel=lab, legend=:topleft), new, col)
            for (col, lab) in metrics]
p07 = plot(panels07..., layout=(1, 3), size=(1800, 560), left_margin=10Plots.mm, bottom_margin=8Plots.mm,
           plot_title="Does a blown-up φ̂ also damage the fit? All seeds, 51 locations", plot_titlefontsize=12)
save(p07, "07_recovery_blown_vs_converged.png")

#========================================================
08 ACCURACY OF PHI_HAT WHEN IT CONVERGES
=========================================================#

p08 = plot(xticks=XCAT, yscale=:log10, xlabel="True noise level", ylabel="φ̂ / true noise level",
           title="Converged seeds: estimated / true noise level (51 locations; label = median)",
           legend=false, size=(950, 540), yticks=([0.5, 1, 2, 5, 10, 100], ["0.5", "1", "2", "5", "10", "100"]))
hline!(p08, [1.0], color=C_TRUTH, lw=1.5)
for (i, n) in enumerate(NOISE_LEVELS)
    ratio = new[(new.noise .== n) .& .!new.blown, :phi_hat] ./ n
    boxplot!(p08, fill(i, length(ratio)), ratio, bar_width=0.5, color=C_CONV, fillalpha=0.7,
             linecolor=C_MUTED, ms=1.5, msw=0)
    annotate!(p08, i + 0.32, median(ratio), text(@sprintf("%.3f", median(ratio)), 7, C_MUTED, :left))
end
save(p08, "08_phi_accuracy_converged.png")

#========================================================
09 EXAMPLE RECOVERY — INFECTIOUS AND BETA FOR BLOWN-UP VS CONVERGED SEEDS
=========================================================#

population  = POPULATION[EXAMPLE_LOCATION]
I_grid      = collect(range(0, 1; length=1000))
true_beta_t = beta_exp(EXAMPLE_LOCATION, mu_true)
true_beta_I = beta_exp(EXAMPLE_LOCATION, I_grid .* population)
I_N_max     = maximum(mu_true) / population
days        = 1:TRAIN_LENGTH

rows09 = []
for n in EXAMPLE_NOISES
    s   = sort(new[(new.noise .== n) .& (new.location .== EXAMPLE_LOCATION), :], :seed)
    y_n = synth(n)["infectious"][1:TRAIN_LENGTH]
    counts = "$(count(s.status .== "converged")) converged, $(count(s.status .== "low")) with φ̂ → 0, $(count(s.status .== "high")) with φ̂ → ∞"
    pi_ = plot(xlabel="Day", ylabel="Infectious", title="noise = $(n): infectious\n($(counts))", legend=:topright)
    pt  = plot(xlabel="Day", ylabel="β(t)  (log scale)", title="noise = $(n): β(t)", legend=false, yscale=:log10)
    pI  = plot(xlabel="I/N", ylabel="β  (log scale)", title="noise = $(n): β vs I/N", legend=false, yscale=:log10)
    scatter!(pi_, days, y_n, ms=1.5, msw=0, color=C_DATA, label="Noisy data")
    for (st, lab, c) in STATUSES
        sel = first(s[s.status .== st, :], N_EXAMPLE_LINES)
        for (k, row) in enumerate(eachrow(sel))
            d = JLD2.load(seed_file(new_sim, n, EXAMPLE_LOCATION, row.seed))
            a = st == "converged" ? 0.35 : 0.6
            plot!(pi_, days, d["prediction"][3, 1:TRAIN_LENGTH], color=c, alpha=a, lw=1, label=k == 1 ? lab : "")
            plot!(pt, days, max.(d["beta_prediction"][1:TRAIN_LENGTH], 1e-6), color=c, alpha=a, lw=1)
            plot!(pI, I_grid, max.(d["y_hat"], 1e-6), color=c, alpha=a, lw=1)
        end
    end
    plot!(pi_, days, mu_true, color=C_TRUTH, lw=2, label="Truth")
    plot!(pt, days, true_beta_t, color=C_TRUTH, lw=2)
    plot!(pI, I_grid, true_beta_I, color=C_TRUTH, lw=2)
    vline!(pI, [I_N_max], color=C_MUTED, ls=:dot)   # largest I/N seen in training
    append!(rows09, [pi_, pt, pI])
end
p09 = plot(rows09..., layout=(length(EXAMPLE_NOISES), 3), size=(1800, 540 * length(EXAMPLE_NOISES)),
           left_margin=10Plots.mm, bottom_margin=6Plots.mm,
           plot_title="$(EXAMPLE_LOCATION): up to $(N_EXAMPLE_LINES) seeds per group (black = truth; dotted line = largest I/N in training; β below 10⁻⁶ drawn at 10⁻⁶)",
           plot_titlefontsize=12)
save(p09, "09_example_recovery_$(EXAMPLE_LOCATION).png")

#========================================================
10 COMPARISON WITH THE OLD FIXED-r NOISE RUNS
=========================================================#

common = intersect(unique(old.location), unique(new.location))
groups10 = [("Old runs, true r fixed", C_OLD, filter(r -> r.location in common, best_old)),
            ("New, no clamp: seed multistart picks (mostly φ̂ → 0)", C_LOW, filter(r -> r.location in common, best_new)),
            ("New, no clamp: best converged seed", C_CONV, filter(r -> r.location in common, best_new_conv))]
panels10 = []
for (col, lab) in metrics
    sp = plot(xticks=XCAT, yscale=:log10, xlabel="True noise level", ylabel=lab, legend=:topleft)
    for (i, n) in enumerate(NOISE_LEVELS), (g, (glab, gcol, gdf)) in enumerate(groups10)
        v = gdf[gdf.noise .== n, col]
        isempty(v) && continue
        boxplot!(sp, fill(i + (g - 2) * 0.26, length(v)), v, bar_width=0.24, color=gcol, fillalpha=0.75,
                 linecolor=C_MUTED, ms=1.5, msw=0, label=i == 1 ? glab : "")
    end
    push!(panels10, sp)
end
p10 = plot(panels10..., layout=(1, 3), size=(1800, 560), left_margin=10Plots.mm, bottom_margin=8Plots.mm,
           plot_title="Lowest-validation-loss seed per location, $(length(common)) locations shared with the old runs " *
                      "(old runs also differ in solver, loss and code)",
           plot_titlefontsize=12)
save(p10, "10_comparison_old_fixed_r.png")

println("\nAll figures saved to $(SAVE_DIR)")
