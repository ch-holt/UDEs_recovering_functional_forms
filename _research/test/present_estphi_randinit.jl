#========================================================
PRESENT RESULTS — JOINTLY ESTIMATED DISPERSION, LOSS VARIANTS

Runs: beta_exp, train_length=365, AutoTsit5(Rosenbrock23()) with default
tolerances, random (every-5th-point) validation split, noise level phi
estimated jointly with the NN as log_phi, random log-uniform phi_init per seed.
One sweep per loss_variant (see LOSS_VARIANTS in src/loss.jl), folders:
  autotsit5_UDE_single_beta=beta_exp_..._noise=<n><suffix>
with the suffixes in VARIANTS below. Variants not on disk are skipped.

Same figures as _research/present_estphi_noclamp.jl, with the variants side
by side (2×2, shared y-axis) in plots/sims/test/ude_single/estphi_randinit_comparison/.
Figures that had several panels per variant are split into one file per panel:
  01  phi_hat vs true noise, every seed, pooled over all 51 locations
  02  blow-up rate by noise level, split by direction
  03a/b validation loss of blown-up vs converged seeds (rank; raw, example location)
  04a/b what multistart selects (top seed's phi_hat; share blown up among kept seeds)
  06a/b training-loss traces for example seeds (whole run; zoom on Adam -> LBFGS)
  07_*  infectious / beta recovery (NMSE) of blown-up vs converged seeds, one file per metric;
        07_recovery_MS=0.8_* the same for the seeds multistart keeps at MS = 0.8
  08  phi_hat / true noise for converged seeds
  09  infectious, beta(t) and beta vs I/N, example location: one row per variant,
      one file per example noise level
  10_*  comparison with the old fixed-r noise runs, one file per metric
Figure 05 (Float64 error of loss_negbin vs r) does not depend on the variant
and is not repeated (see estphi_noclamp/05_loss_precision_vs_r.png).

Seed status, as in the no-clamp script: "blown up" if phi_hat is outside
[1e-3, 10]. For poisson_switch, seeds whose final r is in the Poisson regime
(phi_hat not identified) get their own status, "poisson". For bounded, seeds
held at the lower or upper bound on phi get their own statuses (see bound_lo / BOUND_HI);
for poisson_switch_bounded only the upper bound (its floor on r) applies.
Note: for the bounded variants, phi_hat is 1/sqrt(r_used) (recomputed by scripts/multistart.jl),
so it lies in [~1e-3, 10] and the bound statuses, not the [1e-3, 10] test, flag its
seeds held at a bound.

Plus:
  11 seeds lost, blow-up rates and multistart picks, one line/box per variant
Plus summary_by_variant_noise.csv with the headline numbers.

Seeds "lost" = no results.jld2 for that seed (LBFGS line-search assertion,
or the whole location timing out), out of 100 seeds x 51 locations.
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
const MS_RECOVERY      = 0.8             # multistart limit for the kept-seed version of figure 07
const N_EXAMPLE_LINES  = 30              # max seeds drawn per group in figure 09
const TRAIN_LENGTH     = 365
const N_ADAM           = 2500
const N_SEEDS          = 100             # seeds per location
const N_LOCATIONS      = 51

const SIMS_ROOT = datadir("exp_pro", "sims", "test", "ude_single")
# The earlier figures with the 7 variants up to bounded1e5_newinit are kept in
# estphi_randinit_comparison_archive_7variants_2026-10-16/; this folder now holds SHOW_VARIANTS only
const COMPARE_DIR = plotsdir("sims", "test", "ude_single", "estphi_randinit_comparison")
mkpath(COMPARE_DIR)

# (name, sim_name suffix, label, colour) in a fixed order; colours follow the variant in every figure
const VARIANTS = [("bounded",                  "_estphi_randinit_bounded",                  "Bounded",                  "#2a78d6"),
                  ("bigfloat",                 "_estphi_randinit_bigfloat",                 "BigFloat",                 "#eb6834"),
                  ("poissonswitch",            "_estphi_randinit_poissonswitch",            "Poisson switch",           "#1baf7a"),
                  ("unbounded",                "_estphi_randinit_unbounded",                "Unbounded (control)",      "#eda100"),
                  ("poissonswitch_newbounds",  "_estphi_randinit_poissonswitch_newbounds",  "Poisson switch, new init", "#e87ba4"),
                  ("bounded_newinit",          "_estphi_randinit_bounded_newinit",          "Bounded, new init",        "#008300"),
                  ("bounded1e5_newinit",       "_estphi_randinit_bounded1e5_newinit",       "Bounded (r ≤ 1e5), new init", "#e34948"),
                  ("poissonswitchbounded_newinit", "_estphi_randinit_poissonswitchbounded_newinit", "Poisson switch + floor on r, new init", "#4a3aa7"),
                  ("stable_newinit",           "_estphi_randinit_stable_newinit",           "Stable NB loss, new init", "#2a78d6")]

# The variants shown (at most 8, the size of the categorical palette). Colours stay with the variant,
# so the dropped ones keep their colours in the archived figures; "Stable" reuses the blue of the old
# "Bounded", which is not shown here
const SHOW_VARIANTS = ["unbounded", "poissonswitch_newbounds", "bounded1e5_newinit",
                       "poissonswitchbounded_newinit", "stable_newinit"]

# Bounded variants (plotting name => loss_variant), for the bound statuses below
const BOUNDED_VARIANTS = Dict("bounded" => "bounded", "bounded_newinit" => "bounded", "bounded1e5_newinit" => "bounded_1e5")
# Variants with the floor on r (r = ... + 1e-2, so phi_hat <= 10): the bounded ones and poisson_switch_bounded
has_r_floor(vname) = haskey(BOUNDED_VARIANTS, vname) || vname == "poissonswitchbounded_newinit"

sim_name(suffix, n) = "autotsit5_UDE_single_beta=beta_exp_adam=2500_lbfgs=2000_traindata=$(TRAIN_LENGTH)_noise=$(n)$(suffix)"
old_sim(n) = "UDE_single_beta=beta_exp_adam=2500_lbfgs=2000_traindata=$(TRAIN_LENGTH)_noise=$(n)"

# Colours for seed status, as in the no-clamp script
const C_CONV  = "#2a78d6"   # converged seeds
const C_LOW   = "#eb6834"   # blown up, phi -> 0 (r -> infinity)
const C_HIGH  = "#4a3aa7"   # blown up, phi -> infinity (r -> 0)
const C_POIS  = "#e87ba4"   # poisson_switch only: final r in the Poisson regime
const C_LOWB  = "#eda100"   # bounded only: phi held at its lower bound (r at its cap)
const C_HIGHB = "#1baf7a"   # bounded only: phi held at its upper bound (r at its floor)
const C_OLD   = "#1baf7a"   # old fixed-r runs
const C_TRUTH = "#0b0b0b"
const C_MUTED = "#52514e"
const C_DATA  = "#a9a8a3"
const MS_BLUES = ["#a9c9f0", "#6aa2e6", "#2a78d6", "#18508f"]   # sequential, one hue

# (status, legend label, colour) in a fixed order
const ALL_STATUSES = [("converged", "Converged",                   C_CONV),
                      ("low",       "Blown up, φ̂ → 0",            C_LOW),
                      ("high",      "Blown up, φ̂ → ∞",            C_HIGH),
                      ("poisson",   "Poisson regime (φ̂ not identified)", C_POIS),
                      ("lower_bound", "At lower bound on φ (r at its cap)",   C_LOWB),
                      ("upper_bound", "At upper bound on φ (r at its floor)", C_HIGHB)]

# Bounded variants only: a seed is at a bound when the bound term dominates its part of the transform
# (see r_bounded in src/loss.jl, phi_hat = 1/sqrt(r_used)), with phi_min = BOUNDED_PHI_MIN[loss_variant]:
#   lower: exp(log_phi) < phi_min, i.e. phi_used < 2 phi_min, so phi_hat < 2 phi_min
#   upper: 1/phi_used^2 < 1e-2,    i.e. r_used < 2e-2,        so phi_hat > 1/sqrt(2e-2) ≈ 7.07
bound_lo(vname) = 2 * BOUNDED_PHI_MIN[BOUNDED_VARIANTS[vname]]
const BOUND_HI = 1 / sqrt(2e-2)

default(dpi=200, titlefontsize=11, guidefontsize=10, tickfontsize=8, legendfontsize=8,
        gridalpha=0.12, foreground_color_legend=nothing, background_color_legend=nothing,
        left_margin=4Plots.mm, bottom_margin=4Plots.mm)

const NX   = length(NOISE_LEVELS)
const XCAT = (1:NX, string.(NOISE_LEVELS))
noise_index(n) = findfirst(==(n), NOISE_LEVELS)
function status_of(phi, pregime, vname)
    pregime === true && return "poisson"
    haskey(BOUNDED_VARIANTS, vname) && phi < bound_lo(vname) && return "lower_bound"
    has_r_floor(vname) && phi > BOUND_HI && return "upper_bound"
    return phi < PHI_LO ? "low" : phi > PHI_HI ? "high" : "converged"
end
save(p, dir, name) = (savefig(p, joinpath(dir, name)); println("Saved $(name)"))

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
            # Older summaries have no poisson_regime column
            hasproperty(df, :poisson_regime) || (df[!, :poisson_regime] = fill(missing, nrow(df)))
            push!(parts, select(df, :noise, :location, :seed, :nmse_traj_true, :nmse_beta,
                                :nmse_I_grid, :best_val_loss, :phi_hat, :feasible, :poisson_regime))
        end
    end
    isempty(parts) && return DataFrame()
    return vcat(parts...; cols=:union)
end

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

# Categorical x position with horizontal jitter, so neighbouring noise levels never overlap
jitter(rng, noises; w=0.28) = noise_index.(noises) .+ w .* (2 .* rand(rng, length(noises)) .- 1)

# Side-by-side boxes, one per status, at every noise level
function status_boxes!(sp, df, col, statuses; width=0.8 / length(statuses))
    k = length(statuses)
    for (i, n) in enumerate(NOISE_LEVELS), (j, (st, lab, c)) in enumerate(statuses)
        v = df[(df.noise .== n) .& (df.status .== st), col]
        isempty(v) && continue
        boxplot!(sp, fill(i + (j - (k + 1) / 2) * width, length(v)), v, bar_width=0.9width, color=c,
                 fillalpha=0.75, linecolor=C_MUTED, ms=1, msw=0, label=i == 1 ? lab : "")
    end
    return sp
end

old      = load_summaries(old_sim)
best_old = best_by_val(old)

# Truth for the example location, used by figure 09
synth(n) = JLD2.load(datadir("exp_pro", "synthetic_data", "synthetic_trajectories_HQ_beta_exp",
                             "synthetic_$(EXAMPLE_LOCATION)", "noise=$(n).jld2"))
mu_true = Float64.(synth(0.0)["infectious"][1:TRAIN_LENGTH])

seed_file(sim, n, loc, seed) = joinpath(SIMS_ROOT, sim(n), "synthetic_$(loc)", "simulation_seed=$(seed)", "results.jld2")
neg_loss(tl) = [x < 0 ? -x : NaN for x in tl]

metrics = [(:nmse_traj_true, "NMSE infectious vs truth (days 1–365)"),
           (:nmse_beta,      "NMSE β(t) vs truth"),
           (:nmse_I_grid,    "NMSE β vs I/N (full 0–1 grid)")]

#========================================================
PER-VARIANT PANELS (same content as the no-clamp figures)

Each variant's panels are built here and returned by name; the figures below
put the same panel of every variant side by side.
=========================================================#

function variant_panels(vname, suffix, vlabel)
    vsim(n) = sim_name(suffix, n)

    new = load_summaries(vsim)
    new.status = status_of.(new.phi_hat, new.poisson_regime, vname)
    new.blown  = new.status .!= "converged"
    # Only show the statuses this variant has
    statuses = filter(s -> s[1] in new.status, ALL_STATUSES)
    println("\n$(vlabel): $(nrow(new)) seeds over $(length(unique(new.location))) locations")

    transform!(groupby(new, [:noise, :location]),
               :best_val_loss => (v -> length(v) == 1 ? zeros(1) : (ordinalrank(v) .- 1) ./ (length(v) - 1)) => :val_rank)

    best_new      = best_by_val(new)
    best_new_conv = best_by_val(new; filter_fn = r -> r.status == "converged")
    P = Dict{String, Any}()

    #---------------- SUMMARY TABLE ----------------
    summary_df = DataFrame(variant=String[], noise=Float64[], seeds=Int[], pct_lost=Float64[],
                           pct_blown=Float64[], pct_phi_to_0=Float64[], pct_phi_to_inf=Float64[],
                           pct_poisson=Float64[], pct_lower_bound=Float64[], pct_upper_bound=Float64[],
                           best_val_seed_blown=String[], n_best_blown=Int[],
                           median_ratio_converged=Float64[], median_ratio_best=Float64[],
                           median_nmse_beta_best=Float64[])
    for n in NOISE_LEVELS
        s = new[new.noise .== n, :]
        b = best_new[best_new.noise .== n, :]
        c = s[s.status .== "converged", :]
        nrow(s) == 0 && continue
        push!(summary_df, (vname, n, nrow(s), 100(1 - nrow(s) / (N_SEEDS * N_LOCATIONS)),
                           100mean(s.blown), 100mean(s.status .== "low"), 100mean(s.status .== "high"),
                           100mean(s.status .== "poisson"), 100mean(s.status .== "lower_bound"),
                           100mean(s.status .== "upper_bound"), "$(count(b.blown))/$(nrow(b))", count(b.blown),
                           isempty(c.phi_hat) ? NaN : median(c.phi_hat ./ n), median(b.phi_hat ./ n),
                           median(b.nmse_beta)))
    end

    #---------------- 01 PHI_HAT VS TRUE NOISE ----------------
    rng = MersenneTwister(1)
    phi_axis = (yscale=:log10, ylims=(PHI_DISPLAY_MIN / 10, 1e22), yticks=10.0 .^ (-160:20:0))

    p01 = plot(; xticks=XCAT, xlims=(0.4, NX + 0.6), xlabel="True noise level", ylabel="Estimated noise level φ̂",
               title=vlabel, legend=:bottomleft, phi_axis...)
    hline!(p01, [PHI_LO, PHI_HI], color=C_MUTED, ls=:dot, lw=1, label="Proposed clamp bounds (φ ∈ [10⁻³, 10])")
    for (st, lab, c) in statuses
        s = new[new.status .== st, :]
        scatter!(p01, jitter(rng, s.noise), clamp.(s.phi_hat, PHI_DISPLAY_MIN, 1e21), ms=1.5, msw=0,
                 alpha=st == "converged" ? 0.25 : 0.35, color=c, label="$(lab) ($(nrow(s)) seeds)")
    end
    for (i, n) in enumerate(NOISE_LEVELS)
        plot!(p01, [i - 0.38, i + 0.38], [n, n], color=C_TRUTH, lw=2, label=i == 1 ? "True noise level" : "")
    end
    P["01"] = p01

    #---------------- 02 BLOW-UP RATE BY NOISE LEVEL ----------------
    p02 = groupedbar(string.(summary_df.noise),
                     [summary_df.pct_phi_to_0 summary_df.pct_phi_to_inf summary_df.pct_poisson summary_df.pct_lower_bound summary_df.pct_upper_bound],
                     bar_position=:stack, color=[C_LOW C_HIGH C_POIS C_LOWB C_HIGHB], linecolor=:white, lw=1,
                     label=["φ̂ → 0  (r → ∞)" "φ̂ → ∞  (r → 0)" "Poisson regime" "At lower bound on φ" "At upper bound on φ"],
                     xlabel="True noise level", ylabel="% of seeds not converged", title=vlabel, legend=:topleft)
    for (i, v) in enumerate(summary_df.pct_blown)
        annotate!(p02, i - 0.5, v + 1.5, text(@sprintf("%.1f%%", v), 7, C_MUTED))
    end
    P["02"] = p02

    #---------------- 03 VALIDATION LOSS — BLOWN-UP VS CONVERGED ----------------
    p03a = plot(xticks=XCAT, xlabel="True noise level", ylabel="Rank of validation loss within location\n(0 = lowest, 1 = highest)",
                title=vlabel, legend=:top, legend_columns=length(statuses), ylims=(-0.02, 1.15))
    P["03a"] = status_boxes!(p03a, new, :val_rank, statuses)

    ex = sort(new[(new.noise .== TRACE_NOISE) .& (new.location .== EXAMPLE_LOCATION), :], :best_val_loss)
    negval = -ex.best_val_loss
    p03b = plot(yscale=:log10, xlabel="Seed, ordered by validation loss (lowest first)",
                ylabel="−(best validation loss)", legend=:topright, title=vlabel)
    for (st, lab, c) in statuses
        idx = findall((ex.status .== st) .& (negval .> 0))   # log axis needs a positive value
        isempty(idx) || scatter!(p03b, idx, negval[idx], ms=4, msw=0, color=c, label=lab)
    end
    P["03b"] = p03b

    #---------------- 04 WHAT MULTISTART SELECTS ----------------
    p04a = plot(; xticks=XCAT, xlims=(0.4, NX + 0.6), xlabel="True noise level", ylabel="φ̂ of the lowest-validation-loss seed",
                title="$(vlabel)  (numbers: locations where the top seed is not converged)",
                legend=:bottomleft, phi_axis...)
    hline!(p04a, [PHI_LO, PHI_HI], color=C_MUTED, ls=:dot, label="Proposed clamp bounds")
    for (st, lab, c) in statuses
        s = best_new[best_new.status .== st, :]
        nrow(s) == 0 && continue
        scatter!(p04a, jitter(rng, s.noise), clamp.(s.phi_hat, PHI_DISPLAY_MIN, 1e21), ms=3.5, msw=0, alpha=0.75, color=c, label=lab)
    end
    for (i, n) in enumerate(NOISE_LEVELS)
        plot!(p04a, [i - 0.38, i + 0.38], [n, n], color=C_TRUTH, lw=2, label=i == 1 ? "True noise level" : "")
        k = findfirst(==(n), summary_df.noise)
        isnothing(k) || annotate!(p04a, i, 1e15, text(summary_df.best_val_seed_blown[k], 8, C_MUTED))
    end
    P["04a"] = p04a

    # Replicates scripts/multistart.jl: feasible seeds sorted by val loss, drop the worst floor(n*MS)
    ms_pct = fill(NaN, length(MS_LIMITS), NX)
    for (j, n) in enumerate(NOISE_LEVELS), (k, ms) in enumerate(MS_LIMITS)
        pct = Float64[]
        for g in groupby(new[new.noise .== n, :], :location)
            sub = sort(filter(r -> coalesce(r.feasible, false), g), :best_val_loss)
            nr = nrow(sub); nr == 0 && continue
            kept = sub[1:(nr - floor(Int, nr * ms)), :]
            push!(pct, 100mean(kept.blown))
        end
        isempty(pct) || (ms_pct[k, j] = mean(pct))
    end
    p04b = plot(xticks=XCAT, xlabel="True noise level", ylabel="% of kept seeds that are not converged\n(mean over locations)",
                title=vlabel, legend=:right, ylims=(0, 105))
    for (k, ms) in enumerate(MS_LIMITS)
        plot!(p04b, 1:NX, ms_pct[k, :], color=MS_BLUES[k], lw=2, marker=:circle, ms=4, msw=0,
              label="MS = $(ms) (keep best $(round(Int, 100(1 - ms)))%)")
    end
    P["04b"] = p04b

    #---------------- 06 TRAINING-LOSS TRACES — EXAMPLE SEEDS ----------------
    # The losses are negative here (the constant sum log k! is dropped from the NLL), so plot
    # -loss on a log axis: higher = lower loss. Any non-negative value cannot be shown and is reported.
    trace_pool = sort(new[(new.noise .== TRACE_NOISE) .& (new.location .== EXAMPLE_LOCATION), :], :best_val_loss)
    p06a = plot(xlabel="Iteration", ylabel="−(training loss)  (log scale; higher = lower loss)", yscale=:log10,
                title=vlabel, legend=:bottomright)
    p06b = plot(xlabel="Iteration", ylabel="−(training loss)  (log scale)", yscale=:log10, legend=false,
                title=vlabel, xlims=(N_ADAM - 10, N_ADAM + 30))
    trace_values = Float64[]
    for (st, lab, c) in statuses
        k = findfirst(==(st), trace_pool.status)
        isnothing(k) && continue
        row = trace_pool[k, :]
        tl  = JLD2.load(seed_file(vsim, TRACE_NOISE, EXAMPLE_LOCATION, row.seed), "train_losses")
        l   = @sprintf("%s (seed %d, φ̂ = %.2g, %d LBFGS iterations)", lab, row.seed, row.phi_hat, length(tl) - N_ADAM - 1)
        n_nonneg = count(>=(0), tl)
        n_nonneg > 0 && println("Figure 06 ($(vlabel)): seed $(row.seed) has $(n_nonneg) non-negative training losses, not shown on the log axis")
        plot!(p06a, neg_loss(tl), color=c, lw=1.5, label=l)
        plot!(p06b, neg_loss(tl), color=c, lw=1.5, marker=:circle, ms=2.5, msw=0)
        append!(trace_values, filter(isfinite, neg_loss(tl)))
    end
    if !isempty(trace_values)
        # vline! spans ±Inf, which a log axis cannot draw, so give the switch line explicit finite ends
        y_switch = (minimum(trace_values) / 2, maximum(trace_values) * 2)
        for sp in (p06a, p06b)
            plot!(sp, [N_ADAM, N_ADAM], collect(y_switch), color=C_MUTED, ls=:dash, lw=1.5, label="Adam → LBFGS")
        end
    end
    P["06a"] = p06a
    P["06b"] = p06b

    #---------------- 07 RECOVERY — BLOWN-UP VS CONVERGED ----------------
    for (col, lab) in metrics
        P["07_$(col)"] = status_boxes!(plot(xticks=XCAT, yscale=:log10, xlabel="True noise level", ylabel=lab,
                                            title=vlabel, legend=:topleft), new, col, statuses)
    end
    # The same, for the seeds multistart keeps at MS_RECOVERY (replicates scripts/multistart.jl:
    # feasible seeds sorted by validation loss, drop the worst floor(n * MS))
    kept_ms = DataFrame()
    for g in groupby(new, [:noise, :location])
        sub = sort(filter(r -> coalesce(r.feasible, false), g), :best_val_loss)
        nr = nrow(sub); nr == 0 && continue
        append!(kept_ms, sub[1:(nr - floor(Int, nr * MS_RECOVERY)), :])
    end
    kept_statuses = filter(s -> s[1] in kept_ms.status, ALL_STATUSES)
    for (col, lab) in metrics
        P["07ms_$(col)"] = status_boxes!(plot(xticks=XCAT, yscale=:log10, xlabel="True noise level", ylabel=lab,
                                              title="$(vlabel)  ($(nrow(kept_ms)) kept seeds)", legend=:topleft),
                                         kept_ms, col, kept_statuses)
    end

    #---------------- 08 ACCURACY OF PHI_HAT WHEN IT CONVERGES ----------------
    p08 = plot(xticks=XCAT, yscale=:log10, xlabel="True noise level", ylabel="φ̂ / true noise level",
               title=vlabel, legend=false, yticks=([0.5, 1, 2, 5, 10, 100], ["0.5", "1", "2", "5", "10", "100"]))
    hline!(p08, [1.0], color=C_TRUTH, lw=1.5)
    for (i, n) in enumerate(NOISE_LEVELS)
        ratio = new[(new.noise .== n) .& .!new.blown, :phi_hat] ./ n
        isempty(ratio) && continue
        boxplot!(p08, fill(i, length(ratio)), ratio, bar_width=0.5, color=C_CONV, fillalpha=0.7,
                 linecolor=C_MUTED, ms=1.5, msw=0)
        annotate!(p08, i + 0.32, median(ratio), text(@sprintf("%.3f", median(ratio)), 7, C_MUTED, :left))
    end
    P["08"] = p08

    #---------------- 09 EXAMPLE RECOVERY ----------------
    population  = POPULATION[EXAMPLE_LOCATION]
    I_grid      = collect(range(0, 1; length=1000))
    true_beta_t = beta_exp(EXAMPLE_LOCATION, mu_true)
    true_beta_I = beta_exp(EXAMPLE_LOCATION, I_grid .* population)
    I_N_max     = maximum(mu_true) / population
    days        = 1:TRAIN_LENGTH

    for n in EXAMPLE_NOISES
        s   = sort(new[(new.noise .== n) .& (new.location .== EXAMPLE_LOCATION), :], :seed)
        y_n = synth(n)["infectious"][1:TRAIN_LENGTH]
        counts = join(["$(count(s.status .== st)) $(lowercase(lab))" for (st, lab, _) in statuses], ", ")
        pi_ = plot(xlabel="Day", ylabel="Infectious", title="$(vlabel): infectious\n($(counts))", legend=:topright)
        pt  = plot(xlabel="Day", ylabel="β(t)  (log scale)", title="$(vlabel): β(t)", legend=false, yscale=:log10)
        pI  = plot(xlabel="I/N", ylabel="β  (log scale)", title="$(vlabel): β vs I/N", legend=false, yscale=:log10)
        scatter!(pi_, days, y_n, ms=1.5, msw=0, color=C_DATA, label="Noisy data")
        for (st, lab, c) in statuses
            sel = first(s[s.status .== st, :], N_EXAMPLE_LINES)
            for (k, row) in enumerate(eachrow(sel))
                d = JLD2.load(seed_file(vsim, n, EXAMPLE_LOCATION, row.seed))
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
        P["09_$(n)"] = [pi_, pt, pI]
    end

    #---------------- 10 COMPARISON WITH THE OLD FIXED-r NOISE RUNS ----------------
    common = intersect(unique(old.location), unique(new.location))
    groups10 = [("Old runs, true r fixed", C_OLD, filter(r -> r.location in common, best_old)),
                ("Seed multistart picks", C_LOW, filter(r -> r.location in common, best_new)),
                ("Best converged seed", C_CONV, filter(r -> r.location in common, best_new_conv))]
    for (col, lab) in metrics
        sp = plot(xticks=XCAT, yscale=:log10, xlabel="True noise level", ylabel=lab, title=vlabel, legend=:topleft)
        for (i, n) in enumerate(NOISE_LEVELS), (g, (glab, gcol, gdf)) in enumerate(groups10)
            v = gdf[gdf.noise .== n, col]
            isempty(v) && continue
            boxplot!(sp, fill(i + (g - 2) * 0.26, length(v)), v, bar_width=0.24, color=gcol, fillalpha=0.75,
                     linecolor=C_MUTED, ms=1.5, msw=0, label=i == 1 ? glab : "")
        end
        P["10_$(col)"] = sp
    end
    P["n_common"] = length(common)

    return summary_df, best_new, P
end

#========================================================
BUILD THE PANELS FOR EVERY VARIANT ON DISK
=========================================================#

present  = filter(v -> v[1] in SHOW_VARIANTS && any(n -> isdir(joinpath(SIMS_ROOT, sim_name(v[2], n))), NOISE_LEVELS), VARIANTS)
summaries  = DataFrame[]
best_seeds = Dict{String, DataFrame}()
panels     = Dict{String, Dict{String, Any}}()
for (vname, suffix, vlabel, _) in present
    s, b, P = variant_panels(vname, suffix, vlabel)
    push!(summaries, s)
    best_seeds[vname] = b
    panels[vname]     = P
end
all_summary = vcat(summaries...)
CSV.write(joinpath(COMPARE_DIR, "summary_by_variant_noise.csv"), all_summary)
show(all_summary; allrows=true); println()

#========================================================
FIGURES 01–10: THE SAME PANEL FOR EVERY VARIANT, SIDE BY SIDE

2×2 for four variants (2×3 once a fifth is on disk). The y-axis is shared
across the variants so the panels compare directly, except for the raw losses
(03b, 06), where the artificially low losses of the φ̂ → 0 seeds would flatten
the other variants. Each panel keeps its own legend, because the statuses (and
seed counts) differ between variants; where the legend is identical it is only
drawn in the first panel.
=========================================================#

const NV        = length(present)
const GRID      = (2, cld(NV, 2))
const PANEL_W   = 1000
const PANEL_H   = 720

function side_by_side(key, name, title; link=:all, legend_first_only=false)
    sps = [panels[v[1]][key] for v in present]
    if legend_first_only
        for sp in sps[2:end]
            plot!(sp, legend=false)
        end
    end
    p = plot(sps..., layout=GRID, link=link, size=(PANEL_W * GRID[2], PANEL_H * GRID[1]),
             left_margin=12Plots.mm, bottom_margin=16Plots.mm, top_margin=10Plots.mm,
             plot_title=title, plot_titlefontsize=13)
    save(p, COMPARE_DIR, name)
end

side_by_side("01",  "01_phi_hat_vs_true_noise.png",
             "φ̂ for every seed (51 locations × 100 seeds per noise level)")
# Shared y-limit for the bar heights, with room for the labels
ymax02 = 1.15maximum(all_summary.pct_blown)
for v in present
    plot!(panels[v[1]]["02"], ylims=(0, ymax02))
end
side_by_side("02",  "02_blowup_rate_by_noise.png",
             "Seeds whose φ̂ left the range [10⁻³, 10], are held at a bound on φ, or stayed in the Poisson regime")
side_by_side("03a", "03a_val_loss_rank_blown_vs_converged.png",
             "Rank of each seed's validation loss among its location's seeds")
side_by_side("03b", "03b_val_loss_example.png",
             "$(EXAMPLE_LOCATION), noise = $(TRACE_NOISE): raw validation losses (y-axis not shared)"; link=:none)
side_by_side("04a", "04a_multistart_top_seed_phi.png",
             "φ̂ of the seed multistart ranks first, one point per location")
side_by_side("04b", "04b_multistart_kept_not_converged.png",
             "Share of non-converged seeds in the multistart-kept set"; legend_first_only=true)
side_by_side("06a", "06a_training_traces_example.png",
             "$(EXAMPLE_LOCATION), true noise = $(TRACE_NOISE): training loss of the lowest-val-loss seed of each status (y-axis not shared)";
             link=:none)
side_by_side("06b", "06b_training_traces_zoom.png",
             "Zoom on iterations $(N_ADAM - 10)–$(N_ADAM + 30), the Adam → LBFGS switch (markers = iterations; y-axis not shared)";
             link=:none)
for (col, lab) in metrics
    side_by_side("07_$(col)", "07_recovery_$(col).png",
                 "Does a non-converged φ̂ also damage the fit? $(lab), all seeds, 51 locations")
    side_by_side("07ms_$(col)", "07_recovery_MS=$(MS_RECOVERY)_$(col).png",
                 "$(lab): seeds kept by multistart at MS = $(MS_RECOVERY) (best $(round(Int, 100(1 - MS_RECOVERY)))% by validation loss per location; " *
                 "y-axis not shared, as the control's φ̂ → 0 seeds reach ~1e60)"; link=:none)
end
side_by_side("08",  "08_phi_accuracy_converged.png",
             "Converged seeds: estimated / true noise level (label = median)")

# 09: one row per variant, columns infectious / β(t) / β vs I/N, one figure per example noise level
for n in EXAMPLE_NOISES
    sps = reduce(vcat, [panels[v[1]]["09_$(n)"] for v in present])
    p09 = plot(sps..., layout=(NV, 3), size=(1800, 480 * NV), left_margin=10Plots.mm, bottom_margin=6Plots.mm,
               plot_title="$(EXAMPLE_LOCATION), noise = $(n): up to $(N_EXAMPLE_LINES) seeds per group " *
                          "(black = truth; dotted line = largest I/N in training; β below 10⁻⁶ drawn at 10⁻⁶)",
               plot_titlefontsize=13)
    save(p09, COMPARE_DIR, "09_example_recovery_$(EXAMPLE_LOCATION)_noise=$(n).png")
end

n_common = panels[present[1][1]]["n_common"]
for (col, lab) in metrics
    side_by_side("10_$(col)", "10_comparison_old_fixed_r_$(col).png",
                 "Lowest-validation-loss seed per location vs the old fixed-r runs, $(n_common) shared locations " *
                 "(old runs also differ in solver, loss and code)"; legend_first_only=true)
end

#========================================================
11 SIDE-BY-SIDE SUMMARY OF THE VARIANTS
=========================================================#

# One line per variant, across noise levels. Each variant gets its own marker and a small
# horizontal offset, so variants with identical values (e.g. all at 0) stay visible
const VARIANT_MARKERS = [:circle, :rect, :diamond, :utriangle, :dtriangle, :hexagon, :star5]
function variant_lines(col, ylab, title)
    sp = plot(xticks=XCAT, xlabel="True noise level", ylabel=ylab, title=title, legend=false)
    for (j, (vname, _, vlabel, c)) in enumerate(present)
        s = all_summary[all_summary.variant .== vname, :]
        dx = 0.08 * (j - (NV + 1) / 2)
        plot!(sp, noise_index.(s.noise) .+ dx, s[!, col], color=c, lw=2, marker=VARIANT_MARKERS[j], ms=5, msw=0, label=vlabel)
    end
    return sp
end

# One box per variant at every noise level, for the seed multistart picks
function variant_boxes(col, ylab, title)
    sp = plot(xticks=XCAT, xlabel="True noise level", ylabel=ylab, title=title, legend=false, yscale=:log10)
    w = 0.8 / NV
    for (i, n) in enumerate(NOISE_LEVELS), (j, (vname, _, vlabel, c)) in enumerate(present)
        b = best_seeds[vname]
        v = b[b.noise .== n, col]
        isempty(v) && continue
        boxplot!(sp, fill(i + (j - (NV + 1) / 2) * w, length(v)), v, bar_width=0.9w, color=c, fillalpha=0.75,
                 linecolor=C_MUTED, ms=1.5, msw=0, label="")
    end
    return sp
end

p11a = variant_lines(:pct_lost,       "% of seeds lost",   "Seeds with no result (LBFGS crash / timeout)")
# Seeds held at a bound (bounded only) are counted with the direction they were heading
all_summary.pct_low_any  = all_summary.pct_phi_to_0   .+ all_summary.pct_lower_bound
all_summary.pct_high_any = all_summary.pct_phi_to_inf .+ all_summary.pct_upper_bound
p11b = variant_lines(:pct_low_any,  "% of seeds", "φ̂ → 0  (r → ∞), or at the lower bound on φ")
p11c = variant_lines(:pct_high_any, "% of seeds", "φ̂ → ∞  (r → 0), or at the upper bound on φ")
p11d = variant_lines(:n_best_blown,   "Locations (of 51)", "Multistart's top seed is not converged")
p11f = variant_boxes(:nmse_beta,      "NMSE β(t) vs truth", "β recovery of multistart's top seed")
# Legend in the sixth slot: identity is shown by name, not colour alone
p11leg = plot(framestyle=:none, legend=:left, legendfontsize=12, showaxis=false, grid=false)
for (j, (_, _, vlabel, c)) in enumerate(present)
    plot!(p11leg, [NaN], [NaN], color=c, lw=3, marker=VARIANT_MARKERS[j], ms=6, msw=0, label=vlabel)
end
p11 = plot(p11a, p11b, p11c, p11d, p11f, p11leg,
           layout=(2, 3), size=(1800, 1100),
           left_margin=8Plots.mm, bottom_margin=6Plots.mm,
           plot_title="Loss variants for the jointly estimated noise level (beta_exp, 51 locations × 100 seeds; " *
                      "not converged = φ̂ outside [10⁻³, 10], at a bound, or Poisson regime)",
           plot_titlefontsize=12)
save(p11, COMPARE_DIR, "11_variant_comparison.png")

println("\nAll figures saved to $(COMPARE_DIR)")
