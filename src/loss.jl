#=============================================================
NMSE LOSS FUNCTION
==============================================================#

function loss_nmse(pred, data; norm=true)

    # Mean squared error
    normalising_factor = norm ? maximum(data) - minimum(data) : 1.0
    nmse = sum((pred ./ normalising_factor .- data ./ normalising_factor).^2)/length(data)
    return nmse
end

# Stirling remainder: loggamma(z) = (z - 1/2) log z - z + log(2π)/2 + stirling_tail(z), accurate to ~1e-10 for z >= 10
# (written in powers of w = 1/z, so neither it nor its gradient overflows for very large z)
stirling_tail(z) = (w = 1 / z; w * (1 / 12 - w^2 * (1 / 360 - w^2 / 1260)))

# loggamma(y + r) - loggamma(r) - y log r, without the r-sized cancellation of the direct form for large r
lgamma_ratio(y, r) = r < 10 ? loggamma(y + r) - loggamma(r) - y * log(r) :
                              (r + y - 0.5) * log1p(y / r) - y + stirling_tail(r + y) - stirling_tail(r)

# Negative binomial NLL, rearranged so no term grows with r.
# Accurate in Float64 for any r and tends continuously to the Poisson NLL as r -> infinity.
function loss_negbin_stable(pred, data, r)
    pred = max.(pred, eps())
    r = min(r, 1e100)   # beyond ~1e154 parts of the gradient underflow; at 1e100 the NLL already equals the Poisson NLL in Float64
    return sum(-lgamma_ratio.(data, r) .+ (r .+ data) .* log1p.(pred ./ r) .- data .* log.(pred))
end

#=============================================================
LOSS FUNCTION FOR SINGLE DATASET
==============================================================#

# Loss function for infection observations
function loss_infections(p_all, pred_I, data, tpts, noise)
    # If no noise use NMSE, if noise use negbin NLL
    noise == 0 ? loss_nmse(pred_I[tpts], data[tpts]) :
                # limit log_phi before exp so r <= exp(230) ~ 1e100 (see loss_negbin_stable) and the gradient is 0, not NaN, beyond it
                 loss_negbin_stable(pred_I[tpts], data[tpts], exp(-2 * max(p_all.log_phi, -115.0)))
end

# Loss function for behavioural observations
# Note: will need to change this when we introduce noisy data
# Note: need to think about whether we normalise
loss_contacts(pred_c, data, tpts) = loss_nmse(pred_c[tpts], data[tpts], norm=true)

function loss_ude(p_all, predict_ude, data, u0, tpts, noise)
    pred = predict_ude(p_all, u0)
    isnothing(pred) && (println("ODE solve failed"); return Inf)
    return loss_infections(p_all, pred, data, tpts, noise)
end

function loss_ude_behaviour(p_all, predict_ude, data_I, data_c, u0, tpts, noise)
    pred = predict_ude(p_all, u0)
    isnothing(pred) && (println("ODE solve failed"); return Inf)
    return loss_infections(p_all, pred[3, :], data_I, tpts, noise) + loss_contacts(pred[6, :], data_c, tpts)
end

function regularisation(nn_params)
    # L2 penalty on NN weights (regularisation)
    l2_penalty = 1e-6 * sum(abs2, nn_params)

    return l2_penalty
end

