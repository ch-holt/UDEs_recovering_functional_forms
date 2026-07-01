using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using SymbolicUtils
using SymbolicRegression
using DynamicExpressions
using Symbolics
using JLD2

    # Then do LBFGS optimisation
    adtype = Optimization.AutoZygote()
    optfunc   = Optimization.OptimizationFunction(
                 (theta, _) -> Functions.combined_loss_ude(theta, predict_ude, trajectories)[1],
                 adtype)
    optprob = Optimization.OptimizationProblem(optfunc, best_nn_params)

    iter_lbfgs = Ref(0)
    res = Optimization.solve(
        optprob,
        Optim.LBFGS(m=10),
        callback = (state, l) -> begin
            iter_lbfgs[] += 1
            push!(total_losses, l)

            if l < best_loss
                best_loss = l
                best_nn_params = state.u
            end

            iter_lbfgs[] % 50 == 0 && println("LBFGS iter $(iter_lbfgs[]): $l")
            return false
        end,
        maxiters = maxiters_lbfgs
    )

    final_loss = Functions.combined_loss_ude(res.u, predict_ude, trajectories)[1]

    if final_loss < best_loss
        best_nn_params = res.u
    end


# Plot beta rational vs beta exponential
beta_exp_values = [Functions.beta_exp(location, i) for i in 1:1000]
beta_rat_values = [Functions.beta_rational(location, i) for i in 1:1000]

pl = plot(1:1000, beta_exp_values, label="beta exponential", xlabel="I(t)", ylabel="beta(t)", title="Beta functional forms")
plot!(1:1000, beta_rat_values, label="beta rational", ls=:dash)
display(pl)

diff = beta_exp_values .- beta_rat_values
pl_diff = plot(1:1000, diff, label="Difference (exp - rational)", xlabel="I(t)", ylabel="Difference in beta(t)", title="Difference between beta functional forms")
display(pl_diff)

dataset = JLD2.load(datadir("synthetic_trajectories_rational", "synthesised_MA.jld2"))

# Extract infectious individuals and days from the dataset
data = dataset["infectious"]

# Plot beta rational vs beta exponential
beta_exp_values = Functions.beta_exp(location, data)
beta_rat_values = Functions.beta_rational(location, data)

pl = plot(data, beta_exp_values, label="beta exponential", xlabel="I(t)", ylabel="beta(t)", title="Beta functional forms")
plot!(data, beta_rat_values, label="beta rational", ls=:dash)
display(pl)