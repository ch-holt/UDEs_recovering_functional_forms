
module UDE_FUNCTIONAL_FORMS

using DrWatson
using JLD2
using ComponentArrays
using Zygote
using Optimization
using Lux
using DifferentialEquations
using Optimisers
using Optim
using MLJ
using SymbolicRegression
using SymbolicUtils
using DataFrames
using Plots
using ForecastBaselines
using Statistics
using Random
using Distributions

const _ground_truth = JLD2.load(datadir("exp_raw", "estimated_ground_truth_parameters.jld2"))
const POPULATION = _ground_truth["POPULATION"]
const PREVALENCE = _ground_truth["PREVALENCE"]
const R0_REPRODUCTION = _ground_truth["R0_REPRODUCTION"]
const DELTA = _ground_truth["DELTA"]
const ZETA = _ground_truth["ZETA"]

include("ode_model.jl")
include("neural_network.jl")
include("loss.jl")
include("beta_functions.jl")
include("utils.jl")
include("data_loading.jl")
include("training.jl")
include("symbolic_regression.jl")
include("plotting.jl")

export make_seird_functional, run_seird_functional_form, make_seird_nn, make_predict_ude, make_seird_sr
export build_neural_network
export loss_ude, regularisation, loss_nmse, combined_loss_ude_adam, combined_loss_ude_lbfgs
export beta_exp, beta_rational, beta_mixed
export add_neg_bin_noise, extract_best_ude, format_equation_sigfigs, extract_ude, get_seed_folders, find_peak
export train_ude_single_dataset, train_ude_multiple_datasets
export load_trajectories
export build_sr_inputs, run_symbolic_regression
export plot_ensemble_summary
export POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

end