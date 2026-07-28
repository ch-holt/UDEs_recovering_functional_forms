#=============================================================
EXTRACT THE UDE WITH THE LOWEST NMSE
==============================================================# 

function extract_best_ude(sim_name, location, obs, population)
    # Define the root file path
    root = DrWatson.datadir("exp_pro","sims",  "ude_single",sim_name, "synthesised_$(location)")

    # Collect results from all simulations
    results_list = []
    for filename in readdir(root)
        # Only include directories
        if isdir(joinpath(root, filename))
            # Extract results, predictions and losses
            SR_results = JLD2.load(joinpath(root, filename, "results.jld2"))
            pred = SR_results["prediction"]
            # Extract the predicted infectious trajectory for the training data
            i_traj = pred[3, 1:length(obs)]
            nmse = loss_nmse(i_traj, obs)
            push!(results_list, (nmse=nmse, fname=filename, i_traj=i_traj))
        end
    end

    # Find the simulation with the lowest NMSE
    best_idx = argmin(r.nmse for r in results_list)
    best_fname = results_list[best_idx].fname

    # Extract the data but convert to a 1 x N matrix
    I_nn = reshape(results_list[best_idx].i_traj, 1, :)

    # Extract the NN parameters from the best simulation
    best_results = JLD2.load(joinpath(root, best_fname, "results.jld2"))

    return I_nn, best_results
end

#=============================================================
ADD GAUSSIAN NOISE
==============================================================# 

function add_gaussian_noise(noise_SD, data, rng)

    # add Gaussian noise
    sd = noise_SD * max(0.5, maximum(data))
    noise = randn(rng, length(data)) .* sd
    noisy_beta = data .+ noise

    # Remove negative values
    noisy_beta = max.(noisy_beta, 0.0)

    return noisy_beta
end

#=============================================================
FORMAT EQUATION INTO STRING WITH 3SIGFIGS
==============================================================# 

function format_equation_sigfigs(equation_text::AbstractString)
    number_pattern = r"(?<![A-Za-z_])[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
    return replace(equation_text, number_pattern => match -> begin
        number = parse(Float64, match)
        string(round(number, sigdigits=3))
    end)
end