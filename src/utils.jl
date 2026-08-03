#=============================================================
EXTRACT THE UDE WITH THE LOWEST NMSE
==============================================================# 

function extract_ude(root, obs, filename)

    # Extract results, predictions and losses
    results = JLD2.load(joinpath(root, filename, "results.jld2"))
    pred = results["prediction"]
    # Extract the predicted infectious trajectory for the training data
    i_traj = pred[3, 1:length(obs)]

    # Extract the data but convert to a 1 x N matrix
    I_nn = reshape(i_traj, 1, :)

    return I_nn, results
end

function extract_best_ude(root, obs, multistart, MS_limit)

    seed_folders = get_seed_folders(root, multistart, MS_limit)
    

    # Collect results from all simulations
    results_list = []
    for folder in seed_folders

        I_nn, results = extract_ude(root, obs, folder)

        nmse = loss_nmse(vec(I_nn), obs)
        push!(results_list, (nmse=nmse, fname=folder, I_nn=I_nn, results=results))
    end

    # Find the simulation with the lowest NMSE
    best_idx = argmin(r.nmse for r in results_list)

    # Extract the data but convert to a 1 x N matrix
    I_nn = results_list[best_idx].I_nn

    # Extract the NN parameters from the best simulation
    best_results = results_list[best_idx].results

    return I_nn, best_results
end

function get_seed_folders(root, multistart, MS_limit)
        # Find all seed folders kept
    # If multistart then only use seeds_to_keep
    if !multistart
        seed_folders   = filter(f -> occursin(r"simulation_v1+_seed=", f), readdir(root))
    else
        seeds_to_keep = JLD2.load(joinpath(root, "seeds_to_keep_MS=$(MS_limit).jld2"))["seeds_to_keep"]
        seed_folders = ["simulation_v1_seed=$(s)" for s in seeds_to_keep]
    end
    println("Found $(length(seed_folders)) seed simulations in $(root)")
    return seed_folders
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