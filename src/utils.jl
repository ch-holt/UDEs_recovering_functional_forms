#=============================================================
EXTRACT THE UDE
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

    folder = results_list[best_idx].fname

    return I_nn, best_results, folder
end

function get_seed_folders(root, multistart, MS_limit)
        # Find all seed folders kept
    # If multistart then only use seeds_to_keep
    if !multistart
        seed_folders   = filter(f -> occursin(r"simulation_seed=", f), readdir(root))
    else
        seeds_to_keep = JLD2.load(joinpath(root, "seeds_to_keep_MS=$(MS_limit).jld2"))["seeds_to_keep"]
        seed_folders = ["simulation_seed=$(s)" for s in seeds_to_keep]
    end
    println("Found $(length(seed_folders)) seed simulations in $(root)")
    return seed_folders
end

#=============================================================
FIND WHEN TRAJECTORY GOES FLAT FOR TRAIN/VAL
==============================================================# 

function find_flat_start(data; window::Int=14, rel_threshold::Float64=0.01)

    n = length(data)
    peak = maximum(data)

    # Rolling std over windows of length `window`, normalised by the peak
    rel_std = fill(NaN, n)
    for i in 1:(n - window + 1)
        rel_std[i] = std(view(data, i:(i + window - 1))) / peak
    end

    # Last window start whose relative std is still above threshold
    last_active = findlast(x -> !isnan(x) && x >= rel_threshold, rel_std)

    # Series never settles into a flat tail
    isnothing(last_active) && return n

    # Flatness begins right after that window ends
    flat_start = last_active + window - 1
    return min(flat_start, n)
end

function find_peak(data)
    return argmax(data)
end


#=============================================================
ADD NEGATIVE BINOMIAL NOISE
==============================================================# 


function add_neg_bin_noise(data, noise_level)
    # Fix a seed for reproducibility
    rng=Random.seed!(1234)
    if noise_level == 0
        return copy(data), Inf
    end
    r = 1 / noise_level^2
    p = clamp.(r ./ (r .+ data), nextfloat(0.0), 1.0)
    noisy_data = rand.(rng, NegativeBinomial.(r, p))
    return noisy_data, r
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