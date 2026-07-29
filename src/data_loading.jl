#=============================================================
FUNCTION TO LOAD TRAJECTORIES BASED ON LOCATION AND BETA FUNCTION
==============================================================# 

function load_trajectories(locations, beta_function)
    root = datadir("exp_pro","synthetic_data","synthetic_trajectories_$(beta_function)")
    trajectories = []
    for location in locations
        filename = "synthesised_$(location).jld2"
        dataset = JLD2.load(joinpath(root, filename))
        varying_p = ComponentArray(
            population = dataset["varying_p"]["population"],
            prevalence = dataset["varying_p"]["prevalence"],
            delta = dataset["varying_p"]["delta"],
            R0_reproduction = dataset["varying_p"]["R0_reproduction"],
            zeta = dataset["varying_p"]["zeta"]
        )
        push!(trajectories, (filename=filename, data=dataset["infectious"], days=dataset["days"], varying_p=varying_p))
    end
    println("Loaded $(length(trajectories)) trajectories into memory")
    return trajectories
end