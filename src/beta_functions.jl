#=============================================================
EXPONENTIAL BETA FUNCTION
==============================================================# 

function beta_exp(location, obs)
    
    delta = DELTA[location]
    R0_reproduction = R0_REPRODUCTION[location]
    zeta = ZETA[location]

    # Derive other parameters
    # Infectious period of 10 days represented by recovery rate gamma
    gamma = 1/10
    beta0 = R0_reproduction * (gamma + delta)


    true_beta = beta0 .* exp.(-zeta .* delta .* obs)


    return true_beta

end 

#=============================================================
RATIONAL BETA FUNCTION
==============================================================# 

function beta_rational(location, obs)
    
    delta = DELTA[location]
    R0_reproduction = R0_REPRODUCTION[location]
    zeta = ZETA[location]

    # Derive other parameters
    # Infectious period of 10 days represented by recovery rate gamma
    gamma = 1/10
    beta0 = R0_reproduction * (gamma + delta)

    true_beta = beta0 ./ (1 .+ zeta .* delta .* obs)

    return true_beta

end

#=============================================================
MIXED BETA FUNCTION
==============================================================# 

function beta_mixed(location, obs)
    
    delta = DELTA[location]
    R0_reproduction = R0_REPRODUCTION[location]
    zeta = ZETA[location]

    # Derive other parameters
    # Infectious period of 10 days represented by recovery rate gamma
    gamma = 1/10
    beta0 = R0_reproduction * (gamma + delta)

    true_beta = beta0 .* exp.(-zeta .* delta .* obs) ./ (1 .+ zeta .* delta .* obs)

    return true_beta

end

#=============================================================
BASELINE CONTACTS
==============================================================# 

function baseline_contacts(locations, seed)
    path = datadir("exp_raw", "estimated_ground_truth_parameters.jld2")
    ground_truth = JLD2.load(path)

    # Set seed for reproducibility
    rng = Xoshiro(seed)
    # Uniform distribution for c0 values between 6.94 and 10.64
    c0_dist = Uniform(6.94, 10.64)
    # Sort the locations to ensure consistent ordering
    C0 = Dict{String, Float32}(location => rand(rng, c0_dist) for location in sort(collect(locations)))
    ground_truth["C0"] = C0
    JLD2.save(path, ground_truth)
    return C0

end

#=============================================================
EXPONENTIAL CONTACTS
==============================================================# 

function contacts_exp(location, obs)
    
    delta = DELTA[location]
    zeta = ZETA[location]
    c0 = C0[location]

    true_contacts = c0 .* exp.(-zeta .* delta .* obs)


    return true_contacts

end 

#=============================================================
RATIONAL CONTACTS FUNCTION
==============================================================# 

function contacts_rational(location, obs)
    
    delta = DELTA[location]
    zeta = ZETA[location]
    c0 = C0[location]

    true_contacts = c0 ./ (1 .+ zeta .* delta .* obs)

    return true_contacts

end

#=============================================================
MIXED CONTACTS FUNCTION
==============================================================# 

function contacts_mixed(location, obs)
    
    delta = DELTA[location]
    zeta = ZETA[location]  
    c0 = C0[location]

    true_contacts = c0 .* exp.(-zeta .* delta .* obs) ./ (1 .+ zeta .* delta .* obs)

    return true_contacts

end