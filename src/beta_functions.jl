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