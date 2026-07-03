
#========================================================
DEFINE ODE SYSTEM FOR SYNTHETIC DATA GENERATION
=========================================================# 

function make_seird_functional(beta_function, location, sigma, gamma)
    function seird_functional!(du, u, p, t)
        S, E, I, R, D = u
        N = S + E + I + R
        if N <= 0
            du .= 0.0
            return
        end
        beta = beta_function(location, u[3])
        du[1] = -beta * S * I / N
        du[2] = beta * S * I / N - sigma * E      
        du[3] = sigma * E - (gamma + p.delta) * I  
        du[4] = gamma * I                           
        du[5] = p.delta * I
    end
    return seird_functional!
end

#========================================================
DEFINE FUNCTION TO RUN ODE SYSTEM FOR A BETA FORM
=========================================================# 

function run_seird_functional_form(beta_function, location, fixed_p, varying_p, obs_length)

    # Retrieve fixed parameters
    sigma = fixed_p.sigma
    gamma = fixed_p.gamma
    E0 = fixed_p.E0
    R0_recovered = fixed_p.R0_recovered 
    D0 = fixed_p.D0

    # Retrieve estimated parameters
    population = varying_p.population
    prevalence = varying_p.prevalence
    delta = varying_p.delta
    R0_reproduction = varying_p.R0_reproduction
    zeta = varying_p.zeta

    # Derive other parameters
    beta0 = R0_reproduction * (gamma + delta)
    I0 = max(1.0, prevalence * population)
    S0 = population - E0 - I0 - R0_recovered - D0

    # Define initial state
    u0 = [S0, E0, I0, R0_recovered, D0]

    # Define parameters for the ODE solver
    p = ComponentArray(
        beta0 = beta0,
        delta = delta,
        sigma = sigma,
        gamma = gamma,
        zeta = zeta
    )

    # Align solver times with saved labels days = 1:obs_length
    tspan = [1, obs_length]

    seird_functional! = make_seird_functional(beta_function, location, sigma, gamma)
    prob = ODEProblem(seird_functional!, u0, tspan, p)
    sol = solve(prob, Rosenbrock23(), saveat=1.0, dense=false)
    
    return sol
end

#========================================================
DEFINE NUMBER OF NEURAL NETWORK INPUTS
=========================================================# 

nn_inputs(p, I, ::Val{1}) = [I / p.population]

function nn_inputs(p, I, ::Val{4})
    delta_norm = (log(p.delta) - log(1e-6)) / (log(1e-2) - log(1e-6))
    beta0_norm = (p.R0_reproduction - 1.2) / (6.0 - 1.2)
    zeta_norm  = p.zeta / 0.05
    return [beta0_norm, zeta_norm, delta_norm, I / p.population]
end

#========================================================
DEFINE UDE MODEL
=========================================================# 
function make_seird_nn(beta_network, st, sigma, gamma, input_size::Int)
    valn = Val(input_size)
    function seird_nn!(du, u, p, t)
        S, E, I, R, D = u
        N = S + E + I + R

        if N <= 0
            du .= 0.0
            return
        end

        nn_input = nn_inputs(p, I, valn)
        beta = beta_network(nn_input, p.nn_params, st)[1][1]

        du[1] = -beta * S * I / N
        du[2] = beta * S * I / N - sigma * E
        du[3] = sigma * E - (gamma + p.delta) * I
        du[4] = gamma * I
        du[5] = p.delta * I
    end
    return seird_nn!
end

#========================================================
MAKE PREDICTION USING UDE MODEL
=========================================================# 

function make_predict_ude(prob, train_length)
    function predict_ude(p_all, u0)
        new_prob = remake(prob, p = p_all, u0 = u0)
        sol = solve(new_prob, Rosenbrock23(), saveat=1.0, dense=false)
        if sol.retcode != ReturnCode.Success || length(sol.t) < train_length
            return nothing
        end
        return sol[3, 1:train_length]
    end
    return predict_ude
end