
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
DEFINE ODE SYSTEM FOR BASELINE
=========================================================# 

function make_seird_baseline(sigma, gamma, delta)
    function seird_baseline!(du, u, p, t)
        S, E, I, R, D = u
        N = S + E + I + R
        if N <= 0
            du .= 0.0
            return
        end
        du[1] = -p.beta0 * S * I / N
        du[2] = p.beta0 * S * I / N - sigma * E      
        du[3] = sigma * E - (gamma + delta) * I  
        du[4] = gamma * I                           
        du[5] = delta * I
    end
    return seird_baseline!
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
    # Fixed, high-precision ground truth — shared by every solver arm being
    # compared, so the comparison is "which solver best recovers the same
    # dynamics", not "which solver best recovers its own synthetic realisation".
    sol = solve(prob, Vern7(), saveat=1.0, dense=false, abstol=1e-12, reltol=1e-12)

    return sol
end

#========================================================
DEFINE UDE MODEL
=========================================================# 
function make_seird_nn(beta_network, st, sigma, gamma, delta, population)

    function seird_nn!(du, u, p, t)
        S, E, I, R, D = u
        N = S + E + I + R

        if N <= 0
            du .= 0.0
            return
        end

        nn_input = [I / population]
        beta = beta_network(nn_input, p.nn_params, st)[1][1]

        du[1] = -beta * S * I / N
        du[2] = beta * S * I / N - sigma * E
        du[3] = sigma * E - (gamma + delta) * I
        du[4] = gamma * I
        du[5] = delta * I
    end
    return seird_nn!
end

#========================================================
DEFINE BEHAVIOURAL UDE MODEL 1
=========================================================# 
function make_seird_nn_behaviour_model1(contact_network, st, sigma, gamma, delta, population; dI_scale=1.0)
    

    function seird_nn!(du, u, p, t)
        S, E, I, R, D, c = u
        N = S + E + I + R

        if N <= 0
            du .= 0.0
            return
        end

        dI = sigma * E - (gamma + delta) * I
        # dI_scale = 1 (raw) or the initial population N(0)
        nn_input = [I / population, dI / dI_scale, c]

        rho = 1 / (1 + exp(-p.logit_rho))
        du[1] = -rho * c * S * I / N
        du[2] =  rho * c * S * I / N - sigma * E
        du[3] = sigma * E - (gamma + delta) * I
        du[4] = gamma * I
        du[5] = delta * I
        # New equation for contact dynamics
        du[6] = c * contact_network(nn_input, p.nn_params, st)[1][1]  

    end
    return seird_nn!
end

#========================================================
MAKE PREDICTION USING UDE MODEL
=========================================================# 

# Function to make UDE predictions - default return infections only
function make_predict_ude(prob, train_length, solver; states=3)
    function predict_ude(p_all, u0)
        new_prob = remake(prob, p = p_all, u0 = u0)
        sol = solve(new_prob, solver, saveat=1.0, dense=false, verbose=false)
        if sol.retcode != ReturnCode.Success || length(sol.t) < train_length
            return nothing
        end
        # Array(sol) first: Zygote has no adjoint for slicing an ODESolution with a colon
        return Array(sol)[states, 1:train_length]
    end
    return predict_ude
end

#========================================================
DEFINE UDE MODEL
=========================================================# 
function make_seird_sr(mach, r, sigma, gamma, delta, population)

    function seird_sr!(du, u, p, t)
        S, E, I, R, D = u
        N = S + E + I + R

        if N <= 0
            du .= 0.0
            return
        end

        nn_input = reshape([I / population], 1, :)
        beta = predict(mach, (data=nn_input, idx=r.best_idx))[1]

        du[1] = -beta * S * I / N
        du[2] = beta * S * I / N - sigma * E
        du[3] = sigma * E - (gamma + delta) * I
        du[4] = gamma * I
        du[5] = delta * I
    end
    return seird_sr!
end