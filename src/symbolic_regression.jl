#========================================================
GENERATE SYMBOLIC REGRESSION INPUTS/OUTPUTS
=========================================================# 

function build_sr_inputs(p_all, norm_i_traj, number_of_nn_inputs::Int)

    # We do SR on this grid
    I_grid = collect(range(0, 1; length=1000))
    valn = Val(number_of_nn_inputs)

    # Build matrix inputs for the NN (each column is one I value)
    # Multiply by popultion because nn_input normalises
    # Single dataset just has I_grid
    nn_input_1000  = hcat([nn_inputs(p_all, i*p_all.population, valn) for i in I_grid]...)
    # Multiple datasets has I_grid, beta0_norm, zeta_norm, delta_norm
    nn_input_days  = hcat([nn_inputs(p_all, i*p_all.population, valn) for i in norm_i_traj]...)

    # Build SR inputs - the same as NN inputs but formatted as a dataframe
    if number_of_nn_inputs == 1
        SR_input_1000 = reshape(I_grid, :, 1)
    else
        SR_input_1000 = DataFrame(
            I     = I_grid,
            beta0 = fill(nn_input_1000[1,1], 1000),
            zeta  = fill(nn_input_1000[2,1], 1000),
            delta = fill(nn_input_1000[3,1], 1000)
        )
    end

    if number_of_nn_inputs == 1
        SR_input_days = reshape(norm_i_traj, :, 1)
    else
        SR_input_days = DataFrame(
            I     = norm_i_traj,
            beta0 = fill(nn_input_days[1,1], length(norm_i_traj)),
            zeta  = fill(nn_input_days[2,1], length(norm_i_traj)),
            delta = fill(nn_input_days[3,1], length(norm_i_traj))
        )
    end

    return SR_input_1000, SR_input_days, nn_input_1000, nn_input_days, I_grid
end

#========================================================
RUN SYMBOLICREGRESSION.JL
=========================================================# 

function run_symbolic_regression(SR_input, SR_output, output_dir; seed=seed)
    
    model = SRRegressor(
        niterations=1,
        binary_operators=[+, -, *, /],
        unary_operators=[exp],
        maxsize = 20,
        output_directory = output_dir,
        # Make results reproducible by disabling multithreading
        #populations = 15,
        population_size = 50,
        parsimony = 0.1,
        complexity_of_constants = 5,
        parallelism=:serial,
        seed = seed,
        deterministic = true,
        batching = false
    )

    # Create and train model on this data
    mach = machine(model, SR_input, SR_output)
    fit!(mach)

    r =report(mach)

    best_equation=r.equations[r.best_idx]
    best_symbolic = SymbolicRegression.node_to_symbolic(best_equation)
    simplified_equation = SymbolicUtils.simplify(best_symbolic, expand=true)

    # Round to 3 significant figures
    simplified_equation = format_equation_sigfigs(string(simplified_equation))
    
    return mach, r, best_equation, simplified_equation

end





