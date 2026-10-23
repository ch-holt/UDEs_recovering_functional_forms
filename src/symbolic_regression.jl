#========================================================
GENERATE SYMBOLIC REGRESSION INPUTS/OUTPUTS
=========================================================# 

function build_sr_inputs(p_all, population, norm_i_traj, min_i_norm, max_i_norm)

    # We do SR on this grid
    I_grid = collect(range(min_i_norm, max_i_norm; length=1000))
    range_0_1 = collect(range(0, 1; length=1000))

    # Build matrix inputs for the NN (each column is one I value)
    nn_input_1000 = reshape(I_grid, 1, :)
    nn_input_0_1  = reshape(range_0_1, 1, :)
    nn_input_days = reshape(norm_i_traj, 1, :)

    SR_input_1000 = reshape(I_grid, :, 1)
    SR_input_0_1  = reshape(range_0_1, :, 1)
    SR_input_days = reshape(norm_i_traj, :, 1)

    return SR_input_1000, SR_input_0_1, SR_input_days, nn_input_1000, nn_input_0_1, nn_input_days, I_grid, range_0_1
end

#========================================================
RUN SYMBOLICREGRESSION.JL
=========================================================# 

function run_symbolic_regression(SR_input, SR_output, output_dir; seed=seed, parsimony=3, maxsize=20, complexity_of_constants=2)
    
    model = SRRegressor(
        niterations=100,
        binary_operators=[+, -, *, /],
        unary_operators=[exp],
        maxsize = maxsize,
        output_directory = output_dir,
        # Make results reproducible by disabling multithreading
        populations = 15,
        population_size = 50,
        parsimony = parsimony,
        complexity_of_constants = complexity_of_constants,
        parallelism=:serial,
        seed = seed,
        deterministic = true,
        batching = false,
        # Don't print progress for speed (and doing on multiple seeds would merge logs)
        progress = false,
        # Don't write the hall of fame CSV during the search (the final hall of fame is in report(mach),
        # saved in SR_report.jld2); with many jobs writing to the shared filesystem it makes SR ~30x slower
        save_to_file = false,
    )

    # Create and train model on this data
    mach = machine(model, SR_input, SR_output)
    # Disable printing for speed
    fit!(mach, verbosity=0)

    r =report(mach)

    best_equation=r.equations[r.best_idx]
    best_symbolic = SymbolicRegression.node_to_symbolic(best_equation)
    simplified_equation = SymbolicUtils.simplify(best_symbolic, expand=true)

    # Round to 3 significant figures
    simplified_equation = format_equation_sigfigs(string(simplified_equation))
    
    return mach, r, best_equation, simplified_equation

end





