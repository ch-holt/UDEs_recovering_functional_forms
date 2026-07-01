#=============================================================
DEFINE NEURAL NETWORK WITH TWO HIDDEN LAYERS
==============================================================# 

function build_neural_network(rng, hidden_dims, input_size, output_size, 
                            activation_function, final_activation_function)

    beta_network = Lux.Chain(Lux.Dense(input_size=>hidden_dims, activation_function), Lux.Dense(hidden_dims=>hidden_dims, activation_function),
                            Lux.Dense(hidden_dims=>output_size, final_activation_function))

    # Initialise parameters to build the structure for the UDE
    p_nn_temp, st_nn = Lux.setup(rng, beta_network)

    # Convert to ComponentArray for gradient-based optimisation
    # ComponentArray wraps nested parameter structures into flat array keeping named access
    p_nn_temp = ComponentArray(p_nn_temp)

    return beta_network, p_nn_temp, st_nn
end

