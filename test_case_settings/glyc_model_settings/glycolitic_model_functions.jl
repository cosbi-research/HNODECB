"""
    ground_truth_function(du, u, p, t)

Derivative function for the glycolitic model
"""
function ground_truth_function(du, u, p, t)
    
    du[1] = p[1] - (p[2] * u[1] * u[6]) / (1 + (u[6] / p[3])^p[4])
    du[2] = 2 * (p[2] * u[1] * u[6]) / (1 + (u[6] / p[3])^p[4]) - p[5] * u[2] * (p[6] - u[5]) - p[7] * u[2] * u[5]
    du[3] = p[5] * u[2] * (p[6] - u[5]) - p[8] * u[3] * (p[9] - u[6])
    du[4] = p[8] * u[3] * (p[9] - u[6]) - p[10] * u[4] * u[5] - p[11] * (u[4] - u[7])
    du[5] = p[5] * u[2] * (p[6] - u[5]) - p[10] * u[4] * u[5] - p[7] * u[2] * u[5]
    du[6] = -2 * (p[2] * u[1] * u[6]) / (1 + (u[6] / p[3])^p[4]) + 2 * p[8] * u[3] * (p[9] - u[6]) - p[12] * u[6]
    du[7] = p[13] * p[11] * (u[4] - u[7]) - p[14] * u[7]
end

"""
    get_uode_model_function(appr_neural_network, state, original_parameters_opt)

Returns the model derivative function with the neural network approximating the first equation
The mechanistic parameters are scaled by the original parameters
"""
function get_uode_model_function(appr_neural_network, state, original_parameters_opt)

    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state, original_parameters_opt = original_parameters_opt


            û = appr_neural_network(view(u, [1, 6]), p.p_net, st)[1]
            ode_par = p.ode_par .* original_parameters_opt

            # p3 in log scale because it cannot be negative
            p3 = 10^ode_par[3]

            @inbounds du[1] = û[1]
            @inbounds du[2] = 2 * (ode_par[2] * u[1] * u[6]) / (1 + (max(u[6],0) / p3)^ode_par[4]) - ode_par[5] * u[2] * (ode_par[6] - u[5]) - ode_par[7] * u[2] * u[5]
            @inbounds du[3] = ode_par[5] * u[2] * (ode_par[6] - u[5]) - ode_par[8] * u[3] * (ode_par[9] - u[6])
            @inbounds du[4] = ode_par[8] * u[3] * (ode_par[9] - u[6]) - ode_par[10] * u[4] * u[5] - ode_par[11] * (u[4] - u[7])
            @inbounds du[5] = ode_par[5] * u[2] * (ode_par[6] - u[5]) - ode_par[10] * u[4] * u[5] - ode_par[7] * u[2] * u[5]
            @inbounds du[6] = -2 * (ode_par[2] * u[1] * u[6]) / (1 + (max(u[6],0.0) / p3)^ode_par[4]) + 2 * ode_par[8] * u[3] * (ode_par[9] - u[6]) - ode_par[12] *  u[6]
            @inbounds du[7] = ode_par[13] * ode_par[11] * (u[4] - u[7]) - ode_par[14] * u[7]

        end
end

"""
    ground_truth_function_modified(du, u, p, t)

Derivative function for the glycolitic model. In this function the parameter p[3] is in log scale and we force u[6] to be positive to avoid numerical integration error
"""
function ground_truth_function_modified(du, u, p, t)

    du[1] = p[1] - (p[2] * u[1] * u[6]) / (1 + (max(u[6],0) / (10^p[3]))^p[4])
    du[2] = 2 * (p[2] * u[1] * u[6]) / (1 + (max(u[6],0) / (10^p[3]))^p[4]) - p[5] * u[2] * (p[6] - u[5]) - p[7] * u[2] * u[5]
    du[3] = p[5] * u[2] * (p[6] - u[5]) - p[8] * u[3] * (p[9] - u[6])
    du[4] = p[8] * u[3] * (p[9] - u[6]) - p[10] * u[4] * u[5] - p[11] * (u[4] - u[7])
    du[5] = p[5] * u[2] * (p[6] - u[5]) - p[10] * u[4] * u[5] - p[7] * u[2] * u[5]
    du[6] = -2 * (p[2] * u[1] * u[6]) / (1 + (max(u[6],0) / (10^p[3]))^p[4]) + 2 * p[8] * u[3] * (p[9] - u[6]) - p[12] * u[6]
    du[7] = p[13] * p[11] * (u[4] - u[7]) - p[14] * u[7]
end