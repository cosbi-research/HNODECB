"""
    ground_truth_function(du, u, p, t)

Derivative function for the cell apoptosis model.
"""
function ground_truth_function(du, u, p, t)
    du[1] = -p[1]*u[4]*u[1] + p[2]*u[5]
    du[2] = p[3]*u[5] - p[4]*u[2]*u[3] + p[5]*u[6] + p[6]*u[6]
    du[3] = -p[4]*u[2]*u[3] + p[5]*u[6]
    du[4] = p[6]*u[6] - p[1]*u[4]*u[1] +p[2]*u[5] - p[7]*u[4]*u[7] + p[8]*u[8] + p[3]*u[5]
    du[5] = -p[3]*u[5] + p[1]*u[4]*u[1] - p[2]*u[5]
    du[6] = -p[6]*u[6] + p[4]*u[2]*u[3] - p[5]*u[6]
    du[7] = -p[7]*u[7]*u[4] + p[8]*u[8] + p[9]*u[8]
    du[8] = p[7]*u[7]*u[4] - p[8]*u[8] - p[9]*u[8]
end

"""
    ground_truth_function(du, u, p, t)

Derivative function for the cell apoptosis model. Parameters p2 and p4 are fixed to their literature values.
"""
function ground_truth_function_fixed_p2p4(du, u, p, t)
    p4 = 6.8 * 10^-8*3600 * 10^5
    p2 = 1*10^-2*3600

    ode_par = p.*[2.67 * 10^-9 *3600 * 10^5, 1*10^-2*3600, 8* 10^-3*3600, 6.8 * 10^-8*3600 * 10^5, 5*10^-2*3600, 1*10^-3*3600, 7*10^-5*3600 * 10^5, 1.67 * 10^-5*3600, 1.67*10^-4*3600]
    ode_par = p

    du[1] = -ode_par[1]*u[4]*u[1] + p2*u[5]
    du[2] = ode_par[3]*u[5] - p4*u[2]*u[3] +  ode_par[5]*u[6] + ode_par[6]*u[6]
    du[3] = -p4*u[2]*u[3] +  ode_par[5]*u[6]
    du[4] = ode_par[6]*u[6] - ode_par[1]*u[4]*u[1] +p2*u[5] - ode_par[7]*u[4]*u[7] + ode_par[8]*u[8] + ode_par[3]*u[5]
    du[5] = -ode_par[3]*u[5] + ode_par[1]*u[4]*u[1] - p2*u[5]
    du[6] = -ode_par[6]*u[6] + p4*u[2]*u[3] -  ode_par[5]*u[6]
    du[7] = -ode_par[7]*u[7]*u[4] + ode_par[8]*u[8] + ode_par[9]*u[8]
    du[8] = ode_par[7]*u[7]*u[4] - ode_par[8]*u[8] - ode_par[9]*u[8]
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
             
            
            ode_par = p.ode_par.*original_parameters_opt

            û = appr_neural_network(view(u, [1, 4, 5, 6, 7, 8]), p.p_net, st)[1]# Network prediction
            @inbounds du[1] = -ode_par[1]*u[4]*u[1] + ode_par[2]*u[5]
            @inbounds du[2] = ode_par[3]*u[5] - ode_par[4]*u[2]*u[3] + ode_par[5]*u[6] + ode_par[6]*u[6]
            @inbounds du[3] = -ode_par[4]*u[2]*u[3] + ode_par[5]*u[6]
            @inbounds du[4] = û[1]
            @inbounds du[5] = -ode_par[3]*u[5] + ode_par[1]*u[4]*u[1] - ode_par[2]*u[5]
            @inbounds du[6] = -ode_par[6]*u[6] + ode_par[4]*u[2]*u[3] - ode_par[5]*u[6]
            @inbounds du[7] = max(min(-ode_par[7]*u[7]*u[4] + ode_par[8]*u[8] + ode_par[9]*u[8], 10^5), -10^5)
            @inbounds du[8] = max(min(ode_par[7]*u[7]*u[4] - ode_par[8]*u[8] - ode_par[9]*u[8], 10^5), -10^5)
        end
end

"""
    get_uode_model_function(appr_neural_network, state, original_parameters_opt)

Returns the model derivative function with the neural network approximating the first equation
The mechanistic parameters are scaled by the original parameters. Parameters p2 and p4 are fixed to their literature values.
"""
function get_uode_model_function_fixed_p2p4(appr_neural_network, state, original_parameters_opt)
    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state, original_parameters_opt = original_parameters_opt
             
            
            ode_par = p.ode_par.*original_parameters_opt

            p2 = 1*10^-2*3600
            p4 = 6.8 * 10^-8*3600 * 10^5

            û = appr_neural_network(view(u, [1, 4, 5, 6, 7, 8]), p.p_net, st)[1]# Network prediction
            @inbounds du[1] = -ode_par[1]*u[4]*u[1] + p2*u[5]
            @inbounds du[2] = ode_par[2]*u[5] - p4*u[2]*u[3] + ode_par[3]*u[6] + ode_par[4]*u[6]
            @inbounds du[3] = -p4*u[2]*u[3] + ode_par[3]*u[6]
            @inbounds du[4] = û[1]
            @inbounds du[5] = -ode_par[2]*u[5] + ode_par[1]*u[4]*u[1] - p2*u[5]
            @inbounds du[6] = -ode_par[4]*u[6] + p4*u[2]*u[3] - ode_par[3]*u[6]
            @inbounds du[7] = -ode_par[5]*u[7]*u[4] + ode_par[6]*u[8] + ode_par[7]*u[8]
            @inbounds du[8] = ode_par[5]*u[7]*u[4] - ode_par[6]*u[8] - ode_par[7]*u[8]
        end
end