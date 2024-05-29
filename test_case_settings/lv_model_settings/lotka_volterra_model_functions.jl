"""
    ground_truth_function(du, u, p, t)

Derivative function for the lotka volterra model.
"""
function ground_truth_function(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

"""
    get_uode_model_function(appr_neural_network, state, orig_p1)

Returns the derivative function with the neural network approximating the interaction terms
The parameter p1 is scaled by the original p1, the initial point of the optimization
"""
function get_uode_model_function(appr_neural_network, state, orig_p1)
    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state, orig_p1 = orig_p1
            
            p_true = [1.3, 0.9, 0.8, 1.8]
            p1 = p.p1 * orig_p1

            û = appr_neural_network(u, p.p_net, st)[1]# Network prediction

            @inbounds du[1] = p1*u[1] + û[1]
            @inbounds du[2] = -p_true[4]*u[2] + û[2]
        end
end

"""
    get_uode_model_function(appr_neural_network, state, orig_p1)
    
Returns the derivative function with the nn approximating the first derivative
The parameter p1 is fixed to the orig_p1
"""
function get_uode_model_function_with_fixed_p1(appr_neural_network, state, orig_p1)
    #generates the function with the parameters
    f(du, u, p, t) =
        let appr_neural_network = appr_neural_network, st = state, orig_p1 = orig_p1
            
            p_true = [1.3, 0.9, 0.8, 1.8]
            p1 = orig_p1

            û = appr_neural_network(u, p.p_net, st)[1]# Network prediction

            @inbounds du[1] = p1*u[1] + û[1]
            @inbounds du[2] = -p_true[4]*u[2] + û[2]
        end
end