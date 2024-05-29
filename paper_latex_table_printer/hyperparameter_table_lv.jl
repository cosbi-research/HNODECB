#= 
Script to print the table of tuned hyperparameters for the Lotka Volterra UDE model trained on DS_00 and DS_05
=#

cd(@__DIR__)

using DataFrames, CSV, Serialization, Lux, ComponentArrays, Printf
using PyCall

optuna = pyimport("optuna")

########## error 0.0 ################
hyperparameters = deserialize("results_hyperparameter_tuned/lv_00.jld")
hyperparameters_00 = hyperparameters.study.best_params
hyperparameters_00_best_trial = hyperparameters.study.best_trial

########## error 0.05 ###############
hyperparameters = deserialize("results_hyperparameter_tuned/lv_05.jld")
hyperparameters_05 = hyperparameters.study.best_params
hyperparameters_05_best_trial = hyperparameters.study.best_trial

parameter_names_complete = ["\\alpha"]
parameter_names = parameter_names_complete

#read the template file
template_row = read("latex_table_templates/template_hyper_parameters.txt", String)
physical_parameter_row = "\$##par_name##\$ & \$##search_space##\$ & \$##value_00##\$ & \$##value_05##\$\\\\"

#format the float number in scientific notation
function get_formatted_double(num, perc=false)
    if perc
        return @sprintf("%.2f", num*100)
    else
      #return number in scientific notation with 2 decimal places, with 2 \cdot 10^
      return @sprintf("%.2e", num)
    end
  return num
end

function get_formatted_int(num)
    return @sprintf("%d", num)
end

result_tmp = template_row

#hidden_layer_number 
result_tmp = replace(result_tmp, "##h_l_n_00##" => get_formatted_int(hyperparameters_00["num_hidden_layers"]+1))
result_tmp = replace(result_tmp, "##h_l_n_05##" => get_formatted_int(hyperparameters_05["num_hidden_layers"]+1))

#hidden_node_number
result_tmp = replace(result_tmp, "##h_n_n_00##" => get_formatted_int(2 ^ hyperparameters_00["num_hidden_nodes"]))
result_tmp = replace(result_tmp, "##h_n_n_05##" => get_formatted_int(2 ^ hyperparameters_05["num_hidden_nodes"]))

#learning_rate
result_tmp = replace(result_tmp, "##lr_00##" => get_formatted_double(hyperparameters_00["learning_rate_adam"]))
result_tmp = replace(result_tmp, "##lr_05##" => get_formatted_double(hyperparameters_05["learning_rate_adam"]))

#rho
result_tmp = replace(result_tmp, "##rho_00##" => get_formatted_double(hyperparameters_00["ms_continuity_term"]))
result_tmp = replace(result_tmp, "##rho_05##" => get_formatted_double(hyperparameters_05["ms_continuity_term"]))

#k
result_tmp = replace(result_tmp, "##k_00##" => get_formatted_int(hyperparameters_00["ms_group_size"]))
result_tmp = replace(result_tmp, "##k_05##" => get_formatted_int(hyperparameters_05["ms_group_size"]))

#mechanistic parameters
parameter_rows = []
for dict_key in keys(hyperparameters_00)
    if startswith(String(dict_key), "p")

        #parse as int the parameter name
        pos = parse(Int, replace(dict_key, "p" => ""))
        parameter_row = physical_parameter_row

        parameter_row = replace(parameter_row, "##par_name##" => parameter_names[pos])

        #lower_bound
        lb = hyperparameters_05_best_trial.distributions[dict_key].low
        up = hyperparameters_05_best_trial.distributions[dict_key].high

        search_space_as_string = "["*get_formatted_double(lb) * "," * get_formatted_double(up)*"]"

        parameter_row = replace(parameter_row, "##search_space##" => search_space_as_string)
        parameter_row = replace(parameter_row, "##value_00##" => get_formatted_double(hyperparameters_00[dict_key]))
        parameter_row = replace(parameter_row, "##value_05##" => get_formatted_double(hyperparameters_05[dict_key]))

        push!(parameter_rows, parameter_row)
    end
end

result_tmp = replace(result_tmp, "##physical_parameter_row##" => join(parameter_rows, "\r\n"))

result_tmp = replace(result_tmp, "e+00" => "" )
result_tmp = replace(result_tmp, "e+01" => " \\cdot 10^{1}" )
result_tmp = replace(result_tmp, "e+02" => " \\cdot 10^{2}" )
result_tmp = replace(result_tmp, "e+03" => " \\cdot 10^{3}" )
result_tmp = replace(result_tmp, "e+04" => " \\cdot 10^{4}" )
result_tmp = replace(result_tmp, "e+05" => " \\cdot 10^{5}" )
result_tmp = replace(result_tmp, "e+06" => " \\cdot 10^{6}" )
result_tmp = replace(result_tmp, "e+07" => " \\cdot 10^{7}" )
result_tmp = replace(result_tmp, "e+08" => " \\cdot 10^{8}" )
#with negative
result_tmp = replace(result_tmp, "e-01" => " \\cdot 10^{-1}" )
result_tmp = replace(result_tmp, "e-02" => " \\cdot 10^{-2}" )
result_tmp = replace(result_tmp, "e-03" => " \\cdot 10^{-3}" )
result_tmp = replace(result_tmp, "e-04" => " \\cdot 10^{-4}" )
result_tmp = replace(result_tmp, "e-05" => " \\cdot 10^{-5}" )
result_tmp = replace(result_tmp, "e-06" => " \\cdot 10^{-6}" )
result_tmp = replace(result_tmp, "e-07" => " \\cdot 10^{-7}" )
result_tmp = replace(result_tmp, "e-08" => " \\cdot 10^{-8}" )


println("**********************************************")
print(result_tmp)
