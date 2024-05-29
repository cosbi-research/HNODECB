#= 
Script to print the table of estimated mechanistic parameters for the cell apoptosis UDE model trained on DS_00 and DS_05, fixing p2 and p4 to their literature values
and assuming only y5 and y6 as observable variables.
=#

cd(@__DIR__)

using DataFrames, CSV, Serialization, Lux, ComponentArrays, Printf

#includes the specific model functions
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_settings.jl")

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

par_opt_00 = deserialize("results_local_optima_found/cell_ap_opt_00_fixed_p2p4_observable_56.jld")
par_opt_05 = deserialize("results_local_optima_found/cell_ap_opt_05_fixed_p2p4_observable_56.jld")

ci_00 = deserialize("results_Fisher_confidence_intervals/cell_ap_opt_00_fisher_CI_fixed_p2p4_observable_56.jld")
ci_05 = deserialize("results_Fisher_confidence_intervals/cell_ap_opt_05_fisher_CI_fixed_p2p4_observable_56.jld")

# format the float number in scientific notation
function get_formatted_number(num, perc=false)
    if perc
        return @sprintf("%.2f", num*100)
    else
      #return number in scientific notation with 2 decimal places, with 2 \cdot 10^
      return @sprintf("%.2e", num)
    end
  return num
end

template_row = "
\\multirow{2}{*}{#math###parameter_name###math#} & \\multirow{2}{*}{#math###original_value###math#} & #math#DS_{0.00}#math# &  #math###value_00## #math# &  #math###error_00##\\%#math# & #math###identifiable_00###math# & #math###CI_00###math# #end_line#
                & &  #math#DS_{0.05}#math# &  #math###value_05## #math# &  #math###error_05##\\%#math# & #math###identifiable_05###math# & #math###CI_05###math# #end_line#
                \\midrule
"

parameters_00 = par_opt_00.parameters_training.ode_par
parameters_05 = par_opt_05.parameters_training.ode_par
orig_pars = original_parameters[[1, 3, 5, 6, 7, 8, 9]]

error_00 = abs.(parameters_00 - orig_pars) ./ orig_pars
error_05 = abs.(parameters_05 - orig_pars) ./ orig_pars

parameter_names_complete = ["k_1", "k_{d1}", "k_{d2}", "k_{3}", "k_{d3}", "k_{d4}", "k_{5}", "k_{d5}", "k_{d6}"]
parameter_names = parameter_names_complete[[1, 3, 5, 6, 7, 8, 9]]

total_result = ""
for i in 1:length(parameter_names)
  #replace the ##parameter_name## in string
  tmp_string = template_row
  tmp_string = replace(tmp_string, "##parameter_name##" => parameter_names[i])
  #replace the ##original_value##
  tmp_string = replace(tmp_string, "##original_value##" => get_formatted_number(orig_pars[i]))
  #replace the ##value_00##
  tmp_string = replace(tmp_string, "##value_00##" => get_formatted_number(parameters_00[i]))
  #replace the ##error_00##
  tmp_string = replace(tmp_string, "##error_00##" => get_formatted_number(error_00[i], true))

  if ci_00[i] == false
    tmp_string = replace(tmp_string, "##identifiable_00##"  =>  "")
    tmp_string = replace(tmp_string, "##CI_00##" => "")
  else
    tmp_string = replace(tmp_string, "##CI_00##" => "("*get_formatted_number(ci_00[i][1])* ", "* get_formatted_number(ci_00[i][2]) * ")")
    tmp_string = replace(tmp_string, "##identifiable_00##"  =>  "\\checkmark")
  end
  #replace the ##identifiable_00##
  #replace the ##CI_00##
  
  #replace the ##value_05##
  tmp_string = replace(tmp_string, "##value_05##" => get_formatted_number(parameters_05[i]))
  #replace the ##error_05##
  tmp_string = replace(tmp_string, "##error_05##" =>  get_formatted_number(error_05[i], true))
  #replace the ##identifiable_05##
  if ci_05[i] == false
    tmp_string = replace(tmp_string, "##identifiable_05##"  =>  "")
    tmp_string = replace(tmp_string, "##CI_05##" => "")
  else
    tmp_string = replace(tmp_string, "##CI_05##" => "("*get_formatted_number(ci_05[i][1])* ", "* get_formatted_number(ci_05[i][2]) * ")")
    tmp_string = replace(tmp_string, "##identifiable_05##"  =>  "\\checkmark")
  end

  total_result *= tmp_string
end

total_result = replace(total_result, "#math#" => "\$")
total_result = replace(total_result, "#end_line#" => "\\\\ \n")

total_result = replace(total_result, "e+00" => "" )
total_result = replace(total_result, "e+01" => " \\cdot 10^{1}" )
total_result = replace(total_result, "e+02" => " \\cdot 10^{2}" )
total_result = replace(total_result, "e+03" => " \\cdot 10^{3}" )
total_result = replace(total_result, "e+04" => " \\cdot 10^{4}" )
total_result = replace(total_result, "e+05" => " \\cdot 10^{5}" )
total_result = replace(total_result, "e+06" => " \\cdot 10^{6}" )
total_result = replace(total_result, "e+07" => " \\cdot 10^{7}" )
total_result = replace(total_result, "e+08" => " \\cdot 10^{8}" )
#with negative
total_result = replace(total_result, "e-01" => " \\cdot 10^{-1}" )
total_result = replace(total_result, "e-02" => " \\cdot 10^{-2}" )
total_result = replace(total_result, "e-03" => " \\cdot 10^{-3}" )
total_result = replace(total_result, "e-04" => " \\cdot 10^{-4}" )
total_result = replace(total_result, "e-05" => " \\cdot 10^{-5}" )
total_result = replace(total_result, "e-06" => " \\cdot 10^{-6}" )
total_result = replace(total_result, "e-07" => " \\cdot 10^{-7}" )
total_result = replace(total_result, "e-08" => " \\cdot 10^{-8}" )


#print in console 
print("**********************************************")
print(total_result)
