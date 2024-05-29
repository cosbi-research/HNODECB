#= 
Script to print the table of estimated mechanistic parameters for the yeast glycolysis UDE model trained on DS_00 and DS_05,
assuming that only y5 and y6 are observed.
=#

cd(@__DIR__)

using DataFrames, CSV, Serialization, Lux, ComponentArrays, Printf

#includes the specific model functions
include("../test_case_settings/glyc_model_settings/glycolitic_model_functions.jl")
include("../test_case_settings/glyc_model_settings/glycolitic_model_settings.jl")

par_opt_00 = deserialize("results_local_optima_found/glyc_opt_00_observables_56.jld")
par_opt_05 = deserialize("results_local_optima_found/glyc_opt_05_observables_56.jld")

ci_00 = deserialize("results_Fisher_confidence_intervals/glyc_opt_00_fisher_CI_observable_56.jld")
ci_05 = deserialize("results_Fisher_confidence_intervals/glyc_opt_05_fisher_CI_observable_56.jld")

#format the float number in scientific notation
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

parameters_00 = par_opt_00.parameters_training.ode_par[2:end]
parameters_05 = par_opt_05.parameters_training.ode_par[2:end]
orig_pars = original_parameters[2:end]

parameters_00[2] = 10 .^ parameters_00[2]
parameters_05[2] = 10 .^ parameters_05[2]


error_00 = abs.(parameters_00 - orig_pars) ./ orig_pars
error_05 = abs.(parameters_05 - orig_pars) ./ orig_pars

index_CI_false = [k for k in keys(ci_00) if ci_00[k]!=false]
mean(error_00[index_CI_false .- 1])

index_CI_false = [k for k in keys(ci_05) if ci_05[k]!=false]
mean(error_05[index_CI_false .- 1])

parameter_names_complete = ["k_1" "K_1" "q" "k_2" "N" "k_6" "k_3" "A" "k_4" "\\kappa" "k_5" "\\psi" "k"]
parameter_names = parameter_names_complete

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

  if ci_00[i+1] == false
    tmp_string = replace(tmp_string, "##identifiable_00##"  =>  "")
    tmp_string = replace(tmp_string, "##CI_00##" => "")
  else
    tmp_string = replace(tmp_string, "##CI_00##" => "("*get_formatted_number(max(ci_00[i+1][1],0.00))* ", "* get_formatted_number(ci_00[i+1][2]) * ")")
    tmp_string = replace(tmp_string, "##identifiable_00##"  =>  "\\checkmark")
  end
  #replace the ##identifiable_00##
  #replace the ##CI_00##
  
  #replace the ##value_05##
  tmp_string = replace(tmp_string, "##value_05##" => get_formatted_number(parameters_05[i]))
  #replace the ##error_05##
  tmp_string = replace(tmp_string, "##error_05##" =>  get_formatted_number(error_05[i], true))
  #replace the ##identifiable_05##
  if ci_05[i+1] == false
    tmp_string = replace(tmp_string, "##identifiable_05##"  =>  "")
    tmp_string = replace(tmp_string, "##CI_05##" => "")
  else
    tmp_string = replace(tmp_string, "##CI_05##" => "("*get_formatted_number(max(ci_05[i+1][1],0.00))* ", "* get_formatted_number(ci_05[i+1][2]) * ")")
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
