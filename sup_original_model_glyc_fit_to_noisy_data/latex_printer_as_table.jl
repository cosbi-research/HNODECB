#= 
Script to print the fit results to latex table format
=# 

cd(@__DIR__)

using Serialization, DataFrames, CSV, Printf


#includes the specific model functions
include("../test_case_settings/glyc_model_settings/glycolitic_model_functions.jl")
include("../test_case_settings/glyc_model_settings/glycolitic_model_settings.jl")

results = deserialize("fit_glyc_e0.05.jld")

template_row = "\$#par_name#\$ & \$#target#\$ & \$#estimate#\$ & \$#error#\\%\$\\\\"

function get_formatted_number(num, perc=false)
  if perc
      return @sprintf("%.2f", num*100)
  else
    #return number in scientific notation with 2 decimal places, with 2 \cdot 10^
    return @sprintf("%.2e", num)
  end
return num
end

parameter_names_complete = ["J_0" "k_1" "K_1" "q" "k_2" "N" "k_6" "k_3" "A" "k_4" "\\kappa" "k_5" "\\psi" "k"]

results_string = []
for row_number in 1:size(results)[1]
  row = results[row_number, :]
  par_name = parameter_names_complete[row_number]

  target = get_formatted_number(row[1])
  estimate = get_formatted_number(row[2])
  error = get_formatted_number(row[3], true)

  tmp_string = replace(template_row, "#par_name#" => par_name)
  tmp_string = replace(tmp_string, "#target#" => target)
  tmp_string = replace(tmp_string, "#estimate#" => estimate)
  tmp_string = replace(tmp_string, "#error#" => error)

  push!(results_string, tmp_string)
end

#join the results with a new line
total_result = join(results_string, "\n")
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


println("**************************************************")
println(total_result)