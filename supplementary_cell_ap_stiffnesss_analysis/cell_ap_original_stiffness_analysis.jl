#=
This script is used to determine the identifiability of the parameters of the original cell apoptosis model
assuming different levels of observability.
=#

cd(@__DIR__)

using ComponentArrays, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using StableRNGs, Zygote, LinearAlgebra, SciMLSensitivity, Optimization

#includes the specific model functions
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_settings.jl")

original_trajectory = deserialize("../datasets/e0.0/data/ode_data_cell_apoptosis.jld")
original_parameters_df = deserialize("../datasets/e0.0/data/pert_df_cell_apoptosis.jld")

column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

integrator = TRBDF2(autodiff=false);
abstol = 1e-8
reltol = 1e-7

sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#settings for the training set sampling
tspan = (initial_time_training, end_time_training)
#get the model function of the original model
function derivative_function_for_jacobian(u)

  p = original_parameters

  du_1 = -p[1]*u[4]*u[1] + p[2]*u[5]
  du_2 = p[3]*u[5] - p[4]*u[2]*u[3] + p[5]*u[6] + p[6]*u[6]
  du_3 = -p[4]*u[2]*u[3] + p[5]*u[6]
  du_4 = p[6]*u[6] - p[1]*u[4]*u[1] +p[2]*u[5] - p[7]*u[4]*u[7] + p[8]*u[8] + p[3]*u[5]
  du_5 = -p[3]*u[5] + p[1]*u[4]*u[1] - p[2]*u[5]
  du_6 = -p[6]*u[6] + p[4]*u[2]*u[3] - p[5]*u[6]
  du_7 = -p[7]*u[7]*u[4] + p[8]*u[8] + p[9]*u[8]
  du_8 = p[7]*u[7]*u[4] - p[8]*u[8] - p[9]*u[8]

  return [du_1, du_2, du_3, du_4, du_5, du_6, du_7, du_8]
end

#compute the jacobian of the model function
function model_jacobian_computator(u)
  J = zeros(8, 8)
  u = u[1:8]
  for i in 1:8
    J[i, :] .= Zygote.gradient(x -> derivative_function_for_jacobian(x)[i], u)[1]
  end
  return J
end

#compute the jacobian of the model function
eigenvalues_total = zeros(8, size(original_trajectory)[2])
for i in 1:size(original_trajectory)[2]
  u0 = original_trajectory[1:8, i]
  model_jacobian = model_jacobian_computator(u0)
  eigenvalues, eigenvectors = eigen(model_jacobian)
  eigenvalues_total[:, i] .= eigenvalues
end

stiffness_index = zeros(4, size(original_trajectory)[2])
stiffness_index[1,:] = original_parameters_df.t
delta_t = original_parameters_df.t[2] - original_parameters_df.t[1]

for i in 1:size(original_trajectory)[2]
  stiffness_index[2, i] = maximum(abs.(eigenvalues_total[:, i])) / minimum(abs.(eigenvalues_total[:, i])) * delta_t
  stiffness_index[3, i] =  maximum(abs.(eigenvalues_total[:, i]))
  stiffness_index[4, i] =  minimum(abs.(eigenvalues_total[:, i]))
end

stiffness_index = stiffness_index[:, 1:5]

#read a text file "template_latex_table.txt"
time_placeholder = "#TIME_PLACEHOLDER#"
sr_placeholder = "#SR_PLACEHOLDER#"
maxe_placeholder = "#MAXE_PLACEHOLDER#"
mine_placeholder = "#MINE_PLACEHOLDER#"

latex_table = read("template_latex_table.txt", String)

#function to round a double in scentific notation to 2 decimal digits

function round_to_2_digits(x::Float64)
  # Calculate the exponent
  if x == Inf
    return "\$ \\infty \$"
  elseif x > 0
    exponent = floor(Int, log10(x))
  else
    exponent = 0.0
  end

  # Calculate the mantissa
  mantissa = x / (10^(exponent+0.0))
  
  # Round the mantissa to 2 decimal places
  rounded_mantissa = round(mantissa, digits=2)
  
  if exponent == 0
    return string(rounded_mantissa)
  else
    # Return the formatted string with \cdot 10^
    #exponent = floor(Int, exponent)
    return "\$ $(rounded_mantissa) \\cdot 10^{$(exponent)} \$"
  end
end


for i in 1:size(stiffness_index)[2]
  latex_table = replace(latex_table, time_placeholder => string(stiffness_index[1, i]) * " & " * time_placeholder)
  latex_table = replace(latex_table, sr_placeholder => string(round_to_2_digits(stiffness_index[2, i])) * " & " * sr_placeholder)
  latex_table = replace(latex_table, maxe_placeholder => string(round_to_2_digits(stiffness_index[3, i])) * " & " * maxe_placeholder)
  latex_table = replace(latex_table, mine_placeholder => string(round_to_2_digits(stiffness_index[4, i])) * " & " * mine_placeholder)
end

latex_table = replace(latex_table, time_placeholder => "")
latex_table = replace(latex_table, sr_placeholder => "")
latex_table = replace(latex_table, maxe_placeholder => "")
latex_table = replace(latex_table, mine_placeholder => "")

#print the string on terminal
print(latex_table)
