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

column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

integrator = TRBDF2(autodiff=false);
abstol = 1e-8
reltol = 1e-7

sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

ode_data = deserialize("../datasets/e0.0_doubled/data/ode_data_cell_apoptosis.jld")
ode_data_sd = deserialize("../datasets/e0.0_doubled/data/ode_data_std_cell_apoptosis.jld")
solution_dataframe = deserialize("../datasets/e0.0_doubled/data/pert_df_cell_apoptosis.jld")
solution_sd_dataframe = deserialize("../datasets/e0.0_doubled/data/pert_df_sd_cell_apoptosis.jld")

#settings for the training set sampling
tspan = (initial_time_training, end_time_training)
tsteps = solution_dataframe.t

#get the parameters estimated
tspan = (initial_time_training, end_time_training)

adtype = Optimization.AutoZygote()
par = original_parameters
prob_uode_pred = ODEProblem{true}(ground_truth_function_fixed_p2p4, original_u0, (0, maximum(solution_dataframe.t)))

###########################################################################################################################
function model(params, final_time, observables)
  prob_uode_pred = ODEProblem{true}(ground_truth_function_fixed_p2p4, original_u0, (0, final_time))
  solutions = solve(prob_uode_pred, integrator, p=params, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
  return Array(solutions)[observables, end]
end

# get the spectral decomposition of the Hessian of the cost function with respect to the parameters
function get_Fisher_Information_Matrix_Eigen_Decomposition(parameters_to_consider, observables)
  sensitivity_y2 = Zygote.jacobian(p -> model(p, tsteps[2], observables), parameters_to_consider)[1]
  sensitivity_y3 = Zygote.jacobian(p -> model(p, tsteps[3], observables), parameters_to_consider)[1]
  sensitivity_y4 = Zygote.jacobian(p -> model(p, tsteps[4], observables), parameters_to_consider)[1]
  sensitivity_y5 = Zygote.jacobian(p -> model(p, tsteps[5], observables), parameters_to_consider)[1]
  sensitivity_y6 = Zygote.jacobian(p -> model(p, tsteps[6], observables), parameters_to_consider)[1]
  sensitivity_y7 = Zygote.jacobian(p -> model(p, tsteps[7], observables), parameters_to_consider)[1]
  sensitivity_y8 = Zygote.jacobian(p -> model(p, tsteps[8], observables), parameters_to_consider)[1]
  sensitivity_y9 = Zygote.jacobian(p -> model(p, tsteps[9], observables), parameters_to_consider)[1]
  sensitivity_y10 = Zygote.jacobian(p -> model(p, tsteps[10], observables), parameters_to_consider)[1]
  sensitivity_y11 = Zygote.jacobian(p -> model(p, tsteps[11], observables), parameters_to_consider)[1]
  sensitivity_y12 = Zygote.jacobian(p -> model(p, tsteps[12], observables), parameters_to_consider)[1]
  sensitivity_y13 = Zygote.jacobian(p -> model(p, tsteps[13], observables), parameters_to_consider)[1]
  sensitivity_y14 = Zygote.jacobian(p -> model(p, tsteps[14], observables), parameters_to_consider)[1]
  sensitivity_y15 = Zygote.jacobian(p -> model(p, tsteps[15], observables), parameters_to_consider)[1]
  sensitivity_y16 = Zygote.jacobian(p -> model(p, tsteps[16], observables), parameters_to_consider)[1]
  sensitivity_y17 = Zygote.jacobian(p -> model(p, tsteps[17], observables), parameters_to_consider)[1]
  sensitivity_y18 = Zygote.jacobian(p -> model(p, tsteps[18], observables), parameters_to_consider)[1]
  sensitivity_y19 = Zygote.jacobian(p -> model(p, tsteps[19], observables), parameters_to_consider)[1]
  sensitivity_y20 = Zygote.jacobian(p -> model(p, tsteps[20], observables), parameters_to_consider)[1]
  sensitivity_y21 = Zygote.jacobian(p -> model(p, tsteps[21], observables), parameters_to_consider)[1]
  sensitivity_y22 = Zygote.jacobian(p -> model(p, tsteps[22], observables), parameters_to_consider)[1]
  sensitivity_y23 = Zygote.jacobian(p -> model(p, tsteps[23], observables), parameters_to_consider)[1]
  sensitivity_y24 = Zygote.jacobian(p -> model(p, tsteps[24], observables), parameters_to_consider)[1]
  sensitivity_y25 = Zygote.jacobian(p -> model(p, tsteps[25], observables), parameters_to_consider)[1]
  sensitivity_y26 = Zygote.jacobian(p -> model(p, tsteps[26], observables), parameters_to_consider)[1]
  sensitivity_y27 = Zygote.jacobian(p -> model(p, tsteps[27], observables), parameters_to_consider)[1]
  sensitivity_y28 = Zygote.jacobian(p -> model(p, tsteps[28], observables), parameters_to_consider)[1]
  sensitivity_y29 = Zygote.jacobian(p -> model(p, tsteps[29], observables), parameters_to_consider)[1]
  sensitivity_y30 = Zygote.jacobian(p -> model(p, tsteps[30], observables), parameters_to_consider)[1]
  sensitivity_y31 = Zygote.jacobian(p -> model(p, tsteps[31], observables), parameters_to_consider)[1]
  sensitivity_y32 = Zygote.jacobian(p -> model(p, tsteps[32], observables), parameters_to_consider)[1]
  sensitivity_y33 = Zygote.jacobian(p -> model(p, tsteps[33], observables), parameters_to_consider)[1]
  sensitivity_y34 = Zygote.jacobian(p -> model(p, tsteps[34], observables), parameters_to_consider)[1]
  sensitivity_y35 = Zygote.jacobian(p -> model(p, tsteps[35], observables), parameters_to_consider)[1]
  sensitivity_y36 = Zygote.jacobian(p -> model(p, tsteps[36],observables), parameters_to_consider)[1]
  sensitivity_y37 = Zygote.jacobian(p -> model(p, tsteps[37], observables), parameters_to_consider)[1]
  sensitivity_y38 = Zygote.jacobian(p -> model(p, tsteps[38], observables), parameters_to_consider)[1]
  sensitivity_y39 = Zygote.jacobian(p -> model(p, tsteps[39], observables), parameters_to_consider)[1]
  sensitivity_y40 = Zygote.jacobian(p -> model(p, tsteps[40], observables), parameters_to_consider)[1]
  sensitivity_y41 = Zygote.jacobian(p -> model(p, tsteps[41], observables), parameters_to_consider)[1]
  sensitivity_y42 = Zygote.jacobian(p -> model(p, tsteps[42], observables), parameters_to_consider)[1]

  sensitivity_matrix = vcat(sensitivity_y2, sensitivity_y3, sensitivity_y4, sensitivity_y5, sensitivity_y6, sensitivity_y7, sensitivity_y8, sensitivity_y9, sensitivity_y10, sensitivity_y11, sensitivity_y12, sensitivity_y13, sensitivity_y14, sensitivity_y15, sensitivity_y16, sensitivity_y17, sensitivity_y18, sensitivity_y19, sensitivity_y20, sensitivity_y21, sensitivity_y22, sensitivity_y23, sensitivity_y24, sensitivity_y25, sensitivity_y26, sensitivity_y27, sensitivity_y28, sensitivity_y29, sensitivity_y30, sensitivity_y31, sensitivity_y32, sensitivity_y33, sensitivity_y34, sensitivity_y35, sensitivity_y36, sensitivity_y37, sensitivity_y38, sensitivity_y39, sensitivity_y40, sensitivity_y41, sensitivity_y42)

  normalization_factor = maximum(ode_data[observables, :], dims=2)
  normalization_matrix = vec(repeat(normalization_factor, 41))
  normalization_matrix = Diagonal(1 ./ (normalization_matrix))
  normalization_matrix = abs2.(normalization_matrix)

  hessian_matrix = sensitivity_matrix' * normalization_matrix * sensitivity_matrix
  hessian_matrix = Symmetric(hessian_matrix)
  eigen_value_decomposition = eigen(hessian_matrix)

  eigen_values = real.(eigen_value_decomposition.values)
  eigen_vectors = real.(eigen_value_decomposition.vectors)'

  eigen_vectors_with_eigen_values = hcat(eigen_vectors, eigen_values)

  return eigen_vectors_with_eigen_values, hessian_matrix
end

  # eigenvectors based identifiability algorithm as in Quaiser, Tom, and Martin Mönnigmann. "Systematic identifiability testing for unambiguous mechanistic modeling–application to JAK-STAT, MAP kinase, and NF-κ B signaling pathway models." BMC systems biology 3 (2009): 1-21.
function get_identifiable_parameters(observables)
  eigen_vectors_normalized_with_eigen_values, hessian_matrix = get_Fisher_Information_Matrix_Eigen_Decomposition(par, observables)
  null_eigenvectors = eigen_vectors_normalized_with_eigen_values[abs.(eigen_vectors_normalized_with_eigen_values[:,end]).<1e-5, :]

  #find the maximum position in each null eigen_vectors
  max_positions = []
  for i in 1:size(null_eigenvectors)[1]
    max_position = argmax(abs.(null_eigenvectors[i, 1:end-1]))
    push!(max_positions, max_position)
  end

  identifiable_parameters = Set([1, 2, 3, 4, 5, 6, 7, 8, 9])
  unidentifiable_parameters = Set([])
  
  for i in 1:size(eigen_vectors_normalized_with_eigen_values)[1]

    #get the matrix 
    tmp_sensitivity_matrix = copy(hessian_matrix)
    tmp_sensitivity_matrix = tmp_sensitivity_matrix[sort(collect(identifiable_parameters)), sort(collect(identifiable_parameters))]

    #get the eigen values
    tmp_sensitivity_matrix = Symmetric(tmp_sensitivity_matrix)
    eigen_value_decomposition = eigen(tmp_sensitivity_matrix)

    eigen_values = real.(eigen_value_decomposition.values)
    eigen_vectors = real.(eigen_value_decomposition.vectors)'

    eigen_vectors_normalized = eigen_vectors
    eigen_vectors_normalized_with_eigen_values = hcat(eigen_vectors_normalized, eigen_values)

    first_eigen_vector = eigen_vectors_normalized_with_eigen_values[1, :]
    eigen_value = first_eigen_vector[end]

    if abs(eigen_value) < 1e-5
      println("The parameter ", i, " is not identifiable")

      components = abs.(first_eigen_vector[1:end-1])
      #get the maximum position
      max_position = argmax(components)
      corresponding_parameter = sort(collect(identifiable_parameters))[max_position]
      push!(unidentifiable_parameters, corresponding_parameter)
      #remove all the unidentifiable_parameters from the identifiable_parameters
      identifiable_parameters = setdiff(identifiable_parameters, unidentifiable_parameters) 

    else
      break
    end
  end

  return identifiable_parameters
end

identifiables_all_observables_variables = get_identifiable_parameters([1,2,3,4,5,6,7,8])
identifiables_all_observables_variables_56 = get_identifiable_parameters([5,6])
identifiables_all_observables_variables_5 = get_identifiable_parameters([5])
identifiables_all_observables_variables_6 = get_identifiable_parameters([6])