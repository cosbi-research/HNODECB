#= 
Script to analyse the results of the Lotka Volterra UDE model trained with fixed (and vartying) values of the mechanistic parameter on DS_0.05
=#

cd(@__DIR__)

using ComponentArrays, Lux, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots

result_folder = "profiler_analysis_e0.05"
if !isdir(result_folder)
  mkdir(result_folder)
end

rng = Random.default_rng()
Random.seed!(rng, 0)

include("../test_case_settings/lv_model_settings/lotka_volterra_model_functions.jl")
include("../test_case_settings/lv_model_settings/lotka_volterra_model_settings.jl")

column_names = ["t", "s1", "s2"]

integrator = Vern7()
abstol = 1e-7
reltol = 1e-6

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
  Lux.Dense(2, 2^3, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2^3, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2^3, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

ode_data = deserialize("../datasets/e0.05/data/ode_data_lotka_volterra.jld")
ode_data_sd = deserialize("../datasets/e0.05/data/ode_data_std_lotka_volterra.jld")
solution_dataframe = deserialize("../datasets/e0.05/data/pert_df_lotka_volterra.jld")
solution_sd_dataframe = deserialize("../datasets/e0.05/data/pert_df_sd_lotka_volterra.jld")

#loads the profiling results
profile_results = deserialize("raw_results/lotka_volterra_0.05.jld")

#see the costs 
validation_costs = [result.validation_resulting_cost for result in profile_results]

iterations = 50
lower_bound = 0.013
upper_bound = 3.0
p1_values = range(lower_bound, upper_bound, length=iterations)
mechanicistic_parameter = [p1_values[result.iterator] for result in profile_results]

#defines a grid over the integration interval to evaluate the neural network contribution during the trajectory
sampling_times = range(0.0, 5.0, length=1000)

local_approximating_neural_network = deepcopy(approximating_neural_network)
p_net, st = Lux.setup(rng, local_approximating_neural_network)

############################################ regularizer_evaluator ############################################
mec_parameters = []
nn_contributions = []
se_costs = []
for i in axes(profile_results,1)

  profile_result = profile_results[i]
  validation_cost = validation_costs[i]

  #discards the results with high validation cost
  if validation_cost > 10
    continue
  end

  parameters= profile_result.parameters_training
  p1_value = p1_values[profile_result.iterator]

  #gets the hybrid derivative function and instatiate the prediction ODE problem
  uode_derivative_function = get_uode_model_function_with_fixed_p1(approximating_neural_network, st, p1_value)
  local_initial_conditions = profile_result.initial_state_training
  prob =  ODEProblem{true}(uode_derivative_function, local_initial_conditions, (0.0, 5))


  solutions = solve(prob, integrator, p=parameters, saveat=sampling_times, abstol=abstol, reltol=reltol)
  solution_as_array = Array(solutions)

  solutions_to_evaluate = Array(solve(prob, integrator, p=parameters, saveat=solution_dataframe.t, abstol=abstol, reltol=reltol))
  
  #error over the trajectory
  se_cost = sum(abs2.(solutions_to_evaluate .- ode_data))
  append!(se_costs, se_cost)

  #contribution of the neural network over the trajectory
  nn_contribution = 0.0
  for element in axes(solution_as_array, 2)
   nn_prediction = approximating_neural_network(solution_as_array[:, element], parameters.p_net, st)[1]
   derivative_y1 = p1_value * solution_as_array[1, element] + nn_prediction[1]
   derivative_y2 = -1.8 * solution_as_array[2, element] + nn_prediction[2]

   derivative_norm = (derivative_y1^2 + derivative_y2^2)
   nn_contribution += sum(abs2.(approximating_neural_network(solution_as_array[:, element], parameters.p_net, st)[1])) / derivative_norm
  end

  append!(mec_parameters, p1_value) 
  append!(nn_contributions, nn_contribution / length(sampling_times))
end 

#create the dataframe with the squared error and the average contribution of the neural network
regularizer_df = DataFrame(mec_parameter=mec_parameters, nn_contribution=nn_contributions, se_cost=se_costs)
serialize(result_folder * "/regularizer_df.jld", regularizer_df)
