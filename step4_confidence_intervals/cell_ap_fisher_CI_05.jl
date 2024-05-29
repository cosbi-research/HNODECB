#= 
Script to derive the confidence intervals for the identifiable mechanistic parameters in the cell apoptosis UDE model trained on DS_05
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes

error_level = "e0.05"

#create the results directory
if !isdir("results")
  mkdir("results")
end


#includes the specific model functions
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_settings.jl")

column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

integrator = TRBDF2(autodiff=false);
abstol = 1e-8
reltol = 1e-7
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
  Lux.Dense(6, 2^3, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2^3, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)
#reads the estimated parameters
par_opt = deserialize("local_optima_found/ca_opt_05.jld")

ode_data = deserialize("../datasets/" * error_level * "/data/ode_data_cell_apoptosis.jld")
ode_data_sd = deserialize("../datasets/" * error_level * "/data/ode_data_std_cell_apoptosis.jld")
solution_dataframe = deserialize("../datasets/" * error_level * "/data/pert_df_cell_apoptosis.jld")
solution_sd_dataframe = deserialize("../datasets/" * error_level * "/data/pert_df_sd_cell_apoptosis.jld")

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)

#settings for the training set sampling
tspan = (initial_time_training, end_time_training)
tsteps = solution_dataframe.t
parameters_optimized = par_opt.parameters_training
uode_derivative_function = get_uode_model_function(approximating_neural_network, par_opt.net_status, 1)


parameters_optimized_def = ComponentArray{eltype(parameters_optimized.p_net)}()
u0 = ComponentArray(par_opt.initial_state_training[:, 1])
parameters_optimized_def = ComponentArray(parameters_optimized_def; u0)
pars = ComponentArray(par_opt.parameters_training)
parameters_optimized_def = ComponentArray(parameters_optimized_def; pars)

adtype = Optimization.AutoZygote()

##########################################################################################################################
##################################################### CONFIDENCE INTERVALS ##############################################
prob_uode_pred = ODEProblem{true}(uode_derivative_function, parameters_optimized_def.u0, (0, maximum(solution_dataframe.t)))

function model(params, final_time)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, params.u0, (0, final_time))
  solutions = solve(prob_uode_pred, integrator, p=params.pars, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
  return Array(solutions)[:, end]
end

function first_point(parameters_to_consider)
  return parameters_to_consider.u0
end

function get_covariance_matrix(parameters_to_consider)
  sensitivity_y1 = Zygote.jacobian(p -> first_point(p), parameters_to_consider)[1]
  sensitivity_y2 = Zygote.jacobian(p -> model(p, tsteps[2]), parameters_to_consider)[1]
  sensitivity_y3 = Zygote.jacobian(p -> model(p, tsteps[3]), parameters_to_consider)[1]
  sensitivity_y4 = Zygote.jacobian(p -> model(p, tsteps[4]), parameters_to_consider)[1]
  sensitivity_y5 = Zygote.jacobian(p -> model(p, tsteps[5]), parameters_to_consider)[1]
  sensitivity_y6 = Zygote.jacobian(p -> model(p, tsteps[6]), parameters_to_consider)[1]
  sensitivity_y7 = Zygote.jacobian(p -> model(p, tsteps[7]), parameters_to_consider)[1]
  sensitivity_y8 = Zygote.jacobian(p -> model(p, tsteps[8]), parameters_to_consider)[1]
  sensitivity_y9 = Zygote.jacobian(p -> model(p, tsteps[9]), parameters_to_consider)[1]
  sensitivity_y10 = Zygote.jacobian(p -> model(p, tsteps[10]), parameters_to_consider)[1]
  sensitivity_y11 = Zygote.jacobian(p -> model(p, tsteps[11]), parameters_to_consider)[1]
  sensitivity_y12 = Zygote.jacobian(p -> model(p, tsteps[12]), parameters_to_consider)[1]
  sensitivity_y13 = Zygote.jacobian(p -> model(p, tsteps[13]), parameters_to_consider)[1]
  sensitivity_y14 = Zygote.jacobian(p -> model(p, tsteps[14]), parameters_to_consider)[1]
  sensitivity_y15 = Zygote.jacobian(p -> model(p, tsteps[15]), parameters_to_consider)[1]
  sensitivity_y16 = Zygote.jacobian(p -> model(p, tsteps[16]), parameters_to_consider)[1]
  sensitivity_y17 = Zygote.jacobian(p -> model(p, tsteps[17]), parameters_to_consider)[1]
  sensitivity_y18 = Zygote.jacobian(p -> model(p, tsteps[18]), parameters_to_consider)[1]
  sensitivity_y19 = Zygote.jacobian(p -> model(p, tsteps[19]), parameters_to_consider)[1]
  sensitivity_y20 = Zygote.jacobian(p -> model(p, tsteps[20]), parameters_to_consider)[1]
  sensitivity_y21 = Zygote.jacobian(p -> model(p, tsteps[21]), parameters_to_consider)[1]

  sensitivity_matrix = vcat(sensitivity_y1, sensitivity_y2, sensitivity_y3, sensitivity_y4, sensitivity_y5, sensitivity_y6, sensitivity_y7, sensitivity_y8, sensitivity_y9, sensitivity_y10, sensitivity_y11, sensitivity_y12, sensitivity_y13, sensitivity_y14, sensitivity_y15, sensitivity_y16, sensitivity_y17, sensitivity_y18, sensitivity_y19, sensitivity_y20, sensitivity_y21)

  normalization_factor = ode_data_sd[:,1]
  normalization_matrix = vec(repeat(normalization_factor, 21))
  normalization_matrix = Diagonal(1 ./ normalization_matrix)
  normalization_matrix = abs2.(normalization_matrix)
  #sensitivity_t_d_o = sensitivity_matrix' * sensitivity_matrix

  observed_FIM = sensitivity_matrix' * normalization_matrix * sensitivity_matrix
  observed_FIM = Symmetric(observed_FIM)

  #invert the observed FIM to get the covariance matrix
  cov = pinv(observed_FIM)
  
  return cov
end

cov = get_covariance_matrix(parameters_optimized_def)

ci_p3 = [parameters_optimized_def.pars.ode_par[3] - 1.96 * sqrt(cov[end-9+3, end-9+3]), parameters_optimized_def.pars.ode_par[3] + 1.96 * sqrt(cov[end-9+3, end-9+3])]
ci_p6 = [parameters_optimized_def.pars.ode_par[6] - 1.96 * sqrt(cov[end-9+6, end-9+6]), parameters_optimized_def.pars.ode_par[6] + 1.96 * sqrt(cov[end-9+6, end-9+6])]

# write a text files with the confidence intervals
open("results/cell_ap_fisher_CI_005.txt", "w") do io
  println(io, "Confidence intervals for the parameters k3 and k6")
  println(io, "Parameter 3: ", ci_p3)
  println(io, "Parameter 6: ", ci_p6)
end

confidence_intervals = Dict()
confidence_intervals[1] = false
confidence_intervals[2] = false
confidence_intervals[3] = ci_p3
confidence_intervals[4] = false
confidence_intervals[5] = false
confidence_intervals[6] = ci_p6
confidence_intervals[7] = false
confidence_intervals[8] = false
confidence_intervals[9] = false

#write the confidence intervals to a file
Serialization.serialize("results/cell_ap_opt_05_fisher_CI.jld", confidence_intervals)