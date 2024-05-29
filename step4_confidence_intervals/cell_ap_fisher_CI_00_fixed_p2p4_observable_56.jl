#= 
Script to derive the confidence intervals for the identifiable mechanistic parameters in the cell apoptosis UDE model trained on DS_00 fixing the parameters p2 and p4 fixed to their literature values
and assumng only y5 and y6 observable.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes

error_level = "e0.0"

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

#observable variables
observables = [5,6]


my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
  Lux.Dense(6, 2^3, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2^3, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform)
)
#reads the estimated parameters
par_opt = deserialize("local_optima_found/cell_ap_opt_00_fixed_p2p4_observable_56.jld")

ode_data = deserialize("../datasets/" * error_level * "_doubled/data/ode_data_cell_apoptosis.jld")
ode_data_sd = deserialize("../datasets/" * error_level * "_doubled/data/ode_data_std_cell_apoptosis.jld")
solution_dataframe = deserialize("../datasets/" * error_level * "_doubled/data/pert_df_cell_apoptosis.jld")
solution_sd_dataframe = deserialize("../datasets/" * error_level * "_doubled/data/pert_df_sd_cell_apoptosis.jld")

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)

#settings for the training set sampling
tspan = (initial_time_training, end_time_training)
tsteps = solution_dataframe.t
parameters_optimized = par_opt.parameters_training
uode_derivative_function = get_uode_model_function_fixed_p2p4(approximating_neural_network, par_opt.net_status, ones(length(parameters_optimized.ode_par)))

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
  return Array(solutions)[observables, end]
end

function first_point(parameters_to_consider)
  return parameters_to_consider.u0[:]
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
  sensitivity_y22 = Zygote.jacobian(p -> model(p, tsteps[22]), parameters_to_consider)[1]
  sensitivity_y23 = Zygote.jacobian(p -> model(p, tsteps[23]), parameters_to_consider)[1]
  sensitivity_y24 = Zygote.jacobian(p -> model(p, tsteps[24]), parameters_to_consider)[1]
  sensitivity_y25 = Zygote.jacobian(p -> model(p, tsteps[25]), parameters_to_consider)[1]
  sensitivity_y26 = Zygote.jacobian(p -> model(p, tsteps[26]), parameters_to_consider)[1]
  sensitivity_y27 = Zygote.jacobian(p -> model(p, tsteps[27]), parameters_to_consider)[1]
  sensitivity_y28 = Zygote.jacobian(p -> model(p, tsteps[28]), parameters_to_consider)[1]
  sensitivity_y29 = Zygote.jacobian(p -> model(p, tsteps[29]), parameters_to_consider)[1]
  sensitivity_y30 = Zygote.jacobian(p -> model(p, tsteps[30]), parameters_to_consider)[1]
  sensitivity_y31 = Zygote.jacobian(p -> model(p, tsteps[31]), parameters_to_consider)[1]
  sensitivity_y32 = Zygote.jacobian(p -> model(p, tsteps[32]), parameters_to_consider)[1]
  sensitivity_y33 = Zygote.jacobian(p -> model(p, tsteps[33]), parameters_to_consider)[1]
  sensitivity_y34 = Zygote.jacobian(p -> model(p, tsteps[34]), parameters_to_consider)[1]
  sensitivity_y35 = Zygote.jacobian(p -> model(p, tsteps[35]), parameters_to_consider)[1]
  sensitivity_y36 = Zygote.jacobian(p -> model(p, tsteps[36]), parameters_to_consider)[1]
  sensitivity_y37 = Zygote.jacobian(p -> model(p, tsteps[37]), parameters_to_consider)[1]
  sensitivity_y38 = Zygote.jacobian(p -> model(p, tsteps[38]), parameters_to_consider)[1]
  sensitivity_y39 = Zygote.jacobian(p -> model(p, tsteps[39]), parameters_to_consider)[1]
  sensitivity_y40 = Zygote.jacobian(p -> model(p, tsteps[40]), parameters_to_consider)[1]
  sensitivity_y41 = Zygote.jacobian(p -> model(p, tsteps[41]), parameters_to_consider)[1]
  sensitivity_y42 = Zygote.jacobian(p -> model(p, tsteps[42]), parameters_to_consider)[1]

  sensitivity_matrix = vcat(sensitivity_y1, sensitivity_y2, sensitivity_y3, sensitivity_y4, sensitivity_y5, sensitivity_y6, sensitivity_y7, sensitivity_y8, sensitivity_y9, sensitivity_y10, sensitivity_y11, sensitivity_y12, sensitivity_y13, sensitivity_y14, sensitivity_y15, sensitivity_y16, sensitivity_y17, sensitivity_y18, sensitivity_y19, sensitivity_y20, sensitivity_y21, sensitivity_y22, sensitivity_y23, sensitivity_y24, sensitivity_y25, sensitivity_y26, sensitivity_y27, sensitivity_y28, sensitivity_y29, sensitivity_y30, sensitivity_y31, sensitivity_y32, sensitivity_y33, sensitivity_y34, sensitivity_y35, sensitivity_y36, sensitivity_y37, sensitivity_y38, sensitivity_y39, sensitivity_y40, sensitivity_y41, sensitivity_y42)

  first_normalization_factor = maximum(ode_data[:,1:end], dims=2)
  normalization_factor = maximum(ode_data[observables,1:end], dims=2)

  normalization_matrix = vec(vcat(repeat(first_normalization_factor, 1), repeat(normalization_factor, 41)))

  normalization_matrix = Diagonal(1 ./ (0.005 .* normalization_matrix))
  normalization_matrix = abs2.(normalization_matrix)
  #sensitivity_t_d_o = sensitivity_matrix' * sensitivity_matrix

  observed_FIM = sensitivity_matrix' * normalization_matrix * sensitivity_matrix
  observed_FIM = Symmetric(observed_FIM)

  #invert the matrix to get the covariance matrix
  cov = pinv(observed_FIM)
  
  return cov
end

cov = get_covariance_matrix(parameters_optimized_def)

ci_p1 = [parameters_optimized_def.pars.ode_par[1] - 1.96 * sqrt(cov[end-7+1, end-7+1]), parameters_optimized_def.pars.ode_par[1] + 1.96 * sqrt(cov[end-7+1, end-7+1])]
ci_p3 = [parameters_optimized_def.pars.ode_par[2] - 1.96 * sqrt(cov[end-7+2, end-7+2]), parameters_optimized_def.pars.ode_par[2] + 1.96 * sqrt(cov[end-7+2, end-7+2])]
ci_p5 = [parameters_optimized_def.pars.ode_par[3] - 1.96 * sqrt(cov[end-7+3, end-7+3]), parameters_optimized_def.pars.ode_par[3] + 1.96 * sqrt(cov[end-7+3, end-7+3])]
ci_p6 = [parameters_optimized_def.pars.ode_par[4] - 1.96 * sqrt(cov[end-7+4, end-7+4]), parameters_optimized_def.pars.ode_par[4] + 1.96 * sqrt(cov[end-7+4, end-7+4])]

# write a text files with the confidence intervals
open("results/cell_ap_fisher_CI_000_fixed_p2p4_observable_56.txt", "w") do io
  println(io, "Confidence intervals for the parameters k1, k3 and k5, k6")
  println(io, "k1: ", ci_p1)
  println(io, "k3: ", ci_p3)
  println(io, "k5: ", ci_p5)
  println(io, "k6: ", ci_p6)
end

confidence_intervals = Dict()
confidence_intervals[1] = false
confidence_intervals[2] = ci_p3
confidence_intervals[3] = ci_p5
confidence_intervals[4] = ci_p6
confidence_intervals[5] = false
confidence_intervals[6] = false
confidence_intervals[7] = false

#write the confidence intervals to a file
Serialization.serialize("results/cell_ap_opt_00_fisher_CI_fixed_p2p4_observable_56.jld", confidence_intervals)

