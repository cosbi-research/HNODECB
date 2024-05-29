#= 
Script to assess the identifiability of the mechanistic parameters in the Lotka Volterra UDE model trained on DS_00
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly

error_level = "e0.0"

#create the plot directory
if !isdir("plots")
  mkdir("plots")
end

include("../test_case_settings/lv_model_settings/lotka_volterra_model_functions.jl")
include("../test_case_settings/lv_model_settings/lotka_volterra_model_settings.jl")

column_names = ["t", "s1", "s2"]

integrator = Vern7()
abstol = 1e-7
reltol = 1e-6
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

# neural network
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
  Lux.Dense(2, 16, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(16, 16, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(16, 2; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)
#reads the estimated parameters
par_opt = deserialize("local_optima_found/lv_opt_00.jld")

ode_data = deserialize("../datasets/e0.0/data/ode_data_lotka_volterra.jld")
ode_data_sd = deserialize("../datasets/e0.0/data/ode_data_std_lotka_volterra.jld")
solution_dataframe = deserialize("../datasets/e0.0/data/pert_df_lotka_volterra.jld")
solution_sd_dataframe = deserialize("../datasets/e0.0/data/pert_df_sd_lotka_volterra.jld")

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)

#define the model
tspan = (initial_time_training, end_time_training)
tsteps = solution_dataframe.t
parameters_optimized = par_opt.parameters_training
uode_derivative_function = get_uode_model_function(approximating_neural_network, par_opt.net_status, 1)

#creates the Component Array with all the parameters needed to calculcate the Jacobian
parameters_optimized = ComponentArray(parameters_optimized)
parameters_optimized_def = ComponentArray{eltype(parameters_optimized.p_net)}()
u0 = ComponentArray(par_opt.initial_state_training[:, 1])
parameters_optimized_def = ComponentArray(parameters_optimized_def; u0)
pars = ComponentArray(parameters_optimized)
parameters_optimized_def = ComponentArray(parameters_optimized_def; pars)

adtype = Optimization.AutoZygote()

##########################################################################################################################
##################################################### IDENTIFIABILITY      ##############################################
prob_uode_pred = ODEProblem{true}(uode_derivative_function, par_opt.initial_state_training, (0, maximum(solution_dataframe.t)))

# functions that compute the model at a specific time
function model(params, final_time)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, params.u0, (0, final_time))
  solutions = solve(prob_uode_pred, integrator, p=params.pars, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
  return Array(solutions)[:, end]
end

function first_point(parameters_to_consider)
  return parameters_to_consider.u0
end

function get_Hessian_Spectrum(parameters_to_consider)
  sensitivity_y1 = Zygote.jacobian(p -> first_point(p), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y2 = Zygote.jacobian(p -> model(p, tsteps[2]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y3 = Zygote.jacobian(p -> model(p, tsteps[3]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y4 = Zygote.jacobian(p -> model(p, tsteps[4]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y5 = Zygote.jacobian(p -> model(p, tsteps[5]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y6 = Zygote.jacobian(p -> model(p, tsteps[6]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y7 = Zygote.jacobian(p -> model(p, tsteps[7]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y8 = Zygote.jacobian(p -> model(p, tsteps[8]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y9 = Zygote.jacobian(p -> model(p, tsteps[9]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y10 = Zygote.jacobian(p -> model(p, tsteps[10]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y11 = Zygote.jacobian(p -> model(p, tsteps[11]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y12 = Zygote.jacobian(p -> model(p, tsteps[12]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y13 = Zygote.jacobian(p -> model(p, tsteps[13]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y14 = Zygote.jacobian(p -> model(p, tsteps[14]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y15 = Zygote.jacobian(p -> model(p, tsteps[15]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y16 = Zygote.jacobian(p -> model(p, tsteps[16]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y17 = Zygote.jacobian(p -> model(p, tsteps[17]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y18 = Zygote.jacobian(p -> model(p, tsteps[18]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y19 = Zygote.jacobian(p -> model(p, tsteps[19]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y20 = Zygote.jacobian(p -> model(p, tsteps[20]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y21 = Zygote.jacobian(p -> model(p, tsteps[21]), parameters_to_consider)[1] .* parameters_to_consider'

  sensitivity_matrix = vcat(sensitivity_y1, sensitivity_y2, sensitivity_y3, sensitivity_y4, sensitivity_y5, sensitivity_y6, sensitivity_y7, sensitivity_y8, sensitivity_y9, sensitivity_y10, sensitivity_y11, sensitivity_y12, sensitivity_y13, sensitivity_y14, sensitivity_y15, sensitivity_y16, sensitivity_y17, sensitivity_y18, sensitivity_y19, sensitivity_y20, sensitivity_y21)

  normalization_factor = maximum(ode_data, dims=2)
  normalization_matrix = vec(repeat(normalization_factor, 21))
  normalization_matrix = Diagonal(1 ./ normalization_matrix)
  normalization_matrix = abs2.(normalization_matrix)

  hessian = sensitivity_matrix' * normalization_matrix * sensitivity_matrix .* 1 / size(solution_dataframe, 1) .* 1 / (size(solution_dataframe, 2) - 1)
  hessian = Symmetric(hessian)
  eigen_value_decomposition = eigen(hessian)

  eigen_values = real.(eigen_value_decomposition.values)
  eigen_vectors = real.(eigen_value_decomposition.vectors)'

  eigen_vectors_with_eigen_values = hcat(eigen_vectors, eigen_values)

  return eigen_vectors_with_eigen_values
end

eigen_vectors_with_eigen_values = get_Hessian_Spectrum(parameters_optimized_def)

################################### delta = 10^-4 ########################################
null_direction_dataframe = eigen_vectors_with_eigen_values[abs.(eigen_vectors_with_eigen_values[:, end]).<1e-4, :]

#computes the projection on the \Chi null space
function get_projection_on_null_space(null_direction_dataframe, par_index)
  #sort the matrix by the eigenvalues
  parameter_versor = zeros(size(parameters_optimized_def))
  parameter_versor[end-1+par_index] = 1

  projection = zeros(size(parameters_optimized_def))
  for i in 1:size(null_direction_dataframe)[1]
    projection += dot(parameter_versor,null_direction_dataframe[i, 1:end-1]') .* null_direction_dataframe[i, 1:end-1]
  end

  return projection
end

#divides the projection norm in the components related to the model parameters
projection_alpha = get_projection_on_null_space(null_direction_dataframe, 1)
nn_components = [sum(abs2.(projection_alpha[3:end-1]))]
alpha_components = [sum(abs2.(projection_alpha[end]))] 
is_components = [sum(abs2.(projection_alpha[1:2]))]

#Plots the results
gr()

plot_font = "Arial"
Plots.default(fontfamily=plot_font)

plt = Plots.plot(xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, dpi=300)
Plots.plot!(plt, legend=false, foreground_color_legend = nothing, size = (400, 500), left_margin = 10mm,  bottom_margin = 1mm, top_margin = 1mm, palette = :mk_12)
groupedbar!(plt, [alpha_components nn_components],
        bar_position = :stack,
        bar_width=0.25,
        xticks=(1, [latexstring("\\pi(\\alpha)"),]),
        xlims=(0.8, 1.2),
        ylims=(0, 1),
        label=[ "α" "NN"])
#plot an horizontal line
hline!(plt, [0.05], color="red", linestyle = :dot, linewidth=2, label="")
hline!(plt, [0.07], color="black", linestyle = :dot, linewidth=2, label="")
hline!(plt, [0.03], color="black", linestyle = :dot, linewidth=2, label="")
yaxis!(plt, "Squared norm",foreground_color_grid=:lightgrey)
xaxis!(plt, foreground_color_grid=:lightgrey)

Plots.svg(plt, "plots/composition_projections_lv_e00_delta_4.svg")

plt = Plots.plot(xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, dpi=300)
Plots.plot!(plt, legend=:outerright, foreground_color_legend = nothing, size = (400, 500), left_margin = 10mm,  bottom_margin = 1mm, top_margin = 1mm, palette = :mk_12)
groupedbar!(plt, [alpha_components nn_components],
        bar_position = :stack,
        bar_width=0.25,
        xticks=(1, [latexstring("\\pi(\\alpha)"),]),
        xlims=(0.8, 1.2),
        ylims=(0, 1),
        label=["α" "NN"])
#plot an horizontal line
hline!(plt, [0.05], color="red", linestyle = :dot, linewidth=2, label="")
yaxis!(plt, "Squared norm",foreground_color_grid=:lightgrey)
xaxis!(plt, foreground_color_grid=:lightgrey)

Plots.svg(plt, "plots/composition_projections_lv_legend.svg")

################################### delta = 10^-5 ########################################
null_direction_dataframe = eigen_vectors_with_eigen_values[abs.(eigen_vectors_with_eigen_values[:, end]).<1e-5, :]

#computes the projection on the \Chi null space
function get_projection_on_null_space(null_direction_dataframe, par_index)
  #sort the matrix by the eigenvalues
  parameter_versor = zeros(size(parameters_optimized_def))
  parameter_versor[end-1+par_index] = 1

  projection = zeros(size(parameters_optimized_def))
  for i in 1:size(null_direction_dataframe)[1]
    projection += dot(parameter_versor,null_direction_dataframe[i, 1:end-1]') .* null_direction_dataframe[i, 1:end-1]
  end

  return projection
end

#divides the projection norm in the components related to the model parameters
projection_alpha = get_projection_on_null_space(null_direction_dataframe, 1)
nn_components = [sum(abs2.(projection_alpha[3:end-1]))]
alpha_components = [sum(abs2.(projection_alpha[end]))] 
is_components = [sum(abs2.(projection_alpha[1:2]))]

#Plots the results
gr()

plot_font = "Arial"
Plots.default(fontfamily=plot_font)

plt = Plots.plot(xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, dpi=300)
Plots.plot!(plt, legend=false, foreground_color_legend = nothing, size = (400, 500), left_margin = 10mm,  bottom_margin = 1mm, top_margin = 1mm, palette = :mk_12)
groupedbar!(plt, [alpha_components nn_components],
        bar_position = :stack,
        bar_width=0.25,
        xticks=(1, [latexstring("\\pi(\\alpha)"),]),
        xlims=(0.8, 1.2),
        ylims=(0, 1),
        label=[ "α" "NN"])
#plot an horizontal line
hline!(plt, [0.05], color="red", linestyle = :dot, linewidth=2, label="")
hline!(plt, [0.07], color="black", linestyle = :dot, linewidth=2, label="")
hline!(plt, [0.03], color="black", linestyle = :dot, linewidth=2, label="")
yaxis!(plt, "Squared norm",foreground_color_grid=:lightgrey)
xaxis!(plt, foreground_color_grid=:lightgrey)

Plots.svg(plt, "plots/composition_projections_lv_e00_delta_5.svg")


################################### delta = 10^-6 ########################################
null_direction_dataframe = eigen_vectors_with_eigen_values[abs.(eigen_vectors_with_eigen_values[:, end]).<1e-6, :]

#computes the projection on the \Chi null space
function get_projection_on_null_space(null_direction_dataframe, par_index)
  #sort the matrix by the eigenvalues
  parameter_versor = zeros(size(parameters_optimized_def))
  parameter_versor[end-1+par_index] = 1

  projection = zeros(size(parameters_optimized_def))
  for i in 1:size(null_direction_dataframe)[1]
    projection += dot(parameter_versor,null_direction_dataframe[i, 1:end-1]') .* null_direction_dataframe[i, 1:end-1]
  end

  return projection
end

#divides the projection norm in the components related to the model parameters
projection_alpha = get_projection_on_null_space(null_direction_dataframe, 1)
nn_components = [sum(abs2.(projection_alpha[3:end-1]))]
alpha_components = [sum(abs2.(projection_alpha[end]))] 
is_components = [sum(abs2.(projection_alpha[1:2]))]

#Plots the results
gr()

plot_font = "Arial"
Plots.default(fontfamily=plot_font)

plt = Plots.plot(xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, dpi=300)
Plots.plot!(plt, legend=false, foreground_color_legend = nothing, size = (400, 500), left_margin = 10mm,  bottom_margin = 1mm, top_margin = 1mm, palette = :mk_12)
groupedbar!(plt, [alpha_components nn_components],
        bar_position = :stack,
        bar_width=0.25,
        xticks=(1, [latexstring("\\pi(\\alpha)"),]),
        xlims=(0.8, 1.2),
        ylims=(0, 1),
        label=[ "α" "NN"])
#plot an horizontal line
hline!(plt, [0.05], color="red", linestyle = :dot, linewidth=2, label="")
hline!(plt, [0.07], color="black", linestyle = :dot, linewidth=2, label="")
hline!(plt, [0.03], color="black", linestyle = :dot, linewidth=2, label="")
yaxis!(plt, "Squared norm",foreground_color_grid=:lightgrey)
xaxis!(plt, foreground_color_grid=:lightgrey)

Plots.svg(plt, "plots/composition_projections_lv_e00_delta_6.svg")

