#= 
Script to assess the identifiability of the mechanistic parameters in the cell apoptosis UDE model trained on DS_00 with p2 and p4 fixed to their literature values
and only y5 and y6 are observable.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots
using Optimization, StableRNGs, LaTeXStrings
using Zygote, StatsPlots, Gadfly, DiffEqFlux

error_level = "e0.0"

#includes the specific model functions
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_settings.jl")

column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

integrator = TRBDF2(autodiff=false);
#decrease the tolerance to have a beter gradient
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

parameters_optimized = par_opt.parameters_training
parameters_optimized_def = ComponentArray{eltype(parameters_optimized.p_net)}()
u0 = ComponentArray(par_opt.initial_state_training[:, 1])
parameters_optimized_def = ComponentArray(parameters_optimized_def; u0)
pars = ComponentArray(par_opt.parameters_training)
parameters_optimized_def = ComponentArray(parameters_optimized_def; pars)

adtype = Optimization.AutoZygote()

##########################################################################################################################
##################################################### IDENTIFIABILITY      ##############################################
prob_uode_pred = ODEProblem{true}(uode_derivative_function, parameters_optimized_def.u0, (0, maximum(solution_dataframe.t)))
function model(params, final_time)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, params.u0, (0, final_time))
  solutions = solve(prob_uode_pred, integrator, p=params.pars, saveat=[0, final_time], abstol=abstol, reltol=reltol, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
  return Array(solutions)[observables, end]
end

function first_point(parameters_to_consider)
  return parameters_to_consider.u0[:]
end

function get_Hessian_Spectrum(parameters_to_consider)
  sensitivity_y1 = Zygote.jacobian(p -> first_point(p), parameters_to_consider)[1]
  sensitivity_y2 = Zygote.jacobian(p -> model(p, tsteps[2]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y3 = Zygote.jacobian(p -> model(p, tsteps[3]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y4 = Zygote.jacobian(p -> model(p, tsteps[4]), parameters_to_consider)[1].* parameters_to_consider'
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
  sensitivity_y22 = Zygote.jacobian(p -> model(p, tsteps[22]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y23 = Zygote.jacobian(p -> model(p, tsteps[23]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y24 = Zygote.jacobian(p -> model(p, tsteps[24]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y25 = Zygote.jacobian(p -> model(p, tsteps[25]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y26 = Zygote.jacobian(p -> model(p, tsteps[26]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y27 = Zygote.jacobian(p -> model(p, tsteps[27]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y28 = Zygote.jacobian(p -> model(p, tsteps[28]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y29 = Zygote.jacobian(p -> model(p, tsteps[29]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y30 = Zygote.jacobian(p -> model(p, tsteps[30]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y31 = Zygote.jacobian(p -> model(p, tsteps[31]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y32 = Zygote.jacobian(p -> model(p, tsteps[32]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y33 = Zygote.jacobian(p -> model(p, tsteps[33]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y34 = Zygote.jacobian(p -> model(p, tsteps[34]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y35 = Zygote.jacobian(p -> model(p, tsteps[35]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y36 = Zygote.jacobian(p -> model(p, tsteps[36]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y37 = Zygote.jacobian(p -> model(p, tsteps[37]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y38 = Zygote.jacobian(p -> model(p, tsteps[38]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y39 = Zygote.jacobian(p -> model(p, tsteps[39]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y40 = Zygote.jacobian(p -> model(p, tsteps[40]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y41 = Zygote.jacobian(p -> model(p, tsteps[41]), parameters_to_consider)[1] .* parameters_to_consider'
  sensitivity_y42 = Zygote.jacobian(p -> model(p, tsteps[42]), parameters_to_consider)[1] .* parameters_to_consider'

  sensitivity_matrix = vcat(sensitivity_y1, sensitivity_y2, sensitivity_y3, sensitivity_y4, sensitivity_y5, sensitivity_y6, sensitivity_y7, sensitivity_y8, sensitivity_y9, sensitivity_y10, sensitivity_y11, sensitivity_y12, sensitivity_y13, sensitivity_y14, sensitivity_y15, sensitivity_y16, sensitivity_y17, sensitivity_y18, sensitivity_y19, sensitivity_y20, sensitivity_y21, sensitivity_y22, sensitivity_y23, sensitivity_y24, sensitivity_y25, sensitivity_y26, sensitivity_y27, sensitivity_y28, sensitivity_y29, sensitivity_y30, sensitivity_y31, sensitivity_y32, sensitivity_y33, sensitivity_y34, sensitivity_y35, sensitivity_y36, sensitivity_y37, sensitivity_y38, sensitivity_y39, sensitivity_y40, sensitivity_y41, sensitivity_y42)

  first_normalization_factor = maximum(ode_data[:,1:end], dims=2)
  normalization_factor = maximum(ode_data[observables,1:end], dims=2)

  normalization_matrix = vec(vcat(repeat(first_normalization_factor, 1), repeat(normalization_factor, 41)))

  normalization_matrix = Diagonal(1 ./ normalization_matrix)
  normalization_matrix = abs2.(normalization_matrix)
  #sensitivity_t_d_o = sensitivity_matrix' * sensitivity_matrix

  hessian = sensitivity_matrix' * normalization_matrix * sensitivity_matrix .* 1 / size(solution_dataframe, 1) .* 1 / (size(solution_dataframe, 2) - 1)
  hessian = Symmetric(hessian)
  eigen_value_decomposition = eigen(hessian)

  eigen_values = real.(eigen_value_decomposition.values)
  eigen_vectors = real.(eigen_value_decomposition.vectors)'

  eigen_vectors_with_eigen_values = hcat(eigen_vectors, eigen_values)

  return eigen_vectors_with_eigen_values
end

eigen_vectors_with_eigen_values = get_Hessian_Spectrum(parameters_optimized_def)
null_direction_dataframe = eigen_vectors_with_eigen_values[abs.(eigen_vectors_with_eigen_values[:, end]).<1e-5, :]

#computes the projection on the \Chi null space
function get_projection_on_null_space(null_direction_dataframe, par_index)
  #sort the matrix by the eigenvalues
  parameter_versor = zeros(size(parameters_optimized_def))
  parameter_versor[end-7+par_index] = 1

  projection = zeros(size(parameters_optimized_def))
  for i in 1:size(null_direction_dataframe)[1]
    projection += dot(parameter_versor,null_direction_dataframe[i, 1:end-1]') .* null_direction_dataframe[i, 1:end-1]
  end

  return projection
end

#analyzes the projections of the mechanistic parameters on the null space
projection_p1 = get_projection_on_null_space(null_direction_dataframe, 1)
projection_p3 = get_projection_on_null_space(null_direction_dataframe, 2)
projection_p5 = get_projection_on_null_space(null_direction_dataframe, 3)
projection_p6 = get_projection_on_null_space(null_direction_dataframe, 4)
projection_p7 = get_projection_on_null_space(null_direction_dataframe, 5)
projection_p8 = get_projection_on_null_space(null_direction_dataframe, 6)
projection_p9 = get_projection_on_null_space(null_direction_dataframe, 7)

function get_squared_component(proj, ind)
  if ind == 0
    return sum(abs2.(proj[9:(end-7)]))
  else
    return sum(abs2.(proj[end-7+ind]))
  end
end

component_nn = [get_squared_component(projection_p1, 0), get_squared_component(projection_p3, 0), get_squared_component(projection_p5, 0), get_squared_component(projection_p6, 0), get_squared_component(projection_p7, 0), get_squared_component(projection_p8, 0), get_squared_component(projection_p9, 0)]
component_p1 = [get_squared_component(projection_p1, 1), get_squared_component(projection_p3, 1), get_squared_component(projection_p5, 1), get_squared_component(projection_p6, 1), get_squared_component(projection_p7, 1), get_squared_component(projection_p8, 1), get_squared_component(projection_p9, 1)]
component_p3 = [get_squared_component(projection_p1, 2), get_squared_component(projection_p3, 2), get_squared_component(projection_p5, 2), get_squared_component(projection_p6, 2), get_squared_component(projection_p7, 2), get_squared_component(projection_p8, 2), get_squared_component(projection_p9, 2)]
component_p5 = [get_squared_component(projection_p1, 3), get_squared_component(projection_p3, 3), get_squared_component(projection_p5, 3), get_squared_component(projection_p6, 3), get_squared_component(projection_p7, 3), get_squared_component(projection_p8, 3), get_squared_component(projection_p9, 3)]
component_p6 = [get_squared_component(projection_p1, 4), get_squared_component(projection_p3, 4), get_squared_component(projection_p5, 4), get_squared_component(projection_p6, 4), get_squared_component(projection_p7, 4), get_squared_component(projection_p8, 4), get_squared_component(projection_p9, 4)]
component_p7 = [get_squared_component(projection_p1, 5), get_squared_component(projection_p3, 5), get_squared_component(projection_p5, 5), get_squared_component(projection_p6, 5), get_squared_component(projection_p7, 5), get_squared_component(projection_p8, 5), get_squared_component(projection_p9, 5)]
component_p8 = [get_squared_component(projection_p1, 6), get_squared_component(projection_p3, 6), get_squared_component(projection_p5, 6), get_squared_component(projection_p6, 6), get_squared_component(projection_p7, 6), get_squared_component(projection_p8, 6), get_squared_component(projection_p9, 6)]
component_p9 = [get_squared_component(projection_p1, 7), get_squared_component(projection_p3, 7), get_squared_component(projection_p5, 7), get_squared_component(projection_p6, 7), get_squared_component(projection_p7, 7), get_squared_component(projection_p8, 7), get_squared_component(projection_p9, 7)]

gr()

plot_font = "arial"
Plots.default(fontfamily=plot_font)

plt = Plots.plot(xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, dpi=300)
Plots.plot!(plt, legend=:outerright, foreground_color_legend = nothing, size = (1500, 500), left_margin = 10mm,  bottom_margin = 10mm, top_margin = 1mm, palette = :mk_12)
groupedbar!(plt, [component_nn component_p1 component_p3 component_p5 component_p6 component_p7 component_p8 component_p9],
        bar_position = :stack,
        bar_width=0.3,
        xticks = ([1, 2, 3, 4, 5, 6, 7], [latexstring("\\pi (k_1)"), latexstring("\\pi(k_{d2})"), latexstring("\\pi (k_{d3})"), latexstring("\\pi (k_{d4})"), latexstring("\\pi (k_{5})"), latexstring("\\pi (k_{d5})"), latexstring("\\pi (k_{d6})")]),
        ylims=(0, 1),
        label=[latexstring("NN") latexstring("k_1") latexstring("k_{d2}") latexstring("k_{d3}") latexstring("k_{d4}") latexstring("k_{5}") latexstring("k_{d5}") latexstring("k_{d6}")])
#plot an horizontal line
hline!(plt, [0.05], color="red", linestyle = :dot, linewidth=2, label="")
yaxis!(plt, "Squared norm",)

Plots.svg(plt, "plots/composition_projections_cell_ap_e00_fixed_p2p4_observable_56.svg")
