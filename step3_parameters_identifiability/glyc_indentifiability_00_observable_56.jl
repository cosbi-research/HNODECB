#= 
Script to assess the identifiability of the mechanistic parameters in the Yeast glycolyusis UDE model trained on DS_00 assuming only y5 and y6 observables
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux, Zygote, StatsPlots, LaTeXStrings, Gadfly, ColorSchemes

error_level = "e0.0"

include("../test_case_settings/glyc_model_settings/glycolitic_model_functions.jl")
include("../test_case_settings/glyc_model_settings/glycolitic_model_settings.jl")

column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]

integrator = TRBDF2(autodiff=false);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#observable variables
observables = [5,6]

#neural networks
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
      Lux.Dense(2, 16, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(16, 16, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(16, 16, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(16, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)
#reads the estimated parameters
par_opt = deserialize("local_optima_found/glyc_opt_00_observables_56.jld")

ode_data = deserialize("../datasets/"*error_level * "_doubled/data/ode_data_glycolysis.jld")
ode_data_sd = deserialize("../datasets/"*error_level * "_doubled/data/ode_data_std_glycolysis.jld")
solution_dataframe = deserialize("../datasets/"*error_level * "_doubled/data/pert_df_glycolysis.jld")
solution_sd_dataframe = deserialize("../datasets/"*error_level * "_doubled/data/pert_df_sd_glycolysis.jld")

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)

#settings for the training set sampling
tspan = (initial_time_training, end_time_training)
tsteps = solution_dataframe.t
parameters_optimized = par_opt.parameters_training
uode_derivative_function = get_uode_model_function(approximating_neural_network, par_opt.net_status, ones(length(parameters_optimized.ode_par)))

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
  parameter_versor[end-14+par_index] = 1

  projection = zeros(size(parameters_optimized_def))
  for i in 1:size(null_direction_dataframe)[1]
    projection += dot(parameter_versor,null_direction_dataframe[i, 1:end-1]') .* null_direction_dataframe[i, 1:end-1]
  end

  return projection
end

#analyzes the projections of the mechanistic parameters on the null space
projection_p1 = get_projection_on_null_space(null_direction_dataframe, 1)
projection_p2 = get_projection_on_null_space(null_direction_dataframe, 2)
projection_p3 = get_projection_on_null_space(null_direction_dataframe, 3)
projection_p4 = get_projection_on_null_space(null_direction_dataframe, 4)
projection_p5 = get_projection_on_null_space(null_direction_dataframe, 5)
projection_p6 = get_projection_on_null_space(null_direction_dataframe, 6)
projection_p7 = get_projection_on_null_space(null_direction_dataframe, 7)
projection_p8 = get_projection_on_null_space(null_direction_dataframe, 8)
projection_p9 = get_projection_on_null_space(null_direction_dataframe, 9)
projection_p10 = get_projection_on_null_space(null_direction_dataframe, 10)
projection_p11 = get_projection_on_null_space(null_direction_dataframe, 11)
projection_p12 = get_projection_on_null_space(null_direction_dataframe, 12)
projection_p13 = get_projection_on_null_space(null_direction_dataframe, 13)
projection_p14 = get_projection_on_null_space(null_direction_dataframe, 14)

function get_squared_component(proj, ind)
  if ind == 0
    return sum(abs2.(proj[8:(end-14)]))
  else
    return sum(abs2.(proj[end-14+ind]))
  end
end

component_nn = [get_squared_component(projection_p2, 0), get_squared_component(projection_p3, 0), get_squared_component(projection_p4, 0), get_squared_component(projection_p5, 0), get_squared_component(projection_p6, 0), get_squared_component(projection_p7, 0), get_squared_component(projection_p8, 0), get_squared_component(projection_p9, 0), get_squared_component(projection_p10, 0), get_squared_component(projection_p11, 0), get_squared_component(projection_p12, 0), get_squared_component(projection_p13, 0), get_squared_component(projection_p14, 0)]
component_p1 = [get_squared_component(projection_p2, 1), get_squared_component(projection_p3, 1), get_squared_component(projection_p4, 1), get_squared_component(projection_p5, 1), get_squared_component(projection_p6, 1), get_squared_component(projection_p7, 1), get_squared_component(projection_p8, 1), get_squared_component(projection_p9, 1), get_squared_component(projection_p10, 1), get_squared_component(projection_p11, 1), get_squared_component(projection_p12, 1), get_squared_component(projection_p13, 1), get_squared_component(projection_p14, 1)]
component_p2 = [get_squared_component(projection_p2, 2), get_squared_component(projection_p3, 2), get_squared_component(projection_p4, 2), get_squared_component(projection_p5, 2), get_squared_component(projection_p6, 2), get_squared_component(projection_p7, 2), get_squared_component(projection_p8, 2), get_squared_component(projection_p9, 2), get_squared_component(projection_p10, 2), get_squared_component(projection_p11, 2), get_squared_component(projection_p12, 2), get_squared_component(projection_p13, 2), get_squared_component(projection_p14, 2)]
component_p3 = [get_squared_component(projection_p2, 3), get_squared_component(projection_p3, 3), get_squared_component(projection_p4, 3), get_squared_component(projection_p5, 3), get_squared_component(projection_p6, 3), get_squared_component(projection_p7, 3), get_squared_component(projection_p8, 3), get_squared_component(projection_p9, 3), get_squared_component(projection_p10, 3), get_squared_component(projection_p11, 3), get_squared_component(projection_p12, 3), get_squared_component(projection_p13, 3), get_squared_component(projection_p14, 3)]
component_p4 = [get_squared_component(projection_p2, 4), get_squared_component(projection_p3, 4), get_squared_component(projection_p4, 4), get_squared_component(projection_p5, 4), get_squared_component(projection_p6, 4), get_squared_component(projection_p7, 4), get_squared_component(projection_p8, 4), get_squared_component(projection_p9, 4), get_squared_component(projection_p10, 4), get_squared_component(projection_p11, 4), get_squared_component(projection_p12, 4), get_squared_component(projection_p13, 4), get_squared_component(projection_p14, 4)]
component_p5 = [get_squared_component(projection_p2, 5), get_squared_component(projection_p3, 5), get_squared_component(projection_p4, 5), get_squared_component(projection_p5, 5), get_squared_component(projection_p6, 5), get_squared_component(projection_p7, 5), get_squared_component(projection_p8, 5), get_squared_component(projection_p9, 5), get_squared_component(projection_p10, 5), get_squared_component(projection_p11, 5), get_squared_component(projection_p12, 5), get_squared_component(projection_p13, 5), get_squared_component(projection_p14, 5)]
component_p6 = [get_squared_component(projection_p2, 6), get_squared_component(projection_p3, 6), get_squared_component(projection_p4, 6), get_squared_component(projection_p5, 6), get_squared_component(projection_p6, 6), get_squared_component(projection_p7, 6), get_squared_component(projection_p8, 6), get_squared_component(projection_p9, 6), get_squared_component(projection_p10, 6), get_squared_component(projection_p11, 6), get_squared_component(projection_p12, 6), get_squared_component(projection_p13, 6), get_squared_component(projection_p14, 6)]
component_p7 = [get_squared_component(projection_p2, 7), get_squared_component(projection_p3, 7), get_squared_component(projection_p4, 7), get_squared_component(projection_p5, 7), get_squared_component(projection_p6, 7), get_squared_component(projection_p7, 7), get_squared_component(projection_p8, 7), get_squared_component(projection_p9, 7), get_squared_component(projection_p10, 7), get_squared_component(projection_p11, 7), get_squared_component(projection_p12, 7), get_squared_component(projection_p13, 7), get_squared_component(projection_p14, 7)]
component_p8 = [get_squared_component(projection_p2, 8), get_squared_component(projection_p3, 8), get_squared_component(projection_p4, 8), get_squared_component(projection_p5, 8), get_squared_component(projection_p6, 8), get_squared_component(projection_p7, 8), get_squared_component(projection_p8, 8), get_squared_component(projection_p9, 8), get_squared_component(projection_p10, 8), get_squared_component(projection_p11, 8), get_squared_component(projection_p12, 8), get_squared_component(projection_p13, 8), get_squared_component(projection_p14, 8)]
component_p9 = [get_squared_component(projection_p2, 9), get_squared_component(projection_p3, 9), get_squared_component(projection_p4, 9), get_squared_component(projection_p5, 9), get_squared_component(projection_p6, 9), get_squared_component(projection_p7, 9), get_squared_component(projection_p8, 9), get_squared_component(projection_p9, 9), get_squared_component(projection_p10, 9), get_squared_component(projection_p11, 9), get_squared_component(projection_p12, 9), get_squared_component(projection_p13, 9), get_squared_component(projection_p14, 9)]
component_p10 = [get_squared_component(projection_p2, 10), get_squared_component(projection_p3, 10), get_squared_component(projection_p4, 10), get_squared_component(projection_p5, 10), get_squared_component(projection_p6, 10), get_squared_component(projection_p7, 10), get_squared_component(projection_p8, 10), get_squared_component(projection_p9, 10), get_squared_component(projection_p10, 10), get_squared_component(projection_p11, 10), get_squared_component(projection_p12, 10), get_squared_component(projection_p13, 10), get_squared_component(projection_p14, 10)]
component_p11 = [get_squared_component(projection_p2, 11), get_squared_component(projection_p3, 11), get_squared_component(projection_p4, 11), get_squared_component(projection_p5, 11), get_squared_component(projection_p6, 11), get_squared_component(projection_p7, 11), get_squared_component(projection_p8, 11), get_squared_component(projection_p9, 11), get_squared_component(projection_p10, 11), get_squared_component(projection_p11, 11), get_squared_component(projection_p12, 11), get_squared_component(projection_p13, 11), get_squared_component(projection_p14, 11)]
component_p12 = [get_squared_component(projection_p2, 12), get_squared_component(projection_p3, 12), get_squared_component(projection_p4, 12), get_squared_component(projection_p5, 12), get_squared_component(projection_p6, 12), get_squared_component(projection_p7, 12), get_squared_component(projection_p8, 12), get_squared_component(projection_p9, 12), get_squared_component(projection_p10, 12), get_squared_component(projection_p11, 12), get_squared_component(projection_p12, 12), get_squared_component(projection_p13, 12), get_squared_component(projection_p14, 12)]
component_p13 = [get_squared_component(projection_p2, 13), get_squared_component(projection_p3, 13), get_squared_component(projection_p4, 13), get_squared_component(projection_p5, 13), get_squared_component(projection_p6, 13), get_squared_component(projection_p7, 13), get_squared_component(projection_p8, 13), get_squared_component(projection_p9, 13), get_squared_component(projection_p10, 13), get_squared_component(projection_p11, 13), get_squared_component(projection_p12, 13), get_squared_component(projection_p13, 13), get_squared_component(projection_p14, 13)]
component_p14 = [get_squared_component(projection_p2, 14), get_squared_component(projection_p3, 14), get_squared_component(projection_p4, 14), get_squared_component(projection_p5, 14), get_squared_component(projection_p6, 14), get_squared_component(projection_p7, 14), get_squared_component(projection_p8, 14), get_squared_component(projection_p9, 14), get_squared_component(projection_p10, 14), get_squared_component(projection_p11, 14), get_squared_component(projection_p12, 14), get_squared_component(projection_p13, 14), get_squared_component(projection_p14, 14)]

plot_font = "arial"
Plots.default(fontfamily=plot_font)

plt = Plots.plot(xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, dpi=300)
Plots.plot!(plt, legend=nothing, foreground_color_legend = nothing, size = (1500, 500), left_margin = 10mm,  bottom_margin = 10mm, top_margin = 1mm, palette = :mk_12)
groupedbar!(plt, [component_nn component_p2 component_p3 component_p4 component_p5 component_p6 component_p7 component_p8 component_p9 component_p10 component_p11 component_p12 component_p13 component_p14], 
        bar_position = :stack,
        bar_width=0.4,
        xticks=([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [latexstring("\\pi(k_1)"), latexstring("\\pi(K_1)"), latexstring("\\pi(q)"), latexstring("\\pi(k_2)"), latexstring("\\pi(N)"), latexstring("\\pi(k_6)"), latexstring("\\pi(k_3)"), latexstring("\\pi(A)"), latexstring("\\pi(k_4)"), latexstring("\\pi(\\kappa)"), latexstring("\\pi(k_5)"), latexstring("\\pi(\\psi)"), latexstring("\\pi(k)")]),

        ylims=(0, 1),
        label=[latexstring("NN") latexstring("k_1") latexstring("K_1") latexstring("q") latexstring("k_2") latexstring("N") latexstring("k_6") latexstring("k_3") latexstring("A") latexstring("k_4") latexstring("\\kappa") latexstring("k_5") latexstring("\\psi") latexstring("k")]
        )
#plot an horizontal line
hline!(plt, [0.05], color="red", linestyle = :dot, linewidth=2, label="")
yaxis!(plt, "Squared norm",)

Plots.svg(plt, "plots/composition_projections_glyc_e00_observable_56.svg")

plt = Plots.plot(xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, dpi=300)
Plots.plot!(plt, legend=:outerright, foreground_color_legend = nothing, size = (1500, 600), left_margin = 10mm,  bottom_margin = 10mm, top_margin = 1mm, palette = :mk_12)
groupedbar!(plt, [component_nn component_p2 component_p3 component_p4 component_p5 component_p6 component_p7 component_p8 component_p9 component_p10 component_p11 component_p12 component_p13 component_p14], 
        bar_position = :stack,
        bar_width=0.4,
        xticks=([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [latexstring("\\pi(k_1)"), latexstring("\\pi(K_1)"), latexstring("\\pi(q)"), latexstring("\\pi(k_2)"), latexstring("\\pi(N)"), latexstring("\\pi(k_6)"), latexstring("\\pi(k_3)"), latexstring("\\pi(A)"), latexstring("\\pi(k_4)"), latexstring("\\pi(\\kappa)"), latexstring("\\pi(k_5)"), latexstring("\\pi(\\psi)"), latexstring("\\pi(k)")]),

        ylims=(0, 1),
        label=[latexstring("NN") latexstring("k_1") latexstring("K_1") latexstring("q") latexstring("k_2") latexstring("N") latexstring("k_6") latexstring("k_3") latexstring("A") latexstring("k_4") latexstring("\\kappa") latexstring("k_5") latexstring("\\psi") latexstring("k")]
        )
#plot an horizontal line
hline!(plt, [0.05], color="red", linestyle = :dot, linewidth=2, label="")
yaxis!(plt, "Squared norm",)

Plots.svg(plt, "plots/composition_projections_glyc_legend_observable_56.svg")

