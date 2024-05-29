#=
Script to generate the plots for the Lotka-Volterra model with error level e0.0
=#

cd(@__DIR__)

using ComponentArrays, Lux, Serialization, DifferentialEquations, Random, DataFrames, Plots
using StatsPlots, Gadfly, LaTeXStrings


error_level = "e0.0"

include("../test_case_settings/lv_model_settings/lotka_volterra_model_functions.jl")
include("../test_case_settings/lv_model_settings/lotka_volterra_model_settings.jl")
column_names = ["t", "s1", "s2"]

#integration cell_apop_model_settings
integrator = Vern7()
abstol = 1e-7
reltol = 1e-6

#neural network
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
  Lux.Dense(2, 16, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(16, 16, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(16, 2; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

#load the data
ode_data = deserialize("../datasets/e0.0/data/ode_data_lotka_volterra.jld")
ode_data_sd = deserialize("../datasets/e0.0/data/ode_data_std_lotka_volterra.jld")
solution_dataframe = deserialize("../datasets/e0.0/data/pert_df_lotka_volterra.jld")
solution_sd_dataframe = deserialize("../datasets/e0.0/data/pert_df_sd_lotka_volterra.jld")

#get the parameters estimated
par_opt = deserialize("local_optima_found/lv_opt_00.jld")

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)

#############################################################################################################################
#################################### TRAINING VALIDATION SPLIT ##############################################################

#generates the random mask for the training and the valudation data set
shuffled_positions = shuffle(2:size(solution_dataframe)[1])
first_validation = rand(2:5)
validation_mask = [(first_validation + k * 5) for k in 0:3]
training_mask = [j for j in 1:size(solution_dataframe)[1] if !(j in validation_mask)]

#order the points
training_mask = sort(training_mask)
validation_mask = sort(validation_mask)

original_ode_data = deepcopy(ode_data)
original_ode_data_sd = deepcopy(ode_data_sd)
original_solution_dataframe = deepcopy(solution_dataframe)
original_solution_sd_dataframe = deepcopy(solution_sd_dataframe)

#generates the training and solution data structures 
ode_data = original_ode_data[:, training_mask]
ode_data_sd = original_ode_data_sd[:, training_mask]
solution_dataframe = original_solution_dataframe[training_mask, :]
solution_sd_dataframe = original_solution_sd_dataframe[training_mask, :]

#generates the validation and solution data structures
validation_ode_data = original_ode_data[:, validation_mask]
validation_ode_data_sd = original_ode_data_sd[:, validation_mask]
validation_solution_dataframe = original_solution_dataframe[validation_mask, :]
validation_solution_sd_dataframe = original_solution_sd_dataframe[validation_mask, :]

##################################################### model simulation ###################################################

tspan = (initial_time_training, end_time_training)
parameters_optimized = par_opt.parameters_training

uode_derivative_function = get_uode_model_function(approximating_neural_network, par_opt.net_status, parameters_optimized.p1)
parameters_optimized.p1 = 1

prob_uode_pred = ODEProblem{true}(uode_derivative_function, par_opt.initial_state_training, (0, maximum(solution_dataframe.t)))
solutions = solve(prob_uode_pred, Vern7(), p=parameters_optimized, saveat=0.001, abstol=abstol, reltol=reltol)


#ground truth dynamics
prob_ground_truth = ODEProblem{true}(ground_truth_function, original_u0, (0, maximum(solution_dataframe.t)))
simulation_data_ground_truth = solve(prob_ground_truth, Tsit5(), p=original_parameters,  saveat=0.001, abstol=abstol, reltol=reltol)

######################################################## plot the results
if !isdir("plots")
  mkdir("plots")
end

plot_font = "arial"
Plots.default(fontfamily=plot_font)

plts = []
for i in 1:size(solution_dataframe)[2]-1
    #no legend
    plt = Plots.plot(legend=false, left_margin = 200px, xtickfontsize=14pt,ytickfontsize=14pt, xguidefontsize=14pt, yguidefontsize=14pt,legendfontsize=14pt, size=(750px, 450px))
    Plots.plot!(plt, simulation_data_ground_truth.t, simulation_data_ground_truth[i,:], color= "black", linestyle=:dash, legend=false, label="original model", lv_margin=5, linewidth = 4)
    Plots.plot!(plt, solutions.t, solutions[i,:], color= "green", label="prediction", linewidth = 4)
    Plots.scatter!(plt, solution_dataframe.t, Array(solution_dataframe[:, i+1]), yerr = Array(solution_sd_dataframe[:, i+1]), color= "yellow", markerstrokewidth=2, markerstrokecolor="grey37", markersize = 8)
    Plots.scatter!(plt, validation_solution_dataframe.t, Array(validation_solution_dataframe[:, i+1]), yerr = Array(validation_solution_sd_dataframe[:, i+1]), color= "red", markerstrokewidth=2, markerstrokecolor="grey37", markersize = 8)
    xaxis!("time (y)")
    yaxis!("y"*string(i))
    # svae the plot as a PDF
    push!(plts, plt)
end

plt_y1y2 = Plots.plot(plts[1], plts[2], layout=(1, 2), size=(1500, 500), legend=false, bottom_margin = 50px, left_margin=30px, dpi=300)
Plots.svg(plt_y1y2, "plots/lv_plot_"*error_level*"_y1y2.svg")