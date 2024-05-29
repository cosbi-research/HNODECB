#=
Script to generate the plots for the cell apoptosis UDE model trained on a dataset with error level e0.0
=#

cd(@__DIR__)

using ComponentArrays, Lux, Serialization, DifferentialEquations, Random, DataFrames, Plots
using StatsPlots, Gadfly, LaTeXStrings, Plots.PlotMeasures, DiffEqFlux, .Flux


error_level = "e0.05"

#includes the specific model functions
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_settings.jl")
column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

integrator = TRBDF2(autodiff=false);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
  Lux.Dense(6, 2^3, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2^3, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

ode_data = deserialize("../datasets/e0.05/data/ode_data_cell_apoptosis.jld")
ode_data_sd = deserialize("../datasets/e0.05/data/ode_data_std_cell_apoptosis.jld")
solution_dataframe = deserialize("../datasets/e0.05/data/pert_df_cell_apoptosis.jld")
solution_sd_dataframe = deserialize("../datasets/e0.05/data/pert_df_sd_cell_apoptosis.jld")

#get the parameters estimated
par_opt = deserialize("local_optima_found/ca_opt_05.jld")

#############################################################################################################################
#################################### TRAINING VALIDATION SPLIT ##############################################################

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)

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

###################################################################################################################
########################################### model simulation ######################################################

tspan = (initial_time_training, end_time_training)
parameters_optimized = par_opt.parameters_training

uode_derivative_function = get_uode_model_function(approximating_neural_network, par_opt.net_status, deepcopy(parameters_optimized.ode_par))
parameters_optimized.ode_par .= 1

prob_uode_pred = ODEProblem{true}(uode_derivative_function, par_opt.initial_state_training[:,1], (0, maximum(solution_dataframe.t)))
solutions = solve(prob_uode_pred, TRBDF2(autodiff=false), p=parameters_optimized, saveat=0.001, abstol=abstol, reltol=reltol)

prob_ground_truth = ODEProblem{true}(ground_truth_function, original_u0, (0, maximum(solution_dataframe.t)))
simulation_data_ground_truth = solve(prob_ground_truth, TRBDF2(autodiff=false), p=original_parameters,  saveat=0.001, abstol=abstol, reltol=reltol)


#create the plot directory
if !isdir("plots")
  mkdir("plots")
end

plot_font = "arial"
Plots.default(fontfamily=plot_font)

plts = []
for i in 1:size(solution_dataframe)[2]-1
    #no legend

    plt = Plots.plot(legend=false, xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18,  yguidefonthalign=:left, size=(750, 450))
    Plots.plot!(plt, simulation_data_ground_truth.t, simulation_data_ground_truth[i,:], color= "black", linewidth=4, linestyle=:dashdot)
    Plots.plot!(plt, solutions.t, solutions[i,:], color= "green", linewidth=4)
    Plots.scatter!(plt, solution_dataframe.t, Array(solution_dataframe[:, i+1]), yerr = Array(solution_sd_dataframe[:, i+1]), color= "yellow",markerstrokewidth=2, markerstrokecolor="grey37", markersize = 8)
    Plots.scatter!(plt, validation_solution_dataframe.t, Array(validation_solution_dataframe[:, i+1]), yerr = Array(validation_solution_sd_dataframe[:, i+1]), color= "red", markerstrokewidth=2, markerstrokecolor="grey37", markersize = 8)

    xaxis!(plt, "time (h)")
    yaxis!(plt, "y"*string(i) * " (" * latexstring("10^5") * " molecules/cell)")

    push!(plts, plt)
end

#plot all in one with 2 plots per row 
plt = Plots.plot(plts..., layout=(4, 2), size=(1500, 500*4), legend=false, left_margin = 20mm, dpi = 300)

savefig(plt, "plots/cell_ap_plot_"*error_level*"_summary.svg")


using Plots.PlotMeasures:px
plt_y1y2 = Plots.plot(plts[1], plts[2], layout=(1, 2), size=(1500, 500), legend=false, bottom_margin = 50px, left_margin=30px, dpi=300)

savefig(plt_y1y2, "plots/cell_ap_plot_"*error_level*"_y1y2.svg")