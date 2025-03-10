#=
Script to generate the plots for the Yeast Glycolysis UDE model trained on a dataset with error level e0.05
assuming that only the variables y5 and y6 are observable
=#

cd(@__DIR__)

using ComponentArrays, Lux, Serialization, DifferentialEquations, Random, DataFrames, Plots
using StatsPlots, Gadfly, LaTeXStrings, Plots.PlotMeasures, DiffEqFlux, .Flux

error_level = "e0.05"

include("../test_case_settings/glyc_model_settings/glycolitic_model_functions.jl")
include("../test_case_settings/glyc_model_settings/glycolitic_model_settings.jl")

column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]

integrator = TRBDF2(autodiff=false);
abstol = 1e-7
reltol = 1e-6

observables = [5,6]

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
      Lux.Dense(2, 2^4, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^4, 2^4, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^4, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

ode_data = deserialize("../datasets/e0.05_doubled/data/ode_data_glycolysis.jld")
ode_data_sd = deserialize("../datasets/e0.05_doubled/data/ode_data_std_glycolysis.jld")
solution_dataframe = deserialize("../datasets/e0.05_doubled/data/pert_df_glycolysis.jld")
solution_sd_dataframe = deserialize("../datasets/e0.05_doubled/data/pert_df_sd_glycolysis.jld")

#get the parameters estimated
par_opt = deserialize("local_optima_found/glyc_opt_05_observables_56.jld")

#############################################################################################################################
#################################### TRAINING VALIDATION SPLIT ##############################################################

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)

#generates the random mask for the training and the valudation data set
shuffled_positions = shuffle(2:size(solution_dataframe)[1])
first_validation = rand(2:5)
validation_mask = [(first_validation + k * 5) for k in 0:6]
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

 ##################################################################################################################
 ################################### Simulations ##################################################################
tspan = (initial_time_training, end_time_training)
parameters_optimized = par_opt.parameters_training

uode_derivative_function = get_uode_model_function(approximating_neural_network, par_opt.net_status, deepcopy(parameters_optimized.ode_par))
parameters_optimized.ode_par .= 1


prob_uode_pred = ODEProblem{true}(uode_derivative_function, par_opt.initial_state_training[:,1], (0, maximum(solution_dataframe.t)))
solutions = solve(prob_uode_pred, TRBDF2(autodiff=false), p=parameters_optimized, saveat=0.001, abstol=abstol, reltol=reltol)
#ground truth dynamics
prob_ground_truth = ODEProblem{true}(ground_truth_function, original_u0, (0, maximum(solution_dataframe.t)))
simulation_data_ground_truth = solve(prob_ground_truth, TRBDF2(autodiff=false), p=original_parameters,  saveat=0.001, abstol=abstol, reltol=reltol)

#create the plot directory
if !isdir("plots")
  mkdir("plots")
end

plot_font = "Arial"
Plots.default(fontfamily=plot_font)

plts = []
for i in 1:size(solution_dataframe)[2]-1

    #no legend
    plt = Plots.plot(legend=false, xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18,  yguidefonthalign=:left, size=(750, 450))
    Plots.plot!(plt, simulation_data_ground_truth.t, simulation_data_ground_truth[i,:], color= "black", linewidth=4, linestyle=:dashdot)
    Plots.plot!(plt, solutions.t, solutions[i,:], color= "green", linewidth=4)
    if i in observables
      #plot the training and validation data
      Plots.scatter!(plt, solution_dataframe.t, Array(solution_dataframe[:, i+1]), yerr = Array(solution_sd_dataframe[:, i+1]), color= "yellow",markerstrokewidth=2, markerstrokecolor="grey37", markersize = 8)
      Plots.scatter!(plt, validation_solution_dataframe.t, Array(validation_solution_dataframe[:, i+1]), yerr = Array(validation_solution_sd_dataframe[:, i+1]), color= "red", markerstrokewidth=2, markerstrokecolor="grey37", markersize = 8)
    end

    xaxis!("time (min)")
    yaxis!("y"*string(i) * " (mM)")

    savefig(plt, "plots/glyc_partially_observable_plot_"*error_level*"_s"*string(i)*".svg")

    push!(plts, plt)
end

plt = Plots.plot(plts..., layout=(4, 2), size=(1500, 500*4), legend=false, left_margin = 20mm, dpi = 300)

savefig(plt, "plots/glyc_partially_observable_plot_"*error_level*"_summary.svg")

using Plots.PlotMeasures

plt_y5y6 = Plots.plot(plts[5], plts[6], layout=(1, 2), size=(1500, 500), legend=false, bottom_margin = 50px, left_margin=30px, dpi=300)

savefig(plt_y5y6, "plots/glyc_partially_observable_plot_"*error_level*"_y5y6.svg")