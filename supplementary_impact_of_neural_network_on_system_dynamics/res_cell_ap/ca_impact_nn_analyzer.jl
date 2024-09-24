#=
Script to generate the plots for the Lotka-Volterra model with error level e0.0
=#

cd(@__DIR__)

using ComponentArrays, Lux, Serialization, DifferentialEquations, Random, DataFrames, Plots
using StatsPlots, LaTeXStrings

checking_epochs = [200,700,3000]
variables_to_plot = [1, 5]

error_level = "e0.0"

include("../../test_case_settings/cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../../test_case_settings/cell_apoptosis_settings/cell_apop_model_settings.jl")

column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]

#deserialize the results
results = deserialize("ca_00.jld")
results = results[1]

#plots the training cost behaviour
training_epochs = results.training_epochs[results.training_epochs.>0]
training_costs = results.training_costs[1:length(training_epochs)]

training_epochs = training_epochs .- 1

#plot the training cost behaviour
plot_font = "arial"
Plots.default(fontfamily=plot_font)

using Plots.PlotMeasures:px

plt = Plots.plot(legend=false, xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18,  yguidefonthalign=:left, bottommargin=12mm, leftmargin=10mm, size=(1500, 500))
Plots.plot!(plt, training_epochs, log10.(training_costs), linewidth=4)
xaxis!(plt, "epoch")
yaxis!(plt, "log10(loss)")
vline!(plt, checking_epochs, linewidth=2, linestyle=:dash) 

#save the cost plot
savefig(plt, "training_cost_plot.svg")


########################################################################################################
################################ EPOCH PLOTS ###########################################################

#integration cell_apop_model_settings
integrator = TRBDF2(autodiff=true);
abstol = 1e-7
reltol = 1e-6

#neural network
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
  Lux.Dense(6, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(8, 8, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(8, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform)
)

rng = StableRNG(0)
p_net, st = Lux.setup(rng, approximating_neural_network)

uode_derivative_function = get_uode_model_function_with_only_nn(approximating_neural_network, st)

#load the data
ode_data = deserialize("../../datasets/e0.0/data/ode_data_cell_apoptosis.jld")
ode_data_sd = deserialize("../../datasets/e0.0/data/ode_data_std_cell_apoptosis.jld")
solution_dataframe = deserialize("../../datasets/e0.0/data/pert_df_cell_apoptosis.jld")
solution_sd_dataframe = deserialize("../../datasets/e0.0/data/pert_df_sd_cell_apoptosis.jld")

checkpoint_plots = []
for checking_epoch in checking_epochs
  #parameters optimization at the epoch
  index = checking_epoch/100 + 1
  #convert to int
  index = Int(index)
  par_opt = results.training_parameterizations[index]

  u0 = par_opt.u0[:,1]

  tspan=extrema(solution_dataframe.t)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, u0, tspan)


  prob = remake(
      prob_uode_pred;
      p=par_opt.p,
      tspan=extrema(solution_dataframe.t),
      u0= 10 .^ par_opt.u0[:, 1]
      #u0 = original_u0#u0=ode_data[:, 1]
  )
  solutions = solve(prob, integrator, p=par_opt.p, saveat=0.01, abstol=abstol, reltol=reltol,
  sensealg=sensealg, verbose=false)

  prob_ground_truth = ODEProblem{true}(ground_truth_function, original_u0, (0, maximum(solution_dataframe.t)))
  simulation_data_ground_truth = solve(prob_ground_truth, Tsit5(), p=original_parameters,  saveat=0.001, abstol=abstol, reltol=reltol)

  tmp_checkpoint_plot = []

  #iterate over the variables 
  for i in 1:size(solution_dataframe)[2]-1
    #solve the ODE
    using Plots.PlotMeasures:pt
    plt = Plots.plot(legend=false, left_margin=5mm, bottom_margin=5mm, xtickfontsize=14,ytickfontsize=14, xguidefontsize=14, yguidefontsize=14,legendfontsize=14, size=(750, 450))
    Plots.plot!(plt, simulation_data_ground_truth.t, simulation_data_ground_truth[i,:], color= "black", linestyle=:dash, legend=false, label="original model", lv_margin=5, linewidth = 4)
    Plots.plot!(plt, solutions.t, solutions[i,:], color= "green", label="prediction", linewidth = 4)
    Plots.scatter!(plt, solution_dataframe.t, Array(solution_dataframe[:, i+1]), yerr = Array(solution_sd_dataframe[:, i+1]), color= "yellow", markerstrokewidth=1, markerstrokecolor="grey37", markersize = 8)
    xaxis!("time (y)")
    yaxis!("y"*string(i))

    push!(tmp_checkpoint_plot, plt)
  end

  push!(checkpoint_plots, tmp_checkpoint_plot)
end 

dynamics_plot = Plots.plot(checkpoint_plots[1][variables_to_plot[1]], checkpoint_plots[1][variables_to_plot[2]], checkpoint_plots[2][variables_to_plot[1]], checkpoint_plots[2][variables_to_plot[2]], checkpoint_plots[3][variables_to_plot[1]], checkpoint_plots[3][variables_to_plot[2]], layout=(3, 2), size=(1500, 500*3), legend=false, bottom_margin = 50px, left_margin=30px, dpi=300)
Plots.svg(dynamics_plot, "cell_ap_plot_dynamics_training.svg")