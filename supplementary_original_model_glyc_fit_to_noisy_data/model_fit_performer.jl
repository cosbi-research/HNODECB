#= 
Script to fit the original yeast glycolysis model parameter to noisy data. 
=# 

cd(@__DIR__)

using ComponentArrays, SciMLSensitivity, Serialization, DifferentialEquations, Random, DataFrames, CSV, Plots
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs, ForwardDiff, Zygote 

error_level = "e0.05"

#includes the specific model functions
include("../test_case_settings/glyc_model_settings/glycolitic_model_functions.jl")
include("../test_case_settings/glyc_model_settings/glycolitic_model_settings.jl")

ode_data = deserialize("../datasets/e0.05/data/ode_data_glycolysis.jld")
ode_data_sd = deserialize("../datasets/e0.05/data/ode_data_std_glycolysis.jld")
solution_dataframe = deserialize("../datasets/e0.05/data/pert_df_glycolysis.jld")
solution_sd_dataframe = deserialize("../datasets/e0.05/data/pert_df_sd_glycolysis.jld")

tspan = extrema(solution_dataframe.t)
glyc_ode_problem = ODEProblem{true}(ground_truth_function_modified, original_u0, tspan)

#cost function (no need to use a stiff solver in the original case)
function cost_function(pars)
  solutions = Array(solve(glyc_ode_problem, Vern7(); p=pars, saveat=Array(solution_dataframe.t), reltol=1e-7, abstol=1e-6, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))) 
  if size(solutions,2) != size(ode_data, 2)
    return Inf
  end
  return sum(abs2, (solutions .- ode_data) ./ ode_data_sd)
end

#callback function 
cb = function (p, l)
  println("Current parameters: ", p)
  println("Current loss: ", l)
  return false
end

#defines the optization problem
adtype = Optimization.AutoZygote()
optf = OptimizationFunction((x, p) -> cost_function(x), adtype)
original_parameters_modified = deepcopy(original_parameters)
lb = original_parameters_modified .* 0.8
ub = original_parameters_modified .* 1.2
original_parameters_modified[3] = log10.(original_parameters_modified[3])
lb[3] = log10.(lb[3])
ub[3] = log10.(ub[3])
prob = OptimizationProblem(optf, original_parameters_modified, lb = lb, ub = ub)

sol = solve(prob, LBFGS(); callback=cb, maxiters=1000)

parameters_optimized = sol.u
parameters_optimized[3] = 10^parameters_optimized[3]

relative_error = abs.(parameters_optimized .- original_parameters) ./ original_parameters
df_error = DataFrame(original = original_parameters, optimized = parameters_optimized, relative_error = relative_error)
serialize("fit_glyc_e0.05.jld", df_error)