cd(@__DIR__)

using ComponentArrays, Lux, Serialization, DiffEqFlux, DataFrames, Lux

original_parameters = 1.3
lower_bounds = 10^-2.0 * original_parameters
upper_bounds = 10^2.0 * original_parameters

#needed to load the results
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

results = deserialize("lv_00.jld")
results_not_failed = filter(x -> x.status != "failed", results)
costs = []
for result in results_not_failed
    if result.status != "failed"
        push!(costs, result.validation_resulting_cost)
    end
end
best_optimization = argmin(costs)
res = deepcopy(results_not_failed[best_optimization])
res.parameters_training.p1 = min(max(res.parameters_training.p1, lower_bounds), upper_bounds)
serialize("lv_opt_00.jld", res)
################################################################################################################################
results = deserialize("lv_05.jld")
results_not_failed = filter(x -> x.status != "failed", results)
costs = []
for result in results_not_failed
    if result.status != "failed"
        push!(costs, result.validation_resulting_cost)
    end
end
best_optimization = argmin(costs)
res = deepcopy(results_not_failed[best_optimization])
res.parameters_training.p1 = min(max(res.parameters_training.p1, lower_bounds), upper_bounds)
serialize("lv_opt_05.jld", res)