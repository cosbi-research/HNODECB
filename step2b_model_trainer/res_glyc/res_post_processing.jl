cd(@__DIR__)

using ComponentArrays, Lux, Serialization, DiffEqFlux, DataFrames, Lux

# literature values and upper/lower optimization bound
real_values = Float64[2.5, 100, 0.52, 4, 6, 1, 12, 16, 4, 100, 13, 1.28, 0.1, 1.8]
lower_bounds = 10^-1.0 .* real_values
upper_bounds = 10^1.0 .* real_values
lower_bounds[3] = log10(lower_bounds[3])
upper_bounds[3] = log10(upper_bounds[3])

#needed to load the results
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

results = deserialize("glyc_00.jld")
results_not_failed = filter(x -> x.status != "failed", results)
costs = []
for result in results_not_failed
    if result.status != "failed"
        push!(costs, result.validation_resulting_cost)
    end
end
best_optimization = argmin(costs)
res = deepcopy(results_not_failed[best_optimization])
res.parameters_training.ode_par = min.(max.(res.parameters_training.ode_par, lower_bounds), upper_bounds)
serialize("glyc_opt_00.jld", res)
################################################################################################################################

results = deserialize("glyc_05.jld")
results_not_failed = filter(x -> x.status != "failed", results)
costs = []
for result in results_not_failed
    if result.status != "failed"
        push!(costs, result.validation_resulting_cost)
    end
end
best_optimization = argmin(costs)
res = deepcopy(results_not_failed[best_optimization])
res.parameters_training.ode_par = min.(max.(res.parameters_training.ode_par, lower_bounds), upper_bounds)
serialize("glyc_opt_05.jld", res)