cd(@__DIR__)

using ComponentArrays, Lux, Serialization, DiffEqFlux, DataFrames, Lux

# literature values and upper/lower optimization bound
real_values = Float64[2.67 * 10^-9 *3600 * 10^5, 1*10^-2*3600, 8* 10^-3*3600, 6.8 * 10^-8*3600 * 10^5, 5*10^-2*3600, 1*10^-3*3600, 7*10^-5*3600 * 10^5, 1.67 * 10^-5*3600, 1.67*10^-4*3600]
lower_bounds = 10^-2.0 .* real_values
upper_bounds = 10^2.0 .* real_values

#needed to load the results
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

results = deserialize("ca_00.jld")
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
serialize("ca_opt_00.jld", res)
################################################################################################################################

results = deserialize("ca_05.jld")
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
serialize("ca_opt_05.jld", res)