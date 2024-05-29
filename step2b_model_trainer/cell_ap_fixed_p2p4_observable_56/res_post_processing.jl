cd(@__DIR__)

using ComponentArrays, Lux, Serialization, DiffEqFlux

results = deserialize("ca_00.jld")
results_not_failed = filter(x -> x.status != "failed", results)
costs = []
for result in results_not_failed
    if result.status != "failed"
        push!(costs, result.validation_resulting_cost)
    end
end
best_optimization = argmin(costs)
serialize("cell_ap_opt_00_fixed_p2p4_observable_56.jld", results_not_failed[best_optimization])
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
serialize("cell_ap_opt_05_fixed_p2p4_observable_56.jld", results_not_failed[best_optimization])