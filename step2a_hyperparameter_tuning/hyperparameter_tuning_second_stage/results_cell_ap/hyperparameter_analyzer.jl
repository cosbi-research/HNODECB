#= 
Script to analyze the tuned hyperparameters of the cell apoptosis UDE model
=#

cd(@__DIR__)

using ComponentArrays, Serialization, PyCall
optuna = pyimport("optuna")

hyperparameters = deserialize("cell_ap_05.jld")
hyperparameters_05 = hyperparameters.study.best_params
println(hyperparameters_05)
