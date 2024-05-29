#= 
Script to analyze the tuned hyperparameters of the yeast glycolysis UDE model
=#

cd(@__DIR__)

using ComponentArrays, Serialization, PyCall
optuna = pyimport("optuna")

########## error 0.0 ################
hyperparameters = deserialize("glyc_05.jld")
hyperparameters_05 = hyperparameters.study.best_params
println(hyperparameters_05)
