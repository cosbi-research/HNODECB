#= 
Script to analyze the tuned hyperparameters of the yeast glycolysis UDE model
=#

cd(@__DIR__)

using ComponentArrays, Serialization, PyCall
optuna = pyimport("optuna")

########## error 0.0 ################
hyperparameters = deserialize("glyc_00.jld")
hyperparameters_00 = hyperparameters.study.best_params
println(hyperparameters_00)

########## error 0.05 ###############
hyperparameters = deserialize("glyc_00.jld")
hyperparameters_05 = hyperparameters.study.best_params
println(hyperparameters_05)

