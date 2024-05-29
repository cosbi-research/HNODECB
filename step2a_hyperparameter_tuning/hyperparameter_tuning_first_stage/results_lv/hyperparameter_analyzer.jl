#= 
Script to analyze the tuned hyperparameters of the Lotka Volterra UDE model
=#

cd(@__DIR__)

using ComponentArrays, Serialization, PyCall

optuna = pyimport("optuna")

########## error 0.0 ################
hyperparameters = deserialize("lv_00.jld")
hyperparameters_00 = hyperparameters.study.best_params
println(hyperparameters_00)

########## error 0.05 ###############
hyperparameters = deserialize("lv_05.jld")
hyperparameters_05 = hyperparameters.study.best_params
println(hyperparameters_05)
