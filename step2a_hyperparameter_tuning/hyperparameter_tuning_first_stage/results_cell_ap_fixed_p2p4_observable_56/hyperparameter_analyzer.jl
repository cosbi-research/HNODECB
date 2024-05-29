#= 
Script to analyze the tuned hyperparameters of the cell apoptosis UDE model fixing the parameters p2 and p4 to their literature values,
assuming that only y5 and y6 are observable
=#

cd(@__DIR__)

using ComponentArrays, Serialization, PyCall
optuna = pyimport("optuna")

########## error 0.0 ################
hyperparameters = deserialize("cell_ap_00.jld")
hyperparameters_00 = hyperparameters.study.best_params
println(hyperparameters_00)

########## error 0.05 ###############
hyperparameters = deserialize("cell_ap_05.jld")
hyperparameters_05 = hyperparameters.study.best_params
println(hyperparameters_05)

