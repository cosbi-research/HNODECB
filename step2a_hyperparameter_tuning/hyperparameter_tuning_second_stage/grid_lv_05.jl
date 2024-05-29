#= 
Script to run the L2 regularizing coefficient tuning for the Lotka-Volterra model using the TPE algorithm with error level 5%
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux

#python library for hyperparameter optimization 
using PyCall
optuna = pyimport("optuna")

result_folder = "results_lv"
if !isdir(result_folder)
  mkdir(result_folder)
end
result_name_string = "lv_00.jld"

error_level = "e0.0"

#inlcudes the model settings
include("../../test_case_settings/lv_model_settings/lotka_volterra_model_functions.jl")
include("../../test_case_settings/lv_model_settings/lotka_volterra_model_settings.jl")
column_names = ["t", "s1", "s2"]

#settings for non-stiff problems
integrator = Vern7()
abstol = 1e-7
reltol = 1e-6
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

#loads the data
ode_data = deserialize("../../datasets/e0.05/data/ode_data_lotka_volterra.jld")
ode_data_sd = deserialize("../../datasets//e0.05/data/ode_data_std_lotka_volterra.jld")
solution_dataframe = deserialize("../../datasets//e0.05/data/pert_df_lotka_volterra.jld")
solution_sd_dataframe = deserialize("../../datasets//e0.05/data/pert_df_sd_lotka_volterra.jld")

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

#settings for the training set sampling
tspan = (initial_time_training, end_time_training)
tsteps = range(tspan[1], tspan[2], length=21)
stepsize = (tspan[2] - tspan[1]) / (21 - 1)
lentgth_tsteps = length(tsteps)


#############################################################################################################################
#################################### TRAINING VALIDATION SPLIT ##############################################################

#generates the random mask for the training and the valudation data set
shuffled_positions = shuffle(2:size(solution_dataframe)[1])
first_validation = rand(2:5)
validation_mask = [(first_validation + k * 5) for k in 0:3]
training_mask = [j for j in 1:size(solution_dataframe)[1] if !(j in validation_mask)]

#order the points
training_mask = sort(training_mask)
validation_mask = sort(validation_mask)

original_ode_data = deepcopy(ode_data)
original_ode_data_sd = deepcopy(ode_data_sd)
original_solution_dataframe = deepcopy(solution_dataframe)
original_solution_sd_dataframe = deepcopy(solution_sd_dataframe)

#generates the training and solution data structures 
ode_data = original_ode_data[:, training_mask]
ode_data_sd = original_ode_data_sd[:, training_mask]
solution_dataframe = original_solution_dataframe[training_mask, :]
solution_sd_dataframe = original_solution_sd_dataframe[training_mask, :]

#generates the validation and solution data structures
validation_ode_data = original_ode_data[:, validation_mask]
validation_ode_data_sd = original_ode_data_sd[:, validation_mask]
validation_solution_dataframe = original_solution_dataframe[validation_mask, :]
validation_solution_sd_dataframe = original_solution_sd_dataframe[validation_mask, :]

######################################################################################
######################################################################################
# hyper-parameters tuned in the first stage
original_p1 = 0.581051
original_learning_rate_adam = 0.0071014

approximating_neural_network = Lux.Chain(
  Lux.Dense(2, 2^3, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2^3, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2^3, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

ms_group_size = 5
ms_continuity_term = 6.52233
######################################################################################
######################################################################################

seed = abs(rand(rng, Int))

#constants used in the optimization
tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)

#objective function got the grid optimization
function objective(trial)

  l2_regularization = trial.suggest_uniform("l2_regularization", -100, 100)
  learning_rate_adam = original_learning_rate_adam
  
  rng = StableRNG(seed)
  tmp_approximating_neural_network = deepcopy(approximating_neural_network)
  p_net, st = Lux.setup(rng, tmp_approximating_neural_network)

  #UDE derivative function
  uode_derivative_function = get_uode_model_function(tmp_approximating_neural_network, st, original_p1)
  initial_p1 = 1
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)


  ranges = DiffEqFlux.group_ranges(datasize, ms_group_size)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

  #loss function for the comparison among the parameters and the predictions
  function loss_function(data, deviation, pred)
    original_cost = sum(abs2.(data .- pred) ./ abs2.(deviation))
    return 1/size(data, 2) * sum(original_cost)
  end

  function loss_multiple_shooting(θ)
    #unpack the parameters
    p = θ.p
    tsteps = tmp_steps
    prob = prob_uode_pred
    solver = integrator
    initial_point_parameters = θ.u0

    function unstable_check(dt, u, p, t)
      if any(abs.(u) .> 1e7)
        return true
      end
      return false
    end

    initial_points = initial_point_parameters

    # multiple shooting predictions
    sols = [
      solve(
        remake(
          prob;
          p=p,
          tspan=(tsteps[first(rg)], tsteps[last(rg)]),
          u0=initial_points[:, first(rg)]
          #u0=ode_data[:, first(rg)]
        ),
        solver;
        saveat=tsteps[rg],
        reltol=reltol,
        abstol=abstol,
        sensealg=sensealg,
        unstable_check=unstable_check,
        verbose=false
      ) for rg in ranges
    ]
    # Abort and return infinite loss if one of the integrations failed
    for i in 1:length(sols)
      if size(Array(sols[i]))[2] != length(ranges[i])
        return Inf
      end
    end

    group_predictions = Array.(sols)
    # SE component of the cost function
    loss = 0
    for (i, rg) in enumerate(ranges)
      u = ode_data[:, rg]
      std = ode_data_sd[:, rg]
      û = group_predictions[i]
      loss += loss_function(u, std, û)
    end

    # Continuity component of the loss
    for (i, rg) in enumerate(ranges)
      if i == 1
        continue
      end

      u0 = group_predictions[i-1][:, end]
      u1 = group_predictions[i][:, 1]
      loss += ms_continuity_term * sum(abs2, u0 - u1)
    end

    #l2 regularization term
    loss += l2_regularization * (sum(abs2, p.p_net.layer_1.weight) + sum(abs2, p.p_net.layer_2.weight) + sum(abs2, p.p_net.layer_3.weight) + sum(abs2, p.p_net.layer_4.weight))

    return loss
  end

  #callback function to observe training
  epoch = 1 
  function callback(θ, l, training_epochs, training_costs, num_epoch_to_finish, stuck) 
      
    println("Epoch: " * string(epoch) * " - Loss: " * string(l) * " - Parameters: " * string(θ.p.p1))
    #this is necessary to have the cost at the last iteration, not the best cost found during the optimization
    if epoch == num_epoch_to_finish
      return true
    end

    training_epochs[epoch] = epoch
    training_costs[epoch] = l

    #early exit if the optimization is stuck in a too high cost region
    if epoch > 10 && minimum(training_costs[(epoch-5):(epoch)]) > 10^6
      stuck[1] = true
      return true
    end
  
    epoch += 1

    return false
  end

  #validation loss function
  function validation_loss_function(θ)

    loss = 0.0
    validation_df = validation_solution_dataframe
    validation_sd_dataframe = validation_solution_sd_dataframe

    try
      times_consdiered = validation_df.t
      max_time = extrema(times_consdiered)[2]

      prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

      par = θ.p
      init_par = θ.u0
      prob = remake(
        prob_uode_pred;
        p=par,
        tspan=(0, max_time),
        u0=init_par[:,1]
      )

      function unstable_check(dt, u, p, t)
        if any(abs.(u) .> 1e7)
          return true
        end
        return false
      end

      #select the elements of tsteps greater than initial_time and less than final_time
      solutions = solve(prob, integrator, p=par, saveat=validation_df.t, abstol=abstol, reltol=reltol,
        sensealg=sensealg, unstable_check=unstable_check, verbose=false)
      x = Array(solutions)

      if size(x)[2] != size(validation_df)[1]
        return Inf
      end

      #computes the quadratic loss
      loss = 1 / size(validation_df, 1) * sum(abs2.(Array(validation_df[:, 2:end])' .- x) ./ abs2.(Array(validation_sd_dataframe[:, 2:end])'))
    catch
      loss = Inf
    end

    return loss
  end

  #defining the optimization ad method
  adtype = Optimization.AutoZygote()

  p_net = ComponentArray(p_net)
  p1 = ComponentArray(initial_p1)
  p = ComponentArray{eltype(p_net)}()
  p = ComponentArray(p; p_net)
  p = ComponentArray(p; p1)
  starting_point_in = ComponentVector{Float64}(p=p, u0=deepcopy(ode_data))

  res = nothing
  validation_resulting_cost = nothing

  training_epochs = zeros(Int, 50000)
  training_costs = zeros(50000)

  optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, starting_point_in)
  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  ##################### ADAM ###########
  stuck = [false]
  res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, 2000, stuck), maxiters=2001)

  if stuck[1] == true
    validation_resulting_cost = Inf
  else
    validation_resulting_cost = validation_loss_function(res.u)
  end

  θ_to_memorize = deepcopy(res.u)
  global trial_parameters
  push!(trial_parameters, θ_to_memorize)

  println("Cost: " * string(validation_resulting_cost) * " - " * string(trial.params))
  
  return validation_resulting_cost
end

# grid optimization  
global trial_parameters = []
search_space = Dict(
  "l2_regularization" => [1.0, 0.1, 0.01, 0.001, 0]
)
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective)

#save the optimization results
result = (
  study=study,
  trial_parameters=trial_parameters,
  error_level=error_level
)

serialize(result_folder * "/" * result_name_string, result)