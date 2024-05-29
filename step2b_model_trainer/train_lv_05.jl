#= 
Script to run the training of the Lotka Volterra UDE model on the dataset with error level 0.05.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics, Printf, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux

result_name_string = "lv_05.jld"

folder_name = "res_lv"
#creates the directory to save the results
if !isdir(folder_name)
  mkdir(folder_name)
end

error_level = "e0.05"

#inlcudes the model settings
include("../test_case_settings/lv_model_settings/lotka_volterra_model_functions.jl")
include("../test_case_settings/lv_model_settings/lotka_volterra_model_settings.jl")
column_names = ["t", "s1", "s2"]

#settings for non-stiff problems
integrator = Vern7()
abstol = 1e-7
reltol = 1e-6
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

#loads the data
ode_data = deserialize("../datasets/e0.05/data/ode_data_lotka_volterra.jld")
ode_data_sd = deserialize("../datasets/e0.05/data/ode_data_std_lotka_volterra.jld")
solution_dataframe = deserialize("../datasets/e0.05/data/pert_df_lotka_volterra.jld")
solution_sd_dataframe = deserialize("../datasets/e0.05/data/pert_df_sd_lotka_volterra.jld")


############################ TUNED HYPERPARAMETERS ############################
l2_regularization = 1.0
learning_rate_adam = 0.0071014

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

approximating_neural_network = Lux.Chain(
  Lux.Dense(2, 2^3, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2^3, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2^3, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^3, 2; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

pretrained_p1 = 0.581051
group_size = 5
continuity_term = 6.52233

####################################################################################################################

rng = Random.default_rng()
Random.seed!(rng, 0)

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

#constants used in the optimization
Random.seed!(rng, 0)
tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)
tspan = extrema(original_solution_dataframe.t)

results = []

#function to train the uode model
function train_uode_model(seed, iterator)

  #gets the time to monitor the training time
  initial_time = time()

  rng = StableRNG(seed)
  tmp_neural_network = deepcopy(approximating_neural_network)
  p_net, st = Lux.setup(rng, tmp_neural_network)

  #UDE derivative function
  uode_derivative_function = get_uode_model_function(approximating_neural_network, st, pretrained_p1)
  initial_p1 = 1
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

  #multiple shooting ranges
  ranges = DiffEqFlux.group_ranges(datasize, group_size)

  #loss function for the comparison among the parameters and the predictions
  function loss_function(data, deviation, pred)
    original_cost = sum(abs2.(data .- pred) ./ abs2.(deviation))
    return 1 / size(data, 2) * sum(original_cost)
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

    # Multiple shooting predictions
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
      loss += continuity_term * sum(abs2, u0 - u1)
    end

    #l2 regularization term
    loss += l2_regularization * (sum(abs2, p.p_net.layer_1.weight) + sum(abs2, p.p_net.layer_2.weight) + sum(abs2, p.p_net.layer_3.weight) + sum(abs2, p.p_net.layer_4.weight))

    return loss
  end

  #callback function to observe training and populating the cost array
  function callback(θ, l, training_epochs, training_costs, best_training_parameters)

    epoch = extrema(training_epochs)[2] + 1
    println("********************************Epoch " * string(epoch) * " -- cost: " * string(l) * "")

    #max 2 hours for an optimization
    if time() - initial_time > 60 * 3 * 60
      println("Too slow optimization")
      return true
    end

    #print the cost every 20 epochs
    if epoch % 20 == 0
      println("Iteration " * string(epoch) * " -- cost: " * string(l) * "")
    end

    training_epochs[epoch] = epoch
    training_costs[epoch] = l

    #keeps track of the best solution, to have a uniform behaviour if exiting when stuck
    if epoch > 1 && l < minimum(training_costs[1:(epoch-1)])
      best_training_parameters[1] = deepcopy(θ)
    end

    #if the cost is too high, the optimization is stuck in a non integrable region, stop it
    if epoch > 200 && minimum(training_costs[(epoch-5):(epoch)]) > 10^6
      println("Stuck in non integrability region")
      return true
    end

    return false
  end

  #validation loss function
  function validation_loss_function(θ)

    loss = 0.0
    validation_df = validation_solution_dataframe
    validation_df_std = validation_solution_sd_dataframe

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
        u0=init_par[:, 1]
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

      loss = 1 / size(validation_df, 1) * sum(abs2.(Array(validation_df[:, 2:end])' .- x) ./ abs2.(Array(validation_df_std[:, 2:end])'))
    catch
      loss = Inf
    end

    return loss
  end

  #defining the optimization procedures
  adtype = Optimization.AutoZygote()

  training_epochs = zeros(Int, 50000)
  training_costs = zeros(50000)

  p_net = ComponentArray(p_net)
  p1 = ComponentArray(initial_p1)
  p = ComponentArray{eltype(p_net)}()
  p = ComponentArray(p; p_net)
  p = ComponentArray(p; p1)
  starting_point_in = ComponentVector{Float64}(p=p, u0=deepcopy(ode_data))

  best_training_parameters = [starting_point_in]

  optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, starting_point_in)
  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  ##################### ADAM ###########
  res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, best_training_parameters), maxiters=10000)

  best_parameterization = best_training_parameters[1]
  #computes the cost on validation
  validation_resulting_cost = validation_loss_function(best_parameterization)

  #rescale the parameters
  best_parameterization.p.p1 = best_parameterization.p.p1 * pretrained_p1

  #saves the results  
  result = (
    parameters_training=best_parameterization.p,
    initial_state_training=best_parameterization.u0[:, 1],
    net_status=st,
    error_level=error_level,
    validation_resulting_cost=validation_resulting_cost,
    status="success"
  )

  result
end

multiseeds = 10

#lock for the threads to push! the results
lock_results = ReentrantLock()
Threads.@threads for iterator in 1:multiseeds
  try
    #train the model 
    random_seed = abs(rand(rng, Int))
    result = train_uode_model(random_seed, iterator)

    # Acquire the lock before pushing the results
    lock(lock_results)
    push!(results, result)
    unlock(lock_results)

  catch ex
    showerror(stdout, ex)
    # Acquire the lock before modifying the array
    lock(lock_results)
    push!(results, (status = "failed"))
    unlock(lock_results)
  end
end

serialize(folder_name * "/" * result_name_string, results)
