#= 
Script to train the Lotka Volterra UDE model for fixed (and vartying) values of the mechanistic parameter on DS_0.00
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics, Printf, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs, QuasiMonteCarlo
using DiffEqFlux, .Flux

result_name_string = "lotka_volterra_0.0.jld"

folder_name = "raw_results"
#creates the directory to save the files
if !isdir(folder_name)
  mkdir(folder_name)
end


problem = "lotka_volterra"
error_level = "e0.0"
error_level_num = parse(Float64, error_level[2:end])

l2_regularization = 0.0

include("../test_case_settings/lv_model_settings/lotka_volterra_model_functions.jl")
include("../test_case_settings/lv_model_settings/lotka_volterra_model_settings.jl")

column_names = ["t", "s1", "s2"]

integrator = Vern7()
abstol = 1e-7
reltol = 1e-6

sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))

# using the hyperparameters from the hyperparameter tuning for error 0%
learning_rate_adam =  0.031242

# neural network to replace the interaction terms
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
  Lux.Dense(2, 16, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(16, 16, tanh; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(16, 2; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

group_size = 2
continuity_term = 0.00261295

ode_data = deserialize("../datasets/"*error_level * "/data/ode_data_lotka_volterra.jld")
ode_data_sd = deserialize("../datasets/"*error_level * "/data/ode_data_std_lotka_volterra.jld")
solution_dataframe = deserialize("../datasets/"*error_level * "/data/pert_df_lotka_volterra.jld")
solution_sd_dataframe = deserialize("../datasets/"*error_level * "/data/pert_df_sd_lotka_volterra.jld")

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)

#settings for the training set sampling
tspan = (initial_time_training, end_time_training)
tsteps = range(tspan[1], tspan[2], length=21)
stepsize = (tspan[2] - tspan[1]) / (21 - 1)
lentgth_tsteps = length(tsteps)

######################################################### TRAINING - VALIDATION SPLIT #########################################################
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

#########################################################################################################################################
#defines the grid of parameters to integrate the UDE model
steps = 50
lower_bound = 0.013
upper_bound = 3.0
p1_values = range(lower_bound, upper_bound, length=steps)


results = []

#function that traing the model with the prey birth rate fixed to the the value p1_values[iterator]
#the seed is used to initialize the neural network
function train_uode_model(seed, iterator)

  initial_time = time()

  rng = StableRNG(seed)
  #sets the seed
  # Multilayer FeedForward
  local_approximating_neural_network = deepcopy(approximating_neural_network)
  p_net, st = Lux.setup(rng, local_approximating_neural_network)

  #gets the hybrid derivative function and instatiate the prediction ODE problem
  p1_value = p1_values[iterator]
  uode_derivative_function = get_uode_model_function_with_fixed_p1(approximating_neural_network, st, p1_value)

  tmp_steps = solution_dataframe.t
  datasize = size(ode_data, 2)
  #defines the groups for the multiple shooting
  ranges = DiffEqFlux.group_ranges(datasize, group_size)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

  function loss_function(data, deviation, pred)
    original_cost = sum(abs2.(data .- pred))

    return 1/size(data, 2) * sum(original_cost)
  end

  function loss_multiple_shooting(θ)
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

    num_evaluations = sum([sol.stats.nf for sol in sols])

    # Abort and return infinite loss if one of the integrations failed
    for i in 1:length(sols)
      if size(Array(sols[i]))[2] != length(ranges[i])
        println("Divergent trajectory here num evaluations: " * string(num_evaluations))
        return Inf, num_evaluations
      end
    end

    group_predictions = Array.(sols)
    # Calculate multiple shooting loss (distance from experimental points)
    loss = 0
    for (i, rg) in enumerate(ranges)
      u = ode_data[:, rg]
      std = ode_data_sd[:, rg]
      û = group_predictions[i]
      loss += loss_function(u, std, û)
    end

    #Calculates the continuity loss (evaluating the distance between the end of the first group and the start of the second as in the paper)
    for (i, rg) in enumerate(ranges)
      if i == 1
        continue
      end

      u0 = group_predictions[i-1][:, end]
      u1 = group_predictions[i][:, 1]
      loss += continuity_term * sum(abs2, u0 - u1)
    end

    #l2 regularization term
    loss += l2_regularization * sum(abs2, p.p_net)

    return loss, num_evaluations
  end

  #callback function to observe training and populating the cost array
  function callback(θ, l, num_evaluation, training_function_evaluation, training_epochs, training_costs) 

    epoch = extrema(training_epochs)[2] + 1
    println("********************************Epoch " * string(epoch) * " -- cost: " * string(l) * "")

    training_function_evaluation[epoch] = num_evaluation

    #max 1.5 hours for an optimization
    if time() - initial_time > 60 * 1.5 * 60
      println("Too slow optimization")
      return true
    end

    training_epochs[epoch] = epoch
    training_costs[epoch] = l

    #exit if the the training is stuck in regions with high cost
    if epoch > 200 && minimum(training_costs[(epoch-5):(epoch)]) > 10^6
      println("Stuck in non integrability region")
      return true
    end

    return false
  end

  function validation_loss_function(θ)

    loss = 0.0
    validation_df = validation_solution_dataframe

    try
      prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

      par = θ.p
      init_par = θ.u0[:,1]
      prob = remake(
        prob_uode_pred;
        p=par,
        tspan=extrema(vcat(0.0, validation_df.t)),
        u0=Array(init_par)
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

      loss = 1 / size(validation_df, 1) * sum(abs2.(Array(validation_df[:, 2:end])' .- x))

    catch
      loss = Inf
    end

    return loss
  end

  #defining the optimization procedures
  adtype = Optimization.AutoZygote()
  #######first optimization with 0.6 lenght intervals

  #generates a vectors of -1 with length 100 and type integrater
  training_epochs = zeros(Int, 50000)
  training_costs = zeros(50000)
  training_function_evaluation = zeros(50000)

  validation_epochs = zeros(Int, 30000)
  validation_costs = zeros(30000)
  validation_costs .= Inf

  p_net = ComponentArray(p_net)
  p = ComponentArray{eltype(p_net)}()
  p = ComponentArray(p; p_net)
  starting_point_in = ComponentVector{Float64}(p=p, u0=deepcopy(ode_data))

  datasize = size(ode_data, 2)

  res = nothing
  validation_resulting_cost = nothing

  #defines the integration problem
  optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, starting_point_in)
  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  ##################### ADAM ###########
  res = Optimization.solve(optprob, opt, callback=(θ, l, num_evaluation) -> callback(θ, l, num_evaluation, training_function_evaluation, training_epochs, training_costs), maxiters=10000)
  ##################### LBFGS ##########
  optprob_LBFGS = remake(optprob, u0=res.u)
  res = Optimization.solve(optprob_LBFGS, Optim.LBFGS(), callback=(θ, l, num_evaluation) -> callback(θ, l, num_evaluation, training_function_evaluation, training_epochs, training_costs), maxiters=4000, allow_f_increases=false)

  best_parameterization = res.u
  validation_resulting_cost = validation_loss_function(res.u)

  #saves the results  
  result = (net=approximating_neural_network,
    initial_optimization_point=starting_point_in,
    parameters_training=best_parameterization.p,
    initial_state_training=best_parameterization.u0[:, 1],
    net_status=st,
    validation_resulting_cost=validation_resulting_cost,
    status="success",
    iterator=iterator
  )

  result
end

#lock for push! the results during threads
lock_results = ReentrantLock()

#performs the optimizations with fixed p1 and saves the results in the results array
Threads.@threads for iterator in 1:steps
  try
    #train the model
    random_seed = abs(rand(rng, Int))
    result = train_uode_model(random_seed, iterator)

    # inserts the result
    lock(lock_results)
    push!(results, result)
    unlock(lock_results)

  catch ex
    # inserts the failed result
    showerror(stdout, ex)
    # Acquire the lock before modifying the array
    lock(lock_results)
    push!(results, (status="failed", task_name=result_name_string, iterator=iterator, tspan=tspan, tsteps=tsteps))
    unlock(lock_results)
  end
end

serialize(folder_name * "/" * result_name_string, results)