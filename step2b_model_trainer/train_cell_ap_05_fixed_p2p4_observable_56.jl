#= 
Script to run the training of the cell apoptosis UDE model on the dataset with error level 0.05, fixing p2 and p4 to their literature values and
assuming that only the variables y5 and y6 are observable
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics, Printf, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux

result_name_string = "ca_05.jld"

folder_name = "res_cell_ap_fixed_p2p4_observable_56"
#creates the directory to save the results
if !isdir(folder_name)
  mkdir(folder_name)
end

error_level = "e0.05"

#includes the model settings
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../test_case_settings/cell_apoptosis_settings/cell_apop_model_settings.jl")
column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

observables = [5, 6]

#settings for stiff problems
integrator = TRBDF2(autodiff=true);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#loads the data
ode_data = deserialize("../datasets/e0.05_doubled/data/ode_data_cell_apoptosis.jld")
ode_data_sd = deserialize("../datasets/e0.05_doubled/data/ode_data_std_cell_apoptosis.jld")
solution_dataframe = deserialize("../datasets/e0.05_doubled/data/pert_df_cell_apoptosis.jld")
solution_sd_dataframe = deserialize("../datasets/e0.05_doubled/data/pert_df_sd_cell_apoptosis.jld")

############################ TUNED HYPERPARAMETERS ############################
l2_regularization = 0.1
learning_rate_adam = 0.00411764

p1 = 2.0817
p3 = 1238.37
p5 = 15918.5
p6 = 50.0665
p7 = 2.27564e6
p8 = 2.42737
p9 = 31.3963

pretrained_ode_pars = [p1, p3, p5, p6, p7, p8, p9]

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)
approximating_neural_network = Lux.Chain(
  Lux.Dense(6, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform)
)

group_size = 10
continuity_term = 0.00496526

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)

#############################################################################################################################
#################################### TRAINING VALIDATION SPLIT ##############################################################

shuffled_positions = shuffle(2:size(solution_dataframe)[1])
first_validation = rand(2:5)
validation_mask = [(first_validation + k * 5) for k in 0:6]
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

#constants used in the optimization
tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)
tspan = (initial_time_training, end_time_training)

results = []

#function to train the uode model
function train_uode_model(seed, iterator)

  #gets the time to monitor the training time
  initial_time = time()

  rng = StableRNG(seed)
  local_approximating_neural_network = deepcopy(approximating_neural_network)
  p_net, st = Lux.setup(rng, local_approximating_neural_network)

  #gets the hybrid derivative function and instatiate the prediction ODE problem
  uode_derivative_function = get_uode_model_function_fixed_p2p4(approximating_neural_network, st, pretrained_ode_pars)
  initial_ode_pars = ones(length(pretrained_ode_pars))
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

  #multiple shooting ranges
  ranges = DiffEqFlux.group_ranges(datasize, group_size)

  function loss_function(data, deviation, pred)
    original_cost = sum(abs2.(data[observables, 1:end] .- pred[observables, 1:end]) ./ abs2.(deviation[observables, 1:end]))
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

    initial_points = 10 .^ initial_point_parameters

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

    #initial state
    loss += 1 / size(group_predictions[1], 2) * sum(abs2.(group_predictions[1][Not(observables), 1] .- ode_data[Not(observables), 1]) ./ abs2.(ode_data_sd[Not(observables), 1]))

    #l2 regularization term
    loss += l2_regularization * (sum(abs2, p.p_net.layer_1.weight) + sum(abs2, p.p_net.layer_2.weight) + sum(abs2, p.p_net.layer_3.weight) + sum(abs2, p.p_net.layer_4.weight) + sum(abs2, p.p_net.layer_5.weight) + sum(abs2, p.p_net.layer_6.weight))

    return loss
  end

  #callback function to observe training and populating the cost array
  function callback(θ, l, training_epochs, training_costs, best_training_parameters)

    epoch = extrema(training_epochs)[2] + 1
    println("********************************Epoch " * string(epoch) * " -- cost: " * string(l) * " seed " * string(seed) * "********************************")

    #max 3.5 hours for an optimization
    if time() - initial_time > 60 * 3.5 * 60
      println("Too slow optimization")
      return true
    end

    training_epochs[epoch] = epoch
    training_costs[epoch] = l

    #keep track of the best solution, to have a uniform behaviour if exiting when stuck
    if epoch == 1 || l < minimum(training_costs[1:(epoch-1)])
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
    validation_sd_df = validation_solution_sd_dataframe

    try
      times_consdiered = validation_df.t
      max_time = extrema(times_consdiered)[2]

      prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

      par = θ.p
      init_par = 10 .^ θ.u0
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

      if size(x)[2] != size(validation_df, 1)
        return Inf
      end

      loss = 1 / size(validation_df, 1) * sum(abs2.((Array(validation_df[:, 2:end])')[observables, :] .- x[observables, :]) ./ abs2.((Array(validation_sd_df[:, 2:end])')[observables, :]))

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
  ode_par = ComponentArray(initial_ode_pars)
  p = ComponentArray{eltype(p_net)}()
  p = ComponentArray(p; p_net)
  p = ComponentArray(p; ode_par)
  # do not let the intermediate values to be negative
  u0 = deepcopy(ode_data)
  u0 = max.(u0, 10^(-7))
  u0 = log10.(u0)
  starting_point_in = ComponentVector{Float64}(p=p, u0=u0)

  best_training_parameters = [starting_point_in]

  optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, starting_point_in)
  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  ##################### ADAM ###########
  res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, best_training_parameters), maxiters=10000)

  best_parameterization = best_training_parameters[1]
  validation_resulting_cost = validation_loss_function(best_parameterization)

  # rescale the parameters
  best_parameterization.p.ode_par = best_parameterization.p.ode_par .* pretrained_ode_pars
  initial_values_to_save = 10 .^ best_parameterization.u0

  #saves the results  
  result = (
    parameters_training=best_parameterization.p,
    initial_state_training=initial_values_to_save,
    net_status=st,
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