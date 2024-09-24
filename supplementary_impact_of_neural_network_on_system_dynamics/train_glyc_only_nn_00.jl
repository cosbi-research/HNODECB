#= 
Script to run the training of the Yeast Glycolysis UDE model on the dataset with error level 0.0.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, CSV, Plots, Statistics, Printf, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux

result_name_string = "glyc_00.jld"

folder_name = "res_glyc"
#creates the directory to save the results
if !isdir(folder_name)
  mkdir(folder_name)
end

error_level = "e0.0"

#inlcudes the model settings
include("../test_case_settings/glyc_model_settings/glycolitic_model_functions.jl")
include("../test_case_settings/glyc_model_settings/glycolitic_model_settings.jl")
column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]

#settings for non-stiff problems
integrator = TRBDF2(autodiff=false);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

#loads the data
ode_data = deserialize("../datasets/e0.0/data/ode_data_glycolysis.jld")
ode_data_sd = deserialize("../datasets/e0.0/data/ode_data_std_glycolysis.jld")
solution_dataframe = deserialize("../datasets/e0.0/data/pert_df_glycolysis.jld")
solution_sd_dataframe = deserialize("../datasets/e0.0/data/pert_df_sd_glycolysis.jld")

############################ TUNED HYPERPARAMETERS ############################
learning_rate_adam = 0.005
group_size = 3
continuity_term = 0.0

my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

approximating_neural_network = Lux.Chain(
  Lux.Dense(2, 2^4, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^4, 2^4, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^4, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

#initialize the random function to set the seeds and to have replicabile behavious
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

normalization_factor = maximum(original_ode_data, dims=2) - minimum(original_ode_data, dims=2)
normalization_factor_training = repeat(normalization_factor, 1, size(ode_data)[2])
normalization_factor_validation = repeat(normalization_factor, 1, size(validation_ode_data)[2])

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
  tmp_neural_network = deepcopy(approximating_neural_network)
  p_net, st = Lux.setup(rng, tmp_neural_network)

  #UDE derivative function
  uode_derivative_function = get_uode_model_function_with_only_nn(approximating_neural_network, st)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

  #multiple shooting ranges
  ranges = DiffEqFlux.group_ranges(datasize, group_size)

  #loss function for the comparison among the parameters and the predictions
  function loss_function(data, deviation, pred)
    original_cost = sum(abs2.(data .- pred) ./ abs2.(deviation))
    return 1 / size(data, 2) * sum(original_cost)
  end

  function loss_multiple_shooting(θ, sensealg, integrator)
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
      std = normalization_factor_training[:, rg]
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

    return loss
  end

  #callback function to observe training and populating the cost array
  function callback(θ, l, training_epochs, training_costs, best_training_parameters, training_parameterizations, epoch_parameterizations) #callback function to observe training

    epoch = extrema(training_epochs)[2] + 1

    if epoch % 100 == 1

      push!(training_parameterizations, deepcopy(θ))
      append!(epoch_parameterizations, epoch)

      println("********************************Epoch " * string(epoch) * " -- cost: " * string(l) * "")
      
      prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

      solution_dataframe_to_plot = deepcopy(solution_dataframe)
      validation_solution_dataframe_to_plot = deepcopy(validation_solution_dataframe)

      #solution_dataframe_to_plot = solution_dataframe_to_plot[solution_dataframe_to_plot.t .<=extrema(training_intervals[i].times)[2], :]
      #validation_solution_dataframe_to_plot = validation_solution_dataframe_to_plot[validation_solution_dataframe_to_plot.t .<= extrema(training_intervals[i].times)[2], :]

      prob = remake(
      prob_uode_pred;
      p=θ.p,
      tspan=extrema(solution_dataframe_to_plot.t),
      u0= 10 .^ θ.u0[:, 1]
      #u0 = original_u0#u0=ode_data[:, 1]
      )



      function unstable_check(dt, u, p, t)
        if  any(abs.(u) .> 1e4) 
          return true
        end
        return false
      end

      l2_cost = 0.0

      solutions = solve(prob, integrator, p=θ.p, saveat=0.01, abstol=abstol, reltol=reltol,
      sensealg=sensealg, unstable_check=unstable_check, verbose=false)
      plt = Plots.plot(solutions, layout=7)
      Plots.scatter!(plt, solution_dataframe_to_plot.t, Array(solution_dataframe_to_plot[:, 2:end]), layout=7)
      Plots.scatter!(plt, validation_solution_dataframe_to_plot.t, Array(validation_solution_dataframe_to_plot[:, 2:end]), color= "green", layout=7)

      title!(plt, string(round(l2_cost,digits=2)))
      display(plt)
   
    end

    #max 3.5 hours for an optimization
    if time() - initial_time > 60 * 3.5 * 60
      println("Too slow optimization")
      return true
    end

    #populates the training costs  
    training_epochs[epoch] = epoch
    training_costs[epoch] = l

    #keep track of the best solution, to have a uniform behaviour if exiting when stuck
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

      loss = 1 / size(validation_df, 1) * sum(abs2.(Array(validation_df[:, 2:end])' .- x) ./ abs2.(normalization_factor_validation))

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
  p = ComponentArray{eltype(p_net)}()
  p = ComponentArray(p; p_net)
  u0 = deepcopy(ode_data)
  # do not let the intermediate values to be negative
  u0 = max.(u0, 10^(-7))
  u0 = log10.(u0)
  starting_point_in = ComponentVector{Float64}(p=p, u0=u0)

  best_training_parameters = [starting_point_in]

  optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x, sensealg, TRBDF2(autodiff=false)), adtype)
  optprob = Optimization.OptimizationProblem(optf, starting_point_in)
  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  ##################### ADAM ###########
  training_parameterizations, epoch_parameterizations = [], []
  res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, best_training_parameters, training_parameterizations, epoch_parameterizations), maxiters=3000)

  # there is probably a bug in the LBFGS function and sometimes it fails with an exception, use Interpolating Adjoint because with Quadrature it fails
  #try
  #  optf2 = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), Vern7()), adtype)
  #  optprob2 = Optimization.OptimizationProblem(optf2, best_training_parameters[1])
  #  res = Optimization.solve(optprob2, Optim.LBFGS(), callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, best_training_parameters, training_parameterizations, epoch_parameterizations), maxiters=5000, allow_f_increases=true)
  #catch
  #  println("BFGS failed with error")
  #end


  best_parameterization = best_training_parameters[1]
  validation_resulting_cost = validation_loss_function(best_parameterization)

  # rescale the parameters
  initial_values_to_save = 10 .^ best_parameterization.u0

  #saves the results  
  result = (
    parameters_training=best_parameterization.p,
    initial_state_training=initial_values_to_save,
    net_status=st,
    validation_resulting_cost=validation_resulting_cost,
    status="success",
    training_parameterizations = training_parameterizations,
    epoch_parameterizations = epoch_parameterizations,
    training_epochs = training_epochs,
    training_costs = training_costs
  )

  result
end

multiseeds = 10

#lock for the threads to push! the results
lock_results = ReentrantLock()
for iterator in 1:multiseeds
  try
    #train the model 
    random_seed = abs(rand(rng, Int))
    if iterator != 6
      continue
    end
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
