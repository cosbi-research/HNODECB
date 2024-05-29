#= 
Script to run the hyperparameter optimization for the Yeast Glycolysis model using the TPE algorithm with error level 0.0%,
assuming that only y5 and y6 are observable.
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs
using DiffEqFlux, .Flux

#python library for hyperparameter optimization 
using PyCall
optuna = pyimport("optuna")

result_folder = "results_glyc_observable_56"
if !isdir(result_folder)
  mkdir(result_folder)
end
result_name_string = "glyc_00.jld"

error_level = "e0.0"

observables = [5,6]

#inlcudes the model settings
include("../../test_case_settings/glyc_model_settings/glycolitic_model_functions.jl")
include("../../test_case_settings/glyc_model_settings/glycolitic_model_settings.jl")
column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]

#settings for stiff problems
integrator = TRBDF2(autodiff=false);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))

ode_data = deserialize("../../datasets/e0.0_doubled/data/ode_data_glycolysis.jld")
ode_data_sd = deserialize("../../datasets/e0.0_doubled/data/ode_data_std_glycolysis.jld")
solution_dataframe = deserialize("../../datasets/e0.0_doubled/data/pert_df_glycolysis.jld")
solution_sd_dataframe = deserialize("../../datasets/e0.0_doubled/data/pert_df_sd_glycolysis.jld")

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)


#############################################################################################################################
#################################### TRAINING VALIDATION SPLIT ##############################################################

#settings for the training set sampling
tspan = (initial_time_training, end_time_training)
tsteps = range(tspan[1], tspan[2], length=42)
stepsize = (tspan[2] - tspan[1]) / (42 - 1)
lentgth_tsteps = length(tsteps)

#generates the random mask for the training and the valudation data set
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

######################################################################################################################################
######################################################################################################################################

results = []

#boundaries for the mechanistic parameters
real_values = [2.5, 100, 0.52, 4, 6, 1, 12, 16, 4, 100, 13, 1.28, 0.1, 1.8]
lower_bounds = 10^-1.0 .* real_values
upper_bounds = 10^1.0 .* real_values

lower_bounds[3] = log10(lower_bounds[3])
upper_bounds[3] = log10(upper_bounds[3])

#computes the min-max variation of each timeseries to normalize the objective function
normalization_factor = maximum(original_ode_data, dims=2) - minimum(original_ode_data, dims=2)
normalization_factor_training = repeat(normalization_factor, 1, size(ode_data)[2])
normalization_factor_validation = repeat(normalization_factor, 1, size(validation_ode_data)[2])

#constants used during the objective computation
tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)

#objective function got the TPE optimization
function objective(original_ode_par, learning_rate_adam, num_hidden_layers, num_hidden_nodes, ms_group_size, ms_continuity_term, seed)

  if num_hidden_layers == 1
    approximating_neural_network = Lux.Chain(
      Lux.Dense(2, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    )
  elseif num_hidden_layers == 2
    approximating_neural_network = Lux.Chain(
      Lux.Dense(2, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    )
  elseif num_hidden_layers == 3
    approximating_neural_network = Lux.Chain(
      Lux.Dense(2, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    )
  elseif num_hidden_layers == 4
    approximating_neural_network = Lux.Chain(
      Lux.Dense(2, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    )
  elseif num_hidden_layers == 5
    approximating_neural_network = Lux.Chain(
      Lux.Dense(2, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 2^num_hidden_nodes, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
      Lux.Dense(2^num_hidden_nodes, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
    )
  end

  initial_time = time()

  #initialize the neural network
  rng = StableRNG(seed)
  local_approximating_neural_network = deepcopy(approximating_neural_network)
  p_net, st = Lux.setup(rng, local_approximating_neural_network)

  #gets the hybrid derivative function and instatiate the prediction ODE problem
  uode_derivative_function = get_uode_model_function(local_approximating_neural_network, st, original_ode_par)
  # mechanistic parmeter scaled to the initial value
  initial_parameters = ones(14)

  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

  ranges = DiffEqFlux.group_ranges(datasize, ms_group_size)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

  function loss_function(data, deviation, pred)
    original_cost = sum(abs2.(data[observables,:] .- pred[observables,:]) ./ abs2.(deviation[observables,:]))
    return 1/size(data, 2) * sum(original_cost)
  end

  function loss_multiple_shooting(θ)
    #unpack the parameters
    p = θ.p
    tsteps = tmp_steps
    prob = prob_uode_pred
    solver = integrator
    initial_point_parameters = 10 .^ θ.u0

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
    # Calculate multiple shooting loss (distance from experimental points)
    loss = 0
    for (i, rg) in enumerate(ranges)
      u = ode_data[:, rg]
      std = normalization_factor_training[:, rg]
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
      loss += ms_continuity_term * sum(abs2, u0 - u1)
    end

    loss += 1 / size(group_predictions[1], 2) * sum(abs2.(group_predictions[1][Not(observables), 1] .- ode_data[Not(observables), 1]) ./ abs2.(normalization_factor_training[Not(observables), 1]))

    return loss
  end

  #callback function to observe training
  epoch = 1 
  function callback(θ, l, training_epochs, training_costs, num_epoch_to_finish, stuck) 
      
    println("Epoch: " * string(epoch) * " - Loss: " * string(l))
    #this is necessary to have the cost at the last iteration, not the best cost found during the optimization
    if epoch == num_epoch_to_finish
      return true
    end

    training_epochs[epoch] = epoch
    training_costs[epoch] = l

    #max 4 minutes for each single trial, otherwise it is considered stuck
    if time() - initial_time > 4 * 60
      println("Too slow optimization")
      return true
    end

    #early exit if the optimization is stuck in a too high cost region
    if epoch > 10 && minimum(training_costs[(epoch-5):(epoch)]) > 10^6
      stuck[1] = true
      return true
    end
  
    epoch += 1

    return false
  end

  function validation_loss_function(θ)

    loss = 0.0
    validation_df = validation_solution_dataframe

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
        u0=10 .^ init_par[:,1]
        #u0=ode_data[:, 1]
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

      if size(validation_df, 1) != size(x, 2)
        return Inf
      end

      loss = 1 / size(validation_df, 1) * sum(abs2.((Array(validation_df[:, 2:end])')[observables,:] .- x[observables,:]) ./ abs2.(normalization_factor_validation[observables,:]))

      solutions = solve(prob, integrator, p=par, abstol=abstol, reltol=reltol, sensealg=sensealg, unstable_check=unstable_check, verbose=false)
    catch exc
      println("Exception during validation loss function")
      println(exc)
      loss = Inf
    end

    return loss
  end

  #defining the optimization ad method
  adtype = Optimization.AutoZygote()

  training_epochs = zeros(Int, 50000)
  training_costs = zeros(50000)

  p_net = ComponentArray(p_net)
  ode_par = ComponentArray(initial_parameters)
  p = ComponentArray{eltype(p_net)}()
  p = ComponentArray(p; p_net)
  p = ComponentArray(p; ode_par)
  u0 = deepcopy(ode_data)
  # do not allow the initial state to be negative
  u0 = max.(u0, 10^-7)
  u0 = log10.(u0)
  starting_point_in = ComponentVector{Float64}(p=p, u0=u0)

  optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, starting_point_in)
  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  ##################### ADAM ###########
  stuck = [false]
  res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, 500, stuck), maxiters=501)

  validation_resulting_cost = nothing
  if stuck[1] == true
    println("Stuck optimization")
    validation_resulting_cost = Inf
  else
    validation_resulting_cost = validation_loss_function(res.u)
  end

  return res.u, validation_resulting_cost
end

global trial_parameters = []
# OPTUNA TPE optimization 
study = optuna.create_study(sampler=optuna.samplers.TPESampler(consider_prior=false, n_startup_trials=200, multivariate=true, seed=0))

  for optuna_iteration in 1:500

    trial = study.ask()
  
    # select the trial hyperparameters
    original_p1 = trial.suggest_float("p1", lower_bounds[1], upper_bounds[1])
    original_p2 = trial.suggest_float("p2", lower_bounds[2], upper_bounds[2])
    original_p3 = trial.suggest_float("p3", lower_bounds[3], upper_bounds[3])
    original_p4 = trial.suggest_float("p4", lower_bounds[4], upper_bounds[4])
    original_p5 = trial.suggest_float("p5", lower_bounds[5], upper_bounds[5])
    original_p6 = trial.suggest_float("p6", lower_bounds[6], upper_bounds[6])
    original_p7 = trial.suggest_float("p7", lower_bounds[7], upper_bounds[7])
    original_p8 = trial.suggest_float("p8", lower_bounds[8], upper_bounds[8])
    original_p9 = trial.suggest_float("p9", lower_bounds[9], upper_bounds[9])
    original_p10 = trial.suggest_float("p10", lower_bounds[10], upper_bounds[10])
    original_p11 = trial.suggest_float("p11", lower_bounds[11], upper_bounds[11])
    original_p12 = trial.suggest_float("p12", lower_bounds[12], upper_bounds[12])
    original_p13 = trial.suggest_float("p13", lower_bounds[13], upper_bounds[13])
    original_p14 = trial.suggest_float("p14", lower_bounds[14], upper_bounds[14])
  
    original_ode_pars = [original_p1, original_p2, original_p3, original_p4, original_p5, original_p6, original_p7, original_p8, original_p9, original_p10, original_p11, original_p12, original_p13, original_p14]
  
    ms_group_size =  trial.suggest_int("ms_group_size", 2, 10)
    ms_continuity_term = trial.suggest_float("ms_continuity_term", 0.001, 1000.0, step=nothing, log=true)
  
    learning_rate_adam = trial.suggest_float("learning_rate_adam", 1e-5, 1e-1, step=nothing, log=true)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
    num_hidden_nodes = trial.suggest_int("num_hidden_nodes", 2, 4)
  
    seed = abs(rand(rng, Int))

    #since the results are seed dependent, we run the optimization three times and we keep the best result
    results_1 = nothing
    try
      results_1 = objective(original_ode_pars, learning_rate_adam, num_hidden_layers, num_hidden_nodes, ms_group_size, ms_continuity_term, seed)
    catch
      results_1 = (nothing, Inf, -1)
    end
  
    seed = abs(rand(rng, Int))
    results_2 = nothing
    try
      results_2 = objective(original_ode_pars, learning_rate_adam, num_hidden_layers, num_hidden_nodes, ms_group_size, ms_continuity_term, seed)
    catch
      results_2 = (nothing, Inf, -1)
    end
    
    results_3 = nothing
    try 
      seed = abs(rand(rng, Int))
      results_3 = objective(original_ode_pars, learning_rate_adam, num_hidden_layers, num_hidden_nodes, ms_group_size, ms_continuity_term, seed)
    catch
      results_3 = (nothing, Inf, -1)
    end

    result = results_1
    if results_2[2] < result[2]
      result = results_2
    end
    if results_3[2] < result[2]
      result = results_3
    end

    if result[2] == Inf
      result = (nothing, Inf)
    end
  
    cost = result[2]
  
    try
      θ_to_memorize = deepcopy(result[1])
      global trial_parameters
      θ_to_memorize.p.ode_par  = θ_to_memorize.p.ode_par .* original_ode_pars
      push!(trial_parameters, θ_to_memorize)
    
      #rescale the parameters to obtain the actual value
      ode_par_optimized = result[1].p.ode_par .* original_ode_pars
  
      study.tell(trial, cost)
  
      result_params = copy(trial.params)

      result_params["p1"] = min(max(ode_par_optimized[1], lower_bounds[1]),upper_bounds[1])
      result_params["p2"] = min(max(ode_par_optimized[2], lower_bounds[2]),upper_bounds[2])
      result_params["p3"] = min(max(ode_par_optimized[3], lower_bounds[3]),upper_bounds[3])
      result_params["p4"] = min(max(ode_par_optimized[4], lower_bounds[4]),upper_bounds[4])
      result_params["p5"] = min(max(ode_par_optimized[5], lower_bounds[5]),upper_bounds[5])
      result_params["p6"] = min(max(ode_par_optimized[6], lower_bounds[6]),upper_bounds[6]) 
      result_params["p7"] = min(max(ode_par_optimized[7], lower_bounds[7]),upper_bounds[7])
      result_params["p8"] = min(max(ode_par_optimized[8], lower_bounds[8]),upper_bounds[8])
      result_params["p9"] = min(max(ode_par_optimized[9], lower_bounds[9]),upper_bounds[9])
      result_params["p10"] = min(max(ode_par_optimized[10], lower_bounds[10]),upper_bounds[10])
      result_params["p11"] = min(max(ode_par_optimized[11], lower_bounds[11]),upper_bounds[11])
      result_params["p12"] = min(max(ode_par_optimized[12], lower_bounds[12]),upper_bounds[12])
      result_params["p13"] = min(max(ode_par_optimized[13], lower_bounds[13]),upper_bounds[13])
      result_params["p14"] = min(max(ode_par_optimized[14], lower_bounds[14]),upper_bounds[14])
    
      result_trial = optuna.create_trial(
        params=result_params,
        distributions=trial.distributions,
        value=cost,
        user_attrs = Dict("optuna_iteration" => optuna_iteration)
      )
    
      println("Iteration: " * string(optuna_iteration) * " - " * string(result_trial.value) * " - " * string(result_trial.params))
    
      study.add_trial(result_trial) 
    catch

      study.tell(trial, cost)
  
      result_params = copy(trial.params)

      result_params["p1"] = original_ode_pars[1]
      result_params["p2"] = original_ode_pars[2]
      result_params["p3"] = original_ode_pars[3]
      result_params["p4"] = original_ode_pars[4]
      result_params["p5"] = original_ode_pars[5]
      result_params["p6"] = original_ode_pars[6]
      result_params["p7"] = original_ode_pars[7]
      result_params["p8"] = original_ode_pars[8]
      result_params["p9"] = original_ode_pars[9]
      result_params["p10"] = original_ode_pars[10]
      result_params["p11"] = original_ode_pars[11]
      result_params["p12"] = original_ode_pars[12]
      result_params["p13"] = original_ode_pars[13]
      result_params["p14"] = original_ode_pars[14]
  
      result_trial = optuna.create_trial(
        params=result_params,
        distributions=trial.distributions,
        value=cost,
        user_attrs = Dict("validation_int_step_number" => result[3], "optuna_iteration" => optuna_iteration)
      )
      
      study.add_trial(result_trial)  
    end
  
    best_optimization = study.best_trial
  end

#save the optimization results
result = (
  study=study,
  trial_parameters=trial_parameters,
  error_level=error_level
)

serialize(result_folder * "/" * result_name_string, result)