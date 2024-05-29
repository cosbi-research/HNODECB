#= 
Script to run the L2 regularization tuning for the cell apoptosis model using the TPE algorithm with error level 5.0%,
fixing p2 and p4 to their literature values, assuming that only y5 and y6 are observable
=#

cd(@__DIR__)

using ComponentArrays, Lux, SciMLSensitivity, Serialization, DifferentialEquations, LinearAlgebra, Random, DataFrames, Base.Threads, Dates
using Optimization, OptimizationOptimisers, OptimizationOptimJL, StableRNGs, InvertedIndices
using DiffEqFlux, .Flux

#python library for hyperparameter optimization 
using PyCall
optuna = pyimport("optuna")

result_folder = "results_cell_ap_fixed_p2p4_observable_56"
if !isdir(result_folder)
  mkdir(result_folder)
end
result_name_string = "cell_ap_05.jld"

error_level = "e0.05"

#observable variables
observables = [5, 6] 

#inlcudes the model settings
include("../../test_case_settings/cell_apoptosis_settings/cell_apop_model_functions.jl")
include("../../test_case_settings/cell_apoptosis_settings/cell_apop_model_settings.jl")
column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
 
#settings for stiff problems
integrator = TRBDF2(autodiff=false);
abstol = 1e-7
reltol = 1e-6
sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))
  
ode_data = deserialize("../../datasets/e0.05_doubled/data/ode_data_cell_apoptosis.jld")
ode_data_sd = deserialize("../../datasets/e0.05_doubled/data/ode_data_std_cell_apoptosis.jld")
solution_dataframe = deserialize("../../datasets/e0.05_doubled/data/pert_df_cell_apoptosis.jld")
solution_sd_dataframe = deserialize("../../datasets/e0.05_doubled/data/pert_df_sd_cell_apoptosis.jld")

#initialize the random function to set the seeds and to have replicabile behavious
rng = Random.default_rng()
Random.seed!(rng, 0)
my_glorot_uniform(rng, dims...) = Lux.glorot_uniform(rng, dims...)

#############################################################################################################################
#################################### TRAINING VALIDATION SPLIT ##############################################################

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


#########################################################################################################
################################# Hyper-parameters from stage 1 ##########################################

p1 = 2.0817
p3 = 1238.37
p5 = 15918.5
p6 = 50.0665
p7 = 2.27564e6
p8 = 2.42737
p9 = 31.3963

original_ode_par = [p1, p3, p5, p6, p7, p8, p9]

approximating_neural_network = Lux.Chain(
  Lux.Dense(6, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 2^2, gelu; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
  Lux.Dense(2^2, 1; init_weight=my_glorot_uniform, init_bias=my_glorot_uniform),
)

original_learning_rate_adam = 0.00411764
ms_group_size = 10
ms_continuity_term = 0.00496526

############################################################################################################

#constants used in the optimization
tmp_steps = solution_dataframe.t
datasize = size(ode_data, 2)
tspan = (initial_time_training, end_time_training)

#objective function depends on the seed, attempt with threee different seeds
function objective(trial)

  global rng
  seed = abs(rand(rng, Int))
  resulting_loss_1 = internal_objective(trial, seed)

  seed = abs(rand(rng, Int))
  resulting_loss_2 = internal_objective(trial, seed)

  seed = abs(rand(rng, Int))
  resulting_loss_3 = internal_objective(trial, seed)

  resulting_loss = minimum([resulting_loss_1, resulting_loss_2, resulting_loss_3])

  return resulting_loss

end

#internal objective function got the grid optimization
function internal_objective(trial, seed)


  l2_regularization = trial.suggest_uniform("l2_regularization", -100, 100)
  learning_rate_adam = original_learning_rate_adam

  initial_time = time()

  rng = StableRNG(seed)

  tmp_approximating_neural_network = deepcopy(approximating_neural_network)
  p_net, st = Lux.setup(rng, tmp_approximating_neural_network)

  #UDE derivative function
  uode_derivative_function = get_uode_model_function_fixed_p2p4(approximating_neural_network, st, original_ode_par)
  initial_ode_par = ones(length(original_ode_par))

  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

  # MS settings
  ranges = DiffEqFlux.group_ranges(datasize, ms_group_size)
  prob_uode_pred = ODEProblem{true}(uode_derivative_function, original_u0, tspan)

  #loss function for the comparison among the parameters and the predictions
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
          u0=10 .^ initial_points[:, first(rg)]
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
    loss += 1 / size(group_predictions[1], 2) * sum(abs2.(group_predictions[1][Not(observables), 1] .- ode_data[Not(observables), 1]) ./ abs2.(ode_data_sd[Not(observables), 1]))

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

    loss += l2_regularization * (sum(abs2, p.p_net.layer_1.weight) + sum(abs2, p.p_net.layer_2.weight) + sum(abs2, p.p_net.layer_3.weight) + sum(abs2, p.p_net.layer_4.weight) + sum(abs2, p.p_net.layer_5.weight) + sum(abs2, p.p_net.layer_6.weight))

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

    #attempt too slow
    if time() - initial_time > 60 * 15
      println("Too slow integration")
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
        u0=10 .^ init_par[:,1]
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
      loss = 1 / size(validation_df, 1) * sum(abs2.((Array(validation_df[:, 2:end])')[observables,:] .- x[observables,:]) ./ abs2.((Array(validation_sd_dataframe[:, 2:end])')[observables,:]))
    catch
      loss = Inf
    end

    return loss
  end

 #defining the optimization ad method
  adtype = Optimization.AutoZygote()

  p_net = ComponentArray(p_net)
  ode_par = ComponentArray(initial_ode_par)
  p = ComponentArray{eltype(p_net)}()
  p = ComponentArray(p; p_net)
  p = ComponentArray(p; ode_par)
  u0 = deepcopy(ode_data)
  u0 = max.(u0, 10^-7)
  # do not allow the initial state to be negative
  u0 = log10.(u0)
  starting_point_in = ComponentVector{Float64}(p=p, u0=u0)

  #generates a vectors of -1 with length 100 and type integrater
  training_epochs = zeros(Int, 50000)
  training_costs = zeros(50000)

  optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, starting_point_in)
  opt = OptimizationOptimisers.Adam(learning_rate_adam)

  ##################### ADAM ###########
  stuck = [false]
  res = Optimization.solve(optprob, opt, callback=(θ, l) -> callback(θ, l, training_epochs, training_costs, 2000, stuck), maxiters=2001)

  validation_resulting_cost = nothing
  if stuck[1] == true
    println("Stuck optimization")
    validation_resulting_cost = Inf
  else
    validation_resulting_cost = validation_loss_function(res.u)
  end

  return validation_resulting_cost
end

global trial_parameters = []
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