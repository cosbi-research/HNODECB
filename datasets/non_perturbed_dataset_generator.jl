"""
generate_non_perturbed_training_set(round_truth_function, u0, p, tspan, save_at ; column_names, integrator)

Solve the problems and returns the solution at the given times as a DataFrame
"""
function generate_non_perturbed_training_set(ground_truth_function, u0, p, tspan, save_at ; column_names = ["t", "s1", "s2", "s3", "s4", "s5", "s6", "s7"], integrator = Tsit5())
    [:t, :s1, :s2, :s3, :s4, :s5, :s6, :s7]
    #defining the problem and integrating in the desidered time interval
    ode_problem = ODEProblem(ground_truth_function, u0, tspan, p)
    solutions = solve(ode_problem, integrator, saveat=save_at, reltol=1e-8, abstol=1e-8)

    if solutions.retcode != :Success
        #saving the data
        error("Integration with this parameters was not succesfull")
    else
        solutions_as_matrix = Array(solutions)
        #put save_at as first row of solutions_as_matrix
        solutions_as_matrix = Array(transpose([transpose(vec(save_at)); solutions_as_matrix]))

        #create a dataframe converting solutions_as_matrix with column names t, s1, s2, s3, s4, s5, s6, s7
        df = DataFrame(solutions_as_matrix, column_names)
        return df
    end
end