using CSV, DataFrames, DifferentialEquations, Optim, Plots

function read_data(filename::String)
    lines = readlines(filename)
    df = DataFrame(A=Float64[], B=Float64[], time=Float64[], platelet=Float64[])
    i = 1
    n = length(lines)

    while i <= n
        line = strip(lines[i])
        if startswith(line, "Daily dosage:")
            i += 1
            A_val = parse(Float64, split(strip(lines[i]), ":")[end])
            i += 1
            B_val = parse(Float64, split(strip(lines[i]), ":")[end])
            i += 1

            while i <= n && !startswith(strip(lines[i]), "time,platelet")
                i += 1
            end

            i += 1
            while i <= n && occursin(",", strip(lines[i]))
                t_p = split(strip(lines[i]), ",")
                time_val = parse(Float64, t_p[1])
                platelet_val = parse(Float64, t_p[2])
                push!(df, (A_val, B_val, time_val, platelet_val))
                i += 1
            end
        else
            i += 1
        end
    end
    return df
end

# Read the data from 'data1.txt'
datafile = "data1.txt"
df = read_data(datafile)

"""
r: growth rate of platelets
K: carrying capacity of platelets
α: drug A effect coefficient
β: drug B effect coefficient
γ: platelet decay rate
λ: drug A decay rate
μ: drug B decay rate
A: dosage of drug A
B: dosage of drug B
p: platelet level
t: time
"""

# Define the time-dependent ODE model
function time_dependent_ode!(dP, P, p, t)
    r, K, α, β, γ, λ, μ, A, B = p

    # Calculate the drug effect over time
    drug_effect = α * A * exp(-λ * t) + β * B * exp(-μ * t)

    # ODE model
    dP[1] = r * P[1] * (K - P[1]) + drug_effect - γ * P[1]
end

# Define the loss function
function loss(params)
    r, K, α, β, γ, λ, μ = params
    loss_value = 0.0

    for g in groupby(df, [:A, :B]) # Group by drug A and B
        A, B = g.A[1], g.B[1]
        P0 = g.platelet[1]
        tspan = (minimum(g.time), maximum(g.time))

        # Define the ODE problem
        prob = ODEProblem(time_dependent_ode!, [P0], tspan, (r, K, α, β, γ, λ, μ, A, B))

        # Use Tsitouras 5th order Runge-Kutta method to solve the ODE
        sol = solve(prob, Tsit5(), saveat=0.5)
        predicted = sol[1, :]
        real = [g.platelet[i] for i in 1:length(predicted)]
        loss_value += sum((predicted - real) .^ 2) # MSE as loss function

    end

    return loss_value

end

# Optimize the loss function
result = optimize(loss, [0.1, 3.0, 1.0, 1.0, 0.1, 0.2, 0.3]) # Initial guess for parameters
r, K, α, β, γ, λ, μ = Optim.minimizer(result)
println("ODE parameters fitting completed: r=$r, K=$K, α=$α, β=$β, γ=$γ, λ=$λ, μ=$μ")

# Predict the platelet levels for a new patient with drug dosages A=2.1, B=2.4
A_new, B_new = 2.1, 2.4
times = 0.0:0.5:10.0
P0_new = 1.0
tspan_new = (0.0, 10.0)
prob_new = ODEProblem(time_dependent_ode!, [P0_new], tspan_new, (r, K, α, β, γ, λ, μ, A_new, B_new))
sol_new = solve(prob_new, Tsit5(), saveat=0.5)
pred_df = DataFrame(time=collect(times), platelet_ode=sol_new[1, :])

# Plot the true data and the predicted curve
plot(title="Platelet Level Over Time for Different Drug Dosages", xlabel="Time (months)", ylabel="Relative Platelet Level", legend=:topleft, legendfontsize=6)

for g in groupby(df, [:A, :B])
    plot!(g.time, g.platelet, label="A=$(g.A[1]), B=$(g.B[1])", lw=2)
end

plot!(pred_df.time, pred_df.platelet_ode, label="Predicted (A=2.1, B=2.4)", lw=3, linecolor=:blue, linestyle=:dot)

savefig("platelet_prediction_logistic.png")

# Save the prediction results to a CSV file
open("results(logistic).csv", "w") do io
    println(io, "Daily dosage:")
    println(io, "  drug A: $A_new")
    println(io, "  drug B: $B_new")
    println(io, "\ntime,platelet")

    for row in eachrow(pred_df)
        println(io, "$(row.time),$(row.platelet_ode)")
    end
end

