using CSV, DataFrames, GLM, Plots, StatsBase, StatsModels

"""
readdata(filename::String) -> DataFrame

Read 'data1.txt' and return a DataFrame with the following columns:
  - A: dosage of drug A
  - B: dose of drug B
  - time: time (in months)
  - platelet: relative platelet level
"""

function read_data(filename::String)
    lines = readlines(filename) # The total number of lines in 'data1.txt'
    df = DataFrame(A=Float64[], B=Float64[], time=Float64[], platelet=Float64[]) # Create a dataframe to store the data
    i = 1 # Initialize the line index
    n = length(lines)

    while i <= n
        line = strip(lines[i]) # Read a line and remove leading and trailing whitespaces
        if startswith(line, "Daily dosage:")
            println("Patient records found, parsing...")

            # Parse the dosage of drug A and B
            i += 1
            A_val = parse(Float64, split(strip(lines[i]), ":")[end])
            i += 1
            B_val = parse(Float64, split(strip(lines[i]), ":")[end])
            i += 1

            println("Dosage parsed: A = $A_val, B = $B_val")

            # i += 1 until the line starts with "time,platelet"
            while i <= n && !startswith(strip(lines[i]), "time,platelet")
                i += 1
            end

            # Find the header of the time-platelet table
            println("Time-Platelet table found, parsing...")
            i += 1

            # Parse the time-platelet table
            # Check if the line contains a comma, if so, parse the line
            while i <= n && occursin(",", strip(lines[i]))
                try
                    t_p = split(strip(lines[i]), ",")
                    time_val = parse(Float64, t_p[1])
                    platelet_val = parse(Float64, t_p[2])
                    push!(df, (A_val, B_val, time_val, platelet_val))
                    println("Time-Platelet parsed: time=$time_val, platelet=$platelet_val")
                catch e
                    println("Parse error: ", lines[i], " Error: ", e)
                end
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

# Print the DataFrame to check if the data is correctly parsed
println(df) 


# Train a polynomial regression model to predict platelet levels
# Choose polynomial regression including time, dose (A, B), and their interactions
formula = @formula(platelet ~ (time + time^2) + A + B + A*B + A^2 + B^2 + time*A + time*B)
model = lm(formula, df)

println("Regression model training completed. The coefficients are:")
println(coeftable(model))

# Plot the true data of different drug dosages (A, B)
plot(title="Platelet Level Over Time for Different Drug Dosages",
     xlabel="Time (months)", ylabel="Relative Platelet Level", legend=:topleft, legendfontsize=6)

for g in groupby(df, [:A, :B])
    plot!(g.time, g.platelet, label="A=$(g.A[1]), B=$(g.B[1])", lw=2)
end

# Predict the platelet levels for a new patient with drug dosages A=2.1, B=2.4
A_new = 2.1
B_new = 2.4
times = 0.0:0.5:10.0
pred_df = DataFrame(time = collect(times), A = fill(A_new, length(times)), B = fill(B_new, length(times)))
pred_df.platelet = predict(model, pred_df)

# Plot the predicted platelet levels for the new patient
plot!(pred_df.time, pred_df.platelet, label="Predicted (A=2.1, B=2.4)", lw=3, linecolor=:black, linestyle=:dash)

# Save the plot as an image
savefig("platelet_prediction_polynomial.png")

# Save the prediction results to a CSV file
open("results(polynomial).csv", "w") do io
    println(io, "Daily dosage:")
    println(io, "  drug A: $A_new")
    println(io, "  drug B: $B_new")
    println(io, "\ntime,platelet")

    for row in eachrow(pred_df)
        println(io, "$(row.time),$(row.platelet)")
    end
end
