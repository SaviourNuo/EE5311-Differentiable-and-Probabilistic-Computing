import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

# Define a function to get the winner and loser of the same match 
def get_winner_loser(row):
    if row["winner"] == row["player1"]:
        loser = row["player2"]
    else:
        loser = row["player1"]
    return pd.Series([row["winner"], loser], index=["winner", "loser"])

df = pd.read_csv("data2.csv") # Read the provided data into a dataframe
df_new = df.apply(get_winner_loser, axis=1) # Create a new dataframe that only has two columns: the winner and loser of the same match
players = sorted(set(df_new["winner"]) | set(df_new["loser"])) # Get the names of all players
players_to_index = {name: i for i, name in enumerate(players)} # Map from names to indices
N = len(players) # Number of players

mu_init = np.full(len(players), 20.0) # Initialize mu for all players
log_sigma_init = np.full(len(players), np.log(5.0)) # Initialize log(sigma) for all players
x_init = np.concatenate([mu_init, log_sigma_init]) # Construct a variable vector for optimization

def log_likelihood(x, df, player_to_index):
    total = 0.0

    for row in df.itertuples(): # For every match 
        w = row.winner # Get the winner's name
        l = row.loser # Get the loser's name 
        idx_w = player_to_index[w] # Locate the index of the winner
        idx_l = player_to_index[l] # Locate the index of the loser

        mu_w = x[idx_w] # Get the mu of the winner (intrinsic skill level) 
        mu_l = x[idx_l] # mu for loser
        sigma_w = np.exp(x[idx_w + N]) # Get the sigma of the winner (performance variability)
        sigma_l = np.exp(x[idx_l + N]) # sigma for loser

        demoinator = np.sqrt(sigma_w ** 2 + sigma_l ** 2)
        numerator = mu_w - mu_l
        z = numerator / demoinator
        prob = norm.cdf(z) 

        total += np.log(prob + 1e-10) # Sum of the log likelihood for all matches, plus 1e-10 to prevent possible overflow
    return total

def neg_log_likelihood(x):
    return -log_likelihood(x, df_new, players_to_index)

# minimize negative log likelihood, start from x_init, use L-BFGS-B method, max iterations set to 1000 and display the process
results = minimize(fun=neg_log_likelihood, x0=x_init, method="L-BFGS-B", options={"maxiter":1000, "disp": True})

x_opt = results.x # Return the optimal solution for mu and sigma
mu_values = x_opt[:N]   
log_sigma_values = x_opt[N:]    
sigma_values = np.exp(log_sigma_values)  

# Match each player's name with their mu and sigma
for i, name in enumerate(players):
    mu = mu_values[i]
    sigma = sigma_values[i]
    print(f"{name:10s} | μ = {mu:.2f} | σ = {sigma:.2f}")

# Find mu and sigma for player Gloria and Ingrid separately
idx_G = players_to_index["Gloria"]
idx_I = players_to_index["Ingrid"]
mu_G = x_opt[idx_G]
sigma_G = np.exp(x_opt[idx_G + len(players)])
mu_I = x_opt[idx_I]
sigma_I = np.exp(x_opt[idx_I + len(players)])

# Run 1000 times of simulation of 50 games, count the number of games Gloria wins each time and draw a histogram
sim_results_mle = []
for _ in range(1000):
    x_g = np.random.normal(mu_G, sigma_G, size=50)
    x_i = np.random.normal(mu_I, sigma_I, size=50)
    wins = np.sum(x_g > x_i)
    sim_results_mle.append(wins)

plt.figure(figsize=(10, 6))
plt.hist(sim_results_mle, bins=20, color="lightcoral", alpha=0.8)
plt.title("Gloria's Wins in 50 Matches (MLE Simulation)", fontsize=14)
plt.xlabel("Wins")
plt.ylabel("Frequency")
plt.xticks(range(min(sim_results_mle), max(sim_results_mle) + 1))
plt.grid(True)
plt.show(block=False)

plt.figure(figsize=(12, 6))
x_range = np.linspace(0, 40, 1000)

for i, player in enumerate(players):
    mu = mu_values[i]
    sigma = sigma_values[i]
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x_range - mu)**2 / (2 * sigma**2))
    plt.plot(x_range, y, label=player)

plt.title("Distribution of Each Player's Performance", fontsize=14)
plt.xlabel("Performance", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

