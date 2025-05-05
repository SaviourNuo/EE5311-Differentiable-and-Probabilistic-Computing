import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import TruncatedNormal
from numpyro.infer import MCMC, NUTS
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
players = sorted(set(df_new["winner"]) | set(df_new["loser"])) # Create a new dataframe that only has two columns: the winner and loser of the same match
players_to_index = {name: i for i, name in enumerate(players)} # Map from names to indices
N = len(players) # Number of players

# Convert names to index values
winner_idx = df_new["winner"].map(players_to_index).values
loser_idx = df_new["loser"].map(players_to_index).values

# Define the Bayesian model
def model(winner_idx, loser_idx, N_players):
    mu = numpyro.sample("mu", TruncatedNormal(low=0.0, loc=20.0, scale=5.0).expand([N_players])) # 95% of players with intrinsic skill level between 10-30
    sigma = numpyro.sample("sigma", TruncatedNormal(low=0.0, loc=5.0, scale=2.0).expand([N_players])) # 95% of players with performance variability between 10-30

    mu_w = mu[winner_idx]
    mu_l = mu[loser_idx]
    sigma_w = sigma[winner_idx]
    sigma_l = sigma[loser_idx]

    denominator = jnp.sqrt(sigma_w ** 2 + sigma_l ** 2)
    numerator = mu_w - mu_l
    z = numerator / denominator
    prob_win = norm.cdf(z) 

    numpyro.sample("obs", dist.Bernoulli(probs=prob_win), obs=jnp.ones_like(winner_idx))
    '''
    Observe that in every game in history, the winner has won (win_rate = 1).
    So we use this model to explain these 1 based on the predictied win rate prob_win
    '''

# Perform Bayesian inference using Markov Chain Monte Carlo sampling
kernel = NUTS(model) # Use No-U-Turn Sampler, it can adjust the trajectory length, adaptive step size without manual adjustment
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000) # First 500 times for warm up and stablize, 1000 times to sample and use them as posterior distribution
mcmc.run(jax.random.PRNGKey(0), winner_idx=winner_idx, loser_idx=loser_idx, N_players=N) # Perform sampling
trace = mcmc.get_samples() # Get sampling results

# Every player has 1000 samples of mu and sigma
mu_samples = trace["mu"]
sigma_samples = trace["sigma"]

# Posterior simulation of wins between Gloria and Ingrid
gloria_idx = players_to_index["Gloria"]
ingrid_idx = players_to_index["Ingrid"]

mu_gloria = mu_samples[:, gloria_idx]
sigma_gloria = sigma_samples[:, gloria_idx]
mu_ingrid = mu_samples[:, ingrid_idx]
sigma_ingrid = sigma_samples[:, ingrid_idx]

sim_results_bys = []
for mg, sg, mi, si in zip(mu_gloria, sigma_gloria, mu_ingrid, sigma_ingrid):
    x_g = np.random.normal(mg, sg, size=50)
    x_i = np.random.normal(mi, si, size=50)
    wins = np.sum(x_g > x_i)
    sim_results_bys.append(wins)

plt.hist(sim_results_bys, bins=20, alpha=0.7, color='skyblue')
plt.title("Gloria's Wins in 50 Matches (Bayesian Posterior Simulation)")
plt.xlabel("Wins")
plt.ylabel("Frequency")
plt.xticks(range(min(sim_results_bys), max(sim_results_bys) + 1))  
plt.grid(True)
plt.show(block=False)

# Players are sorted by their names in alphabetical order
players_sorted = sorted(players)
sorted_indices = np.array([players_to_index[name] for name in players_sorted]) 

# Compute the mean and confidence interval of μ
mu_mean = trace["mu"].mean(axis=0)[sorted_indices]
mu_low = np.percentile(trace["mu"][:, sorted_indices], 2.5, axis=0)
mu_high = np.percentile(trace["mu"][:, sorted_indices], 97.5, axis=0)

# Plot the confidence interval of μ for different players
plt.figure(figsize=(10, 6))
plt.errorbar(mu_mean, range(len(players_sorted)), xerr=[mu_mean - mu_low, mu_high - mu_mean], fmt='o', color='royalblue', ecolor='steelblue', capsize=3)
plt.yticks(ticks=range(len(players_sorted)), labels=players_sorted)
plt.title("Posterior Mean and 95% Credible Interval for μ", fontsize=14)
plt.xlabel("μ")
plt.gca().invert_yaxis() 
plt.grid(True)
plt.show(block=False)

# Compute the mean and confidence interval of σ
sigma_mean = trace["sigma"].mean(axis=0)[sorted_indices]
sigma_low = np.percentile(trace["sigma"][:, sorted_indices], 2.5, axis=0)
sigma_high = np.percentile(trace["sigma"][:, sorted_indices], 97.5, axis=0)

# Plot the confidence interval of σ for different players
plt.figure(figsize=(10, 6))
plt.errorbar(sigma_mean, range(len(players_sorted)), xerr=[sigma_mean - sigma_low, sigma_high - sigma_mean], fmt='o', color='darkorange', ecolor='orangered', capsize=3)
plt.yticks(ticks=range(len(players_sorted)), labels=players_sorted)
plt.title("Posterior Mean and 95% Credible Interval for σ", fontsize=14)
plt.xlabel("σ")
plt.gca().invert_yaxis() 
plt.grid(True)
plt.show()