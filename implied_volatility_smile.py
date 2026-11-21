from implied_volatility import *
import matplotlib.pyplot as plt

def smile_vol(market_prices, S, strikes, T, r, option_type="call"):
    X=[]
    Y=[]
    for price, K in zip(market_prices, strikes):
        implied_vol_for_k = implied_volatility_brent(price, S, K, T, r, option_type)
        Y.append(implied_vol_for_k)
    return np.array(Y)


def plot_volatility_smile(strikes, implied_vols):
    plt.figure(figsize=(9, 5))

    plt.plot(strikes, implied_vols * 100, marker="o")

    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.title("Implied Volatility Smile")
    plt.grid(True)

    plt.show()





# Market parameters
S = 100
T = 1
r = 0.03

# Different strikes
strikes = np.array([80, 90, 100, 105, 110, 120])

# Synthetic volatility smile
true_vols = np.array([0.28, 0.24, 0.21, 0.20, 0.19, 0.21])

# Generate synthetic market prices
market_prices = [
    black_scholes_call(S, K, T, r, sigma)
    for K, sigma in zip(strikes, true_vols)
]


# Recover implied volatilities
implied_vols = smile_vol(
    market_prices, S, strikes, T, r, "call"
)

print("Strikes:", strikes)
print("Original vols:", true_vols)
print("Recovered vols:", implied_vols)

# Plot the smile
plot_volatility_smile(strikes, implied_vols)