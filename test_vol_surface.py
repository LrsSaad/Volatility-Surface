
import numpy as np

from call_put_pricer import black_scholes_call

from implied_volatility_surface import (
    calculate_volatility_surface,
    calculate_surface_log_moneyness,
    plot_vol_surface
)

# Market parameters
S = 100
r = 0.03

# Strikes and maturities
strikes = np.array([80, 90, 100, 105, 110, 120])
maturities = np.array([0.25, 0.5, 1.0, 2.0])

# Generate synthetic market prices
market_prices = np.zeros((len(maturities), len(strikes)))

true_vols = np.zeros_like(market_prices)

for i, T in enumerate(maturities):

    forward = S * np.exp(r * T)

    for j, K in enumerate(strikes):

        k = np.log(K / forward)

        # Artificial volatility surface
        sigma = 0.19 + 0.025 * np.exp(-T) + 0.08 * (k + 0.05)**2

        true_vols[i, j] = sigma

        market_prices[i, j] = black_scholes_call(
            S, K, T, r, sigma
        )

# Recover implied volatility surface
implied_vols = calculate_volatility_surface(
    market_prices, S, strikes, maturities, r
)

# Calculate log-moneyness
log_moneyness = calculate_surface_log_moneyness(
    strikes, maturities, S, r
)

print("Original volatility surface:")
print(true_vols)

print("\nRecovered volatility surface:")
print(implied_vols)

# Visualize
plot_vol_surface(
    log_moneyness, maturities, implied_vols
)
