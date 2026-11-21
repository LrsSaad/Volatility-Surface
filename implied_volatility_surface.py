import numpy as np
import matplotlib.pyplot as plt
from implied_volatility import implied_volatility_brent

def calculate_volatility_surface(market_prices, S, strikes, maturities, r, option_type="call"):
    matrix = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes): 
            implied_vol_for_K_T = implied_volatility_brent(
                market_prices[i,j], S, K, T, r, option_type
                )
            matrix[i,j] = implied_vol_for_K_T
            
    return matrix



def calculate_surface_log_moneyness(strikes, maturities, S, r):
    forward_prices = S * np.exp(r * maturities)

    log_moneyness = np.log(
        strikes[np.newaxis, :] / forward_prices[:, np.newaxis]
    )

    return log_moneyness



def plot_vol_surface(log_moneyness, maturities, implied_vols):
    T_grid = np.repeat(
        maturities[:, np.newaxis],
        implied_vols.shape[1],
        axis=1
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        log_moneyness,
        T_grid,
        implied_vols * 100,
        cmap="viridis"
    )

    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied Volatility (%)")

    ax.set_title("Implied Volatility Surface")

    plt.show()