from call_put_pricer import *

from scipy.optimize import brentq


def implied_volatility_brent(
    market_price, S, K, T, r, option_type="call"
):

    if S <= 0 or K <= 0 or T <= 0:
        raise ValueError("S, K and T must be positive")

    option_type = option_type.lower()

    if option_type == "call":
        price_function = black_scholes_call
        lower_bound = max(0, S - K * np.exp(-r * T))
        upper_bound = S

    elif option_type == "put":
        price_function = black_scholes_put
        lower_bound = max(0, K * np.exp(-r * T) - S)
        upper_bound = K * np.exp(-r * T)

    else:
        raise ValueError("option_type must be call or put")

    if not lower_bound <= market_price <= upper_bound:
        raise ValueError("Market price violates Black-Scholes bounds")

    if np.isclose(market_price, lower_bound):
        return 0.0

    if np.isclose(market_price, upper_bound):
        raise ValueError("No finite implied volatility exists at upper bound")

    def objective(sigma):
        return price_function(S, K, T, r, sigma) - market_price

    return brentq(objective, 1e-8, 5.0)




# Market parameters

S = 100
K = 105
T = 1
r = 0.03
market_price = 10

iv = implied_volatility_brent(
    market_price, S, K, T, r, "call"
)

print("Implied Volatility:", iv)
print("Implied Volatility (%):", iv * 100)
