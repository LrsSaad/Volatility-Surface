import numpy as np
from scipy.stats import norm

def calculate_d1_d2(S, K, T, r, sigma):

    d1 = (
        np.log(S / K) + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2


def black_scholes_greeks(S, K, T, r, sigma, option_type):

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T and sigma must be positive")

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)

    # Common quantities
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-r * T)

    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Gamma (same for Call and Put)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))

    # Vega (for a 1.00 change in volatility)
    vega = S * pdf_d1 * np.sqrt(T)

    # Theta (per year)
    common_theta = -S * pdf_d1 * sigma / (2 * np.sqrt(T))

    if option_type == "call":
        theta = common_theta - r * K * discount * norm.cdf(d2)
        rho = K * T * discount * norm.cdf(d2)

    else:
        theta = common_theta + r * K * discount * norm.cdf(-d2)
        rho = -K * T * discount * norm.cdf(-d2)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }


# Market parameters
S = 100
K = 105
T = 1
r = 0.03
sigma = 0.20

call_greeks = black_scholes_greeks(S, K, T, r, sigma, "call")
put_greeks = black_scholes_greeks(S, K, T, r, sigma, "put")

print("CALL GREEKS")
for greek, value in call_greeks.items():
    print(f"{greek}: {value:.6f}")

print("\nPUT GREEKS")
for greek, value in put_greeks.items():
    print(f"{greek}: {value:.6f}")
