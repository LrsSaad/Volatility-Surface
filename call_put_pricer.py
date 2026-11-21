
import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T and sigma must be positive")
    
    d1 = (
        np.log(S / K) + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    call_price = (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    return call_price


def black_scholes_put(S, K, T, r, sigma):

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T and sigma must be positive")
    
    d1 = (
        np.log(S / K) + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    put_price = (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )

    return put_price


# Market parameters

S = 100
K = 105
T = 1
r = 0.03
sigma = 0.20


call = black_scholes_call(S, K, T, r, sigma)
put = black_scholes_put(S, K, T, r, sigma)

print("Call price:", call)
print("Put price:", put)

# Call Put Parity
print(S - K*np.exp(-r*T))
print(call-put)





