import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, r, sigma, T):
    
    if T <= 0:
        return max(S-K,0)

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    return call_price