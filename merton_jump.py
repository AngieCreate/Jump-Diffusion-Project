import numpy as np
from math import factorial
from black_scholes import black_scholes_call

def merton_jump_call(S, K, r, sigma, T, lam, muJ, delta, N=50):
    price = 0

    # expected jump size
    k = np.exp(muJ + 0.5*delta**2) - 1

    for i in range(N):
        weight = np.exp(-lam*T)*(lam*T)**i / factorial(i)
        sigma_i = np.sqrt(sigma**2 + (i*delta**2)/T)
        r_i = r - lam*k + (i*(muJ + 0.5*delta**2))/T
        price += weight * black_scholes_call(S, K, r_i, sigma_i, T)

    return price