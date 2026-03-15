import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from black_scholes import black_scholes_call
from merton_jump import merton_jump_call

st.title("Merton Jump Diffusion Option Pricing Simulator")

# SIDEBAR PARAMETERS
st.sidebar.header("Model Parameters")

maturity_months = st.sidebar.selectbox(
    "Maturity (Months)",
    [0.25,0.5,1,2,3,4,6]
)

T = maturity_months / 12

sigma = st.sidebar.slider(
    "Diffusion Volatility σ",
    0.05,0.80,0.30
)

lam = st.sidebar.slider(
    "Jump Intensity λ",
    0.0,5.0,1.0
)

muJ = st.sidebar.slider(
    "Mean Jump Size μJ",
    -0.5,0.5,-0.10
)

delta = st.sidebar.slider(
    "Jump Volatility δ",
    0.01,1.0,0.25
)

r = st.sidebar.slider(
    "Risk Free Rate",
    0.0,0.10,0.03
)

# STOCK LIST
stocks = ["MSFT","AMZN","NVDA","GOOG"]

selected_stocks = st.multiselect(
    "Select stocks to display",
    stocks,
    default=["MSFT"]
)

# DOWNLOAD PRICES
prices = {}

for ticker in stocks:
    data = yf.download(ticker, period="1d")
    prices[ticker] = float(data["Close"].iloc[-1])

def get_market_options(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options

    if len(expirations) == 0:
        return None, None

    exp = expirations[0]
    chain = stock.option_chain(exp)
    calls = chain.calls

    return exp, calls


st.subheader("Strike Range (80% – 120% of S₀)")

# Dynamic stock layout
if len(selected_stocks) == 1:
    cols = [st]
else:
    cols = st.columns(2)


for i, ticker in enumerate(selected_stocks):
    col = cols[i % len(cols)]
    S0 = prices[ticker]

    strike_range = np.linspace(
        0.80*S0,
        1.20*S0,
        40
    )

    bs_prices = []
    merton_prices = []

    for K in strike_range:
        bs = black_scholes_call(S0,K,r,sigma,T)
        mj = merton_jump_call(
            S0,K,r,sigma,T,
            lam,muJ,delta
        )

        bs_prices.append(bs)
        merton_prices.append(mj)

    exp, calls = get_market_options(ticker)

    market_strikes = []
    market_prices = []

    if calls is not None:
        for _, row in calls.iterrows():
            K = row["strike"]
            price = row["lastPrice"]
            if 0.85*S0 <= K <= 1.20*S0:
                market_strikes.append(K)
                market_prices.append(price)

    # PLOT
    fig, ax = plt.subplots()

    ax.plot(
        strike_range,
        bs_prices,
        label="Black-Scholes",
        linewidth=2
    )

    ax.plot(
        strike_range,
        merton_prices,
        label="Merton Jump Diffusion",
        linewidth=2
    )

    if len(market_strikes) > 0:
        ax.scatter(
            market_strikes,
            market_prices,
            label="Market Prices",
            s=40
        )

    ax.axvline(
        x=S0,
        linestyle="--",
        linewidth=2,
        color="black",
        label=f"ATM (K = S₀ = {round(S0,2)})"
    )

    ax.axvspan(
        strike_range.min(),
        S0,
        alpha=0.08
    )

    ax.text(
        S0 * 0.88,
        max(bs_prices) * 0.85,
        "ITM (Call)",
        fontsize=11,
        weight="bold"
    )

    ax.text(
        S0 * 1.05,
        max(bs_prices) * 0.85,
        "OTM (Call)",
        fontsize=11,
        weight="bold"
    )

    ax.set_title(f"{ticker} Call Option Prices (S₀ = {round(S0,2)}, T = {maturity_months} months)")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Call Option Price ($)")

    ax.legend()

    col.pyplot(fig)

# Calculator

st.subheader("Option Calculator")

calc_stock = st.selectbox(
    "Stock",
    stocks
)

S0 = prices[calc_stock]

exp, calls = get_market_options(calc_stock)

strike_input = st.number_input(
    "Strike Price",
    value=float(round(S0))
)

model_choice = st.selectbox(
    "Pricing Model",
    ["Market", "Black-Scholes", "Merton Jump Diffusion"]
)

bs_price = black_scholes_call(S0,strike_input,r,sigma,T)

merton_price = merton_jump_call(
    S0,strike_input,r,sigma,T,
    lam,muJ,delta
)

market_price = None

if calls is not None:
    calls["diff"] = abs(calls["strike"] - strike_input)
    closest = calls.sort_values("diff").iloc[0]
    market_price = closest["lastPrice"]

# Display Results

if model_choice == "Market":
    price = market_price
elif model_choice == "Black-Scholes":
    price = bs_price
else:
    price = merton_price


if price is not None:
    break_even = strike_input + price
    st.metric("Option Price", f"${price:.2f}")
    st.metric("Break Even Stock Price", f"${break_even:.2f}")

else:
    st.write("No market option available for this strike.")