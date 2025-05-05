# Delta Hedging App

Based on an essential excerpt from my final CQF (Certificate of Quantitative Finance) project which focused on Optimal Delta Hedging in a Dynamic context, I would like to build a simple and easy to deploy Dynamic Delta Hedging App via [Streamlit](https://streamlit.io/).

> [!NOTE]
For full definitions, please have a look into my Dynamic Delta Hedging Jupiter notebook at the [link](https://github.com/dom00mi/Option-Pricing/blob/main/Delta%20Hedging/Dynamic%20Delta%20Hedging.ipynb). 

> ## App Code
```
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from delta_hedger import BlackScholes 
import plotly.graph_objects as go
import plotly.express as px
# App title
st.title('Dynamic Delta Hedging App')

# Sidebar Inputs
st.sidebar.header('Input Parameters')

spot = st.sidebar.number_input('Initial Stock Price (Spot)', value=100.0)
K = st.sidebar.number_input('Strike Price (K)', value=100.0)
r = st.sidebar.number_input('Risk-Free Rate (r)', value=0.05, step=0.01)
T = st.sidebar.number_input('Time to Expiry (Years)', value=1.0)
timesteps = st.sidebar.number_input('Time Steps', value=252, step=10)
nsim = st.sidebar.number_input('Number of Simulations', value=100, step=100)
number_of_shares=st.sidebar.number_input('Number of Shares (to delta-hedge)', value=1, step=1)
option_type= st.sidebar.selectbox(
    'Select Option Type',
    ['Call', 'Put']
)
pricing_method= st.sidebar.selectbox(
    'Select Option Pricing Method',
    ['Black Scholes', 'Monte Carlo']
)
sde_simulation_type = st.sidebar.selectbox(
    'Select SDE Simulation Type',
    ['Exact GBM','Euler-Maruyama', 'Milstein']
)

hedging_type=st.sidebar.selectbox(
    'Select Hedging Manner',
    ['Actual','Implied', 'Delta-Hedging']
)

implied_vol=st.sidebar.number_input('Implied Volatility', value=0.2, step=0.01)
actual_vol=st.sidebar.number_input('Actual Volatility', value=0.2, step=0.01)

delta_hedger = BlackScholes(
    spot=spot,
    strike=K,
    r=r,
    T=T,
    option_type=option_type,
    timesteps=timesteps,
    nsim=nsim,
    number_shares=number_of_shares,
    SDE_sim_type=sde_simulation_type,
    hedging_type=hedging_type,
    pricing_method=pricing_method,
    implied_vol=implied_vol,
    actual_vol=actual_vol
)


# Button to Show Distribution
if st.button('Show Option Price Distribution'):
    #option_price = delta_hedger.option_pricer()
    #st.success(f'Option Price: $ {option_price:.4f}')
    # Create a Matplotlib figure
    fig = delta_hedger.plot_option_price_distribution_plotly()
    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
    
if st.button('Show Asset Price Paths'):
    fig = delta_hedger.asset_path_plot_plotly()
    st.plotly_chart(fig)
    
if st.button('Show P&L Distribution'):
    fig = delta_hedger.pandl_distribution_plotly()
    st.plotly_chart(fig)
if st.button('Analyse Greeks and P&L over time'):
    delta_pandl = delta_hedger.delta_pandl_plotly()
    gamma_pandl = delta_hedger.gamma_pandl_plotly()
    st.plotly_chart(delta_pandl)
    st.plotly_chart(gamma_pandl)
if st.button('Analyse P&L and compare with Underlying and Option Price over time'):
    underlying_option, underlying_pandl = delta_hedger.delta_hedging_plots_plotly()
    st.plotly_chart(underlying_option)
    st.plotly_chart(underlying_pandl)
```

> [!IMPORTANT]
> ### How to Use the App?



> 1. Save the files locally in your folder, as an example `C:\Users\Your_Username`.
>
> 2. Assuming that you have installed  [Anaconda](https://www.anaconda.com/), please launch ![Screenshot 2025-05-05 151452](https://github.com/user-attachments/assets/15d4185f-b428-4f0d-ad3b-2d140e5196d6) in your computer

> 
> 3. Once launched Anaconda, run the following commands: `cd C:\Users\Your_Username` and `streamlit run delta_hedging_app.py`
>
> 4. This will launch a Local URL: http://localhost:8501 where you can run the app.

You can click on:

- Show Option Price Distribution
- Show Asset Price Paths
- Show P&L Distribution
- Analyse Greeks and P&L over time
- Analyse P&L and compare with Underlying and Option Price over time


### Show Option Price Distribution
![Screenshot 2025-05-05 145742](https://github.com/user-attachments/assets/4559edd0-a8fb-4c6e-9f66-5bb8bcd734be)



### Show Asset Price Paths
![Screenshot 2025-05-05 145853](https://github.com/user-attachments/assets/17b8f8cf-f6ad-4a3c-9987-03bbccc1ff11)

### Show P&L Distribution
![Screenshot 2025-05-05 150516](https://github.com/user-attachments/assets/ad8c8f9a-b3e8-49d1-a712-d53992dd79c8)

### Analyse Greeks and P&L over time
![Screenshot 2025-05-05 150959](https://github.com/user-attachments/assets/4cb5a7fa-9d74-40f5-b8a6-22b4590e6996)

### Analyse P&L and compare with Underlying and Option Price over time
![Screenshot 2025-05-05 151059](https://github.com/user-attachments/assets/d1e84b0a-f657-4ba3-8569-05a94c37f496)

