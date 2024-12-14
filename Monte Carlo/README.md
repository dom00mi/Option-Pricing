# Monte-Carlo-Asian-and-Lookback-Option-Pricer
In this mini-research, I will be attempting to use the Monte Carlo scheme to price exotic options (Asian and lookback options) the risk-neutral density  𝑄  . and simulating the asset price process through the Geometric Bronwian Motion SDE in its Euler-Maruyama form.

## Sections:



### 1. The Underlying Asset Price Process:


- ###### 1.1 Introduction on GBM:
    - Numerical Approaches for SDEs
    - The Euler-Maruyama Numerical Scheme    
 
- ###### 1.2 Monte Carlo Simulated Asset Paths
    - Monte Carlo Simulations with Different Volatility Levels
    - Simulation Datasets for the Euler-Maruyama scheme at the boundaries of our volatility scenarios   
    - Simulation Datasets for the exact GBM at the boundaries of our volatility scenarios
    - Monte Carlo Simulations with Different Risk-free Levels
    - Geometric Brownian Motion (GBM) vs the Euler Maruyama scheme
    
    
### 2. Asian and Lookback Option Pricing:

- ###### 2.1 Asian Options 
    - Key Definitions and Types
    - Asians Payoffs vs Plain Vanilla
- ###### 2.2 Lookback Options
    - Key Definitions and Types
    - Lookback Payoffs 
    
- ###### 2.3 The Asian and Lookback Option Pricer
    - Asian and Lookback Options Prices at Initial Conditions

- ###### 2.4 The Relationship between Fixed Strike Asian Option and its parameters:
    - Fixed Strike Asian Options vs Volatility
    - Fixed Strike Asian Options vs Underlying
    - Fixed Strike Asian Option vs Strikes
    - Fixed Strike Asian Option vs Risk-free Rate
    
- ###### 2.5 The Relationship between Fixed Strike Lookback Options and its parameters:

    - Fixed Strike Lookback Options vs Volatility
    - Fixed Strike Lookback Options vs Underlying
    - Fixed Strike Lookback Option vs Strikes
    - Fixed Strike Lookback Option vs Risk-free Rate
    
- ###### 2.6 The Relationship between Floating Strike Asian Options and its parameters:

    - Floating Strike Asian Options vs Volatility
    - Floating Strike Asian Options vs Underlying
    - Floating Strike Asian Option vs Strikes
    - Floating Strike Asian Option vs Risk-free Rate 
    
- ###### 2.7 The Relationship between Floating Lookback Options and its parameters:

    - Floating Strike Lookback Options vs Volatility
    - Floating Strike Lookback Options vs Underlying
    - Floating Strike Lookback Option vs Strikes
    - Floating Strike Lookback Option vs Risk-free Rate
    
 
### 3. A Further Analysis - Option Pricing Surfaces:
    
- ###### 3.1 Fixed Strike Asian Options Surfaces

    - Fixed Strike Asian Options vs Strike and Time to Expiry
    - Fixed Strike Asian Options vs Volatility and Risk-free rates  
    
- ###### 3.2 Fixed Strike Lookback Options Surfaces 

    - Fixed Strike Lookback Options vs Strike and Time to Expiry
    - Fixed Strike Lookback Options vs Volatility and Risk-free rates

- ###### 3.3 Floating Strike Asian Options Surfaces 

    - Floating Strike Asian Options vs Strike and Time to Expiry
    - Floating Strike Asian Options vs Volatility and Risk-free rates
    
- ###### 3.4 Floating Strike Lookback Options Surfaces 

    - Floating Strike Lookback Options vs Strike and Time to Expiry
    - Floating Strike Lookback Options vs Volatility and Risk-free rates
    
  
   
### 4. Conclusions 

- ###### 5.1 Final Observations
    - Monte Carlo Pricing as a valid tool for complex payoffs
    - Monte Carlo framework and its computational effort
    - Benefits and Drawbacks of choosing Monte Carlo compared to a Finite Difference approach
    - Existence of more Closed-Form solutions
    


### 5. References



# Importing data analysis libraries
import pandas as pd
import numpy as np

# Importing Charting libraries
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# import math library
import math

# statistical library
import scipy.stats as stats
# option strategies
import opstrat as op
# To create tables
from tabulate import tabulate
