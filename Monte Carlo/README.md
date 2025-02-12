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
    
  
   
### 4/5. Conclusions 

- ###### 5.1 Final Observations
    - Monte Carlo Pricing as a valid tool for complex payoffs
    - Monte Carlo framework and its computational effort
    - Benefits and Drawbacks of choosing Monte Carlo compared to a Finite Difference approach
    - Existence of more Closed-Form solutions
    

To see the full code, please visit my Jupyter notebook!

### Some Visualizations:


![vol1](https://github.com/user-attachments/assets/8b74ba96-4d36-4479-8ce3-fbe7aab8d4fc)



![vol2](https://github.com/user-attachments/assets/6e3e70c9-0196-4acf-a1e3-8c8be1eb08f9)




![vol3](https://github.com/user-attachments/assets/dedbdd40-9f91-4172-a993-a59e6453d6cd)




![vol4](https://github.com/user-attachments/assets/1dcc2840-3d9f-4af1-9494-d0fe7e936fff)



![vol5](https://github.com/user-attachments/assets/5cded808-4d58-4b4d-b99d-da62385d74ce)

