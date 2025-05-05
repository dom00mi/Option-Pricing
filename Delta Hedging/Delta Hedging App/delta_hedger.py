import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.precision", 5)
import seaborn as sns
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings('ignore')  
import yfinance as yf
import math
from math import log, e
from scipy.stats import norm

#Statistical Models
import statsmodels.api as sm 

from tqdm import tqdm
import pandas as pd
import os

class BlackScholes:
    """
    This class is based on the Black-Scholes Option Pricing model and serves as:
    - A Black-Scholes like pricing tool to obtain option prices
    - An Advanced Greeks Computational Tool
    - An Integration from the Black-Scholes framework to the Geometric Brownian Motion Simulated Asset Paths
    """
    def __init__(self, spot, strike, r, T, option_type, timesteps, nsim, number_shares, SDE_sim_type, hedging_type, pricing_method, implied_vol, actual_vol):
        # Spot Price
        self.spot = spot
        # Option Strike
        self.strike = strike
        # Interest Rate
        self.r = r
        # Days To Expiration
        self.T = T
        # Time-steps
        self.ts = timesteps
        # Number of Simulations
        self.N = nsim
        # Initial Number of Shares 
        self.number_shares = number_shares
        # Option type
        self.option_type = option_type
        # Hedging Type 
        self.hedging_type = hedging_type
        # Pricing Methodology
        self.pricing_method = pricing_method
        # Actual Volatility
        self.actual_vol = actual_vol
        # Implied Volatility
        self.implied_vol = implied_vol

        if self.strike == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        # Monte Carlo Simulation Type for the SDE for the Underlying Asset
        self.SDE_sim_type = SDE_sim_type
        
        
        assert option_type in ['Call', 'Put'], "option_type must be 'Call' or 'Put'"
        assert pricing_method in ['Black Scholes', 'Monte Carlo'], "Invalid pricing method"
        
    def single_MonteCarlo_simulator(self):
        np.random.seed(2024)
        dt = self.T / self.ts
        if self.SDE_sim_type == 'Exact GBM':
            S = np.zeros(self.ts+1)  # Initialize S array with zeros
            t = np.linspace(0, self.T, self.ts+1)
            W = np.random.standard_normal(size=self.ts+1)
            W = np.cumsum(W) * np.sqrt(dt)
            S = self.spot * np.exp((self.r - 0.5 * self.actual_vol**2) * t + self.actual_vol * W)
        elif self.SDE_sim_type == 'Euler-Maruyama':
            t = np.linspace(0, self.T, self.ts+1)
            S = np.zeros(self.ts+1)  # Initialize S array with zeros
            S[0] = self.spot
            for i in range(self.ts):
                # change in Brownian Motion
                # w is the Brownian Motion (a Standard Normal Variable scaled by square root of time)
                w = np.random.standard_normal() * np.sqrt(dt)
                # Euler-Maruyama method
                S[i+1] = S[i] * (1 + self.r * dt + self.actual_vol * w)

        # elif self.SDE_sim_type == 'Milstein':
        #     # Milstein Approximation
        #     X_mil, X = [], self.spot
        #     # Brownian Motion terms
        #     dB = np.sqrt(dt) * np.random.randn(self.N)
        #     B  = np.cumsum(dB)
        #     for j in range(self.N):
        #         X += self.r*X*dt + self.actual_vol*X*dB[j] + 0.5*self.actual_vol**2 * X * (dB[j] ** 2 - dt)
        #         X_mil.append(X)
        #     #S = np.array(X_mil)
        else:
            raise ValueError("Invalid SDE Simulation Type, please either choose Exact GBM, Euler-Maruyama or Milstein")
                
        return t, S
    
    def multiple_MonteCarlo_simulator(self):
        '''This function is our Asset Path Monte Carlo Simulator in the Risk-Neutral Setting, based on the following chosen logics:
        - Exact Geometric Brownian Motion
        - Euler-Maruyama Approximation
        - Milstein Scheme (not implemented)
        '''
        
        np.random.seed(2024)
        dt = self.T / self.ts
        
        if self.SDE_sim_type == 'Exact GBM':
            t = np.linspace(0, self.T, self.ts+1)
            S = np.zeros((self.ts+1, self.N))  # Initialize S array with zeros
            # defining the Brownian Motion
            W = np.random.standard_normal(size=(self.ts+1, self.N))
            W = np.cumsum(W, axis=0) * np.sqrt(dt)
            S = self.spot * np.exp((self.r - 0.5 * self.actual_vol**2) * t[:, np.newaxis] + self.actual_vol * W)
        elif self.SDE_sim_type == 'Euler-Maruyama':
            S = np.zeros((self.ts, self.N))  # Initialize S array with zeros
            S[0] = self.spot
            for i in range(0, self.ts-1):
                w = np.random.standard_normal(self.N) * np.sqrt(dt)
                S[i+1] = S[i] * (1 + self.r * dt + self.actual_vol * w)
                    
        elif self.SDE_sim_type == 'Milstein':
            S = np.zeros((self.ts+1, self.N))  # Initialize S array with zeros
            S[0] = self.spot
            for i in range(self.ts):
                w = np.random.standard_normal(self.N) * np.sqrt(dt)
                S[i+1] = S[i] * (1 + self.r * dt + self.actual_vol * w + 0.5 * self.actual_vol**2 * (w**2 - dt))
                
        else:
            raise ValueError("Invalid SDE Simulation Type, please either choose Exact GBM, Euler-Maruyama or Milstein")


        return S  
            

    def option_pricer(self):
        '''This function returns an option price depending on the following pricing techiques:
        - Black-Scholes 
        - Monte Carlo
        
        The functions computes standard European Call and Put options, based on inputted implied volatility
        '''
        
        # Initialize the price, so price is defined
        price = None
        
        if self.pricing_method == 'Black Scholes':
            if self.implied_vol == 0 or self.T == 0:
                call = max(0.0, self.spot - self.strike)
                put = max(0.0, self.strike - self.spot)
            else:
                self.d1 = (math.log(self.spot / self.strike) + 
                      (self.r + (self.implied_vol**2) / 2) * self.T) / (self.implied_vol * math.sqrt(self.T))
                self.d2 = self.d1 - self.implied_vol * math.sqrt(self.T)
                
                if self.option_type == 'Call':
                    price = self.spot * norm.cdf(self.d1) - self.strike * math.exp(-self.r * self.T) * norm.cdf(self.d2)
                elif self.option_type == 'Put':
                    price = self.strike * math.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.spot * norm.cdf(-self.d1)
                else:
                    raise ValueError("Invalid option type. Choose 'Call' or 'Put'.")
                
        elif self.pricing_method == 'Monte Carlo':
            S = self.multiple_MonteCarlo_simulator()
            if self.option_type == 'Call':
                price = np.exp(-self.r*self.T)*np.mean(np.maximum(0, S[-1]-self.strike))
            elif self.option_type == 'Put':
                price = np.exp(-self.r*self.T) * np.mean(np.maximum(0, self.strike-S[-1]))
            else:
                raise ValueError("Invalid option type. Choose 'Call' or 'Put'.")    
        
        else:
            raise ValueError("Invalid Pricing Method")
        
        if price is None:
            raise ValueError("Price was not computed. Please check the inputs and methods.")
            
        return price

    def delta_greeks(self):
        '''This function includes all the Delta Greeks (including mixed partial derivatives)
        Delta
        Elasticity
        Charm - DdeltaDtime
        Vanna - DDeltaDvol
        '''
        
        # Initialize the Greeks
        delta = None
        elasticity = None
        charm = None
        vanna = None
        
        
        if self.hedging_type == 'Actual':
            vol = self.actual_vol
        elif self.hedging_type == 'Implied':
            vol = self.implied_vol
        elif self.hedging_type == 'Delta-Hedging':
            vol = self.actual_vol = self.implied_vol
        else:
            raise ValueError("Invalid Hedging Type. Choose 'Actual', 'Implied' or 'Delta-Hedging'.")

        if vol == 0 or self.T == 0:
            if self.option_type == 'Call' and self.spot > self.strike:
                delta = 1.0
            else:
                delta = 0.0
            if self.option_type == 'Put' and self.spot < self.strike:
                delta = -1.0
            else:
                delta = 0.0
        else:
            self.d1 = (math.log(self.spot / self.strike) + 
                      (self.r + (vol**2) / 2) * self.T) / (vol * math.sqrt(self.T))
            self.d2 = self.d1 - vol * math.sqrt(self.T)

            if self.option_type == 'Call':
                delta = norm.cdf(self.d1)
                charm = -math.exp(-self.r * self.T) * (norm.cdf(self.d1) * (-self.d2 / (2 * self.T)) - self.r * norm.cdf(self.d1))
            elif self.option_type == 'Put':
                delta = -norm.cdf(-self.d1)
                charm = -math.exp(-self.r * self.T) * (norm.cdf(self.d1) * (-self.d2 / (2 * self.T)) + self.r * norm.cdf(-self.d1))

        price = self.option_pricer()
        elasticity = delta * self.spot / price
        vanna = -math.exp(-self.r * self.T) * self.d2 * norm.cdf(self.d1) / vol
        
        if delta is None and elasticity is None and charm is None and vanna is None:
            raise ValueError("Greeks were not computed. Please check the inputs and methods.")


        return [delta, elasticity, charm, vanna]

    def gamma_greeks(self):
        '''This function calculates all the Gamma Greeks, namely:
        - Gamma
        - Zomma
        - Speed
        - Color
        '''
        
        if self.hedging_type == 'Actual':
            vol = self.actual_vol
        elif self.hedging_type == 'Implied':
            vol = self.implied_vol
        elif self.hedging_type == 'Delta-Hedging':
            vol = self.actual_vol = self.implied_vol
        else:
            raise ValueError("Invalid Hedging Type. Choose 'Actual', 'Implied' or 'Delta-Hedging'.")
            
        self.d1 = (math.log(self.spot / self.strike) + 
                      (self.r + (vol**2) / 2) * self.T) / (vol * math.sqrt(self.T))
        self.d2 = self.d1 - vol * math.sqrt(self.T)

        gamma = norm.pdf(self.d1)*1/(self.spot*vol*math.sqrt(self.T))
        # Zomma or DGammaDVol
        zomma = gamma*(self.d1*self.d2-1)/vol
        #Speed or DGammaDSpot
        speed = -gamma*(1+self.d1/vol*math.sqrt(self.T))/self.spot
        #Color or DgammaDtime
        color = gamma*(self.r+(1-self.d1*self.d2)/self.T)
        
        
        return [gamma,zomma,speed,color]
    
    
    def vega_greeks(self):
        '''This function calculates the Vega Greeks, namely
        
        
        It includes also Theta and Rho, not previously included in the aforementioned functions
        '''
        
        # Initialize the Greeks
        vega = None
        vomma = None
        theta = None
        rho = None
        
        if self.hedging_type == 'Actual':
            vol = self.actual_vol
        elif self.hedging_type == 'Implied':
            vol = self.implied_vol
        elif self.hedging_type == 'Delta-Hedging':
            vol = self.actual_vol = self.implied_vol
        else:
            raise ValueError("Invalid Hedging Type. Choose 'Actual', 'Implied' or 'Delta-Hedging'.")
            
        self.d1 = (math.log(self.spot / self.strike) + 
                      (self.r + (vol**2) / 2) * self.T) / (vol * math.sqrt(self.T))
        self.d2 = self.d1 - vol * math.sqrt(self.T)
        
        # Vega
        vega = self.strike*math.exp(-self.r*self.T)*norm.pdf(self.d2)*math.sqrt(self.T)
        #Vomma or DVegaDVol
        vomma = vega*self.d1*self.d2/vol
        
        
        #Theta and Rho
        if self.option_type == 'Call':
            theta = -self.spot*norm.pdf(self.d1)*vol/(2*math.sqrt(self.T))-self.r*self.strike*math.exp(-self.r*self.T)*norm.cdf(self.d2)
            rho = self.strike*self.T*math.exp(-self.r*self.T)*norm.cdf(self.d2)
        elif self.option_type == 'Put':
            theta = -self.spot*norm.pdf(self.d1)*vol/(2*math.sqrt(self.T))+self.r*self.strike*math.exp(-self.r*self.T)*norm.cdf(-self.d2)
            rho = -self.strike*self.T*math.exp(-self.r*self.T)*norm.cdf(-self.d2)
        
        
        if vega is None and vomma is None and theta is None and rho is None:
            raise ValueError("Greeks were not computed. Please check the inputs and methods.")

        return [vega,vomma,theta,rho]
        
        
    def asset_path_plot_plotly(self):
        """This method plots multiple simulated asset paths using Plotly."""
        asset = self.multiple_MonteCarlo_simulator()
        fig = go.Figure()
        
        # Plot only up to 10 paths for clarity and max 1000
        num_paths_to_plot = max(10, min(asset.shape[1],1000))
        for i in range(num_paths_to_plot):
            fig.add_trace(go.Scatter(
            x=np.arange(asset.shape[0]),
            y=asset[:, i],
            mode='lines',
            name=f'Path {i+1}'
        ))
            
        # Layout customization
        fig.update_layout(
        title=f'Asset Path Simulated via {self.SDE_sim_type} Scheme',
        xaxis_title='Time step',
        yaxis_title='Stock Price S',
        template='plotly_white',
        width=900,
        height=500
    )
        return fig


    def option_sim_data_generator(self):
        '''This function will create a dataframe populated with option prices, Greeks, portfolio metrics (e.g. P&L) at each
        time step, based on a single Monte Carlo simulation'''
        t, S = self.single_MonteCarlo_simulator()
        dt = self.T / self.ts

        # Initializing all the various lists
        
        option_prices = []
        # Delta Greeks
        deltas = []
        elasticities = []
        charms = []
        vannas = []
        # Gamma Greeks 
        gammas = []
        zommas = []
        speeds = []
        colors = []
        # P&L related lists
        cashflows = []
        cash_account = []

        for spot in S:
            bs = BlackScholes(spot, self.strike, self.r, self.T, self.option_type, self.ts, self.N, self.number_shares, self.SDE_sim_type, self.hedging_type, 
                                   self.pricing_method, self.implied_vol, self.actual_vol)
            option_prices.append(bs.option_pricer())
            delta_greeks = bs.delta_greeks()
            gamma_greeks = bs.gamma_greeks()
            deltas.append(delta_greeks[0])
            elasticities.append(delta_greeks[1])
            charms.append(delta_greeks[2])
            vannas.append(delta_greeks[3])
            gammas.append(gamma_greeks[0])
            zommas.append(gamma_greeks[1])
            speeds.append(gamma_greeks[2])
            colors.append(gamma_greeks[3])

        # Calculate initial portfolio value
        initial_portfolio_value = option_prices[0] - deltas[0] * S[0]
        cash_account.append(-initial_portfolio_value)
        cashflows.append(0)  # Initial cashflow is zero
        

        for i in range(1, len(t)):
            cashflow = (deltas[i] - deltas[i - 1]) * S[i]
            cash_account_value = cash_account[-1] * math.exp(self.r * dt) + cashflow
            cashflows.append(cashflow)
            cash_account.append(cash_account_value)

        option_df = pd.DataFrame({
            'Time Step': t,
            'Underlying Price': S,
            'Option Price': option_prices,
            'Delta': deltas,
            'Elasticity': elasticities,
            'Charm': charms,
            'Vanna': vannas,
            'Gamma': gammas,
            'Zomma': zommas,
            'Speed': speeds,
            'Color': colors,
            'Replicating Cashflow': cashflows,
            'Cash Account': cash_account
        })

        option_df['Portfolio'] = option_df['Option Price'] - option_df['Delta'] * option_df['Underlying Price']
        option_df['Cumulative P&L'] = option_df['Portfolio'] + option_df['Cash Account']
        option_df['Interest'] = option_df['Cumulative P&L']*(math.exp(self.r*dt)-1)
        
        #Option Delta Hedging
        option_df['Delta %'] = option_df['Delta']*100
        
        #Total Delta Position
        option_df['Total Delta Position'] = (option_df['Delta %'].shift(1) - option_df['Delta %'])*self.number_shares
        option_df['Total Delta Position'] = option_df['Total Delta Position'].astype(float).round(1)
        abs_total_delta_position = option_df['Total Delta Position'].abs().round(1).astype(str)
        
        # Strategy Adjustments using temporary absolute values
        option_df['Contract Adjustments'] = np.where(option_df['Total Delta Position'] > 0,
                                             'Short ' + abs_total_delta_position + ' Stock',
                                             'Long ' + abs_total_delta_position + ' Stock')

        

        return option_df
    
    
    
    def multiple_sim_generator(self):
        """This method generates a dataframe of option prices based on the given simulations of underlying prices."""
        mult_underlying_sim = pd.DataFrame(self.multiple_MonteCarlo_simulator())
        mult_option_px_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        
        for i in range(mult_underlying_sim.shape[0]):
            for j in range(mult_underlying_sim.shape[1]):
                spot_price = mult_underlying_sim.iloc[i, j]
                bs_temp = BlackScholes(spot=spot_price, strike=self.strike, r=self.r, T=self.T, 
                                   timesteps=self.ts, nsim=self.N, number_shares=self.number_shares, 
                                   option_type=self.option_type, SDE_sim_type=self.SDE_sim_type, 
                                   hedging_type=self.hedging_type, implied_vol=self.implied_vol, 
                                   actual_vol=self.actual_vol, pricing_method = self.pricing_method)
                mult_option_px_sim.iloc[i, j] = bs_temp.option_pricer()
                
        return mult_option_px_sim
    
    
    def multiple_sim_delta(self):
        """This method generates a dataframe of deltas, gammas based on the given simulations of underlying prices."""
        mult_underlying_sim = pd.DataFrame(self.multiple_MonteCarlo_simulator())
        mult_delta_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        
        for i in range(mult_underlying_sim.shape[0]):
            for j in range(mult_underlying_sim.shape[1]):
                spot_price = mult_underlying_sim.iloc[i, j]
                bs_temp = BlackScholes(spot=spot_price, strike=self.strike, r=self.r, T=self.T, 
                                   timesteps=self.ts, nsim=self.N, number_shares=self.number_shares, 
                                   option_type=self.option_type, pricing_method = self.pricing_method, SDE_sim_type=self.SDE_sim_type, 
                                   hedging_type=self.hedging_type, implied_vol=self.implied_vol, 
                                   actual_vol=self.actual_vol)
                mult_delta_sim.iloc[i, j] = bs_temp.delta_greeks()[0]
                
        print(f' Deltas DataFrame across {self.N} simulations, when hedging with {self.hedging_type}')
                
        return mult_delta_sim
    
    def multiple_sim_gamma(self):
        # Our starting dataframe (i.e. the simulated asset prices)
        mult_underlying_sim = pd.DataFrame(self.multiple_MonteCarlo_simulator())
        dt = self.T / self.ts
        # The dataframe for option gammas
        mult_gamma_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        for i in range(mult_underlying_sim.shape[0]):
            for j in range(mult_underlying_sim.shape[1]):
                spot_price = mult_underlying_sim.iloc[i, j]
                # Calling an instance of the class
                bs_temp = BlackScholes(spot=spot_price, strike=self.strike, r=self.r, T=self.T, 
                                       timesteps=self.ts, nsim=self.N, number_shares=self.number_shares, 
                                       option_type=self.option_type, SDE_sim_type=self.SDE_sim_type, 
                                       hedging_type=self.hedging_type, implied_vol=self.implied_vol, pricing_method = self.pricing_method, 
                                       actual_vol=self.actual_vol)
                # gamma dataframe creation
                mult_gamma_sim.iloc[i, j] = bs_temp.gamma_greeks()[0]
                
        return mult_gamma_sim
    
    
    def multiple_pandlsim(self):
        '''This function builds a set of dataframes of dimension: Time step x Number of Simulations for each of the following:
        - Option Price
        - Delta
        - Gamma
        - Portfolio Stats'''
        
        dt = self.T / self.ts
        # Our starting dataframe (i.e. the simulated asset prices)
        mult_underlying_sim = pd.DataFrame(self.multiple_MonteCarlo_simulator())
        # All the dataframes
        mult_option_px_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        mult_delta_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        mult_gamma_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        mult_port_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        mult_cash_acc_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        mult_cash_flow_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        mult_pandl_sim = pd.DataFrame(index=mult_underlying_sim.index, columns=mult_underlying_sim.columns)
        
        for i in range(mult_underlying_sim.shape[0]):
            for j in range(mult_underlying_sim.shape[1]):
                spot_price = mult_underlying_sim.iloc[i, j]
                bs_temp = BlackScholes(spot=spot_price, strike=self.strike, r=self.r, T=self.T, 
                                       timesteps=self.ts, nsim=self.N, number_shares=self.number_shares, 
                                       option_type=self.option_type, SDE_sim_type=self.SDE_sim_type, 
                                       hedging_type=self.hedging_type, implied_vol=self.implied_vol, pricing_method = self.pricing_method,  
                                       actual_vol=self.actual_vol)
                # option price dataframe creation
                mult_option_px_sim.iloc[i, j] = bs_temp.option_pricer()
                # delta dataframe creation
                mult_delta_sim.iloc[i, j] = bs_temp.delta_greeks()[0]
                # gamma dataframe creation
                mult_gamma_sim.iloc[i, j] = bs_temp.gamma_greeks()[0]
                # portfolio dataframe creation
                mult_port_sim.iloc[i, j] = mult_option_px_sim.iloc[i, j] - mult_delta_sim.iloc[i, j] * spot_price
                if i == 0:
                    mult_cash_acc_sim.iloc[i, j] = - mult_port_sim.iloc[i, j]
                    mult_cash_flow_sim.iloc[i, j] = 0
                else:
                    mult_cash_flow_sim.iloc[i, j] = (mult_delta_sim.iloc[i, j] - mult_delta_sim.iloc[i-1, j]) * mult_underlying_sim.iloc[i, j]
                    mult_cash_acc_sim.iloc[i, j] = mult_cash_acc_sim.iloc[i-1, j] * math.exp(-self.r * dt) + mult_cash_flow_sim.iloc[i, j]
                    
                #P&L calculation
                mult_pandl_sim.iloc[i, j] = mult_port_sim.iloc[i, j] + mult_cash_acc_sim.iloc[i, j] 
        
        return mult_pandl_sim
                
        
    def plot_option_price_distribution_plotly(self):
        """This method plots an interactive statistical distribution of option prices using Plotly."""
        option_prices = self.multiple_sim_generator().values.flatten()
        mean_price = np.mean(option_prices)
        median_price = np.median(option_prices)
        
        fig = go.Figure()
        # Histogram
        fig.add_trace(go.Histogram(
        x=option_prices, nbinsx=50, marker_color='blue', opacity=0.7,
        name='Option Prices'))
        
        # Mean line
        fig.add_vline(
        x=mean_price, line_dash='dash', line_color='red',
        annotation_text=f"Mean: {mean_price:.2f}", annotation_position="top left"
    )
        # Median line
        fig.add_vline(
        x=median_price, line_dash='dash', line_color='green',
        annotation_text=f"Median: {median_price:.2f}", annotation_position="top right"
    )
        # Layout
        fig.update_layout(
        title="Statistical Distribution of Option Prices",
        xaxis_title="Option Price",
        yaxis_title="Frequency",
        bargap=0.1,
        template="plotly_white"
    )
        return fig    
            
        
    def pandl_distribution_plotly(self):
        '''This function plots the distribution of the Profit & Loss using Plotly.'''
        pandl = self.multiple_pandlsim()
        terminal_pandl = pandl.iloc[-1, :]  # Extract terminal P&L (last time step)
        mean_pandl = np.mean(terminal_pandl)
        median_pandl = np.median(terminal_pandl)
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
        x=terminal_pandl,
        nbinsx=50,
        marker_color='blue',
        opacity=0.7,
        name='P&L'
    ))
        # Mean line
        fig.add_vline(
        x=mean_pandl,
        line=dict(color='red', dash='dash'),
        annotation_text=f'Mean: {mean_pandl:.2f}',
        annotation_position="top",
        name='Mean'
    )
        # Median line
        fig.add_vline(
        x=median_pandl,
        line=dict(color='green', dash='dash'),
        annotation_text=f'Median: {median_pandl:.2f}',
        annotation_position="top",
        name='Median'
    )
        fig.update_layout(
        title=f'Statistical Distribution of P&L ({self.N} simulations, hedging with {self.hedging_type})',
        xaxis_title='P&L',
        yaxis_title='Frequency',
        bargap=0.2,
        template='plotly_white',
        width=900,
        height=500
    )
        return fig

        
    
    def delta_pandl_plotly(self):
        '''This function plots Delta evolution and P&L over time using Plotly.'''
        pandl = self.multiple_pandlsim().mean(axis=1)
        delta = self.multiple_sim_delta().mean(axis=1)
        
        fig = go.Figure()
        
        # P&L trace
        fig.add_trace(go.Scatter(
        x=np.arange(len(pandl)),
        y=pandl,
        name='P&L',
        yaxis='y1',
        line=dict(color='blue')
    ))
        # Delta trace
        fig.add_trace(go.Scatter(
        x=np.arange(len(delta)),
        y=delta,
        name='Delta',
        yaxis='y2',
        line=dict(color='red')
    ))
        # Layout with dual axis
        fig.update_layout(
        title=f'Delta and Average P&L over Time (Hedging: {self.hedging_type})',
        xaxis=dict(title='Time Step'),
        yaxis=dict(title='P&L', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
        yaxis2=dict(title='Delta', overlaying='y', side='right', titlefont=dict(color='red'), tickfont=dict(color='red')),
        legend=dict(x=0.5, y=1.1, orientation='h'),
        template='plotly_white',
        width=900,
        height=500
    )
        return fig
    
    def gamma_pandl_plotly(self):
        '''This function plots Gamma evolution and P&L over time using Plotly.'''
        pandl = self.multiple_pandlsim().mean(axis=1)
        gamma = self.multiple_sim_gamma().mean(axis=1)
        
        fig = go.Figure()
        
        # P&L trace
        fig.add_trace(go.Scatter(
        x=np.arange(len(pandl)),
        y=pandl,
        name='P&L',
        yaxis='y1',
        line=dict(color='blue')
    ))
        # Gamma trace
        fig.add_trace(go.Scatter(
        x=np.arange(len(gamma)),
        y=gamma,
        name='Gamma',
        yaxis='y2',
        line=dict(color='red')
    ))
        # Layout with dual axis
        fig.update_layout(
        title=f'Gamma and Average P&L over Time (Hedging: {self.hedging_type})',
        xaxis=dict(title='Time Step'),
        yaxis=dict(title='P&L', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
        yaxis2=dict(title='Gamma', overlaying='y', side='right', titlefont=dict(color='red'), tickfont=dict(color='red')),
        legend=dict(x=0.5, y=1.1, orientation='h'),
        template='plotly_white',
        width=900,
        height=500
    )
        return fig

        
    
    def delta_hedging_plots(self):
        
        option_df = BlackScholes(self.spot, self.strike, self.r, self.T, self.option_type, self.ts, self.N, self.number_shares, self.SDE_sim_type, self.hedging_type, 
                                   self.pricing_method, self.implied_vol, self.actual_vol).option_sim_data_generator()

        # Extract columns from the dataframe
        t = option_df['Time Step']
        S = option_df['Underlying Price']
        option_prices = option_df['Option Price']
        deltas = option_df['Delta']
        profit_and_loss = option_df['Cumulative P&L']
        
        # Plotting Asset and Option
        fig, ax1 = plt.subplots(figsize=(16, 8))
        
        color = 'tab:blue'
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Underlying Price', color=color)
        line1, = ax1.plot(t, S, color=color, label='Underlying Price')
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Option Price', color=color)
        line2, = ax2.plot(t, option_prices, color=color, label='Option Price')
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.title(f'Underlying Price and Option Price over Time when Hedging with {self.hedging_type}')
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.show()
        
        # Plotting P&L and Underlying
        fig2, ax1 = plt.subplots(figsize=(16, 8))
        color = 'tab:blue'
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Underlying Price', color=color)
        line1, = ax1.plot(t, S, color=color, label='Underlying Price')
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Cumulative P&L', color=color)
        line2, = ax2.plot(t, profit_and_loss, color=color, label='Cumulative P&L')
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.title(f'Underlying Price and Cumulative P&L over Time when Hedging with {self.hedging_type}')
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.show()
        
    def delta_hedging_plots_plotly(self):
        option_df = BlackScholes(
        self.spot, self.strike, self.r, self.T, self.option_type, self.ts, self.N, 
        self.number_shares, self.SDE_sim_type, self.hedging_type, self.pricing_method, 
        self.implied_vol, self.actual_vol
    ).option_sim_data_generator()
        
        # Extract columns
        t = option_df['Time Step']
        S = option_df['Underlying Price']
        option_prices = option_df['Option Price']
        deltas = option_df['Delta']
        profit_and_loss = option_df['Cumulative P&L']
        
        # Plot 1: Underlying Price vs Option Price
        fig1 = go.Figure()
        
        # Underlying Price
        fig1.add_trace(go.Scatter(
        x=t, y=S, name='Underlying Price', yaxis='y1', line=dict(color='blue')
    ))
        # Option Price
        fig1.add_trace(go.Scatter(
        x=t, y=option_prices, name='Option Price', yaxis='y2', line=dict(color='red')
    ))
        # Layout
        fig1.update_layout(
        title=f'Underlying Price and Option Price over Time (Hedging: {self.hedging_type})',
        xaxis=dict(title='Time Step'),
        yaxis=dict(title='Underlying Price', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
        yaxis2=dict(title='Option Price', overlaying='y', side='right', titlefont=dict(color='red'), tickfont=dict(color='red')),
        legend=dict(x=0.5, y=1.15, orientation='h'),
        template='plotly_white',
        width=900,
        height=500
    )
        # Plot 2: Underlying Price vs Cumulative P&L
        fig2 = go.Figure()
        
        # Underlying Price
        fig2.add_trace(go.Scatter(
        x=t, y=S, name='Underlying Price', yaxis='y1', line=dict(color='blue')
    ))
        # Cumulative P&L
        fig2.add_trace(go.Scatter(
        x=t, y=profit_and_loss, name='Cumulative P&L', yaxis='y2', line=dict(color='red')
    ))
        # Layout
        fig2.update_layout(
        title=f'Underlying Price and Cumulative P&L over Time (Hedging: {self.hedging_type})',
        xaxis=dict(title='Time Step'),
        yaxis=dict(title='Underlying Price', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
        yaxis2=dict(title='Cumulative P&L', overlaying='y', side='right', titlefont=dict(color='red'), tickfont=dict(color='red')),
        legend=dict(x=0.5, y=1.15, orientation='h'),
        template='plotly_white',
        width=900,
        height=500
    )
        return fig1, fig2


    
    def greeks_plot_generator(self):
        
        option_df = BlackScholes(self.spot, self.strike, self.r, self.T, self.option_type, self.ts, self.N, self.number_shares, self.SDE_sim_type, self.hedging_type, 
                                   self.pricing_method, self.implied_vol, self.actual_vol).option_sim_data_generator()
        # Extract columns from the dataframe
        t = option_df['Time Step']
        S = option_df['Underlying Price']
        option_prices = option_df['Option Price']
        deltas = option_df['Delta']
        charms = option_df['Charm']
        
        
        # Plotting Delta and Charm
        fig, ax3 = plt.subplots(figsize=(16, 8))
        
        color = 'tab:green'
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Delta', color=color)
        line3, = ax3.plot(t, deltas, color=color, label='Delta')
        ax3.tick_params(axis='y', labelcolor=color)
        ax4 = ax3.twinx()
        color = 'tab:purple'
        ax4.set_ylabel('Charm', color=color)
        line4, = ax4.plot(t, charms, color=color, label='Charm')
        ax4.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.title('Delta and Charm over Time')
        lines = [line3, line4]
        labels = [line.get_label() for line in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        plt.show()
        
        #Plotting Delta and Gamma
        gammas = option_df['Gamma']
        fig1, ax4 = plt.subplots(figsize=(16, 8))
        color = 'tab:green'
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Delta', color=color)
        line3, = ax4.plot(t, deltas, color=color, label='Delta')
        ax4.tick_params(axis='y', labelcolor=color)
        
        ax5 = ax4.twinx()
        color = 'tab:purple'
        ax4.set_ylabel('Gamma', color=color)
        line4, = ax4.plot(t, gammas, color=color, label='Gamma')
        ax4.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title('Delta and Gamma over Time')