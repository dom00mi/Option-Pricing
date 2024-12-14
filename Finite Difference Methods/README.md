# Finite Difference Methods for Option Pricing

### Intro:


Finite Difference Methods (FDM) are widely recognized numerical techniques in computational finance, frequently used to solve differential equations by approximating them as difference equations. Known for their simplicity and historical significance, FDM represents one of the earliest approaches for addressing differential equations, with their application to numerical problems dating back to the 1950s.

In this quick project, I would like to show you how to apply Finite Difference Methods in the context of Option Pricing. FDM shares similarities with (binomial) tree models in its conceptual framework. However, instead of discretizing asset prices and time using a tree structure, FDM discretizes on a grid, dividing both time and asset price into small intervals and calculating values at every grid point.

Below the primary approaches to FDM:

- #### Explicit
- #### Implicit
- #### Crank-Nicolson methods


The explicit method is straightforward to implement but is sensitive to the size of time and asset steps, which can lead to convergence issues. It is generally less stable compared to implicit or Crank-Nicolson methods. The implicit method, while more stable, requires solving a system of equations, making it computationally intensive. The Crank-Nicolson approach combines features of both explicit and implicit methods, offering a balance between stability and computational efficiency.

Finite Difference Methods are particularly suitable for low-dimensional problems, typically up to four dimensions, making them ideal for scenarios like pricing options or solving partial differential equations (PDEs) related to financial models.

### Additional Interesting Insights about FDM:

1. #### Flexibility in Applications: 

FDM is not limited to finance; it is also extensively used in engineering, physics, and other sciences for tasks like heat transfer, fluid dynamics, and electromagnetic field modeling.

2. #### Grid Customization: 

Advanced implementations allow for adaptive grids, where grid spacing changes dynamically based on the problem's complexity, improving both accuracy and efficiency.

3. #### Boundary Condition Management: 

FDM can accommodate complex boundary conditions, such as Dirichlet or Neumann boundaries, which are common in real-world financial and physical systems.

4. #### Integration with Stochastic Models: 

FDM can be combined with stochastic differential equations to model scenarios with uncertainty, broadening its applicability in risk analysis.

5. #### Historical Evolution: 

While its roots trace back to the mid-20th century, modern advancements in computational power have significantly enhanced the usability of FDM, enabling it to tackle more sophisticated problems with higher precision.

6. #### Comparison with Finite Element Methods (FEM):

While both FDM and FEM solve differential equations, FDM is preferred for problems with simpler geometries, whereas FEM excels in complex geometrical domains.

In this repo, I would like to show you how to apply Finite Difference Methods in the context of Option Pricing.


## Black Scholes PDE 

In this section, let's take a look on the famous Black-Scholes Partial Differential Equation:

\begin{equation}
{\frac {\partial V}{\partial t}}+{\frac {1}{2}}\sigma ^{2}S^{2}{\frac {\partial ^{2}V}{\partial S^{2}}}+rS{\frac {\partial V}{\partial S}} - rV = 0
\end{equation}


This has a well-known closed-form solution, but let's pretend to use a Numerical Method to obtain the solution, hence differentiation using the grid.


##### Time to Maturity Step:

$t=T - k\delta t$ 

where $0 \leq k \leq K$

##### Asset Step:

$S=i\delta s$

where $0 \leq i \leq I$

Here i and k are respective steps in the grid and we can write the value of the option at each grid points as:

$V^k_i = V(iδS,T −kδt)$


### Greeks Approximation

The Black-Scholes equation can be written as in terms of the Greeks, as:

$Θ+\frac{1}{2}σ^2S^2Γ+rS∆−rV =0$

##### Theta:

${\displaystyle {\frac {\partial V}{\partial t}} = \lim_{h \to 0} \frac{V(S,t+h)-V(S,t)}{h} }$

which becomes in terms of grid points:

${\displaystyle {\frac {\partial V}{\partial t}} \approx \frac{V^k_i - V^{k+1}_i}{\delta t}}$


##### Delta:

${\displaystyle {\frac {\partial V}{\partial S}} \approx \frac{V^k_{i+1}- V^{k}_{i-1}}{2 \delta S}}$


##### Gamma:

${\displaystyle {\frac {\partial ^2 V}{\partial S^2}} \approx \frac{V^k_{i+1}-2V^k_i + V^{k}_{i-1}}{\delta S^2}}$


And now let's get to the code!

