## Data generation

### Model formulation

We use a compartmental model that stratifies the population into 5 compartments; susceptible $(S)$, exposed $(E)$, infectious $(I)$, recovered $(R)$ and deceased $(D)$. We represent the number of individuals in each compartment at time $t$ with $S(t), E(t), I(t), R(t), D(t)$ respectively, and the active population at time $t$ with size $N(t) = S(t) + E(t) + I(t) + R(t)$. 

*Why do I use the active population here? Because deceased individuals aren't contributing to transmission or depletion of susceptibles? Is this standard in epidemiology?*

The disease dynamics are modelled by the following system of ordinary differential equations (ODEs):

$$
\begin{aligned}

\frac{dS}{dt} &= -\beta(t)\frac{S(t)I(t)}{N}, \\ 
\frac{dE}{dt} &= \beta(t)\frac{S(t)I(t)}{N} - \sigma E(t), \\
\frac{dI}{dt}&= \sigma E(t) - (\gamma+\delta)I(t), \\
\frac{dR}{dt} &= \gamma I(t), \\
\frac{dD}{dt}&= \delta I(t)
\end{aligned}
$$
where $\sigma$ is the incubation rate, $\gamma$ is the recovery rate, and $\delta$ is the disease-induced mortality rate. We incorporate endogenous behaviour by introducing a feedback loop represented by a time-varying transmission rate $\beta(t)$ that varies depending on disease prevalence at time $t$. 

*Should I actually be writing $\beta(I)$ instead of $\beta(t)$?*

### Time-varying transmission rate

We will model the transmission rate as a monotonically decreasing function of the number of infections $I(t)$, reflecting adaptive protective behaviours in the population (as infections increase, transmission decreases due to the population's response to perceived risk). 

The true generating functional form is given by:

$$ \beta(t) = \beta_0  e ^{-\zeta \delta I(t)} \tag{1}$$
where $\beta_0$ is the baseline transmission rate in the absence of behavioural response, $\zeta$ is the strength of the behavioural response, where a higher $\zeta$ corresponds to a bigger difference in the population's behaviour when compared with the baseline, and $\delta$ and $I(t)$ are as above.

The baseline transmission rate is given by:
$$
\beta_0 = R_0(\gamma + \delta) \tag{2}
$$
where $R_0$ is the basic reproduction number. Note that when the strength of the behavioural response, $\zeta = 0$, we have that $\beta(t) = \beta_0$ for all $t$. This is analagous to a model that does not incorporate endogenous behaviour.

### Parameters and initial conditions

To synthesise the epidemic trajectories we used parameters calibrated to COVID-19 data taken from the COVID-19 Data Repository by the Centre for Systems Science and Engineering (CSSE) at John Hopkins University (JHU) [7] for each of the 51 US states. 

We held the incubation rate $\sigma$ and the recovery rate $\gamma$ fixed across all trajectories, and their values $\sigma = 1/3$ day$^{-1}$ [1,2] and $\gamma = 1/10$ day$^{-1}$ [3,4] were chosen based on early SARS-CoV-2 data. To model each synthetic trajectory from disease inception, we also kept $E(0)=1$ and $R(0)=D(0)=0$ (initial exposed, recovered, and deceased population sizes) constant across all trajectories. 

To generate realistic epidemic trajectories, we obtained the initial population size $N(0)$ for each state from the JHU CSSE data [7], and inferred the context-specific parameters using the methodology and code outlined in [5]. As a brief summary, we undertook Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC) with weakly informative priors and took the weighted median of the respective posterior marginal distributions. The code used for the parameter inference is available at [6] and further details of the methodology used can be found in the Supplementary material of [5]. 

The parameters that were estimated, and therefore varied between state trajectories, were the disease prevalence $\pi_0$, the basic reproduction number $R_0$, the disease-induced mortality rate $\delta$ and the behavioural strength $\zeta$. The priors were given by:

$$
R_0 \sim \mathcal{U}(1.2, 6.0), \quad
\log \pi_0 \sim \mathcal{U}(\log 10^{-8}, \log 10^{-3}), \quad
\log \delta \sim \mathcal{U}(\log 10^{-6}, \log 10^{-2}), \quad
\zeta \sim \mathcal{U}(0, 0.05)
$$

where "log-uniform priors were used for $\pi_0$ and $\delta$ to allow these positive parameers to vary over several orders of magnitude without favouring any particular scale a priori, which is appropriate when both are expected to be small and poorly constrained before observing the data" [5]

From these estimated parameters we were then able to derive state-specific baseline transmission rates $\beta_0$ using Equation $(2)$,  initial prevalence given by $I_{0}=\max\{1.0, \pi_0 N(0)\}$, and the initial susceptible population size $S(0) = N(0) - E(0) - I(0) - R(0)-D(0)$.

### Simulation

For each state, we generated 365 days of data of an epidemic trajectory, by solving the ODE system using the `Tsit5()` solver from `DifferentialEquations.jl` with the parameters described above. The number of individuals in each compartment for each state was recorded at daily intervals.

We did not compute a 7-day moving average or add observational noise for Phase 1.

## Universal differential equation (UDE) framework

### Model formulation

In the UDE framework, we attempt to learn the "unknown" time-varying transmission rate by replacing $\beta(t)$ with a neural network approximator $f_{NN}^\theta$, where $\theta$ denotes the trainable parameters of the neural network, namely the weights and biases. We can rewrite the ODE system as follows:
$$
\begin{aligned}
\frac{dS}{dt} &= -f_{NN}^\theta(x(t))\frac{S(t)I(t)}{N}, \\ 
\frac{dE}{dt} &= f_{NN}^\theta(x(t))\frac{S(t)I(t)}{N} - \sigma E(t), \\
\frac{dI}{dt}&= \sigma E(t) - (\gamma+\delta)I(t), \\
\frac{dR}{dt} &= \gamma I(t), \\
\frac{dD}{dt}&= \delta I(t)
\end{aligned}
$$
where $x(t)$ denotes the input parameters for our neural network at time $t$.

*We are currently exploring two variants of the UDE framework to investigate the effect on the neural network's ability to approximate the transmission rate, and the performance of symbolic regression*

### Neural network architecture

The neural network $f_{NN}^\theta$ is a feed-forward neural network implemented in `Lux.jl`. It has 2 neural network layers, with 5 neurons per layer and a Gaussian Error Linear Unit (GELU) activation function. The neural network has a linear output layer with 1 neuron and a softplus final activation function. The final activation function ensures that $\beta(t)\geq 0$ for all $t$, so the learned transmission rate remains epidemiologically plausible.

**What is prior knowledge, could enforce constraints**

We explore two variants of the UDE framework with different training processes and neural network inputs $x(t)$.

***Single-trajectory model***
This model is trained on a singular simulated trajectory (for Massachusetts), and takes the normalised infectious population:
$$
\mathbf{X}(t) = [I(t)/N(t)]
$$

***Multiple-trajectory model***
This model is trained on a multiple simulated trajectories (the simulated epidemics for each US state). The neural network takes 5 parameters as inputs; the parameters that vary across the simulated trajectories $\beta_0, \zeta, \delta$, in addition to the normalised infectious population and normalised time:

$$
\mathbf{X}(t) = [\beta_0, \zeta, \delta, I(t)/N(t),t/T]
$$

where $T=365$ the length of the simulated trajectory.

Using the trajectory-specific parameters as inputs allows the neural network weights and biases $\theta$ to generalise across all 51 trajectories by conditioning the output on the varying input parameters.

**Need to implement: instead of inputting constants: scale based on the prior**

### ODE solver

The UDE system is integrated using the `Rosenbrock23()` solver from `DifferentialEquations.jl` and the model's prediction for each state is saved at daily intervals for $t \in [1, 365]$. We use `DiffEqFlux.jl` to undertake gradient-based optimisation, to update the trainable neural network parameters. 

## Loss function and optimisation

### Loss function

We train the neural network parameters $\theta$ by minimising the mean squared error (MSE) between the predicted infectious count $\hat{I}(t)$ and the observed (simulated) infectious count $I(t)$. The loss function is given by:
$$
\mathrm{MSE} = \frac{1}{T} \sum_{i=1}^T ( \hat{I}_i - I_i )^2
$$
In Phase 1, we exclude regularisation as we are attempting to create ideal conditions for the neural network to accurately approximate the transmission rate.

### Multiple-trajectory training objective

To obtain a combined loss across all of our synthesised trajectories, we evaluate the MSE for each individual trajectory and then sum the total loss. So for our $M=51$ states, the combined loss function to minimise is:
$$
L_{\text{combined}}(\theta)= \sum_{i=1}^ML_{i}(\theta)
$$
where $L_i(\theta)$ is the MSE for trajectory $i$.

We accumulate the gradients of each individual loss with respect to $\theta$ across all trajectories, and use this combined gradient to update our neural network parameters:
$$
\Delta_{\theta}L_{\text{combined}}=\sum_{i=1}^M \Delta L_{i}(\theta)
$$
### Optimiser

We use the Adam optimiser from `Optimisers.jl` with a learning rate of $\eta = 10^{-3}$, running the training process for a maximum of 2,500 iterations. We retain the parameters that result in the lowest loss across all iterations. The model parameters were randomly initialised, and training is stopped if there are 5 consecutive infinite losses as this indicates an unstable parameter region.

We compute the gradient using reverse-mode automatic differentiation via `Zygote.jl`, using the `pullback` function.  The gradient with of the loss with respect to the neural network parameters is used to update the neural network weights.

## Symbolic regression

### Problem formulation

**This section is unclear and needs amending!**

After training the neural network to approximate the transmission rate $\beta(t)$, we will attempt to recover the true functional form that we used to generate the synthetic data used for training, by using symbolic regression, namely, the SINDy method [8].

In traditional uses of the SINDy algorithm, the data required is the time series data and the measured derivative data [8]. However, in our case we modify the algorithm so that it only applies to the unknown function in our equation, $\beta(t)$, and we replace the derivative data with the output of our neural network $f_{NN}^\theta(x(t))$. Since we have an approximation of the function, we are able to generate many input-output pairs [9].

We take a set of input-output pairs; evaluating both the neural network inputs and the neural network's approximation of the transmission rate at each time point:
$$
\{\mathbf{X},f_{NN}^\theta(\mathbf{X})\}_{t=1}^T
$$
We then use `DataDrivenDiffEq.jl` and `DataDrivenSparse.jl` to frame this as a sparse regression problem [8,9] :
$$
f_{NN}^\theta(\mathbf{X})=\Theta(\mathbf{X}) \Xi
$$
where $\Theta(X)$ is the library of candidate functions that could comprise the functional form for the transmission rate, and $\Xi = [\xi_{1}, \xi_{2}, \dots, \xi_{n}]$ is the sparse vector of coefficients determining which of the candidate functions are active in the functional form. We want to find the fewest terms in the library that can describe the data, hence why we call it sparse.

Once we have determined the sparse coefficient vector $\Xi(X)$, we can construct each element of the functional form:

$$
\hat{\beta}_{{SR}}(t)_k \approx f_{NN_{k}}^\theta(\mathbf{X}(t))\approx\Theta(\mathbf{X}(t)^T) \xi_k
$$
and so we have that the SINDy approximation of the transmission rate can be represented by:

$$
\hat{\beta}_{{SR}}(t) \approx f_{NN}^\theta(\mathbf{X}(t))\approx\Xi^T(\Theta(\mathbf{X}(t)^T))^T
$$
*~={red}
1. *Should I be representing beta as a function of $\beta(I)$ or $\beta(\beta_0, \zeta, \delta, I(t))$*=~

### Candidate function library

We use different basis libraries for each variant of the framework:

***Single-trajectory:***
The library consists of:
- Polynomial terms in $u = I$ up to degree 3: $\{u, u^2, u^3\}$
- Exponential terms:
	1. $\exp(−ζδN · u_\text{scaled})$ 
	2. $\exp(−ζ · u_\text{scaled})$ 
	3. $\exp(−δ ·u_\text{scaled})$ 
	4. $\exp(−ζδ ·u_\text{scaled})$ 
	5. $\exp(−ζN ·u_\text{scaled})$ 
	6. $\exp(−δN ·u_\text{scaled})$

We use is $u_{\text{scaled}} = I/N$ in the exponential argument to prevent overflow in the basis. The sparse coefficient should be able to counteract this by multiplying by coefficient $N$. We assume $\zeta$ and $\delta$ to be known, due to the limitation of SINDy that it can only produce a linear combination of nonlinear terms so is unable to change the coefficient within the exponential argument.

***Multiple-trajectory:***
The input vector to the basis is $u = (\beta_0, \zeta, \delta, I/N)$
The library consists of:
1. $\exp(-\beta_0 I/N)$  
2. $\exp(-\zeta I/N)$  
3. $\exp(-\delta I/N)$  
4. $\exp(-\beta_0 \zeta I/N)$  
5. $\exp(-\beta_0 \delta I/N)$  
6. $\exp(-\zeta \delta I/N)$  
7. $\exp(-\beta_0 \zeta \delta I/N)$  
8. $\beta_0 \exp(-\zeta \delta I/N)$
### Sparse regression algorithm

We identify the sparse coefficient vector using the Sequentially Thresholded Least Squares (STLSQ) algorithm, with a shrinking cut-off of $\lambda=10^{-2}$. STLSQ iteratively applies (until convergence) a least-squares algorithm, removing any coefficients $[\xi_{1}, \xi_{2}, \dots, \xi_{n}]$ that are less than the shrinking cut off $\lambda$.

The regression is configured with:
- Maximum iterations: 10,000
- Normalisation: z-score standardisation of inputs and outputs (ZScoreTransform)
- Model selection criterion: Bayesian Information Criterion (BIC) which penalises complexity and favours parsimonious models
- Coefficient precision: rounded to 1 significant digit for interpretability

*Try genetic algorithm "symbolic regression using a genetic algorithm offers a better approach when there are complex constraints on the functional form of the system" and STLSQ better on systems that require large amounts of data points i.e. not our case=~ [10].~={red} Evolutionary genetic algorithm used in=~ [[philippsCurrentStateOpen2025]]*

*Settings above are taken from=~ [[Automatically discover missing physics by embedding ML into differential equations - SciML tutorial]] ~={red}and are not thought through by me.*

*In =~[[Automatically discover missing physics by embedding ML into differential equations - SciML tutorial]], and [10] ~={red}they include: Data split: 90% train / 10% validation, with random shuffled batches of size 30 which makes the result of the regression change every time, unsure why but don't think I need here, INVESTIGATE*

### Validation

We substitute the SINDy representation of the transmission rate into the system of differential equations and solve:

$$
\begin{aligned}

\frac{dS}{dt} &= -\hat{\beta}_{{SR}}\frac{S(t)I(t)}{N}, \\ 
\frac{dE}{dt} &= \hat{\beta}_{{SR}}\frac{S(t)I(t)}{N} - \sigma E(t), \\
\frac{dI}{dt}&= \sigma E(t) - (\gamma+\delta)I(t), \\
\frac{dR}{dt} &= \gamma I(t), \\
\frac{dD}{dt}&= \delta I(t)
\end{aligned}
$$
We compare the epidemic trajectories predicted by the true generating transmission rate, the neural network approximation and the symbolic regression representation. Analagously, we compare the time-varying transmission rate trajectories throughout the epidemic for each scenario.

We use the mean-squared error to evaluate performance:

$$
MSE = \frac{1}{T}\sum_{t=1}^T(\hat{I}(t)-I(t))^2
$$
# References

[1] S. A. Lauer, K. H. Grantz, Q. Bi, F. K. Jones, Q. Zheng, H. R. Meredith, A. S. Azman, N. G. Reich, and J. Lessler, “The incubation period of coronavirus disease 2019 (COVID-19) from publicly reported confirmed cases: estimation and application,” Annals of internal medicine, vol. 172, no. 9, pp. 577–582, 2020.

[2] X. He, E. H. Lau, P. Wu, X. Deng, J. Wang, X. Hao, Y. C. Lau, J. Y. Wong, Y. Guan, X. Tan, et al., “Temporal dynamics in viral shedding and transmissibility of COVID-19,” Nature Medicine, vol. 26, no. 5, pp. 672–675, 2020.

[3] R. W¨olfel, V. M. Corman, W. Guggemos, M. Seilmaier, S. Zange, M. A. M¨uller, D. Niemeyer, T. C. Jones, P. Vollmar, C. Rothe, et al., “Virological assessment of hospitalized patients with COVID-2019,” Nature, vol. 581, no. 7809, pp. 465–469, 2020.

[4] O. Puhach, B. Meyer, and I. Eckerle, “SARS-CoV-2 viral load and shedding kinetics,” Nature Reviews Microbiology, vol. 21, no. 3, pp. 147–161, 2023.

[5] Pant, B. _et al._ (2025) ‘The paradox of neglecting changes in behavior: How standard epidemic models Misestimate both transmissibility and final epidemic size’, _MedRxiv_ [Preprint]. doi:10.64898/2025.12.07.25341782. 

[6] B. Pant, M. Lalovic, I. Z. Kiss, and M. Santillana, “epi-behavior-models: Repository.” https://github.com/markolalovic/epi-behavior-models, 2025. Deposited 24 November 2025.

[7] Johns Hopkins University Center for Systems Science and Engineering, “COVID-19 Data Repository.” https://github.com/CSSEGISandData/COVID-19, 2020. Accessed: 19 October 2025.

[8] Brunton, S.L., Proctor, J.L. and Kutz, J.N. (2016) ‘Discovering governing equations from data by sparse identification of nonlinear dynamical systems’, _Proceedings of the National Academy of Sciences_, 113(15), pp. 3932–3937. doi:10.1073/pnas.1517384113.

[9] Rackauckas, C. _et al._ (2020) ‘Universal differential equations for scientific machine learning’, _arXiv_ [Preprint]. doi:10.21203/rs.3.rs-55125/v1. 

[10] Chesebro, A.G. _et al._ (2025) _Scientific machine learning of chaotic systems discovers governing equations for neural populations_. Available at: https://arxiv.org/html/2507.03631v3#S8 (Accessed: 08 April 2026).