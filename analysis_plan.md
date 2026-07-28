# Can we learn the behavioural response to an epidemic?

**Aim:** Can we learn a behavioural response to an epidemic in idealised conditions?

**Objectives:**
- Under what conditions can we recover an analytical expression for the behavioural response from epidemic data?
- How does noisy observational data affect our ability to recover the behavioural response?
- Can we learn the behavioural response with limited data and use it to make better predictions?
- Can we integrate idealised behavioural data to improve our predictions?
# Background

Mechanistic models aim to mathematically represent the dynamic processes within a system by using known physical, biological or chemical laws. In the context of infectious disease forecasting, mechanistic models are used to explain how and why a disease is spreading [@yeIntegratingArtificialIntelligence2025]. 

During the COVID-19 pandemic, many mechanistic models were developed, but their success in accurately forecasting the pandemic trajectory was varied [@hamiltonIncorporatingEndogenousHuman2024]. It became clear that there was an intrinsic link between the population's behavioural response to the pandemic and how the disease was spreading [@kuwaharaPredictingCOVID19Pandemic2023], but determining the underlying relationship and how to incorporate this response into the models was challenging. Typically, the behaviour was included exogenously via public health guidance or mobility data, however, when incorporated in this way, the behavioural response to the epidemic dynamics are ignored. Accounting for the behavioural response endogenously (when the behaviour is represented as a function of another time-dependent variable or state within the model and is able to alter the epidemiological dynamics [@hamiltonIncorporatingEndogenousHuman2024] [@pantParadoxNeglectingChanges2025]) was identified as an important way to improve forecasting in the post-COVID evaluation [11].

This coupled behaviour-disease dynamic is complex to model and there is a lack of data (theory?) around how the population actually responds [@ryanBehaviourDiseaseTransmission]. However, it has been shown that models which do not incorporate behavioural response typically underestimate $R_0$ and overestimate the final epidemic size [@pantParadoxNeglectingChanges2025]. When behaviour is incorporated the model fits better to observed data and synthetic data experiments showed that this better fit, improved estimation of $R_0$ and the epidemic size is a direct result of the inclusion of behavioural response [@hamiltonIncorporatingEndogenousHuman2024] [@pantParadoxNeglectingChanges2025]. Behavioural dynamics are also important for non-pharmaceutical interventions (NPIs). During the first wave of the COVID-19 pandemic, NPIs were more effective than when introduced during the second wave of the pandemic and this is likely due to the population's behaviour not returning to the pre-pandemic baseline after the first wave. This highlights that people's behaviour changes even when it is not enforced [@sharmaUnderstandingEffectivenessGovernment2021a]. 

==There are three main approaches to incorporate behaviour endogenously==:
- introducing a feedback loop: a time-varying transmission rate that depends on other states
- using game/utility theory where the transmission rate is affected by the introduction of interventions and the populations' decision of whether to adhere to them or not
- considering information/opinion spread where the movement of individuals to and from a protected class if they adhere/reject an NPI is modelled, and how they move between the classes depends on the risk perception which can be influenced by other members of the population [@hamiltonIncorporatingEndogenousHuman2024].

Introducing a feedback loop, where a behavioural response is triggered by the prevalence of the disease, is a relatively straightforward approach to incorporating endogenous behaviour. Analyses have shown that a relatively simple compartmental model with a feedback loop could predict COVID-19 deaths as well as the CDC ensemble [11], and that including the infection rate as a function of prevalence resulted in less error than when this feedback loop was excluded [7].

In recent years, there has been a particular focus on exploring the opportunity to integrate machine learning into mechanistic models, retaining the mechanistic structure of the system but enabling the model to use multiple data streams and big data to improve the accuracy of its prediction. Attempts to integrate machine learning into epidemiological forecasting include using physics-informed neural networks (PINNs), epidemiology-aware AI models (EAAMs), synthetically-trained AI models, and AI-augmented epidemiological models [@yeIntegratingArtificialIntelligence2025]. However, most methods attempt to enhance epidemiological parts of the model, with limited attention given to socio-behavioural mechanisms [@yeIntegratingArtificialIntelligence2025]. 

Neural networks have been used to leverage the vast amounts of historical data at later stages of an epidemic to estimate the epidemiological parameters and to produce epidemic predictions [@nguyenBeCakedExplainableArtificial2022]. ~={red} add more examples here=~. 

*Add here about other methods for discovering unknown equations e.g. basis functions, splines*

However, a relatively new approach to embed neural networks within the mechanistic structure of the dynamical system via a universal differential equation (UDE) framework has emerged. UDEs are differential equation systems that have universal approximators, such as neural networks, embedded within them. Known dynamics of the system can be explicitly included, and the neural networks can use data during training to then learn and approximate unknown components or processes [@rackauckasUniversalDifferentialEquations2021]. This allows the advantages of machine learning and mechanistic modelling to be combined and to complement the flaws of one another by learning unknown components of the system in a data-driven way, while still being able to incorporate prior epidemiological knowledge, constrain the system, interpret the dynamics and assign epidemiological meaning to the parameters and transmission.

Their use has been investigated in the context of infectious disease forecasting; for example, mapping noisy wastewater surveillance data to reported case counts [@schmidWastewaterinformedNeuralCompartmental2026], and approximating the force of infection from neighbouring regions to a target region [@rojas-camposLearningCOVID19Regional2023]. ~={red} add more examples here e.g. [39] and [53] from [6]=~

To further improve the utility of the UDE framework, it is possible to use symbolic regression to represent the neural network approximation in a functional form [@DiscoveringGoverningEquations] [@rackauckasUniversalDifferentialEquations2021]. Symbolic regression ...This enhances interpretability and reduces the black box effect of traditional ML techniques. Moreover, it can give us insights into the unknown underlying mechanisms that we are investigating, allowing us to learn some process or component within the system. As well as providing information about the system we are looking at, the symbolic representation can also help us make better predictions [@ScientificMachineLearning].

*Include here about using the NN approximation as a data augmentation tool for SR*[@ioannouEmpiricalInvestigationNeural2026]

*Add here more detailed info about the different types of symbolic regression e.g. sparse regression vs evolutionary algorithm*

*Add here (?) about where symbolic regression has been used in infectious disease modelling e.g. [@rojas-camposLearningCOVID19Regional2023]*

*Add here about how behavioural scientists are using ML*

The use of the UDE framework for coupled behaviour-disease dynamic has not been extensively researched. The only example at the time of writing is in [@kuwaharaPredictingCOVID19Pandemic2023], where they wanted to investigate the link between the population's response during the COVID-19 pandemic, and the various pandemic waves observed. The timing of interventions made it clear that a large first wave led to strict interventions being implemented, the number of infections decreasing and then restrictions being relaxed leading to a second wave. They used UDEs to learn the interaction between mobility and infections to predict the future trajectory of the epidemic and people's mobility patterns by training the UDE on mobility and infection prevalence data. They then qualitatively compared the performance of the framework when learning biases were included as additional objectives for the neural network to optimise over.

We want to further investigate this behaviour-disease interplay by introducing a purely endogenous feedback loop to represent the risk-driven behaviour of the population, and seeing if, and under what conditions, it is possible to learn the behavioural response via symbolically recovering the time-varying transmission rate. ==The primary objective is to correctly learn a mechanistically meaningful expression for the behavioural response. As a result, we would expect this to improve predictive ability of the model, though this is not explored in depth here. CHANGE THIS==

To learn the prevalence-dependent behaviour we will use a UDE framework, then we will use symbolic regression techniques to recover a functional form for the behavioural response, which will provide information about the underlying behavioural response during an epidemic. We will only attempt to represent the endogenous behavioural response as a feedback loop through a transmission rate that depends on other time-varying states within the model. We will not attempt to simulate more complicated theoretical behavioural models ~={red}add more examples here=~, however, we believe it would be possible to introduce a UDE framework in these cases. For example, in [@ryanBehaviourDiseaseTransmission] they use a health belief model to describe factors that contribute towards an individual assuming a protective behaviour. Future work could investigate if it is possible to learn the rate of behaviour uptake and abandonment in the health belief model, and recover the symbolic form of this relationship with parameter values for the factors contributing towards performing protective behaviour.

***Probably a repeat - is my deleted Ch1 rationale***
As discussed above, there is a clear link between behaviour change and disease spread. Despite this, there is currently limited theory in infectious disease modelling for how to integrate behaviour into our models. During the COVID-19 pandemic, a large amount of data on the population's behaviour was collected and we want to investigate if we can learn elements of the coupled behaviour-disease interplay from this data and integrate this into our models. Since there is limited behavioural theory, it is not possible to use a purely mechanistic model to incorporate what we learn from the data, so we investigate the use of a hybrid model that combines our epidemiological understanding of transmission dynamics with machine learning methods that can learn unknown processes in a data driven approach, namely neural networks within a universal differential equation (UDE) and symbolic regression (SR) framework.

Understanding and approximating the behavioural response may improve models' predictive power [@pantParadoxNeglectingChanges2025], potentially allowing the identification of turning points and better capturing the peak's magnitude. Using UDEs and SR to learn the underlying mechanism of behavioural response will provide an analytical representation of the behavioural response that is interpretable.

# Methods

## Phase 1: create ideal conditions for a symbolic UDE framework

### Data generation

#### Epidemic data

##### Model formulation

We use a compartmental model that stratifies the population into 5 compartments; susceptible $(S)$, exposed $(E)$, infectious $(I)$, recovered $(R)$ and deceased $(D)$. We represent the number of individuals in each compartment at time $t$ with $S(t), E(t), I(t), R(t), D(t)$ respectively, and the active population at time $t$ with size $N(t) = S(t) + E(t) + I(t) + R(t)$. 

The disease dynamics are modelled by the following system of ordinary differential equations (ODEs):

$$
\begin{aligned}

\frac{dS}{dt} &= -\beta(I(t))\frac{S(t)I(t)}{N(t)}, \\ 
\frac{dE}{dt} &= \beta(I(t))\frac{S(t)I(t)}{N(t)} - \sigma E(t), \\
\frac{dI}{dt}&= \sigma E(t) - (\gamma+\delta)I(t), \\
\frac{dR}{dt} &= \gamma I(t), \\
\frac{dD}{dt}&= \delta I(t)
\end{aligned}
$$
where $\sigma$ is the incubation rate, $\gamma$ is the recovery rate, and $\delta$ is the disease-induced mortality rate. The incubation rate is the rate of transition from infected (but not yet contagious), to contagious. We incorporate endogenous behaviour by introducing a feedback loop represented by a time-varying transmission rate $\beta(I(t))$ that varies depending on disease prevalence at time $t$. 
##### Time-varying transmission rate

We will model the transmission rate as a monotonically decreasing function of the number of mortalities at time $t$, $\delta I(t)$, reflecting adaptive protective behaviours in the population (as mortalities increase, transmission decreases due to the population's response to perceived risk). 

We will explore three generating functional forms [@pantParadoxNeglectingChanges2025]:

**Exponential form:**
$$ \beta(I(t)) = \beta_0  e ^{-\zeta \delta I(t)} \tag{1}$$
**Rational form:**
$$ \beta(I(t)) = \frac{\beta_0}{1+\zeta \delta I(t)} \tag{2}$$
**Mixed form:**
$$ \beta(I(t)) = \frac{\beta_0 e ^{-\zeta \delta I(t)}}{1+\zeta \delta I(t)} \tag{3}$$
where $\beta_0$ is the baseline transmission rate in the absence of behavioural response, $\zeta$ is the strength of the behavioural response, and $\delta$ and $I(t)$ are as above. A higher $\zeta$ corresponds to a bigger difference in the population's behaviour when compared with the baseline. 

The baseline transmission rate is given by:
$$
\beta_0 = R_0(\gamma + \delta) \tag{4}
$$
where $R_0$ is the basic reproduction number. Note that when the strength of the behavioural response, $\zeta = 0$, we have that $\beta(I(t)) = \beta_0$ for all $t$. This is analagous to a model that does not incorporate endogenous behaviour.

##### Parameters and initial conditions

To produce meaningful synthetic epidemic trajectories we used parameters calibrated to COVID-19 mortality data taken from the COVID-19 Data Repository by the Centre for Systems Science and Engineering (CSSE) at John Hopkins University (JHU) [7]. The model above was fit to the mortality data independently for each of the 51 US states, assuming a the functional form of the time-varying transmission rate $\beta(I)$ is known.

We held the incubation rate $\sigma$ and the recovery rate $\gamma$ fixed across all trajectories, and their values $\sigma = 1/3$ day$^{-1}$ [1, 2] and $\gamma = 1/10$ day$^{-1}$ [3, 4] were chosen based on early SARS-CoV-2 data. To model each trajectory from disease inception, we also kept $E(0)=1$ and $R(0)=D(0)=0$ (initial exposed, recovered, and deceased population sizes) constant across all trajectories. 

***Maybe change to $E(0)=I(0)$, or $I(0)=0$ and $E(0) = \pi_0 N(0)$***

To generate the epidemic trajectories, we obtained the initial population size $N(0)$ for each state from the JHU CSSE data [7], and inferred the context-specific parameters for each US state using the methodology and code outlined in [@pantParadoxNeglectingChanges2025]. As a brief summary, we used Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC) with weakly informative priors and took the weighted median of the respective posterior marginal distributions. The code used for the parameter inference is available at [6] and further details of the methodology used can be found in the Supplementary material of [@pantParadoxNeglectingChanges2025]. 

The parameters that were estimated, and therefore varied between state trajectories, were the disease prevalence $\pi_0$, the basic reproduction number $R_0$, the disease-induced mortality rate $\delta$ and the behavioural strength $\zeta$. The priors were given by:
$$R_0 \sim \mathcal{U}(1.2, 6.0), \quad
\log \pi_0 \sim \mathcal{U}(\log 10^{-8}, \log 10^{-3}), \quad
\log \delta \sim \mathcal{U}(\log 10^{-6}, \log 10^{-2}), \quad
\zeta \sim \mathcal{U}(0, 0.05)$$
Both $\pi_0$ and $\delta$ are expected to be small and positive, so log-uniform priors are appropriate and allow the parameters to vary across several orders of magnitude as they are weakly identified by prior knowledge [@pantParadoxNeglectingChanges2025].

From these estimated parameters we were then able to derive state-specific baseline transmission rates $\beta_0$ using Equation $(4)$,  initial prevalence given by $I_{0}=\max\{1.0, \pi_0 N(0)\}$, and the initial susceptible population size $S(0) = N(0) - E(0) - I(0) - R(0)-D(0)$.

##### Simulation

==Need to resimulate using Rosenbrock23() solver (or does it need to be consistent e.g. should my synthetic data have the same solver as my UDE framework? because I changed the solver in the UDE framework)==

For each state, we generated 365 days of data of an epidemic trajectory, by solving the ODE system using the `Rosenbrock23()` solver from `DifferentialEquations.jl` with the parameters estimated above. The number of individuals in each compartment for each state was recorded at daily intervals.

#### Behavioural data

##### Model formulation

We assume that the transmission rate $\beta(t)$ is proportional to some behavioural data, say the average number of contacts $c(t)$. So we define:

$$
\beta(c(t))= p \cdot c(t)
$$
where $p$ is the probability of transmission given a contact. For each of our three functional forms for the transmission rate we define a different function for $c(t)$:

**Exponential form:**
$$p=\frac{\beta_{0}}{c_{0}}, \quad c(t) = c_{0}e^{-\zeta \delta I(t)} \tag{5}$$
**Rational form:**
$$ p=\frac{\beta_{0}}{c_{0}},\quad c(t)= \frac{c_0}{1+\zeta \delta I(t)} \tag{6}$$
**Mixed form:**
$$ p=\frac{\beta_{0}}{c_{0}},\quad c(t)= \frac{c_0 e ^{-\zeta \delta I(t)}}{1+\zeta \delta I(t)} \tag{7}$$
where $c_0$ is the baseline contact rate in the absence of an epidemic.

***Over time the baseline moves away from pre-pandemic behaviour e.g. the baseline behaviour at the first wave won't be the same as the baseline behaviour at the second wave***

##### Parameters

We will use the parameters estimated in Phase 1 for $\beta_0, \zeta$ and $\delta$. We will use the POLYMOD dataset [@mossongSocialContactsMixing2008] to estimate a value for $c_0$ that reflects the contact rate in pre-pandemic times. Although POLYMOD was only carried out in European countries, because this is a synthetic study, we don't need the data to fit exactly to the US states but are just choosing a value that is empirically motivated.

We then sample $c_0$ from a uniform distribution, taking the country with the lowest (Germany with $7.95$) and highest (Italy with $19.77$) mean daily contacts from the POLYMOD survey [@mossongSocialContactsMixing2008] as the upper and lower bounds.
$$
c_{0} \sim \text{Uniform}(8, 20)
$$
##### Simulation

For each state we sample a value of $c_0$ from the uniform distribution above, then generated 365 days of a behavioural data trajectory, by inputting the observed number of infectious individuals alongside the estimated parameters for that state. The average number of contacts for each state was recorded at daily intervals.
 
### Universal differential equation (UDE) framework

#### Model formulation

In the UDE framework, we attempt to learn the "unknown" time-varying transmission rate by replacing $\beta(I(t))$ with a neural network approximator $f_{NN}^\theta$, where $\theta = \{W^{(1)},b^{(1)},W^{(2)},b^{(2)},W^{(3)},b^{(3)}\} \in \mathbb{R}^{46}$ denotes the trainable parameters of the neural network, namely the weights and biases. We can rewrite the ODE system as follows:
$$
\begin{aligned}

\frac{dS}{dt} &= -f_{NN}^\theta(\mathbf{X}(t))\frac{S(t)I(t)}{N(0)}, \\ 
\frac{dE}{dt} &= f_{NN}^\theta(\mathbf{X}(t))\frac{S(t)I(t)}{N(0)} - \sigma E(t), \\
\frac{dI}{dt}&= \sigma E(t) - (\gamma+\delta)I(t), \\
\frac{dR}{dt} &= \gamma I(t), \\
\frac{dD}{dt}&= \delta I(t)
\end{aligned}
$$
where $\mathbf{X}(t)$ denotes the input parameters for our neural network at time $t$.
#### Neural network architecture

The neural network $f_{NN}^\theta$ is a feed-forward neural network implemented in `Lux.jl`. It has $2$ neural network layers, with $5$ neurons per layer and a Gaussian Error Linear Unit (`gelu`) activation function. The neural network has a linear output layer with 1 neuron and a `softplus` final activation function. The final activation function ensures that $\beta(I(t))\geq 0$ for all $t$, so the learned transmission rate remains epidemiologically plausible.

We explore two variants of the UDE framework with different training processes and neural network inputs $\mathbf{X}(t)$.
##### Single-dataset model

This model is trained on a single dataset, and takes the infectious population normalised by the initial population (the prevalence) as an input:
$$
\mathbf{X}(t) = [I(t)/N(0)]
$$

##### Multiple-dataset model

This model is trained on multiple datasets simultaneously. The neural network takes $4$ parameters as inputs; the parameters that vary across the simulated trajectories $\beta_0, \zeta, \delta$, in addition to the infectious population. All inputs are normalised; the infectious population is normalised by the initial population, $\beta_0$ and $\zeta$ are rescaled to $[0, 1]$ by their prior bounds, and $\delta$ is rescaled using its log-uniform prior bounds on the log scale. 
$$
\mathbf{X}(t) = [\beta_0^{\text{norm}}, \zeta^{\text{norm}}, \delta^{\text{norm}}, I(t)/N(0)]
$$
Using the dataset-specific parameters as inputs allows the neural network weights and biases $\theta$ to generalise across all datasets by conditioning the output on the varying input parameters. 

##### Behavioural data model

Our UDE framework is the same as in the single dataset setting, however, we now take the infectious population normalised by the initial population (the prevalence) as an input, as well as the behavioural data rescaled to $[0, 1]$ by the upper and lower bounds:
$$
\mathbf{X}(t) = [c(t)^{\text{norm}}, I(t)/N(0)]
$$

#### ODE solver

The UDE system is integrated using the `Rosenbrock23()` solver from `DifferentialEquations.jl` and the model's prediction for each state is saved at daily intervals for $t \in [1, 365]$. We use `DiffEqFlux.jl` to undertake gradient-based optimisation, to update the trainable neural network parameters. 

### Loss function and optimisation

#### Single dataset training objective

We train the neural network parameters $\theta$ by evaluating a loss function consisting of 
a scaled mean-squared error (MSE) term and a regularisation term. 

The scaled MSE is evaluated between the predicted infectious count $\hat{I}(t)$ and the observed (simulated) infectious count $I(t)$ and is given by:
$$
\mathrm{L}^I_{MSE}(\theta) = \frac{1}{T} \sum_{i=1}^T \left(\frac{ \hat{I}_i - I_i }{I_{\text{max}}-I_{\text{min}}}\right)^2
$$
The regularisation term is to prevent the neural network from overfitting, and we use an $L2$ regularisation term given by:
$$
\mathrm{L}_{\text{reg}}​(θ)=\lambda \lvert\lvert \theta \rvert \rvert_{2}^2​,\quad\lambda=10^{−6}
$$

We then minimise the sum of both loss terms to find $\theta^*$ such that:
$$
\theta^* = \arg\min_{\theta}\left[\mathrm{L}^I_{MSE}(\theta) + \mathrm{L}_{reg}(\theta)\right]
$$
#### Multiple dataset training objective

To obtain a combined loss across multiple synthetic datasets, we evaluate the scaled MSE for each individual dataset and then sum the total loss. We then add the $L2$ regularisation term for the neural network parameters. So for our $M$ datasets, we try and find $\theta^*$ such that:
$$
\theta^*= \arg\min_{\theta} \left[\mathrm{L}_{\text{reg}}​(\theta) + \sum_{j=1}^M\mathrm{L}_{MSE,j}^{I}(\theta)\right]
$$
where $\mathrm{L}_{MSEj}^I(\theta)$ is the scaled MSE on the infection data for trajectory $j$.

We accumulate the gradients of each individual loss with respect to $\theta^*$ across all trajectories, and use this combined gradient to update our neural network parameters:
$$
\nabla_{\theta}\mathrm{L}_{\text{combined}}=\sum_{i=1}^M \nabla_{\theta} \mathrm{L}_{i}(\theta)
$$
***Potentially change this to evaluating the gradient of the total loss - but shouldn't make a difference***

#### Behavioural data training objective

To integrate the behavioural data in the UDE training, we will construct a loss function that has the scaled MSE on the infections and the regularisation term as in the single data setting, but we will also include a loss term that evaluates the loss between the observed daily contacts and the predicted daily contacts. 

So the training objective is:
$$
\theta^* = \arg\min_{\theta}\left[\mathrm{L}_{MSE}^I(\theta) + \mathrm{L}_{MSE}^c(\theta)+\mathrm{L}_{reg}(\theta)\right]
$$
where
$$
\mathrm{L}_{MSE}^c(\theta)= \frac{1}{T} \sum_{i=1}^T \left(\frac{ \hat{c}_i - c_i }{c_{\text{max}}-c_{\text{min}}}\right)^2
$$
We will create a prediction for the behavioural data $\hat{c}(t)$, by evaluating the relevant function for $c(t)$ (as in Equations $(5), (6),$ and $(7)$) using the predicted infections.
#### Optimiser

We use two optimisers from `Optimisers.jl`. First we do $2,500$ iterations using the Adam optimiser with a learning rate of  $\eta = 10^{-3}$, then we use L-BFGS with $m=10$ (number of previous iterations stored) for a maximum of $2,000$ iterations. We retain the parameters that result in the lowest loss across all iterations. 

We compute the gradient using reverse-mode automatic differentiation via `Zygote.jl`, using the `pullback` function.  The gradient of the loss with respect to the neural network parameters is used to update the neural network weights.

#### Multistart

There may be certain initialisations of the neural network that do not converge in training, often due to the optimiser getting stuck in a local minima. Multistart is a method to remove these particular initialisations. We randomise a starting point and rerun multiple times, where each initialisation produces a candidate model which is then fully trained. We then only retain the runs which we consider usable by evaluating how well they describe the training data.

The criteria we will use for determining usable runs will be that the validation loss in the lowest 90th percentile across all initialisations.

*How do we decide what is usable...90th percentile, need to check, and maybe add another check that takes the xth lowest percentile of combined training and validation loss like in Schmid*
#### Train / validation split and early stopping

To prevent the neural network overfitting we can use train/validation split. By splitting the data used in training into a training set and a validation set, we can identify at what point the neural network is overfitting to the training data by tracking both the loss on the training data and the validation data. We want to choose the neural network parameters at the point where the validation loss has its global minimum.

We split our data into 80% training and 20% validation. We complete all optimisation iterations, then select the parameters that minimise the validation loss.

### Symbolic regression

After training the neural network to approximate the transmission rate $\beta(I(t))$, we will attempt to recover the true functional form that we used to generate the synthetic data used for training, by using symbolic regression. We explored two methods for recovering the time-varying transmission rate from our observed data, sparse regression via sparse identification of nonlinear dynamical systems (SINDy) [@DiscoveringGoverningEquations] and an evolutionary algorithm via `SymbolicRegression.jl` [@cranmerInterpretableMachineLearning2023]. 
#### Problem formulation

For simplicity, we will use notation for the single trajectory model in the following.

To undertake symbolic regression we use the output of our neural network $f_{NN}^\theta(\mathbf{X}(t))$. Since we have an universal approximation of the function, we are able to generate many input-output pairs, overcoming the data quantity limitation in traditional symbolic regression applications [@rackauckasUniversalDifferentialEquations2021].

We take this set of input-output pairs; evaluating both the neural network inputs and the neural network's approximation of the transmission rate at each time point:
$$
\{\mathbf{X},f_{NN}^\theta(\mathbf{X})\}_{k=1}^K
$$
##### Sparse identification of nonlinear dynamical systems (SINDy)

We then use `DataDrivenDiffEq.jl` and `DataDrivenSparse.jl` to frame this as a sparse regression problem [@DiscoveringGoverningEquations], [@rackauckasUniversalDifferentialEquations2021] :
$$
f_{NN}^\theta(\mathbf{X})=\Theta(\mathbf{X}) \Xi
$$

$$
\beta(\mathbf{I})=\Theta(\mathbf{I}) \Xi
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

A limitation of SINDy is that it can only represent functions as a linear combination of the candidate basis functions. The three functional forms chosen to represent the transmission rate (Equations 1, 2, and 3) are nonlinear in their parameters, meaning no standard library of basis functions can recover them without strong prior knowledge of the functional form. This would require a highly tailored candidate library, defeating the purpose of a data-driven discovery approach. Therefore, we chose to continue with the evolutionary algorithm approach, which requires no prior knowledge of the behaviour-transmission relationship.

**Seeing as SymbolicRegression.jl produces too many terms because the NN overfits - could we do first SymbolicRegression.jl then take the terms in this and create our candidate library and do SINDy?**
##### SymbolicRegression.jl

###### Evolutionary algorithm

The second method we will use is `SymbolicRegression.jl`[@cranmerInterpretableMachineLearning2023], [16].  This library takes a simple evolutionary algorithm based on tournament selection, and introduces some modifications. A brief overview of a simple evolutionary algorithm is as follows [13, 14] :

1. Assume a population of individuals, a fitness function, and a set of mutation operations
2. Randomly select an $n$ sized subset of individuals from the population
3. Run the tournament by evaluating the fitness of every individual in that subset
4. Select the fittest individual as the winner with probability $p$, otherwise remove the individual
5. Create a copy of the selected individual and apply a randomly-selected mutation from a set of possible mutations.
6. Replace the weakest member of the subset with the newly mutated individual.

The changes introduced are:
1. ***Age-regularised evolution*** [15]; replacing the eldest member of the population instead of the weakest preventing early convergence in a local minimum
2. ***Simulated annealing*** [17]; speeds up the search process by moving from areas of high temperature where the probability of rejecting a mutation is low so the population diversifies more, to areas of low temperature where the probability of rejecting a mutation is high so the population narrows on the fittest
3. Embedding the algorithm within an ***evolve-simplify-optimise loop***; applying multiple mutations before simplifying, making equations only accessible via redundant intermediary available to the algorithm and optimising free parameters after the expression structure is found
4. Using an ***adaptive parsimony metric***; takes into account the frecency (frequency and recency) of complexities so that there are a similar number of expressions of each level of complexity ensuring that the algorithm doesn't only explore simple, less accurate expressions or complex, more accurate expressions.
###### Configuration

The algorithm is configured with:
- A function space of expressions built from:
	- Binary operators: $\{+, -, *, /\}$ 
	- Unary operators: $\{\exp\}$
- A complexity budget of $20$ nodes in the expression
- 100 evolutionary cycles
- 15 populations each with 50 members
- A parsimony score of 0.1
- Penalise constants with a complexity of 2
- Serial parallelism, deterministic, and no batching to make results reproducible

We select the neural network approximation that has the lowest scaled MSE on the training data
and use this approximation to augment our observation data. We generate $1,000$ input/output pairs by evaluating the neural network at $1,000$ uniformly distributed points within two intervals:
- $[0, 1]$ - all possible input values ($0 - 100$% of the population infectious)
- $[\min(\beta) - \max (\beta)]$ - just within sample

**Why don't I just do SR on every initialisation and create a distribution of UDE+SR results**
#### Validation

The `SymbolicRegression.jl` algorithm outputs a Pareto frontier of potential analytical functions of varying complexity. It then scores each equation based on the change in log-loss divided by the change in complexity. The algorithm then selects the 'best' equation that balances complexity and accuracy according to its selection criteria that maximises the expression score for each equation that is within 1.5x the minimum loss.

In our validation, we will extract both the equation selected as the optimal trade-off between complexity and accuracy by `SymbolicRegression.jl`, and also the equation closest to our ground truth. *The reason for this choice, is that although more complex equations may have a lower error, they are less interpretable and the motivation for using symbolic regression is to obtain an interpretable representation for the behavioural response. Additionally, more complex terms may tend to zero within the range we are considering. (???)*

**How will I actually do this - surely I can't do manually**

We define a 'successful' regression as meeting the following two criteria:
- The functional form is correct (i.e. the expression structure is the same as the true generating equation)
- Parameters are correct to XX significant figures (or MSE < XX)

We substitute the symbolic regression representation of the transmission rate into the system of differential equations and solve:

$$
\begin{aligned}

\frac{dS}{dt} &= -\hat{\beta}_{{SR}}\frac{S(t)I(t)}{N}, \\ 
\frac{dE}{dt} &= \hat{\beta}_{{SR}}\frac{S(t)I(t)}{N} - \sigma E(t), \\
\frac{dI}{dt}&= \sigma E(t) - (\gamma+\delta)I(t), \\
\frac{dR}{dt} &= \gamma I(t), \\
\frac{dD}{dt}&= \delta I(t)
\end{aligned}
$$
We will refer to this model as the SIR + SR model.

We compare the epidemic trajectories predicted by the true generating transmission rate, the neural network approximation and the SIR + SR model. Analogously, we compare the time-varying transmission rate trajectories throughout the epidemic for each scenario. We use the scaled MSE to evaluate performance of individual initialisations, and the continuous ranked probability score (CRPS) [cite seb] on the log scale for the distribution of all initialisations. 

**Can I use the log scale for the transmission rate or does it only work for infections because it is scoring the reproduction number?**

### Single dataset setting

We run $100$ initialisations of the UDE framework, recording the runtime of each training process. We evaluate the scaled MSE for each individual initialisation, and the CRPS [cite seb] on the ==log scale== for the distribution of all $100$ initialisations. We will perform SR on each trained UDE, and evaluate the scaled MSE for each individual SIR + SR model and the CRPS for the distribution of all $100$ initialisations. We will also extract the symbolic representation of the neural network.

**What is the best way to present this? Take the analytical form of the NN with lowest MSE and compare to the true form? Or do for all and take % similar to true form? or extract most similar structurally**

Should I do CRPS on the log scale?

We complete this for 50 datasets (each representing a US state), for each of the three functional forms for the transmission rate.

**What am I comparing it to? Is it meaningful on its own? I guess I will compare to other states**
### Multiple dataset setting

By training the neural network approximator on multiple datasets, we allow it to learn a transferable representation of the underlying behavioural mechanism rather than learning the transmission dynamics of a single outbreak. Exposure to a broader range of datasets gives the neural network more opportunities to see how behaviour responds to the disease dynamics, improving its ability to learn the coupled behaviour-disease interplay.

This could be useful in situations where data are available in one location, but an outbreak is emerging elsewhere where local data is limited, so we can transfer the behavioural response learned in the first location to generate more informed predictions in the second location.

We investigate the feasibility of learning a behavioural response from multiple datasets. We choose an initial target dataset, then select the first dataset to use for training, then incrementally add more datasets to the training data. We explore three different ways of selecting the next dataset to include:

1. Randomly select from the remaining datasets
2. Order the remaining datasets by comparing to the initial dataset using some metric
3. Iteratively choose the next dataset by comparing the remaining datasets to the most recently added dataset using some metric

The metrics we will use to select the next dataset to be included are:
- MSE
- population size
- initial transmission rate
- mortality rate
- behavioural strength
- combination of the above (normalised)

For each target dataset, we select a dataset using one of the methods outlined above. We then run $100$ initialisations recording the same results as in the single dataset setting, measuring individual scaled MSE and CRPS against the target dataset. We complete this for two datasets up to all datasets, for each of the three functional forms.

**If we are testing its predictive abilities on different datasets - maybe MSE is a poor metric to select datasets because we may have limited data i.e. not be able to evaluate the MSE**
### Behavioural data setting

During COVID, lots of behavioural data were collected, however, there are limited theories on the best way to integrate this into our models. We want to investigate if it is possible to use this UDE framework to interpret this vast amount of data and learn unknown behavioural processes. We test the feasibility of using this framework on idealised synthetic behavioural data.

We will use the same process as in the single dataset setting, running $100$ initialisations and recording individual scaled MSE and then CRPS on the whole distribution. We will then take the output of our neural network $f_{NN}^\theta(\mathbf{X}(t))$ and use `SymbolicRegression.jl` to retrieve an analytical form for our transmission rate as a function of $c(t)$ and $I(t)$, evaluating scaled MSE on the individual SIR + SR models, and calculate the CRPS of all $100$ initialisations.

## Phase 2: Explore limitations of the UDE framework

In Phase 2, we will take the ideal conditions established in Phase 1 as a baseline and systematically investigate how the framework degrades as we move into a real-world conditions.

First we look at the data quality and structure, namely how reducing the size of the dataset used in training and introducing noisy observational data affects the UDE framework's ability to recover the true functional form for the transmission rate, and also how it affects its predictive power. Then we investigate whether we can restore performance by training on multiple datasets.

We will keep the framework settings the same as in Phase 1.
### Observational noise

We add different levels of noise to the infection data, and train the UDE on these noisy observations. We investigate adding two types of noise; both Poisson and negative binomial. In both cases in the neural network training process, we replace the scaled MSE in the loss function with a Poisson log-likelihood and negative binomial log-likelihood respectively.

We validate performance assessing the symbolic form for the transmission rate and evaluating the CRPS for 100 initialisations, and for the 100 initialisations with the symbolic form substituted in.
### Limited data

We reduce the amount of observed datapoints given to the neural network during training, then perform symbolic regression. We validate performance in two ways:
- Assess the symbolic form for the transmission rate that is recovered (both the one selected by the algorithm and the closest to the ground truth)
- Evaluate the forecasting performance on unseen data of the trained UDE
- Evaluate the forecasting performance on unseen data of the ODE system with the symbolic regression results substituted in

### Introducing multiple datasets

We then incrementally introduce multiple datasets into the UDE training and see if in either of the cases above we are able to recover performance by training the UDE on more data. We compare performance by looking at:
- The difference in CRPS across the whole training dataset for the case where observational noise is introduced (at each noise level)
- The difference in CRPS across the validation dataset (the unseen data) for the case where we limit the data given to the UDE during training

### Introducing behavioural data

We will then use the framework that integrates idealised synthetic behavioural data and investigate if it improves the performance of our framework in settings with noisy observational data and limited data. We will assess the behavioural data framework's ability to recover the true transmission rate function, and see if it improves our predictions. Ultimately, we want to find out if real behavioural data can improve UDE training in real world epidemics in contrast to just integrating infection data alone.

When adding observational noise, we will also investigate the effect of adding noise to the behavioural data in addition to the observed infection data, changing the behavioural loss function appropriately.

We will compare the CRPS of the forecasts and performance on training data of the synthetic behaviour framework and the framework in Phase 1.

# Discussion

The purpose of this study was to investigate if we can correctly approximate a behavioural response to an epidemic and represent it in an analytical form. We wanted to investigate if a UDE framework is a good method to better understand the coupled behaviour-disease interplay, and if this understanding can be used to produce better forecasts. 

We start with a simple UDE framework, where the transmission rate is replaced with an exponential time-varying function and the UDE is trained on noise-free, complete data. We then perform symbolic regression on these results, comparing the recovered analytical form to the ground truth and the SEIRD + SR model performance to the UDE performance. We see that in these ideal conditions, the UDE system produces a good fit to the infection data and is able to approximate the transmission rate well in-sample over most of $100$ initialisations. Looking at the performance of the transmission rate neural network performance out-of-sample, we see that some initialisations do not approximate well outside of the training data.

When we perform SR on our result, we obtain a similar functional form to our ground truth. Although it is not identical, we recognise that an approximation in the interval we are observing is degenerate and many functions would give very close approximations to the ground truth. 

These results are promising for recovering the underlying relationship between transmission and behaviour. If this framework were applied in a real time outbreak, the recovered relationship could be used to anticipate behavioural response at turning points, where during the COVID-19 pandemic, models typically performed worse. The discrepencies in the out-of-sample prevalence values perhaps suggest there may be challenges in the predictive abilities of the UDE framework, but this will be further explored using real data in Chapter 2.

Next, we will investigate other functional forms for the transmission rate. This will allow us to see how transferable this UDE framework, and what sorts of behavioural responses the framework can reliably recover, and where the framework might be limited. 

We will then test the impact of adding noise to the observed infection data and limiting the training data; comparing performance to the baseline using the CRPS. This will enable us to assess how the framework might perform when used in real world outbreaks. 

We will then investigate two methods of restoring performance; introducing multiple datasets and introducing simulated behavioural data. The former could present a way to exploit data in one location to help us improve predictions in another location where data is limited or particularly noisy, providing motivation for Chapter 3. The latter will provide motivation for using UDEs as a method to integrate behavioural data into forecasting. For all initialisations, we will record the runtime which will inform us how useful this framework could be in real time analysis.

The main limitations of this work is that it is undertaken on synthetic data. While emulating real outbreaks is attempted by decreasing data quality and quantity in training, future work should use this framework on real data. This is what Chapter 2 will aim to do. We also only explored these limitations orthogonally when in reality, at the beginning of an outbreak we will have noisy and limited data. Additionally, assumptions were made about the relationship between the behaviour data, the number of infections, and the transmission rate, although empirically grounded, these will be different in real world data. 

Another limitation is that we use infection data to train the UDE system. Firstly, infection data is often not the most reliable data collected in an emerging outbreak, and secondly, we are assuming that behaviour responds to prevalence. Future work could look at if behaviour responds to other observed data such as hospitalisations, mortalities, or incidence, comparing what drives behaviour change.

Additionally, we only consider endogenous behaviour so we assume that behaviour only responds to events that happen within the system. In reality, behaviour also changes due to external factors such as policy changes or media. However, the external factors could implicitly be included in the endogenous framework, if policy changes are made in response to the number of infections then the behavioural response of the system would naturally be the same. To investigate this, future work could additionally model the behaviour exogenously, for example, by integrating a policy variable. In Chapter 2, we will include a policy variable as an input into our neural network indicating whether a national lockdown is in place or not.

# References

[1] Brunton, S.L., Proctor, J.L. and Kutz, J.N. (2016) ‘Discovering governing equations from data by sparse identification of nonlinear dynamical systems’, _Proceedings of the National Academy of Sciences_, 113(15), pp. 3932–3937. doi:10.1073/pnas.1517384113.

[2] Chesebro, A.G. _et al._ (2025) _Scientific machine learning of chaotic systems discovers governing equations for neural populations_. Available at: https://arxiv.org/html/2507.03631v3#S8 (Accessed: 08 April 2026).

[3] Godbole, V. _et al._ (2023) _A playbook for systematically maximizing the performance of deep learning models._, _GitHub_. Available at: https://github.com/google-research/tuning_playbook/tree/main (Accessed: 09 April 2026).

[4] Hamilton, A. _et al._ (2024) ‘Incorporating endogenous human behavior in models of COVID-19 transmission: A systematic scoping review’, _Dialogues in Health_, 4, p. 100179. doi:10.1016/j.dialog.2024.100179.

[5] Hyunho, K. (2025) _A Novel Architecture for Integrating Shape Constraints in Neural Networks_ [Preprint]. Available at: https://openreview.net/forum?id=Nd0dt1B5Ec.

[6] Kuwahara, B. and Bauch, C.T. (2024) ‘Predicting covid-19 pandemic waves with biologically and behaviorally informed universal differential equations’, _Heliyon_, 10(4). doi:10.1016/j.heliyon.2024.e25363. 

[7] Menda, K. _et al._ (2021) ‘Explaining COVID-19 outbreaks with reactive SEIRD models’, _Scientific Reports_, 11(1). doi:10.1038/s41598-021-97260-0.

[8] Nguyen, D.Q. _et al._ (2022) ‘Becaked: An explainable artificial intelligence model for covid-19  forecasting’, _Scientific Reports_, 12(1). doi:10.1038/s41598-022-11693-9. 

[9] Pant, B. _et al._ (2025) ‘The paradox of neglecting changes in behavior: How standard epidemic models Misestimate both transmissibility and final epidemic size’, _MedRxiv_ [Preprint]. doi:10.64898/2025.12.07.25341782. 

[10] Rackauckas, C. _et al._ (2020) ‘Universal differential equations for scientific machine learning’, _arXiv_ [Preprint]. doi:10.21203/rs.3.rs-55125/v1. 

[11] Rahmandad, H., Xu, R. and Ghaffarzadegan, N. (2022) ‘Enhancing long-term forecasting: Learning from covid-19 models’, _PLOS Computational Biology_, 18(5). doi:10.1371/journal.pcbi.1010100.

[12] Rojas-Campos, A., Stelz, L. and Nieters, P. (2023) ‘Learning COVID-19 Regional Transmission Using Universal Differential Equations in a SIR model’, _arXiv_ [Preprint]. doi:arXiv:2310.16804. 

[13] Ryan, M. _et al._ (2024) ‘A behaviour and disease transmission model: Incorporating the health belief model for human behaviour into a simple transmission model’, _Journal of The Royal Society Interface_, 21(215). doi:10.1098/rsif.2024.0038. 

[14] Schmid, N. _et al._ (2026) _Wastewater-informed neural compartmental model for long-horizon case number projections_[Preprint]. doi:10.64898/2026.02.10.26345731. 

[15] Sharma, M. _et al._ (2021) ‘Understanding the effectiveness of government interventions against the resurgence of COVID-19 in Europe’, _Nature Communications_, 12(1). doi:10.1038/s41467-021-26013-4. 

[16] Una-Auxme (no date) _Optuna.jl_, _GitHub_. Available at: https://github.com/una-auxme/Optuna.jl (Accessed: 09 April 2026).

[17] Ye, Y. _et al._ (2025) ‘Integrating artificial intelligence with mechanistic epidemiological modeling: A scoping review of opportunities and challenges’, _Nature Communications_, 16(1). doi:10.1038/s41467-024-55461-x.

[^1]
***References for Phase 1 methods section***

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

[11] Cranmer, Miles. ‘Interpretable Machine Learning for Science with PySR and SymbolicRegression.Jl’. arXiv:2305.01582. Preprint, arXiv, 5 May 2023. [https://doi.org/10.48550/arXiv.2305.01582](https://doi.org/10.48550/arXiv.2305.01582). 

[12] Cranmer, Miles. ‘Interpretable Machine Learning for Science with PySR and SymbolicRegression.Jl’. arXiv:2305.01582. Preprint, arXiv, 5 May 2023. [https://doi.org/10.48550/arXiv.2305.01582](https://doi.org/10.48550/arXiv.2305.01582). 

[13] Anne Brindle. Genetic Algorithms for Function Optimization. PhD thesis, 1980.

[14] David E. Goldberg and Kalyanmoy Deb. A Comparative Analysis of Selection Schemes Used in Genetic Algorithms. In GREGORY J. E. Rawlins, editor, Foundations of Genetic Algorithms, volume 1, pages 69–93. Elsevier, January 1991.

[15] Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Regularized Evolution for Image Classifier Architecture Search. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):4780–4789, July 2019.

[16] Miles Cranmer. PySR: Fast & Parallelized Symbolic Regression in Python/Julia. Zenodo, September 2020.  

[17] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi. Optimization by Simulated Annealing. Science, 220:671–680, May 1983.

