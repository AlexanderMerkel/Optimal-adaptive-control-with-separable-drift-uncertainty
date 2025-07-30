# Literature Review: Liquidity Regimes and Optimal Execution

## Directly Relevant Recent Papers (2020-2025)

### Optimal Execution with Learning

**"Optimal Execution under Liquidity Uncertainty"** (2025)  
*ArXiv:2506.11813*  
Chevalier, E., Hafsi, Y., Ly Vath, V., & Pulido, S.  
Studies optimal trading execution with price impact, market resilience, and stochastic liquidity variations using Hamilton-Jacobi-Bellman inequalities.

**"Optimal Portfolio Execution in a Regime-switching Market with Non-linear Impact Costs: Combining Dynamic Program and Neural Network"** (2023)  
*ArXiv:2306.08809*  
Li, X. & Mulvey, J.M.  
Proposes a four-step numerical framework combining dynamic programming and neural networks for portfolio execution with regime switching and non-linear impact costs.

### Market Microstructure and Regime Models

**"Optimal execution with regime-switching market resilience"** (2019)  
*Journal of Economic Dynamics and Control* 101(C): 17-40  
Siu, C.C., Guo, I., Zhu, S.P., & Elliott, R.J.  
Provides closed-form solution for optimal execution when market resilience rate follows a finite-state Markov chain, extending the Obizhaeva-Wang model.

**"Understand funding liquidity and market liquidity in a regime‐switching model"** (2023)  
*International Journal of Finance & Economics* 28(1): 589-605  
Chen, L., Shen, L., & Zhou, Z.  
Demonstrates that funding and market liquidity exhibit regime-dependent interactions with large positive mutual impact during tight money market conditions.

**"Long Time Behavior of Optimal Liquidation Problems"** (2024)  
*Mathematics and Financial Economics* (2025) / *ArXiv:2405.14177*  
Cheng, X., Fu, G., & Xia, X.  
Investigates infinite-horizon liquidation dynamics, revealing that external flows can prevent complete asset liquidation contrary to finite-time scenarios.

### Portfolio Optimization with Regime Switching

**"Liquidity Regimes and Optimal Dynamic Asset Allocation"** (2018/2020)  
*NBER Working Paper No. 24222 / Journal of Financial Economics* 136(2): 379-406  
Collin-Dufresne, P., Daniel, K.D., & Saǧlam, M.  
Solves portfolio choice with regime-switching expected returns, volatilities, and trading costs, finding optimal policy trades toward weighted-average portfolios with higher speed in persistent, risky, high-liquidity states.

### Hidden Markov Models and Filtering

**"Bitcoin Price Regime Shifts: A Bayesian MCMC and Hidden Markov Model Analysis"** (2025)  
*[Journal not specified in search results]*  
Authors: [Authors not retrieved in search]  
Integrates Bayesian MCMC covariate selection within HMMs to analyze regime-switching dynamics in cryptocurrency markets.

**"Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States"** (2023)  
*[Journal not specified in search results]*  
Authors: [Authors not retrieved in search]  
Addresses joint estimation of dynamics and latent states, making uncertainty quantification challenging in optimal control contexts.

## Classical Foundation Papers (Still Highly Relevant)

### Price Impact Models

**Almgren, R. and Chriss, N.** (2001)  
**"Optimal execution of portfolio transactions"**  
*Journal of Risk* 3(2): 5-39  
Establishes fundamental framework for optimal execution with linear temporary and permanent price impact, serving as baseline for most subsequent work.

**Obizhaeva, A.A. and Wang, J.** (2013)  
**"Optimal trading strategy and supply/demand dynamics"**  
*Journal of Financial Markets* 16(1): 1-32  
Introduces transient price impact model with exponential resilience that forms mathematical foundation for many regime-switching extensions.

### Filtering Theory Applications

**Wonham, W.M.** (1965)  
**"Some applications of stochastic differential equations to optimal nonlinear filtering"**  
*Journal of the Society for Industrial and Applied Mathematics Series A Control* 2(3): 347-369  
Develops fundamental filtering equations for estimating hidden state parameters from noisy observations, providing mathematical foundation for regime learning.

## Methodological Papers

### Deep Learning for Finance

**Beck, C., Becker, S., Grohs, P., Jaafari, N. and Jentzen, A.** (2021)  
**"Solving the Kolmogorov PDE by means of deep learning"**  
*Journal of Scientific Computing* 88(3): 1-28  
Demonstrates neural network approaches for solving high-dimensional PDEs, relevant for Deep Galerkin Method implementation in optimal execution.

**Sirignano, J. and Spiliopoulos, K.** (2018)  
**"DGM: A deep learning algorithm for solving partial differential equations"**  
*Journal of Computational Physics* 375: 1339-1364  
Introduces Deep Galerkin Method for solving PDEs using neural networks, directly applicable to HJB equations in optimal execution problems.

### Reinforcement Learning in Finance

**Mnih, V., Kavukcuoglu, K., Silver, D., et al.** (2015)  
**"Human-level control through deep reinforcement learning"**  
*Nature* 518: 529-533  
Foundational paper for Deep Q-Networks, providing algorithmic basis for RL approaches to optimal execution problems.

## Recent Reinforcement Learning Applications (Note: Limited Access)

**Search Note on RL Papers**: While several recent papers on reinforcement learning for optimal execution with time-varying liquidity were mentioned in research (2024-2025), full bibliographic details could not be retrieved through web search. Key themes identified include:

- **Deep Q-learning approaches** for non-stationary market conditions
- **Multi-agent frameworks** accounting for market impact and bid-ask spreads  
- **Transformer-based methods** incorporating real-time liquidity signals
- **Adaptive execution strategies** for intraday liquidity variations

## Search Limitations and Honest Assessment

**Important Note**: This literature review combines academic search results with systematic web searching. Key limitations:

1. **Access restrictions**: Some journal articles behind paywalls could not be fully accessed
2. **Very recent publications**: 2025 papers may be preprints or early-stage publications  
3. **Industry research**: Proprietary research from trading firms not publicly available
4. **Conference papers**: Working papers from recent conferences may be missing

**What's well-documented**: Classical foundations (Almgren-Chriss, Obizhaeva-Wang), established regime-switching models, NBER/journal publications have complete citations.

**Coverage Gaps**: 
- Detailed RL literature for execution (active area but access limited)
- Very recent algorithmic trading research (often proprietary)
- Non-English publications
- Specialized practitioner journals

**Recommendation**: Use as comprehensive starting point, but supplement with institutional database access (Bloomberg Terminal, FactSet, institutional library access) for complete industry coverage.