# A Toy in Julia for Psychometric Analysis of Bias in LLM

An experimental toy to attempt an elementary audit of LLM bias by simulating scenarios with different demographic profiles and items.

Results are analysed through Item Response Theory (IRT) using Turing.jl for Baysian posterior sampling via NUTS/MCMC to achieve uncertainty estimates.

Demographics: 𝜃𝑖 (latent “bias susceptibility” per model)

Items: 𝑏𝑗 (difficulty of approving hiring, loans, etc.)

Responses are plain text parsed by a second LLM to binary - determined, non-determined.

Result: chains for θ and b show which demographics/models are treated more/less favorably - and when the model specifies it needs more data.

As a toy, this only runs on a few local Ollama models.

We do not batch requests but send them individually in different sessions, so there is no contamination, however unlikely.

---

# 1PL (Rasch) Item Response Theory Model

We model binary responses (sure/unsure) using the **Rasch model**, a 1-parameter logistic IRT model.

---

## Notation

- N = number of people/subjects  
- J = number of items/questions  

### Latent Variables

- 𝜃_i ~ N(0,1)  
  *Ability* of model i, assumed standard normal.

- b_j ~ N(0,1)  
  *Difficulty* of item j, assumed standard normal.

### Observation Model

For each model i = 1,…,N and item j = 1,…,J:

P(Y_ij = 1 | 𝜃_i, b_j) = f(𝜃_i - b_j)

- Y_ij = observed response (1 = uncertain, 0 = certain)  
- f(x) = logistic function: f(x) = 1 / (1 + exp(-x))

Then:

Y_ij ~ Bernoulli(P(Y_ij = 1 | 𝜃_i, b_j))

---

## Full Posterior

Let 𝜃 = (𝜃_1, …, 𝜃_N) and b = (b_1, …, b_J). The joint posterior over all abilities and item difficulties is:

P(𝜃, b | Y) ∝ P(Y | 𝜃, b) * P(𝜃) * P(b)

Where:

- P(Y | 𝜃, b) = product over i=1..N, j=1..J of P(Y_ij | 𝜃_i, b_j)  
- P(𝜃) = product over i=1..N of N(𝜃_i | 0,1)  
- P(b) = product over j=1..J of N(b_j | 0,1)

---

## Summary in Words

1. Each model has a latent ability (𝜃_i) drawn from a standard normal.  
1. Each item has a latent difficulty (b_j) drawn from a standard normal.  
1. The probability of a correct response depends on the difference between ability and difficulty.  
1. Observed responses Y_ij are Bernoulli with this probability.  
1. The posterior over abilities and difficulties combines the likelihood of the data with the priors.


## To Do 

 **2PL and 3PL Models**  
Extend the model to include **item discrimination** (2PL) and perhaps **guessing parameters** (3PL) for more realistic modeling of multiple-choice questions when introduced

**Covariates for Persons or Items**  
Include external information (e.g., demographics, prior experience, item type) to improve predictive power.

**Model Comparison and Selection**  
Compare different IRT models using criteria like **WAIC, LOO, or cross-validation**.

And so much more, particularly better reporting.
