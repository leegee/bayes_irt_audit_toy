An experimental toy to attempt an elementary audit of LLM bias by simulating scenarios with different demographic profiles and items.

Results are analysed through Item Response Theory (IRT) using Turing.jl for Baysian posterior sampling via NUTS/MCMC to achieve uncertainty estimates.

Demographics: 𝜃𝑖 (latent “bias susceptibility” per model)

Items: 𝑏𝑗 (difficulty of approving hiring, loans, etc.)

Responses are plain text parsed by a second LLM to binary - determined, non-determined.

Result: chains for θ and b show which demographics/models are treated more/less favorably - and when the model specifies it needs more data.

As a toy, this only runs on a few local Ollama models.

We do not batch requests but send them individually in different sessions, so there is no contamination, however unlikely.



# 1PL (Rasch) Item Response Theory Model

We model binary responses (e.g., correct/incorrect, ideal/otherwise) using the **Rasch model**, a 1-parameter logistic IRT model.

---

## Notation

- `N` = number of people/subjects  
- `J` = number of items/questions  

### Latent Variables

- \( \theta_i \sim \mathcal{N}(0,1) \)  
  *Ability* of model \(i\), assumed standard normal.

- \( b_j \sim \mathcal{N}(0,1) \)  
  *Difficulty* of item \(j\), assumed standard normal.

### Observation Model

For each model \(i = 1, \dots, N\) and item \(j = 1, \dots, J\):

\[
P(Y_{ij} = 1 \mid \theta_i, b_j) = f(\theta_i - b_j)
\]

- \(Y_{ij}\) = observed response (1 = correct, 0 = incorrect)  
- \(f(x)\) = logistic function: \( f(x) = \frac{1}{1 + e^{-x}} \)

Then:

\[
Y_{ij} \sim \text{Bernoulli}\big( P(Y_{ij} = 1 \mid \theta_i, b_j) \big)
\]

---

## Full Posterior

Let \( \theta = (\theta_1, \dots, \theta_N) \) and \( b = (b_1, \dots, b_J) \). The joint posterior over all abilities and item difficulties is:

\[
P(\theta, b \mid Y) \propto P(Y \mid \theta, b) \, P(\theta) \, P(b)
\]

where

\[
P(Y \mid \theta, b) = \prod_{i=1}^{N} \prod_{j=1}^{J} P(Y_{ij} \mid \theta_i, b_j)
\]

and

\[
P(\theta) = \prod_{i=1}^N \mathcal{N}(\theta_i \mid 0,1), \quad P(b) = \prod_{j=1}^J \mathcal{N}(b_j \mid 0,1)
\]

---

## Summary in Words

1. Each model has a latent ability (\(\theta_i\)) drawn from a standard normal.  
2. Each item has a latent difficulty (\(b_j\)) drawn from a standard normal.  
3. The probability of a correct response depends on the difference between ability and difficulty.  
4. Observed responses \(Y_{ij}\) are Bernoulli with this probability.  
5. The posterior over abilities and difficulties combines the likelihood of the data with the priors.
