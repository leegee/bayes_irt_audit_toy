An experimental toy to attempt an elementary basic audit of LLM decisions by simulating scenarios with different demographic profiles and items.

Uses the local Ollama to test several models against questions from specific demographics.

Results are analysed through Item Response Theory (IRT) and Baysian posterior sampling via NUTS/MCMC to achieve uncertaint estimates.

Demographics: 𝜃𝑖 (latent “bias susceptibility” per person)

Items: 𝑏𝑗 (difficulty of approving hiring, loans, etc.)

Responses: observed 0/1 outcomes from LLMs

Turing.jl performs Bayesian posterior sampling with NUTS/MCMC

Result: chains for θ and b show which demographics/models are treated more/less favorably.

    P(correct/yes)=f(θ−b)

    θi ∼ N(0,1), bj ​ ∼ N(0,1)

    Yij ∼ Bernoulli( P(Yij​ = 1∣θi​,bj​))

    P(θ,b∣Y) ∝ P( Y∣θ,b) P(θ)P(b)

