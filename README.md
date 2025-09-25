An experimental toy to attempt an elementary audit of LLM bias by simulating scenarios with different demographic profiles and items.

Results are analysed through Item Response Theory (IRT) using Turing.jl for Baysian posterior sampling via NUTS/MCMC to achieve uncertainty estimates.

Demographics: 𝜃𝑖 (latent “bias susceptibility” per person)

Items: 𝑏𝑗 (difficulty of approving hiring, loans, etc.)

Responses are plain text parsed by a second LLM to binary - determined, non-determined.

Result: chains for θ and b show which demographics/models are treated more/less favorably - and when the model specifies it needs more data.

    P(correct/yes)=f(θ−b)

    θi ∼ N(0,1), bj ​ ∼ N(0,1)

    Yij ∼ Bernoulli( P(Yij​ = 1∣θi​,bj​))

    P(θ,b∣Y) ∝ P( Y∣θ,b) P(θ)P(b)

As a toy, this only runs on a few local Ollama models.

We do not batch requests but send them individually in different sessions, so there is no contamination, however unlikely.

