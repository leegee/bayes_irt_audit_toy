module IRT

import Turing
import Distributions
import StatsFuns
import Base.Threads

export fit_irt_model

function fit_irt_model(response_matrix::Matrix{Int})
    n_demo, n_items_total = size(response_matrix)

    Turing.@model function irt_model(response_matrix)
        θ ~ Turing.MvNormal(zeros(n_demo), ones(n_demo))
        b ~ Turing.MvNormal(zeros(n_items_total), ones(n_items_total))
        logit_p = θ .- transpose(b)
        p = @. StatsFuns.logistic(logit_p)
        for i in 1:n_demo, j in 1:n_items_total
            response_matrix[i, j] ~ Distributions.Bernoulli(p[i, j])
        end
    end

    model = irt_model(response_matrix)
    n_chains = Threads.nthreads() > 1 ? Threads.nthreads() : 4
    chain = Turing.sample(model, Turing.NUTS(0.65), 1000;
        tune=500,
        chains=n_chains,
        progress=true,
        threaded_chains=true)
    return chain
end

end # module
