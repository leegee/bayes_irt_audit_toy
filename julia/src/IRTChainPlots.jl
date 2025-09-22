module IRTChainPlots

using JLD2
using StatsPlots
using DataFrames
using Statistics

export plot_irt_chain, plot_all_chains, summarize_irt_chain, plot_ability_heatmap

"""
    plot_irt_chain(chain; param=:θ, demo_names=nothing, item_names=nothing)

Plot posterior distributions of a single parameter from a Turing IRT chain.
`param` can be `:θ` (abilities) or `:b` (item difficulties).
`demo_names` and `item_names` are optional labels for the axes.
"""
function plot_irt_chain(chain; param=:θ, demo_names=nothing, item_names=nothing)
    df_chain = DataFrame(chain)
    param_cols = filter(c -> startswith(c, string(param)), names(df_chain))
    n = length(param_cols)

    labels = param == :θ ? (demo_names === nothing ? string.(1:n) : demo_names) :
             (item_names === nothing ? string.(1:n) : item_names)

    p = plot()
    for (i, col) in enumerate(param_cols)
        density!(p, df_chain[!, col], label=string(labels[i]))
    end
    xlabel!(param == :θ ? "Demographics" : "Items")
    ylabel!("Posterior density")
    title!("Posterior distributions of $(param)")
    return p
end

"""
    plot_all_chains(jld2_file; param=:θ, demo_names=nothing, item_names=nothing)

Plot posterior distributions for all models stored in a JLD2 file containing `all_chains`.
"""
function plot_all_chains(jld2_file; param=:θ, demo_names=nothing, item_names=nothing)
    data = JLD2.load(jld2_file)
    all_chains = data["all_chains"]
    n_models = length(all_chains)

    plots_list = []
    for (model_name, chain) in all_chains
        p = plot_irt_chain(chain; param=param, demo_names=demo_names, item_names=item_names)
        plot!(p, title=string(model_name))
        push!(plots_list, p)
    end

    return plot(plots_list..., layout=(n_models, 1), size=(800, 300 * n_models))
end

"""
    summarize_irt_chain(chain; param=:θ)

Return summary statistics (mean, median, std, 2.5% and 97.5% quantiles) for each parameter in the chain.
"""
function summarize_irt_chain(chain; param=:θ)
    df_chain = DataFrame(chain)
    param_cols = filter(c -> startswith(c, string(param)), names(df_chain))

    summary_df = DataFrame(
        parameter=String[],
        mean=Float64[],
        median=Float64[],
        std=Float64[],
        q2_5=Float64[],
        q97_5=Float64[]
    )

    for col in param_cols
        vals = df_chain[!, col]
        push!(summary_df, (
            parameter=col,
            mean=mean(vals),
            median=median(vals),
            std=std(vals),
            q2_5=quantile(vals, 0.025),
            q97_5=quantile(vals, 0.975)
        ))
    end

    return summary_df
end

"""
    plot_ability_heatmap(jld2_file; demo_names=nothing)

Plot a heatmap of mean abilities (θ) across demographics for all models in a JLD2 file.
"""
function plot_ability_heatmap(jld2_file; demo_names=nothing)
    data = JLD2.load(jld2_file)
    all_chains = data["all_chains"]
    n_models = length(all_chains)
    model_names = collect(keys(all_chains))

    # Assume all chains have same number of demographics
    first_chain = first(values(all_chains))
    n_demo = length(filter(c -> startswith(c, "θ"), names(DataFrame(first_chain))))
    labels = demo_names === nothing ? string.(1:n_demo) : demo_names

    heat_matrix = zeros(n_demo, n_models)
    for (j, model_name) in enumerate(model_names)
        chain = all_chains[model_name]
        df_chain = DataFrame(chain)
        θ_cols = filter(c -> startswith(c, "θ"), names(df_chain))
        for (i, col) in enumerate(θ_cols)
            heat_matrix[i, j] = mean(df_chain[!, col])
        end
    end

    heatmap(
        model_names,
        labels,
        heat_matrix,
        xlabel="Model",
        ylabel="Demographic",
        title="Mean abilities (θ) across models",
        color=:viridis,
        right_margin=10Plots.mm
    )
end

end # module
