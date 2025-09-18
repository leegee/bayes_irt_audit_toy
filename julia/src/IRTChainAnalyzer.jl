module IRTChainAnalyzer

using JLD2
using MCMCChains
using Statistics
using StatsPlots
using Printf

export load_all_and_summarize, plot_posteriors, compare_models, save_summary_and_plots

# --------------------------
# Load and summarize
# --------------------------
function load_all_and_summarize(filename::String)
    all_chains, demographics, items = JLD2.jldopen(filename, "r") do file
        chains = read(file, "all_chains")
        demos = read(file, "demographics")
        its = read(file, "items")
        return chains, demos, its
    end

    summary = Dict{String,Any}()

    for (model_name, chain) in all_chains
        theta_names = filter(n -> startswith(String(n), "θ["), names(chain))
        b_names = filter(n -> startswith(String(n), "b["), names(chain))

        theta_samples = Array(chain[:, theta_names, :])
        b_samples = Array(chain[:, b_names, :])

        theta_mean = mean(theta_samples, dims=1) |> vec
        theta_sd = std(theta_samples, dims=1) |> vec
        b_mean = mean(b_samples, dims=1) |> vec
        b_sd = std(b_samples, dims=1) |> vec

        summary[model_name] = Dict(
            "theta_mean" => theta_mean,
            "theta_sd" => theta_sd,
            "b_mean" => b_mean,
            "b_sd" => b_sd,
            "demographics" => demographics,
            "items" => items
        )
    end

    return summary, demographics, items
end

# --------------------------
# Single model plot
# --------------------------
function plot_posteriors(summary::Dict{String,Any}; model::String="", type::Symbol=:theta)
    if isempty(model)
        model = first(keys(summary))
    end
    data = summary[model]

    if type == :theta
        values_mean = data["theta_mean"]
        values_sd = data["theta_sd"]
        labels = data["demographics"]
        title_txt = "Posterior θ (Demographics) - $model"
    elseif type == :b
        values_mean = data["b_mean"]
        values_sd = data["b_sd"]
        labels = data["items"]
        title_txt = "Posterior b (Items) - $model"
    else
        error("type must be :theta or :b")
    end

    n = length(values_mean)
    bar(1:n, values_mean, yerr=values_sd, xticks=(1:n, labels),
        legend=false, xlabel="", ylabel="Posterior mean ± SD",
        title=title_txt, rotation=45, size=(1000, 400))
end

# --------------------------
# Compare multiple models
# --------------------------
function compare_models(summary::Dict{String,Any}; type::Symbol=:theta, savepath::String="")
    models = keys(summary)
    if type == :theta
        labels = summary[first(models)]["demographics"]
        title_txt = "Comparison of θ (Demographics) across models"
    elseif type == :b
        labels = summary[first(models)]["items"]
        title_txt = "Comparison of b (Items) across models"
    else
        error("type must be :theta or :b")
    end

    n = length(labels)
    colors = palette(:tab10)
    plot(size=(1200, 500), xlabel="", ylabel="Posterior mean", xticks=(1:n, labels), rotation=45, title=title_txt)

    for (i, model) in enumerate(models)
        if type == :theta
            y = summary[model]["theta_mean"]
        else
            y = summary[model]["b_mean"]
        end
        scatter!(1:n, y, label=model, markersize=6, color=colors[(i-1)%length(colors)+1])
    end

    if !isempty(savepath)
        png(savepath)
        @info "Saved comparison plot to $savepath"
    end

    return current()
end

# --------------------------
# Save summary and plots
# --------------------------
function save_summary_and_plots(summary::Dict{String,Any}; prefix::String="")
    # Save the summary as JLD2
    JLD2.@save "$(prefix)_summary.jld2" summary
    @info "Saved summary to $(prefix)_summary.jld2"

    # Save individual model plots
    for model in keys(summary)
        plot_posteriors(summary, model=model, type=:theta)
        png("$(prefix)_theta_$(model).png")
        plot_posteriors(summary, model=model, type=:b)
        png("$(prefix)_b_$(model).png")
    end

    # Save comparison plots
    compare_models(summary, type=:theta, savepath="$(prefix)_theta_comparison.png")
    compare_models(summary, type=:b, savepath="$(prefix)_b_comparison.png")
end

function compare_models(summary::Dict{String,Any}; type::Symbol=:theta, savepath::String="")
    models = collect(keys(summary))
    if type == :theta
        labels = summary[models[1]]["demographics"]
        title_txt = "Comparison of θ (Demographics) across models"
    elseif type == :b
        labels = summary[models[1]]["items"]
        title_txt = "Comparison of b (Items) across models"
    else
        error("type must be :theta or :b")
    end

    n = length(labels)
    colors = palette(:tab10)
    plt = plot(size=(1200, 500), xlabel="", ylabel="Posterior mean ± SD",
        xticks=(1:n, labels), rotation=45, title=title_txt)

    for (i, model) in enumerate(models)
        if type == :theta
            y = summary[model]["theta_mean"]
            yerr = summary[model]["theta_sd"]
        else
            y = summary[model]["b_mean"]
            yerr = summary[model]["b_sd"]
        end

        # Scatter with error bars
        scatter!(1:n, y, yerr=yerr, label=model, color=colors[(i-1)%length(colors)+1], markersize=6)
    end

    if !isempty(savepath)
        png(savepath)
        @info "Saved comparison plot with error bars to $savepath"
    end

    return plt
end

end # module
