module IRTChainAnalyzer

using JLD2
using MCMCChains
using Statistics
using StatsPlots
using Printf
using Measures

export load_all_and_summarize, plot_posteriors, compare_models, save_summary_and_plots

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

    # Safe for 1 item
    plt = bar(1:n, values_mean, yerr=values_sd,
        xticks=(1:n, labels),
        legend=false, xlabel="", ylabel="Posterior mean ± SD",
        title=title_txt, rotation=45, size=(1000, 400))

    return plt
end

function compare_models(summary::Dict{String,Any}; type::Symbol=:theta, savepath::String="")
    models = collect(keys(summary))

    if type == :theta
        labels = summary[models[1]]["demographics"]
        title_txt = "Comparison of θ (Demographics) across models"
        means = [summary[m]["theta_mean"] for m in models]
        sds = [summary[m]["theta_sd"] for m in models]
    elseif type == :b
        labels = summary[models[1]]["items"]
        title_txt = "Comparison of b (Items) across models"
        means = [summary[m]["b_mean"] for m in models]
        sds = [summary[m]["b_sd"] for m in models]
    else
        error("type must be :theta or :b")
    end

    n_items = length(labels)
    n_models = length(models)

    if n_items == 1 || n_models == 1
        # fallback to regular bar plot
        y = vec(hcat(means...))
        yerr = vec(hcat(sds...))
        xticks_labels = n_models == 1 ? labels : models
        plt = bar(1:length(y), y, yerr=yerr,
            xticks=(1:length(y), xticks_labels),
            xlabel="",
            ylabel="Posterior mean ± SD",
            title=title_txt,
            rotation=45,
            size=(1200, 500),
            label=xticks_labels,
            margin=40mm
        )
    else
        means_matrix = hcat(means...)
        sds_matrix = hcat(sds...)
        plt = groupedbar(1:n_items, means_matrix;
            yerr=sds_matrix,
            xticks=(1:n_items, labels),
            group=models,
            xlabel="",
            ylabel="Posterior mean ± SD",
            title=title_txt,
            bar_width=0.7,
            rotation=45,
            size=(1300, 600),
            margin=40mm
        )
    end

    if !isempty(savepath)
        savefig(plt, savepath)
        @info "Saved comparison plot to $(savepath)"
    end

    return plt
end

function save_summary_and_plots(summary::Dict{String,Any}; prefix::String="")
    # Sanitize prefix
    safe_prefix = sanitize_filename(prefix)

    # Save the summary as JLD2
    filepath = "$(safe_prefix)_summary.jld2"
    JLD2.@save filepath summary
    @info "Saved summary to $(filepath)"

    #for model in keys(summary)
    #    plt = plot_posteriors(summary, model=model, type=:theta)
    #    png("$(safe_prefix)_theta_$(sanitize_filename(model)).png")
    #    plt = plot_posteriors(summary, model=model, type=:b)
    #    png("$(safe_prefix)_b_$(sanitize_filename(model)).png")
    #end

    # Save comparison plots
    compare_models(summary, type=:theta, savepath="$(safe_prefix)_theta_comparison.png")
    compare_models(summary, type=:b, savepath="$(safe_prefix)_b_comparison.png")
end

function sanitize_filename(name::AbstractString)
    replace(name, r"[:\\]" => "_")
end

end # module
