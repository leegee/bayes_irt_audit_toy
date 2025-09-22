#!/usr/bin/env julia

using StatsPlots
using JLD2
using CSV, DataFrames

# --- Directories ---
isdir("csv") || mkdir("csv")
isdir("jld2") || mkdir("jld2")
isdir("plots") || mkdir("plots")

include("LLMBiasAudit.jl")
using .LLMBiasAudit

include("IRTChainPlots.jl")
using .IRTChainPlots

plot_prefix = "plots/audit_results"

# --- Load IRT chains, demographics, items ---
@info("Loading IRT chains from file: $all_chains_datafile_path")
data = JLD2.load(all_chains_datafile_path)
all_chains = data["all_chains"]
demographics = data["demographics"]
items = data["items"]

demo_names = [d.name for d in demographics]
item_names = items

# --- Generate plots per model ---
for (model_name, chain) in all_chains
    safe_name = replace(model_name, r":" => "_")

    # θ posterior densities
    p_theta = plot_irt_chain(chain; param=:θ, demo_names=demo_names)
    png_file = "$(plot_prefix)_$(safe_name)_theta.png"
    savefig(p_theta, png_file)
    @info("Saved θ posterior plot: $png_file")

    # b posterior densities
    p_b = plot_irt_chain(chain; param=:b, item_names=item_names)
    png_file = "$(plot_prefix)_$(safe_name)_b.png"
    savefig(p_b, png_file)
    @info("Saved b posterior plot: $png_file")

    # Summary CSV for θ
    summary_df = summarize_irt_chain(chain; param=:θ)
    csv_file = "csv/summary_$(safe_name)_theta.csv"
    CSV.write(csv_file, summary_df)
    @info("Saved θ summary CSV: $csv_file")
end

# --- Heatmap across all models ---
p_heat = plot_ability_heatmap(all_chains_datafile_path; demo_names=demo_names)
png_file = "$(plot_prefix)_theta_heatmap.png"
savefig(p_heat, png_file)
@info("Saved θ heatmap: $png_file")

@info("All plots generated successfully.")
