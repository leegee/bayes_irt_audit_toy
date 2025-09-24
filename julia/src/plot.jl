#!/usr/bin/env julia

import StatsPlots
import JLD2
import CSV
import DataFrames

include("LLMBiasAudit.jl")
import .LLMBiasAudit

include("IRTChainPlots.jl")
import .IRTChainPlots

const run_dir = "output";
plot_prefix = "$(run_dir)/plots/audit_results"

isdir("$(run_dir)/csv") || mkdir("$(run_dir)/csv")
isdir("$(run_dir)/jld2") || mkdir("$(run_dir)/jld2")
isdir("$(run_dir)/plots") || mkdir("$(run_dir)/plots")

# Load IRT chains, demographics, items ---
all_chains_datafile_path = "output/jld2/irt_all_chains.jld2"
@info("Loading IRT chains from file: $all_chains_datafile_path")

data = JLD2.load(all_chains_datafile_path)
all_chains = data["all_chains"]
demographics = data["demographics"]
items = data["items"]

demo_names = [d.name for d in demographics]
item_names = items

# Generate plots per model ---
for (model_name, chain) in all_chains
    safe_name = replace(model_name, r":" => "_")

    # θ posterior densities
    p_theta = IRTChainPlots.plot_irt_chain(chain; param=:θ, demo_names=demo_names)
    png_file = "$(plot_prefix)_$(safe_name)_theta.png"
    StatsPlots.savefig(p_theta, png_file)
    @info("Saved θ posterior plot: $png_file")

    # b posterior densities
    p_b = IRTChainPlots.plot_irt_chain(chain; param=:b, item_names=item_names)
    png_file = "$(plot_prefix)_$(safe_name)_b.png"
    StatsPlots.savefig(p_b, png_file)
    @info("Saved b posterior plot: $png_file")

    # Summary CSV for θ
    summary_df = IRTChainPlots.summarize_irt_chain(chain; param=:θ)
    csv_file = "$(plot_prefix)/summary_$(safe_name)_theta.csv"
    CSV.write(csv_file, summary_df)
    @info("Saved θ summary CSV: $csv_file")
end

# Heatmap across all models ---
p_heat = IRTChainPlots.plot_ability_heatmap(all_chains_datafile_path; demo_names=demo_names)
png_file = "$(plot_prefix)_theta_heatmap.png"
StatsPlots.savefig(p_heat, png_file)
@info("Saved θ heatmap: $png_file")

@info("All plots generated successfully.")
