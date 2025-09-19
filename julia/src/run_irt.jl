#!/usr/bin/env julia

include("LLMBiasAudit.jl")
using .LLMBiasAudit

include("IRTChainAnalyzer.jl")
using .IRTChainAnalyzer

# Load chains, demographics, items, and get summary
summary, demographics, items = IRTChainAnalyzer.load_all_and_summarize("jld2/irt_all_chains.jld2")

# Inspect results for a specific model
# model = "phi3:latest"
# summary[model]["theta_mean"]    # posterior means for demographics
# summary[model]["theta_sd"]      # posterior SDs for demographics
# summary[model]["b_mean"]        # posterior means for items
# summary[model]["b_sd"]          # posterior SDs for items
# summary[model]["demographics"]  # names of demographics
# summary[model]["items"]          # the original items

# Plot and save individual model posteriors and comparison plots
IRTChainAnalyzer.save_summary_and_plots(summary; prefix="plots/audit_results")

