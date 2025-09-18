module LLMBiasAuditViz

using CSV
using DataFrames
using Plots
using StatsPlots

export visualize_responses

"""
    visualize_responses(csv_file::String; outdir::String="plots")

Reads the CSV output from LLMBiasAudit and plots:
1. Bar chart of 0/1 responses by demographic and item
2. Heatmap of the response matrix

Both plots are saved as PNG files in `outdir`.
"""
function visualize_responses(csv_file::String; outdir::String="plots")
    # Create output directory if it doesn't exist
    isdir(outdir) || mkpath(outdir)

    # Load CSV
    df = CSV.read(csv_file, DataFrame)

    # Add a label
    df.demographic_label = string.(df.demographic, " - ", df.name)

    # Short item labels for plotting
    item_labels = ["Managerial job", "Loan", "University"]

    # Ensure categorical ordering
    demographics = unique(df.name)
    items = unique(df.item)

    # --- Bar chart ---
    bar_plot = @df df groupedbar(:demographic_label, :response,
        group=:item,
        bar_position=:dodge,
        xlabel="Demographic",
        ylabel="Response (0=No, 1=Yes)",
        legend=:topright,
        title="Responses by Demographic and Item"
    )

    savefig(bar_plot, joinpath(outdir, "responses_bar.png"))
    println("Grouped bar chart saved to $(joinpath(outdir, "responses_bar.png"))")

    # --- Heatmap ---
    # reshape response vector into n_demo × n_items
    demographics_labels = unique(df.demographic_label)
    matrix = reshape(df.response, length(demographics_labels), length(items))

    heat_plot = heatmap(matrix,
        xlabel="Items",
        ylabel="Demographics",
        xticks=(1:length(items), item_labels),
        yticks=(1:length(demographics_labels), demographics_labels),
        xrotation=30,
        c=:blues,
        size=(800, 400),
        title="Responses Heatmap"
    )

    savefig(heat_plot, joinpath(outdir, "responses_heatmap.png"))
    println("Heatmap saved to $(joinpath(outdir, "responses_heatmap.png"))")
end

end # module
