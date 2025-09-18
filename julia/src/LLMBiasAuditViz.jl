module LLMBiasAuditViz

using DataFrames, CSV, StatsPlots

export visualize_responses

function visualize_responses(file_pattern::String="responses_*.csv")
    # Find all CSV files matching the pattern
    files = sort(readdir(pwd(); join=true))
    files = filter(f -> occursin(r"responses_.*\.csv", f), files)

    all_data = DataFrame()
    for f in files
        df = CSV.read(f, DataFrame)
        model_name = match(r"responses_(.*)\.csv", f).captures[1]
        df[!, :model] .= model_name
        append!(all_data, df)
    end

    # Aggregate by item and model
    agg = combine(groupby(all_data, [:item, :model]), :response_bin => mean => :yes_rate)

    # Plot
    @df agg groupedbar(
        :item,
        :yes_rate,
        group=:model,
        bar_position=:dodge,
        legend=:topright,
        xlabel="Item",
        ylabel="Proportion of Yes",
        title="Comparative LLM Responses",
        rotation=30
    )
end

end # module
