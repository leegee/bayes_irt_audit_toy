module LLMBiasAuditViz

# Import modules without polluting global namespace
import DataFrames
import CSV
import StatsPlots
import Statistics

export visualize_responses

function visualize_responses(file_pattern::Regex=r"^csv/responses_(.*)\.csv$")
    # Find all CSV files matching the pattern
    files = sort(readdir(pwd(); join=true))
    files = filter(f -> occursin(file_pattern, f), files)

    all_data = DataFrames.DataFrame()
    for file in files
        df = CSV.read(file, DataFrames.DataFrame)
        model_name = match(file_pattern, file).captures[1]
        println("Read $(model_name) from $(file)")
        df[!, :model] .= model_name
        DataFrames.append!(all_data, df)
    end

    # Aggregate by item and model
    agg = DataFrames.combine(
        DataFrames.groupby(all_data, [:item, :model]),
        #:column_name => function => output_column_name
        :response_bin => Statistics.mean => :yes_rate
    )

    # Plot
    StatsPlots.@df agg StatsPlots.groupedbar(
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
