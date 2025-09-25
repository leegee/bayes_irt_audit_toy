#!/usr/bin/env julia 

include("LLMBiasAudit.jl")
import .LLMBiasAudit

import CSV
import DataFrames: DataFrame, unstack, Not, names, nrow
import StatsPlots: heatmap
import PlotlyJS
import Dash
import Logging: @error
import FilePathsBase: mkpath

const OUTPUT_BASE_DIR = "output"
mkpath(OUTPUT_BASE_DIR)
mkpath(joinpath(OUTPUT_BASE_DIR, "csv"))
mkpath(joinpath(OUTPUT_BASE_DIR, "plots"))

safe_filename(model_name::String) = replace(model_name, r"[^\w]" => "_")

function load_model_csv(model_name::String; classified::Bool=true)
    file_name = classified ? "responses_$(safe_filename(model_name)).csv" :
                "responses_$(safe_filename(model_name))_raw.csv"
    file_path = joinpath(OUTPUT_BASE_DIR, "csv", file_name)
    if !isfile(file_path)
        @error "CSV file does not exist" file_path
        return DataFrame()  # return empty DataFrame instead of failing
    end
    CSV.read(file_path, DataFrame)
end

function plot_binary_heatmap(df::DataFrame, model_name::String)
    heatmap_df = unstack(df, :demographic, :item_text, :response_bin)
    mat = Matrix{Union{Missing,Int}}(heatmap_df[:, Not(:demographic)])
    mat = coalesce.(mat, 0)

    heatmap(
        mat,
        xticks=(1:size(mat, 2), names(heatmap_df)[2:end]),
        yticks=(1:size(mat, 1), heatmap_df.demographic),
        c=:coolwarm,
        colorbar_title="Binary Label",
        title="Binary Classification Heatmap: $(model_name)"
    )
end

function plot_interactive_heatmap(df::DataFrame, model_name::String)
    heatmap_df = unstack(df, :demographic, :item_text, :response_bin)
    mat = Matrix{Union{Missing,Int}}(heatmap_df[:, Not(:demographic)])
    mat = coalesce.(mat, 0)

    hover_text = ["$(heatmap_df.demographic[i]), $(names(heatmap_df)[j+1]): $(mat[i,j])"
                  for i in 1:size(mat, 1), j in 1:size(mat, 2)]

    trace = PlotlyJS.heatmap(
        z=mat,
        x=names(heatmap_df)[2:end],
        y=heatmap_df.demographic,
        text=hover_text,
        hoverinfo="text+z",
        colorscale="RdBu"
    )
    PlotlyJS.plot(trace)
end

# Interactive raw text table (Dash)
function run_dash_table(df::DataFrame)
    app = Dash.dash()

    app.layout = Dash.html_div() do
        [
            Dash.dcc_dropdown(
                id="demo-filter",
                options=[Dict("label" => d, "value" => d) for d in unique(df.demographic)],
                multi=true,
                placeholder="Select Demographics"
            ),
            Dash.dcc_dropdown(
                id="item-filter",
                options=[Dict("label" => i, "value" => i) for i in unique(df.item_text)],
                multi=true,
                placeholder="Select Items"
            ),
            Dash.html_div(id="table-container")
        ]
    end

    Dash.callback!(app,
        Dash.Output("table-container", "children"),
        Dash.Input("demo-filter", "value"),
        Dash.Input("item-filter", "value")
    ) do selected_demos, selected_items
        try
            # Default to empty arrays if nothing was selected yet
            selected_demos = isnothing(selected_demos) ? String[] : selected_demos
            selected_items = isnothing(selected_items) ? String[] : selected_items

            mask = trues(nrow(df))
            if !isempty(selected_demos)
                mask .&= in.(df.demographic, Ref(selected_demos))
            end
            if !isempty(selected_items)
                mask .&= in.(df.item_text, Ref(selected_items))
            end
            filtered = df[mask, :]

            Dash.html_table(
                vcat(
                    [Dash.html_tr([Dash.html_th(col) for col in names(filtered)])],
                    [Dash.html_tr([Dash.html_td(filtered[row, col]) for col in names(filtered)])
                     for row in 1:nrow(filtered)]
                )
            )
        catch e
            @error "Dash callback error" exception = (e, catch_backtrace())
            return Dash.html_div("Error in callback: $(e)")
        end
    end

    Dash.run_server(app, "127.0.0.1"; debug=true)
end

function main(; classified::Bool=true)
    for model in LLMBiasAudit.default_models
        df = load_model_csv(model; classified=classified)
        if nrow(df) == 0
            @error "Skipping model due to missing CSV" model
            continue
        end
        println("Plotting static heatmap for ", model)
        plot_binary_heatmap(df, model)
        println("Plotting interactive heatmap for ", model)
        plot_interactive_heatmap(df, model)
        println("Launching interactive table for ", model)
        run_dash_table(df)
    end
end

main()
