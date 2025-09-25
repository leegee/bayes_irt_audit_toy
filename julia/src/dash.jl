#!/usr/bin/env julia 

include("LLMBiasAudit.jl")
import .LLMBiasAudit

import CSV
import DataFrames: DataFrame, Not, names, nrow, unstack
import PlotlyJS
import Dash
import Logging: @error
import FilePathsBase: mkpath

const OUTPUT_BASE_DIR = "output"
mkpath(OUTPUT_BASE_DIR)
mkpath(joinpath(OUTPUT_BASE_DIR, "csv"))
mkpath(joinpath(OUTPUT_BASE_DIR, "plots"))

safe_filename(model_name::String) = replace(model_name, r"[^\w]" => "_")

# Load CSV for a model
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

# Create PlotlyJS heatmap figure
function plotly_heatmap_figure(df::DataFrame, model_name::String)
    if nrow(df) == 0
        return PlotlyJS.Plot(PlotlyJS.Layout(title="No data"))
    end

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

    PlotlyJS.Plot(trace, PlotlyJS.Layout(
        title="Binary Classification Heatmap: $(model_name)",
        template="plotly_dark"
    ))
end

# Run single Dash app for all models
function run_dash_app_all_models(model_dfs::Dict{String,DataFrame})
    app = Dash.dash()

    app.layout = Dash.html_div() do
        [
            Dash.dcc_dropdown(
                id="model-selector",
                options=[Dict("label" => m, "value" => m) for m in keys(model_dfs)],
                value=first(keys(model_dfs)),
                placeholder="Select Model"
            ),
            Dash.dcc_dropdown(
                id="demo-filter",
                options=[],  # updated dynamically
                multi=true,
                placeholder="Select Demographics"
            ),
            Dash.dcc_dropdown(
                id="item-filter",
                options=[],  # updated dynamically
                multi=true,
                placeholder="Select Items"
            ),
            Dash.html_div(id="table-container"),
            Dash.dcc_graph(id="heatmap-graph")
        ]
    end

    # Update dropdown options when model changes
    Dash.callback!(app,
        Dash.Output("demo-filter", "options"),
        Dash.Output("item-filter", "options"),
        Dash.Input("model-selector", "value")
    ) do selected_model
        df = model_dfs[selected_model]
        demo_opts = [Dict("label" => d, "value" => d) for d in unique(df.demographic)]
        item_opts = [Dict("label" => i, "value" => i) for i in unique(df.item_text)]
        return demo_opts, item_opts
    end

    # Update table + heatmap when any filter changes
    Dash.callback!(app,
        Dash.Output("table-container", "children"),
        Dash.Output("heatmap-graph", "figure"),
        Dash.Input("model-selector", "value"),
        Dash.Input("demo-filter", "value"),
        Dash.Input("item-filter", "value")
    ) do selected_model, selected_demos, selected_items
        df = model_dfs[selected_model]

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

        table_component = Dash.html_table(
            vcat(
                [Dash.html_tr([Dash.html_th(col) for col in names(filtered)])],
                [Dash.html_tr([Dash.html_td(filtered[row, col]) for col in names(filtered)])
                 for row in 1:nrow(filtered)]
            )
        )

        heatmap_fig = plotly_heatmap_figure(filtered, selected_model)

        return table_component, heatmap_fig
    end

    Dash.run_server(app, "127.0.0.1"; debug=true)
end

# Main function
function main(; classified::Bool=true)
    model_dfs = Dict{String,DataFrame}()
    for model in LLMBiasAudit.default_models
        df = load_model_csv(model; classified=classified)
        if nrow(df) > 0
            model_dfs[model] = df
        else
            @error "Skipping model due to missing CSV" model
        end
    end

    if isempty(model_dfs)
        @error "No data available for any model."
        return
    end

    println("Launching interactive Dash app for all models")
    run_dash_app_all_models(model_dfs)
end

main()
