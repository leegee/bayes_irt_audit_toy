#!/usr/bin/env julia

include("LLMBiasAudit.jl")
import .LLMBiasAudit

import CSV
import Dash
import DataFrames: DataFrame, names, nrow, vcat
import FilePathsBase: mkpath
import Logging: @error
import PlotlyJS
import Statistics: mean

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
        return DataFrame()
    end
    CSV.read(file_path, DataFrame)
end

# Binary heatmap (long-format, handles empty table)
function plotly_binary_heatmap(df::DataFrame, model_name::String)
    if nrow(df) == 0
        return PlotlyJS.Plot(PlotlyJS.Layout(
            title="Binary Responses: $(model_name) (no data)",
            template="plotly_dark"
        ))
    end

    hover_text = ["$(df.demographic[i]), $(df.item_text[i]): $(df.response_bin[i])" for i in 1:nrow(df)]

    trace = PlotlyJS.heatmap(
        z=df.response_bin,
        x=df.item_text,
        y=df.demographic,
        text=hover_text,
        hoverinfo="text+z",
        colorscale="RdBu",
        zmin=0, zmax=1
    )

    PlotlyJS.Plot(trace, PlotlyJS.Layout(
        title="Binary Responses: $(model_name)",
        template="plotly_dark",
        yaxis=PlotlyJS.attr(autorange="reversed")
    ))
end

# Ability heatmap from filtered DataFrame (handles missing/empty values)
function plotly_ability_heatmap_from_df(df::DataFrame, model_names::Vector{String})
    if nrow(df) == 0
        return PlotlyJS.Plot(PlotlyJS.Layout(
            title="Mean abilities (θ) (no data)",
            template="plotly_dark"
        ))
    end

    demographics = unique(df.demographic)
    n_demo = length(demographics)
    n_models = length(model_names)

    # Create a matrix of mean response_bin per demo/model
    heat_matrix = Matrix{Union{Float64,Missing}}(undef, n_demo, n_models)
    for (j, model_name) in enumerate(model_names)
        # select df for this model if df has multiple models, else just use df
        # Here we assume df only has one model loaded, so we just copy the same df
        for (i, demo) in enumerate(demographics)
            vals = df.response_bin[df.demographic.==demo]
            heat_matrix[i, j] = isempty(vals) ? missing : mean(vals)
        end
    end

    hover_text = ["$(demographics[i]), $(model_names[j]): $(ismissing(heat_matrix[i,j]) ? "NA" : round(heat_matrix[i,j], digits=2))"
                  for i in 1:n_demo, j in 1:n_models]

    trace = PlotlyJS.heatmap(
        z=heat_matrix,
        x=model_names,
        y=demographics,
        text=hover_text,
        hoverinfo="text+z",
        colorscale="Viridis"
    )

    PlotlyJS.Plot(trace, PlotlyJS.Layout(
        title="Mean abilities (θ) across models",
        template="plotly_dark",
        yaxis=PlotlyJS.attr(autorange="reversed")
    ))
end



# Load all CSVs into a Dict
function load_all_models()
    Dict(model => load_model_csv(model) for model in LLMBiasAudit.default_models)
end

# Dash app
function run_dash_app()
    all_models_data = load_all_models()
    model_names = collect(keys(all_models_data))
    all_models_dropdown = ["All Models"; model_names]  # Add "All Models" option

    app = Dash.dash()

    app.layout = Dash.html_div() do
        [
            Dash.dcc_dropdown(
                id="model-filter",
                options=[Dict("label" => m, "value" => m) for m in all_models_dropdown],
                value=first(all_models_dropdown),
                clearable=false,
                placeholder="Select model"
            ),
            Dash.dcc_dropdown(
                id="demo-filter",
                options=[],
                multi=true,
                placeholder="Select demographics"
            ),
            Dash.dcc_dropdown(
                id="item-filter",
                options=[],
                multi=true,
                placeholder="Select items"
            ),
            Dash.html_div(id="table-container"),
            Dash.dcc_graph(id="binary-heatmap"),
            Dash.dcc_graph(id="ability-heatmap")
        ]
    end

    # Update demo/item dropdowns when model changes
    Dash.callback!(app,
        Dash.Output("demo-filter", "options"),
        Dash.Output("item-filter", "options"),
        Dash.Input("model-filter", "value")
    ) do selected_model
        combined_df = selected_model == "All Models" ? vcat(values(all_models_data)...) :
                      all_models_data[selected_model]

        demo_opts = [
            Dict("label" => "All", "value" => "__ALL__");
            [Dict("label" => d, "value" => d) for d in unique(combined_df.demographic)]
        ]

        item_opts = [
            Dict("label" => "All", "value" => "__ALL__");
            [Dict("label" => i, "value" => i) for i in unique(combined_df.item_text)]
        ]

        return demo_opts, item_opts
    end

    # Update table and heatmaps dynamically
    Dash.callback!(app,
        Dash.Output("table-container", "children"),
        Dash.Output("binary-heatmap", "figure"),
        Dash.Output("ability-heatmap", "figure"),
        Dash.Input("model-filter", "value"),
        Dash.Input("demo-filter", "value"),
        Dash.Input("item-filter", "value")
    ) do selected_model, selected_demos, selected_items
        selected_demos = isnothing(selected_demos) ? String[] : selected_demos
        selected_items = isnothing(selected_items) ? String[] : selected_items

        # Filter the table
        df = selected_model == "All Models" ? vcat(values(all_models_data)...) :
             all_models_data[selected_model]

        models_for_heatmap = selected_model == "All Models" ? collect(keys(all_models_data)) :
                             [selected_model]

        mask = trues(nrow(df))

        if !("__ALL__" in selected_demos) && !isempty(selected_demos)
            mask .&= in.(df.demographic, Ref(selected_demos))
        end
        if !("__ALL__" in selected_items) && !isempty(selected_items)
            mask .&= in.(df.item_text, Ref(selected_items))
        end

        filtered = df[mask, :]

        # Build table component
        table_component = Dash.html_table(
            vcat(
                [Dash.html_tr([Dash.html_th(col) for col in names(filtered)])],
                [Dash.html_tr([Dash.html_td(filtered[row, col]) for col in names(filtered)]) for row in 1:nrow(filtered)]
            )
        )

        # Build heatmaps from filtered data
        binary_fig = plotly_binary_heatmap(filtered, selected_model)
        ability_fig = plotly_ability_heatmap_from_df(filtered, models_for_heatmap)

        return table_component, binary_fig, ability_fig
    end

    Dash.run_server(app, "127.0.0.1"; debug=true)
end

run_dash_app()
