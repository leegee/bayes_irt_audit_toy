#!/usr/bin/env julia

include("LLMBiasAudit.jl")
import .LLMBiasAudit

import CSV
import Dash
import DataFrames: DataFrame, names, nrow, vcat
import FilePathsBase: mkpath
import JLD2
import Logging: @error
import PlotlyJS
import Statistics: mean

const OUTPUT_BASE_DIR = "output"
mkpath(OUTPUT_BASE_DIR)
mkpath(joinpath(OUTPUT_BASE_DIR, "csv"))
mkpath(joinpath(OUTPUT_BASE_DIR, "plots"))

const ALL_CHAINS_PATH = joinpath(OUTPUT_BASE_DIR, "jld2", "irt_all_chains.jld2")

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

# Binary heatmap (long-format)
function plotly_binary_heatmap(df::DataFrame, model_name::String)
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

# Ability heatmap from JLD2
function plotly_ability_heatmap(jld2_file; selected_demos=nothing, demo_names=nothing, selected_models=nothing)
    data = JLD2.load(jld2_file)
    all_chains = data["all_chains"]
    demographics = data["demographics"]

    model_names = selected_models === nothing ? collect(keys(all_chains)) : selected_models
    full_demo_names = demo_names === nothing ? [d.name for d in demographics] : demo_names
    n_demo = length(full_demo_names)

    # Map selected demographics to indexes
    if selected_demos !== nothing && !isempty(selected_demos)
        idxs = findall(d -> d in selected_demos, full_demo_names)
        labels = full_demo_names[idxs]
    else
        idxs = 1:n_demo
        labels = full_demo_names
    end

    heat_matrix = zeros(length(labels), length(model_names))
    for (j, model_name) in enumerate(model_names)
        df_chain = DataFrame(all_chains[model_name])
        θ_cols = filter(c -> startswith(c, "θ"), names(df_chain))
        θ_cols_filtered = θ_cols[idxs]
        for (i, col) in enumerate(θ_cols_filtered)
            heat_matrix[i, j] = mean(df_chain[!, col])
        end
    end

    hover_text = ["$(labels[i]), $(model_names[j]): $(round(heat_matrix[i,j], digits=2))"
                  for i in 1:length(labels), j in 1:length(model_names)]

    trace = PlotlyJS.heatmap(
        z=heat_matrix,
        x=model_names,
        y=labels,
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
        if selected_model == "All Models"
            combined_df = vcat(values(all_models_data)...)
        else
            combined_df = all_models_data[selected_model]
        end

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

    # Update table and heatmaps
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

        if selected_model == "All Models"
            df = vcat(values(all_models_data)...)
            models_for_heatmap = collect(keys(all_models_data))
        else
            df = all_models_data[selected_model]
            models_for_heatmap = [selected_model]
        end

        mask = trues(nrow(df))

        if "__ALL__" in selected_demos
            mask .&= trues(nrow(df))  # ignore demo filter
        elseif !isempty(selected_demos)
            mask .&= in.(df.demographic, Ref(selected_demos))
        end

        if "__ALL__" in selected_items
            mask .&= trues(nrow(df))  # ignore item filter
        elseif !isempty(selected_items)
            mask .&= in.(df.item_text, Ref(selected_items))
        end

        filtered = df[mask, :]

        table_component = Dash.html_table(
            vcat(
                [Dash.html_tr([Dash.html_th(col) for col in names(filtered)])],
                [Dash.html_tr([Dash.html_td(filtered[row, col]) for col in names(filtered)]) for row in 1:nrow(filtered)]
            )
        )

        binary_fig = plotly_binary_heatmap(filtered, selected_model)
        ability_fig = plotly_ability_heatmap(ALL_CHAINS_PATH; selected_demos=selected_demos, selected_models=models_for_heatmap)

        return table_component, binary_fig, ability_fig
    end

    Dash.run_server(app, "127.0.0.1"; debug=true)
end

run_dash_app()
