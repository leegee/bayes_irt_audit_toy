#!/usr/bin/env julia

include("LLMBiasAudit.jl")
import .LLMBiasAudit

import CSV
import Dash
import DataFrames: DataFrame, names, nrow, vcat, groupby
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

# Safely ensure a :model column exists (in-place)
function ensure_model_column!(df::DataFrame, model_name::String)
    if :model ∉ names(df)
        df[!, :model] = fill(model_name, nrow(df))
    end
    return df
end

# Binary heatmap (long-format)
function plotly_binary_heatmap(df::DataFrame, model_name::String)
    if nrow(df) == 0
        return PlotlyJS.Plot(PlotlyJS.Layout(title="Binary Responses: (no data)", template="plotly_dark"))
    end

    hover_text = ["$(df.demographic[i]), $(df.item_text[i]): $(df.response_bin[i])" for i in 1:nrow(df)]

    trace = PlotlyJS.heatmap(
        z=coalesce.(df.response_bin, 0.0),
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

# Ability heatmap from filtered DataFrame (robust groupby per-demo × per-model)
function plotly_ability_heatmap_from_df(df::DataFrame, selected_models::Vector{String})
    if nrow(df) == 0
        return PlotlyJS.Plot(PlotlyJS.Layout(title="Mean binary responses (no data)", template="plotly_dark"))
    end

    # Ensure model column exists (should already be there for loaded models)
    ensure_model_column!(df, selected_models[1])

    # Keep model order equal to selected_models (important)
    model_names = selected_models

    # Unique demographics in filtered df (preserve appearance order)
    demographics = collect(unique(df.demographic))

    # Compute mean(response_bin) for each (demographic, model) via groupby
    g = groupby(df, [:demographic, :model])
    means = Dict{Tuple{Any,Any},Float64}()
    for sub in g
        key = (sub.demographic[1], sub.model[1])
        vals = coalesce.(sub.response_bin, 0.0)
        means[key] = isempty(vals) ? 0.0 : mean(vals)
    end

    # Assemble heat matrix (demographics rows × model columns)
    heat_matrix = zeros(length(demographics), length(model_names))
    for (i, demo) in enumerate(demographics)
        for (j, model) in enumerate(model_names)
            heat_matrix[i, j] = get(means, (demo, model), 0.0)
        end
    end

    hover_text = ["$(demographics[i]), $(model_names[j]): $(round(heat_matrix[i,j], digits=2))"
                  for i in 1:length(demographics), j in 1:length(model_names)]

    trace = PlotlyJS.heatmap(
        z=heat_matrix,
        x=model_names,
        y=demographics,
        text=hover_text,
        hoverinfo="text+z",
        colorscale="Viridis",
        zmin=0, zmax=1
    )

    PlotlyJS.Plot(trace, PlotlyJS.Layout(
        title="Mean binary responses across models",
        template="plotly_dark",
        yaxis=PlotlyJS.attr(autorange="reversed")
    ))
end

# Load all CSVs into a Dict and guarantee model column and stable order
function load_all_models()
    models = LLMBiasAudit.default_models  # preserve expected order
    d = Dict{String,DataFrame}()
    for model in models
        df = load_model_csv(model)
        if nrow(df) > 0
            df = copy(df)                      # avoid mutating source
            df[!, :model] = fill(model, nrow(df))
            d[model] = df
        else
            @error "Skipping missing CSV" model
        end
    end
    return d
end

# Dash app
function run_dash_app()
    all_models_data = load_all_models()
    model_names = LLMBiasAudit.default_models  # keep canonical order
    # Filter out models with no data
    model_names = [m for m in model_names if haskey(all_models_data, m)]
    all_models_dropdown = ["All Models"; model_names]

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
        combined_df = selected_model == "All Models" ? vcat([all_models_data[m] for m in model_names]...) : all_models_data[selected_model]

        demo_opts = [Dict("label" => "All", "value" => "__ALL__");
            [Dict("label" => d, "value" => d) for d in unique(combined_df.demographic)]]

        item_opts = [Dict("label" => "All", "value" => "__ALL__");
            [Dict("label" => i, "value" => i) for i in unique(combined_df.item_text)]]

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

        # Build combined or single-model df (already has model column)
        if selected_model == "All Models"
            df = vcat([all_models_data[m] for m in model_names]...)
            models_for_heatmap = model_names
        else
            df = all_models_data[selected_model]
            models_for_heatmap = [selected_model]
        end

        # Apply filters
        mask = trues(nrow(df))
        if "__ALL__" ∉ selected_demos && !isempty(selected_demos)
            mask .&= in.(df.demographic, Ref(selected_demos))
        end
        if "__ALL__" ∉ selected_items && !isempty(selected_items)
            mask .&= in.(df.item_text, Ref(selected_items))
        end
        filtered = df[mask, :]

        # Table
        table_component = Dash.html_table(
            vcat(
                [Dash.html_tr([Dash.html_th(col) for col in names(filtered)])],
                [Dash.html_tr([Dash.html_td(filtered[row, col]) for col in names(filtered)]) for row in 1:nrow(filtered)]
            )
        )

        # Figures (both reflect the filtered table)
        binary_fig = plotly_binary_heatmap(filtered, selected_model)
        ability_fig = plotly_ability_heatmap_from_df(filtered, models_for_heatmap)

        return table_component, binary_fig, ability_fig
    end

    Dash.run_server(app, "127.0.0.1"; debug=true)
end

run_dash_app()
