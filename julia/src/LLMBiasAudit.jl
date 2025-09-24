module LLMBiasAudit

include("IRT.jl")
import .IRT

include("LLMBiasData.jl")
import .LLMBiasData

include("ResponseBinaryClassifier.jl")
import .ResponseClassifier

import DataFrames
import CSV
import PromptingTools
import JLD2
import Base.Threads

export main, default_models, all_chains_datafile_path

PromptingTools.OPENAI_API_KEY = ""

const smallest_model = "gemma:2b"

const default_models = [
    "gemma:2b",
    "phi3:latest",
    "mistral:latest",
    "llama3:latest"
]

const ollama_schema = PromptingTools.OllamaSchema()

function perform_audit_query(prompts::Vector{Vector}; model::String="gemma:2b", max_tokens::Int=50)
    lock_obj = ReentrantLock()
    n = length(prompts)
    responses = Vector{String}(undef, n)

    println("Running LLM queries individually (unbatched) on $(model)...")
    Threads.@threads for i in 1:n
        try
            prompt = prompts[i]
            @debug "prompt $(i)/$(n): $(prompt)"

            response = PromptingTools.aigenerate(
                ollama_schema,
                prompt;
                model=model,
                max_tokens=max_tokens,
                api_kwargs=(url="http://localhost",)
            )

            result = ""
            if response !== nothing && hasproperty(response, :content) && !isempty(response.content)
                result = strip(replace(response.content, r"[\n\r\f]+" => " "))
            end

            # thread-safe assignment
            lock(lock_obj) do
                responses[i] = result
            end

        catch e
            lock(lock_obj) do
                responses[i] = ""
            end
            @warn "LLM query failed for prompt $(i): $e"
        end
    end

    return responses
end


function load_or_query_models(models::Vector{String}, run_dir, prompts, prompt_info, use_cache::Bool)
    all_responses_raw = Dict{String,Vector{String}}()

    for model_name in models
        @info("Processing model $(model_name)")
        filename_cached = "$(run_dir)/csv/responses_$(replace(model_name, r"[^\w]" => "_"))_raw.csv"

        if use_cache && isfile(filename_cached)
            df_cache = CSV.read(filename_cached, DataFrames.DataFrame)
            verbose_responses = df_cache.response_text
            @info("Loaded cached verbose responses for $(model_name).")
            @info("Loaded columns:", DataFrames.names(df_cache))
        else
            verbose_responses = perform_audit_query(prompts; model=model_name)
            @info "First 5 responses: ", verbose_responses[1:min(5, end)]

            CSV.write(filename_cached, DataFrames.DataFrame(prompt=prompt_info, response_text=verbose_responses))
            @info("Saved raw verbose responses to cache for $(model_name).")
        end

        all_responses_raw[model_name] = verbose_responses
    end

    return all_responses_raw
end

function classify_and_save(models::Vector{String}, prompt_info, all_responses_raw, run_dir, demographics, items)
    all_responses_bin = Dict{String,Vector{Int}}()

    for model_name in models
        @info("Classifying responses for $(model_name)")
        verbose_responses = all_responses_raw[model_name]
        response_tuples = [(ptype, item, resp) for ((ptype, item), resp) in zip(prompt_info, verbose_responses)]

        raw_class_output, responses_bin = ResponseClassifier.classify_responses_llm(
            response_tuples;
            model=smallest_model
        )

        all_responses_bin[model_name] = responses_bin

        # Save CSV
        safe_model_name = replace(model_name, r"[^\w]" => "_")
        filename_csv = "$(run_dir)/csv/responses_$(safe_model_name).csv"
        CSV.write(filename_csv,
            DataFrames.DataFrame(
                demographic=[demo.name for demo in demographics for _ in items],
                prompt_type=getindex.(response_tuples, 1),
                item_text=getindex.(response_tuples, 2),
                response_text=getindex.(response_tuples, 3),
                classifier_output=raw_class_output,
                response_bin=responses_bin
            )
        )
        @info("Responses saved to '$(filename_csv)'")
    end

    return all_responses_bin
end

function fit_irt_models_and_save(models::Vector{String}, run_dir, demographics, items, all_responses_bin)
    all_chains = Dict{String,Any}()

    for model_name in models
        @info("Fitting IRT for $(model_name)")
        responses_bin = all_responses_bin[model_name]

        response_matrix = reshape(responses_bin, length(demographics), length(items))
        chain = IRT.fit_irt_model(response_matrix)

        safe_model_name = replace(model_name, r"[^\w:]" => "_")
        filename_chain = "$(run_dir)/jld2/irt_chain_$(safe_model_name).jld2"
        JLD2.@save filename_chain chain
        @info("IRT chain saved to '$(filename_chain)'")

        all_chains[model_name] = chain
    end

    return all_chains
end

function main(; models=default_models, run_dir="output", use_cache::Bool=true)
    @info("LLMBiasAudit.main enter: use_cache: $(use_cache), run_dir=$(run_dir), models=$(models)")
    all_chains_datafile_path = "$(run_dir)/jld2/irt_all_chains.jld2"

    demographics = LLMBiasData.define_demographics()
    items = LLMBiasData.define_items()
    prompts, prompt_info = LLMBiasData.generate_audit_prompts(demographics, items)

    all_responses_raw = load_or_query_models(models, run_dir, prompts, prompt_info, use_cache)
    all_responses_bin = classify_and_save(models, prompt_info, all_responses_raw, run_dir, demographics, items)
    all_chains = fit_irt_models_and_save(models, run_dir, demographics, items, all_responses_bin)

    # Save all chains
    JLD2.@save all_chains_datafile_path all_chains demographics items
    @info("All IRT chains saved to '$(all_chains_datafile_path)'")

    return all_responses_raw, all_responses_bin, all_chains
end

end # module
