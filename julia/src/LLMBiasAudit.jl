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

export main, all_chains_datafile_path

PromptingTools.OPENAI_API_KEY = ""

const smallest_model = "gemma:2b"
const all_chains_datafile_path = "jld2/irt_all_chains.jld2"
const default_models = [
    "gemma:2b",
    "phi3:latest",
    "mistral:latest",
    "llama3:latest"
]

const ollama_schema = PromptingTools.OllamaSchema()

# Generate prompts as Message arrays 
function generate_prompts(demographics::Vector{LLMBiasData.Demographic},
    items::Vector{Tuple{String,String}})
    prompts = Vector{Vector}()
    prompt_info = Vector{Tuple{String,String}}()  # store metadata asp tuple (prompt_type, item_text)

    for demo in demographics, (ptype, item_text) in items
        push!(prompts, LLMBiasData.get_auditable_prompt(ptype, "$(demo.name) applying $item_text"))
        push!(prompt_info, (ptype, item_text))
    end

    return prompts, prompt_info
end

# Main LLM query 
function query_ollama_client(
    prompts::Vector{Vector};
    model::String="gemma:2b",
    max_tokens::Int=50
)
    n = length(prompts)
    responses = Vector{String}(undef, n)

    println("Running LLM queries individually (unbatched) on $(model)...")
    Threads.@threads for i in 1:n
        @info "prompt $(i) / $(n)"
        response = PromptingTools.aigenerate(
            ollama_schema,
            prompts[i];
            model=model,
            max_tokens=max_tokens,
            api_kwargs=(url="http://localhost",)
        )
        responses[i] = strip(replace(response.content, r"[\n\r\f]+" => " "))
    end

    return responses
end


function main(; models=default_models, use_cache::Bool=true)
    @info("LLMBiasAudit.main enter")
    demographics = LLMBiasData.define_demographics()
    items = LLMBiasData.define_items()

    prompts, prompt_info = generate_prompts(demographics, items)

    all_responses_raw = Dict{String,Vector{String}}()
    all_responses_bin = Dict{String,Vector{Int}}()
    all_chains = Dict{String,Any}()

    for model_name in models
        @info("Running model $(model_name)")
        safe_model_name = replace(model_name, r"[^A-Za-z0-9:]" => "_")
        filename_csv = "csv/responses_$(safe_model_name).csv"
        filename_cache = "csv/responses_$(safe_model_name)_raw.csv"
        filename_chain = "jld2/irt_chain_$(safe_model_name).jld2"

        # Load cache
        if use_cache && isfile(filename_cache)
            df_cache = CSV.read(filename_cache, DataFrames.DataFrame)
            verbose_responses = df_cache.response_text
            @info("Loaded cached verbose responses for $(model_name).")
        else
            verbose_responses = query_ollama_client(prompts; model=model_name)
            CSV.write(filename_cache, DataFrames.DataFrame(prompt=prompt_info, response_text=verbose_responses))
            @info("Saved raw verbose responses to cache for $(model_name).")
        end
        all_responses_raw[model_name] = verbose_responses

        # Prepare tuples for classifier
        response_tuples = [(ptype, item, resp) for ((ptype, item), resp) in zip(prompt_info, verbose_responses)]

        # Classify
        raw_class_output, responses_bin_union = ResponseClassifier.classify_responses_llm(
            response_tuples;
            model=smallest_model
        )

        # Convert missing -> 0
        responses_bin = [ismissing(x) ? 0 : x for x in responses_bin_union]
        n_missing = count(ismissing, responses_bin_union)
        if n_missing > 0
            @warn "Model $(model_name): $n_missing missing labels replaced with 0 in IRT input."
        end
        all_responses_bin[model_name] = responses_bin
        @info("Binary labels processed for IRT.")

        # Save CSV
        CSV.write(filename_csv,
            DataFrames.DataFrame(
                prompt_type=getindex.(response_tuples, 1),
                item_text=getindex.(response_tuples, 2),
                response_text=getindex.(response_tuples, 3),
                classifier_output=raw_class_output,
                response_bin=responses_bin
            )
        )
        @info("Responses saved to '$(filename_csv)'")

        # Fit IRT
        response_matrix = reshape(responses_bin, length(demographics), length(items))
        chain = IRT.fit_irt_model(response_matrix)
        JLD2.@save filename_chain chain
        @info("IRT chain saved to '$(filename_chain)'")
        all_chains[model_name] = chain
    end

    # Save all chains
    JLD2.@save all_chains_datafile_path all_chains demographics items
    @info("All IRT chains saved to '$(all_chains_datafile_path)'")

    return all_responses_raw, all_responses_bin, all_chains
end

end # module
