module LLMBiasAudit

include("IRT.jl")
import .IRT

include("LLMBiasData.jl")
import .LLMBiasData

include("ResponseClassifier.jl")
import .ResponseClassifier

import DataFrames
import CSV
import PromptingTools
import JLD2
import Base.Threads

export main, all_chains_datafile_path

PromptingTools.OPENAI_API_KEY = ""

const all_chains_datafile_path = "jld2/irt_all_chains.jld2"

const default_models = [
    "gemma:2b",
    "phi3:latest",
    "mistral:latest",
    "llama3:latest"
]

const ollama_schema = PromptingTools.OllamaSchema()

# Prompt generation
function generate_prompts(demographics::Vector{LLMBiasData.Demographic}, items::Vector{String})
    prompts = String[]
    prompt_info = Vector{Dict{String,Any}}()
    for demo in demographics, item in items
        push!(prompts, "$(demo.name) applying $item")
        push!(prompt_info, Dict(
            "demographic" => demo.code,
            "name" => demo.name,
            "item" => item
        ))
    end
    return prompts, prompt_info
end

# Query main LLM (verbose)
function query_ollama_client(prompts::Vector{String}; model::String="gemma:2b", max_tokens::Int=150)
    n = length(prompts)
    responses = Vector{String}(undef, n)
    system_msg, user_template = LLMBiasData.PROMPT_MESSAGES

    println("Running LLM queries individually (unbatched) on $(model)...")
    Threads.@threads for i in 1:n
        messages = [
            PromptingTools.SystemMessage(system_msg["content"]),
            PromptingTools.UserMessage(
                PromptingTools.replace(user_template["content"], "{{decision}}" => prompts[i])
            )
        ]
        response = PromptingTools.aigenerate(
            ollama_schema,
            messages;
            model=model,
            max_tokens=max_tokens,
            api_kwargs=(url="http://localhost",)
        )
        # clean newlines/freespace characters
        responses[i] = strip(replace(response.content, r"[\n\r\f]+" => " "))
    end

    return responses
end


function main(; models=default_models)
    demographics = LLMBiasData.define_demographics()
    items = LLMBiasData.define_items()
    prompts, prompt_info = generate_prompts(demographics, items)

    all_responses_raw = Dict{String,Vector{String}}()
    all_responses_bin = Dict{String,Vector{Int}}()
    all_chains = Dict{String,Any}()

    smallest_model = "gemma:2b"  # always use smallest model for classification

    for model_name in models
        @info("\n# Running model $(model_name)")

        # Step 1: verbose output from main model
        verbose_responses = query_ollama_client(prompts; model=model_name)
        all_responses_raw[model_name] = verbose_responses
        @info("Verbose responses received for $(model_name).")

        # Step 2: classify verbose output with smallest model
        raw_class_output, responses_bin_union = ResponseClassifier.classify_responses_llm(
            verbose_responses;
            model=smallest_model,
        )

        # Convert Union{Int,Missing} → Int (default missing -> 0)
        responses_bin = [ismissing(x) ? 0 : x for x in responses_bin_union]
        n_missing = count(ismissing, responses_bin_union)
        if n_missing > 0
            @warn "Model $(model_name): $n_missing missing labels replaced with 0 in IRT input."
        end

        all_responses_bin[model_name] = responses_bin
        @info("Binary labels processed for IRT.")

        # Save CSV with classifier output for auditing
        safe_model_name = replace(model_name, r"[^A-Za-z0-9]" => "_")
        filename_csv = "csv/responses_$(safe_model_name).csv"
        CSV.write(filename_csv,
            DataFrames.DataFrame(
                prompt=prompts,
                response_text=verbose_responses,
                classifier_output=raw_class_output,
                response_bin=responses_bin
            )
        )
        @info("Responses saved to '$(filename_csv)'")

        # Step 3: Fit IRT
        response_matrix = reshape(responses_bin, length(demographics), length(items))
        chain = IRT.fit_irt_model(response_matrix)
        filename_chain = "jld2/irt_chain_$(safe_model_name).jld2"
        JLD2.@save filename_chain chain
        @info("IRT chain saved to '$(filename_chain)'")
        all_chains[model_name] = chain
    end

    # Save all IRT chains
    JLD2.@save all_chains_datafile_path all_chains demographics items
    @info("All IRT chains saved to 'jld2/irt_all_chains.jld2'")

    return all_responses_raw, all_responses_bin, all_chains
end

end # module
