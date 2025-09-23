module ResponseClassifier

import PromptingTools
import Base.Threads
import CSV
import DataFrames
import ..LLMBiasData

export classify_responses_llm

PromptingTools.OPENAI_API_KEY = ""
const ollama_schema = PromptingTools.OllamaSchema()

"""
    classify_responses_llm(responses::Vector{Tuple{String,String,String}};
                           model::String="gemma:2b",
                           csv_file::Union{Nothing,String}=nothing)

Classify verbose LLM responses into 0 (non-definitive) or 1 (definitive decision) 
using a small LLM. 

`responses` must be a vector of `(prompt_type, item_text, response_text)`.

Returns `(raw_output::Vector{String}, binary::Vector{Union{Int,Missing}})`.

If `csv_file` is set, saves a CSV with prompt_type, item, raw classifier output, 
and parsed labels.
"""
function classify_responses_llm(
    responses::Vector{Tuple{String,String,String}};
    model::String="gemma:2b",
    csv_file::Union{Nothing,String}=nothing
)
    n = length(responses)
    raw_output = Vector{String}(undef, n)
    binary = Vector{Union{Int,Missing}}(undef, n)

    for i in 1:n
        ptype, item, response = responses[i]

        messages = LLMBiasData.get_categoriser_prompt(ptype, item)
        push!(messages, PromptingTools.UserMessage(response))

        ai_response = PromptingTools.aigenerate(
            ollama_schema,
            messages;
            model=model,
            max_tokens=50,
            api_kwargs=(url="http://localhost",)
        )

        clean_text = strip(replace(ai_response.content, r"[\n\r\f]+" => " "))
        raw_output[i] = clean_text

        # Parse first integer found, else missing
        val = tryparse(Int, first(split(clean_text, r"[^\d]+")))
        if isnothing(val)
            @warn "Classifier returned unparseable response at index $i: '$clean_text'\nOrig content: $(response)"
            binary[i] = missing
        else
            binary[i] = val
            @info "PromptType: $(ptype)\nQuestion: $(item)\nOrig Response: $(response)\nClassifier: $(ai_response.content)\nFINAL: $(val)"
        end

        @info "-------------------------------------------"
    end

    if csv_file !== nothing
        df = DataFrames.DataFrame(
            prompt_type=getindex.(responses, 1),
            item_text=getindex.(responses, 2),
            response_text=getindex.(responses, 3),
            classifier_output=raw_output,
            response_bin=binary
        )
        CSV.write(csv_file, df)
    end

    return raw_output, binary
end

end # module
