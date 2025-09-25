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

Classify verbose LLM responses into 0 (definitive Yes/No) or 1 (non-definitive / Unsure) 
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

        # Build categorizer prompt
        messages = LLMBiasData.get_categoriser_prompt(ptype, response)

        # Generate classification via LLM
        ai_response = PromptingTools.aigenerate(
            ollama_schema,
            messages;
            model=model,
            max_tokens=50,
            api_kwargs=(url="http://localhost",)
        )

        clean_text = strip(replace(ai_response.content, r"[\n\r\f]+" => " "))
        raw_output[i] = clean_text

        # Try first parsing the first numeric 
        val = nothing
        for s in split(clean_text, r"[^\d]+")
            if !isempty(s)
                val = tryparse(Int, s)
                if val !== nothing
                    break
                end
            end
        end

        # Fallback textual mapping
        if isnothing(val)
            @warn "Classifier did not return a parsable number"
            lc = lowercase(clean_text)
            if occursin("yes", lc) || occursin("no", lc)
                val = 0
            elseif occursin("unsure", lc)
                val = 1
            else
                @warn "Classifier unparseable at index $i: '$clean_text'\nOriginal: $(response)"
                binary[i] = missing
                continue
            end
        end

        binary[i] = val

        @info "# PROMPTTYPE: $(ptype)\n# QUESTION: $(item)\n# ORIG RESPONSE: $(response)\n# CLASSIFIER: $(ai_response.content)\n# FINAL: $(val)"
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
