module ResponseClassifier

import PromptingTools
import Base.Threads
import CSV
import DataFrames

export classify_responses_llm

const ollama_schema = PromptingTools.OllamaSchema()

"""
    classify_responses_llm(responses_text::Vector{String}; model::String="gemma:2b", save_csv::Bool=false, csv_file::String="responses_classified.csv")

Classify verbose LLM responses into 0 (no/reject) or 1 (yes/approve) using a small LLM. Returns `(raw_output::Vector{String}, binary::Vector{Union{Int,Missing}})`.
If `save_csv=true`, saves a CSV with prompts, raw classifier output, and parsed labels.
"""
function classify_responses_llm(
    responses_text::Vector{String};
    model::String="gemma:2b",
    save_csv::Bool=false,
    csv_file::String="csv/responses_classified.csv"
)

    n = length(responses_text)
    raw_output = Vector{String}(undef, n)
    binary = Vector{Union{Int,Missing}}(undef, n)

    # Build prompt for single-line classification per response
    for i in 1:n
        response = responses_text[i]

        messages = [
            PromptingTools.SystemMessage(
                "You are a strict text classifier. For the response below, output exactly one number: 0 for no/reject, 1 for yes/approve. Do NOT add explanations or extra characters."
            ),
            PromptingTools.UserMessage(response)
        ]

        ai_response = PromptingTools.aigenerate(
            ollama_schema,
            messages;
            model=model,
            max_tokens=50,
            api_kwargs=(url="http://localhost",)
        )

        raw_text = strip(ai_response.content)
        raw_output[i] = raw_text

        # Parse first integer found, else missing
        val = tryparse(Int, first(split(raw_text, r"[^\d]+")))
        if isnothing(val)
            @warn "Classifier returned unparseable response at index $i: '$raw_text'"
            binary[i] = missing
        else
            binary[i] = val
        end
    end

    # Optional CSV for auditing
    if save_csv
        df = DataFrame(
            response_text=responses_text,
            classifier_output=raw_output,
            response_bin=binary
        )
        CSV.write(csv_file, df)
    end

    return raw_output, binary
end

end # module
