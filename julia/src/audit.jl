#!/usr/bin/env julia 

using Base.Threads

const desired_threads = max(Sys.CPU_THREADS - 1, 1)

# Relaunch Julia if needed
if Threads.nthreads() < desired_threads
    println("Restarting Julia with $desired_threads threads...")
    run(`julia -t $desired_threads $(PROGRAM_FILE)`)
    exit()
end

println("Running Julia with $(Threads.nthreads()) threads.")

include("LLMBiasAudit.jl")

LLMBiasAudit.main(
    models=[
        "gemma:2b",
        "phi3:latest",
    ],
    use_cache=false
)
