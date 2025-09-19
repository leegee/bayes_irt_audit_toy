#!/usr/bin/env julia

include("LLMBiasAudit.jl")

LLMBiasAudit.main(models=["phi3:latest"])
