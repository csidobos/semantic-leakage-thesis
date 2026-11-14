# Semantic Leakage in Large Language Models

This repository contains the code and experimental materials for an MSc thesis investigating semantic leakage in large language models (LLMs).

## Research overview

The study examines whether socially relevant information introduced earlier in a prompt influences LLM responses to subsequent questions, even when that information is not directly relevant to the task.

The experiment investigates differences across:

- **Models:** OpenAI, Mistral, Qwen, and Grok
- **Languages:** English and Hungarian
- **Context conditions:** Target, Comparison, Neutral, Pseudo, Nonconnected, and Non-social
- **Textual distance:** Three levels (0, 1, 2)
- **Paraphrasing:** Three variants (p0, p1, p2)
- **Response formats:** Multiple choice (MC) and multiple choice with an additional "Cannot be determined" option (MC+NAA)

The primary outcome is the proportion of stereotypical responses across experimental conditions.

## Repository structure

| Directory / File | Description |
|---|---|
| `data/` | Stimulus materials, generated requests, and run schedule |
| `api/` | Model API integration and connection checks |
| `analysis/` | Data cleaning, validation, descriptive analysis, and visualization scripts |
| `thesis/` | R scripts for bootstrap analysis |
| `generate_*.py` | Experimental prompt and schedule generation |
| `run_*.py` | Scripts for running model experiments |
| `audit_results.py` | Experimental output auditing |

## Data availability

The repository includes experimental stimuli and generated request files. Raw model responses and processed experimental results are not included.

## Notes

The repository preserves the original research project structure and contains scripts used during different stages of the study.
