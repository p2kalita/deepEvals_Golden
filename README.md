# GoldenForge -  Golden Dataset creation using DeepEvals + LLMs (Gemini, GPT-4o-mini, Groq /llama,qwen, ollama/Gemma4 (local) )

Modular DeepEval golden-dataset generator for `support-ticket CSVs`, with
provider switching (`Gemini` / `GPT-4o` / `Sonnet` / `Groq` / `local Ollama`) done
entirely through YAML via `LiteLLM`.


## Layout

```
goldenforge/
├── main.py                    CLI entry point
├── requirements.txt
├── configs/                   YAML provider/model configs (edit these to switch models)
│   ├── Gemini 2.5 Pro - Simple.yaml
│   ├── simple-gpt-4o.yaml
│   ├── simple-sonnet.yaml
│   ├── simple-groq-llama.yaml
│   └── simple-ollama-gemma.yaml
├── src/
│   ├── config_loader.py       YAML -> ModelConfig, and the LiteLLM model-string mapping
│   ├── llm_wrappers.py        LiteLLMDeepEvalLLM + LocalEmbeddingModel
│   ├── document_builder.py    tickets.csv -> one .txt document per row
│   ├── synthesis_configs.py   evolution tiers / styling variants / filtration / context config
│   ├── golden_writer.py       writes all 4 output formats
│   └── pipeline.py            orchestrates everything above
└── outputs/                   generated documents + all golden files land here
```

## Why split it this way

Each file answers exactly one question:

- `config_loader.py` — *which model, and what does LiteLLM call it?*
- `llm_wrappers.py` — *how do I talk to that model, in DeepEval's required shape?*
- `document_builder.py` — *where do the source documents come from?*
- `synthesis_configs.py` — *what does "low/medium/high" and each style actually mean?*
- `golden_writer.py` — *what do I do with the goldens once I have them?*
- `pipeline.py` — *what order do all of these run in?*

Swapping a provider touches only a YAML file. Adding a 5th output format
touches only `golden_writer.py`. Changing what "high complexity" means
touches only `synthesis_configs.py`. Nothing else moves.

## Usage

```bash
pip install -r requirements.txt

# set whichever key your chosen config needs, e.g.:
export GEMINI_API_KEY="sk-ant-..."

# run the full 3 tiers x 4 styles = 12-combination grid
python main.py --config "configs/Gemini 2.5 Flash - Simple.yaml" --tickets tickets.csv

# or just one combination
python main.py --config configs/simple-groq-llama.yaml --tier medium --style technical
```

Each combination writes 4 files into `outputs/`:

```
outputs/goldens_low_simple.json          (DeepEval-native)
outputs/goldens_low_simple.csv           (DeepEval-native)
outputs/goldens_low_simple_clean.json    (id, input, expected_output, context, source_file, evolutions, quality_score)
outputs/goldens_low_simple_clean.csv
```

## Provider notes

| `components.api` in YAML | Needs | Notes |
|---|---|---|
| `google` | `GOOGLE_API_KEY` | Gemini |
| `openai` | `OPENAI_API_KEY` | GPT-4o |
| `anthropic` | `ANTHROPIC_API_KEY` | Sonnet |
| `groq` | `GROQ_API_KEY` | Groq-hosted Llama |
| `ollama` | none (local) | Run `ollama serve` and `ollama pull gemma4` first |




## What's preserved from 04-06-2026_deepEvals_Golden.ipynb notebook , unchanged

- `DeepEvalBaseLLM` wrapper with both plain-text and Pydantic-schema generation paths
- `LocalEmbeddingModel` (sentence-transformers, no OpenAI key, no cost)
- All 7 Evolution types, all 3 complexity tiers (low/medium/high) with original weights
- All 4 styling variants (simple/professional/technical/domain)
- `FiltrationConfig` (quality_threshold=0.5, max_quality_retries=3)
- `ContextConstructionConfig` (chunk_size=1024, chunk_overlap=128, embedder attached here)
- `async_mode=True`, `max_concurrent`, `include_expected_output=True`
- `generate_goldens_from_docs()` used exactly as before
- `EvaluationDataset` wrapping of results
- DeepEval-native JSON/CSV save + clean JSON/CSV save

## What's new (per the brief's requirements)

- LiteLLM as the unified API layer (`config_loader.py`, `llm_wrappers.py`)
- YAML-driven provider switching
- CSV-to-documents generation (`document_builder.py`) replacing the 3 hardcoded sample files
- The 3x4 = 12-combination batch driver (`pipeline.py`)
