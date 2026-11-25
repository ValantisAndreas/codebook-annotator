# codebook-annotator

Annotate text with large language models from a YAML **codebook**, record full provenance for
every item, and evaluate the results against human annotators.

- One codebook file defines the categories and which of their fields are shown to the model.
- Works with local models (Ollama, any OpenAI-compatible server) and APIs (Anthropic, Mistral, OpenAI-compatible).
- Every output record carries the codebook, prompt, config and sample digests, the model digest, token usage and latency.
- Runs resume safely, and a resume is refused if the configuration or the model version changed.
- Evaluation: F1, Jaccard, Hamming, Krippendorff's α against annotators, model and codebook agreement, slices, paired bootstrap CIs, disagreement audits.
- Optional (off by default): verbatim evidence quotes with automatic verification, a second verification pass, 1–10 scores instead of yes/no, few-shot examples.

## Install

Python 3.11 or newer. From this folder:

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -e ".[eval]"        # add [dev] for tests, [docs] for the documentation site

## Quickstart

    cba init ~/my_project && cd ~/my_project        # create projects outside this folder
    cba validate codebook.yaml
    cba run -c run.yaml --dry-run                 # checks codebook, data and prompt, no model calls
    cba run -c run.yaml                           # offline stub provider
    cba run -c run.yaml --run-id llama --provider ollama --model-tag llama3.1:8b --model-digest auto
    cba evaluate --dataset items.jsonl --runs runs/llama.jsonl -o runs/results.xlsx

Turn on the optional features per run, for example:

    cba run -c run.yaml --run-id llama_checked --provider ollama --model-tag llama3.1:8b --model-digest auto --evidence --verify

## Commands

| Command | Purpose |
|---|---|
| `init` | create a starter codebook, items file and run config |
| `validate`, `render`, `templates` | check a codebook, show the exact prompt, list prompt templates |
| `convert` | spreadsheet/CSV with one row per category → codebook YAML |
| `ablate` | base codebook + extra fields → one variant per field and/or all fields |
| `check-invariance` | verify that prompts differ only in the codebook block |
| `run` | annotate a dataset (config files + CLI options) |
| `evaluate`, `bootstrap`, `disagreements`, `annotator-disagreements` | evaluation (see `docs/guides/evaluation.md`) |
| `aggregate`, `overlap`, `strip-text` | dataset preparation and sharing helpers |

Run `cba <command> --help` for all options.

## Documentation

The documentation is an MkDocs site in `docs/`. To read it locally:

    pip install -e ".[docs]"
    mkdocs serve                     # then open http://127.0.0.1:8000

Main pages:

- `docs/getting-started.md`: install, set up a model, first run and evaluation
- `docs/concepts.md`: codebooks, prompts, runs, records, resume and provenance
- `docs/guides/`: writing and comparing codebooks, running, providers, optional features,
  evaluation, preparing data, reproducibility, troubleshooting
- `docs/reference/`: configuration, command line, codebook format, run records, prompt templates
- `examples/`: runnable examples (codebook comparison, ordinal scores, quickstart)

## Contributing

See `CONTRIBUTING.md` for the development setup and the rules that keep runs reproducible,
`SECURITY.md` for API keys and data handling, and `CHANGELOG.md` for changes.

## Authors

- Lavinia Marin
- Valantis Andreas

## Licence

MIT, see `LICENSE`.
