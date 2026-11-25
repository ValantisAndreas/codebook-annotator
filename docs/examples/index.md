# Examples

Each example is a folder under `examples/` in the source code, with its own data and a README.
All data in the examples is invented. Run the commands from inside the example folder.

| Example | Shows |
|---|---|
| [Comparing codebooks](ablation.md) (`examples/ablation`) | A codebook written in a spreadsheet, variants with one added field each, the invariance check, runs of every variant and the Codebook Agreement sheet |
| [Ordinal scores](ordinal.md) (`examples/ordinal`) | A codebook with 1–10 scores, a run, and the comparison with gold scores in the Ordinal sheet |

The folder `examples/quickstart` holds a further small codebook with eight short reviews, gold
labels and a run config for a local Ollama model (`run_ollama.yaml`).

To start a project of your own, `cba init` writes a starter codebook, data file and run config.
See [Getting started](../getting-started.md).
