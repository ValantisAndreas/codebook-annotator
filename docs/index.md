# codebook-annotator

codebook-annotator applies a **codebook** to a collection of texts with a large language model
(LLM), and records exactly how every annotation was made. You then compare the model's labels
with human annotations.

It is written for researchers who code text with a fixed set of categories (values, frames,
topics, moral foundations, feedback types) and want to know how well an LLM can do the same
coding, and how the wording of the codebook changes the result.

## What it does

- **Codebook in, labels out.** You describe your categories in a YAML file (or a spreadsheet).
  The tool turns it into a prompt, sends each text to the model and stores one record per text.
- **Local or remote models.** Ollama on your own machine, any server with an OpenAI-compatible
  API, or the Anthropic and Mistral APIs.
- **Provenance by default.** Every record stores the model, the model digest, the codebook and
  prompt fingerprints, the settings, the token usage and the response time.
- **Safe to interrupt.** A run can be stopped and resumed. The tool refuses to resume when the
  codebook, the prompt or a setting has changed.
- **Evaluation.** F1, Jaccard, Hamming loss and Krippendorff's alpha against the annotators,
  agreement between models and between codebooks, slices by any field, bootstrap confidence
  intervals and disagreement lists.
- **Codebook comparisons.** Build variants of a codebook with one field added at a time, check
  that their prompts differ only in the codebook part, and compare the results.
- **Optional extras**, all off by default: supporting quotes that are checked against the text,
  a second checking pass, 1–10 scores instead of yes/no, and worked examples in the prompt.

## A 30-second example

```bash
cba init my_project
cd my_project
cba run -c run.yaml
cba evaluate --dataset items.jsonl --runs runs/first_run.jsonl -o runs/results.xlsx
```

`cba init` writes a small codebook, four example texts with human labels, and a run config that
uses the offline `stub` provider, so these commands work before you have set up a model.
[Getting started](getting-started.md) continues from here with a real model.

## Where to go next

- [Getting started](getting-started.md): install, set up a model, first run and evaluation.
- [Concepts](concepts.md): the few ideas the tool is built on.
- [Writing a codebook](guides/codebooks.md): the codebook format and how to write good categories.
- [Examples](examples/index.md): complete, runnable walkthroughs.
- [Command-line reference](reference/cli.md) and [configuration reference](reference/configuration.md).
