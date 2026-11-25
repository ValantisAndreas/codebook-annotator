# Concepts

A handful of ideas explain how the tool works. The guides build on them.

## Codebook

A codebook is a YAML file that lists your **categories** and describes each of them with
**fields**: a definition, typical words, a boundary note, or anything else you find useful. The
field names are yours to choose.

```yaml
render:
  - {field: definition, label: Definition}
  - {field: indicators, label: Typical indicators}
categories:
  Praise:
    definition: The writer expresses satisfaction.
    indicators: [great, thank you]
  Complaint:
    definition: The writer reports a problem.
    indicators: [broken, late]
```

The `render` list decides **which fields the model sees, in which order and under which label**.
A field that is not listed in `render` stays in the file but never reaches the prompt. This makes
it easy to keep notes in the codebook, and to build codebook variants that differ in exactly one
field. The full format is in the [codebook format reference](reference/codebook-format.md).

## Prompt

The tool builds two messages for every text:

- The **system prompt**: a role sentence, a task sentence, a heading, one block per category
  (in alphabetical order, with the fields from `render`), and the instructions for the answer
  format (a JSON object with `predictions` and `reasoning`).
- The **user prompt**: "Evaluate the following text and return the JSON object:" followed by the text.

The structure is always the same. Only the codebook part changes between codebooks, which you can
check with `cba check-invariance`. Use `cba render` to see the exact prompt.

## Prompt template

The role sentence, the task sentence, the heading and the example reasoning come from a
**prompt template**. The default, `default_v1`, uses neutral wording. A codebook can replace any of
the four strings with its own `prompt:` block, which is the usual way to adapt the wording to your
domain. Built-in templates never change, see [Prompt templates](reference/templates.md).

## Prompt condition

Some optional features change the prompt: worked examples, evidence quotes, scores instead of
yes/no, and a second checking pass. Each run records its **prompt condition**, for example
`zero_shot` (the default) or `few_shot+evidence`. Runs with different conditions are never mixed
up in the evaluation. See [Optional features](guides/optional-features.md).

## Run

A **run** applies one codebook with one model to one sample of texts. Its settings come from one
or more config files plus command-line options. A run has a name (`run_id`) and writes one file,
`<output_dir>/<run_id>.jsonl`.

## Record

The run file has one **record** per text. A record holds the predictions (true or false per
category), the model's one-sentence reasoning, the raw answer, a parse status, and the provenance
fields below. The format is described in [Run records](reference/run-records.md).

Every record has a `parse_status`:

| Status | Meaning |
|---|---|
| `ok` | The answer was read and every category was present |
| `partial` | The answer was read but some categories were missing (counted as absent) or invalid |
| `failed` | No usable answer (no response, or not readable as JSON) |

## Resume and the config fingerprint

If a run is interrupted, start it again with the same settings. Texts with an `ok` record are
skipped and the others are annotated again. Records that were not `ok` are moved to
`<run_id>.attempts.jsonl`, so nothing is lost.

To make sure a run file never mixes different set-ups, every record carries a **config
fingerprint** (`config_sha256`). It covers the codebook file, the rendered system prompt, the
sample, the model and its digest (also the one read from Ollama with `model_digest: auto`, so a
re-downloaded model counts as a change), and every setting that changes the model's answers. If you resume with a
different fingerprint, the tool stops and asks you to choose a new `run_id`. Settings that only
affect how the run is carried out (concurrency, timeouts, output folder, storage options) are not
part of the fingerprint.

## Provenance

Each record stores, under `run`:

- the dataset name and revision, the sample id and a fingerprint of the sample texts,
- the codebook id, version and file fingerprint,
- the provider, the model name and the model digest (Ollama),
- the prompt template, the prompt condition and a fingerprint of the system prompt,
- temperature, seed, context window and the tool version,
- a timestamp,

and, outside `run`, the token usage, response time and number of model calls for that text.
With these, you can tell later exactly how every label was produced.

## Arms

For the agreement measures, runs are grouped into **arms**. An arm is a codebook (`codebook_id`)
under a prompt condition. Two runs of the same arm (for example on different models) give **model agreement**.
Two arms on the same model give **codebook agreement**. See [Evaluating runs](guides/evaluation.md).
