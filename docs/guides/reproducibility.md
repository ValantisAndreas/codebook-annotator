# Reproducibility and sharing

## What every record stores

Each record carries what is needed to tell later how it was made. The full list is in
[Run records](../reference/run-records.md). In short:

| Question | Fields |
|---|---|
| Which texts? | `run.dataset`, `run.dataset_revision`, `run.sample_id`, `run.sample_sha256` |
| Which codebook? | `run.codebook_id`, `run.codebook_version`, `run.codebook_sha256` |
| Which prompt? | `run.prompt_template`, `run.prompt_condition`, `run.system_prompt_sha256` |
| Which model? | `run.provider`, `run.model_tag`, `run.model_digest`, `run.model_digest_observed` |
| Which settings? | `run.temperature`, `run.seed`, `run.context_window`, `run.config_sha256` |
| Which tool? | `run.tool_version`, `schema_version` |
| When and at what cost? | `run.timestamp`, `usage` |

The `*_sha256` fields are fingerprints: if two runs have the same `system_prompt_sha256`, the model
received exactly the same instructions.

Set `dataset_revision` yourself (for example a version number or a commit hash of your data), as
the tool cannot know it.

## Pin the model

- With Ollama, use `model_digest: auto`. The digest identifies the exact model file. If you later
  give that digest in a config, the run stops when the installed model differs.
- With APIs, use pinned model names with a date or version, not names like `latest`.
- Set `temperature: 0.0` (the default). Some providers still give slightly different answers on
  repeated calls, so keep the run files: they are the result.

## Frozen parts

- **Prompt templates never change.** New wording comes as a new template name. A run that records
  `prompt_template: default_v1` can always be rebuilt with the same wording.
- **The prompt structure is fixed.** With the same codebook, template and features, the system
  prompt is identical byte for byte. `cba render` shows it.
- **The record format only grows.** `schema_version` (currently 2) changes if a field is ever
  renamed or removed.

## What to share

To let others check or repeat your work, share:

1. the codebook files,
2. the run config files,
3. the run files (see below for text),
4. the gold data, if its licence allows,
5. the tool version (`cba --version`).

## Sharing runs without the texts

Run files contain the item texts and the model's full answers by default. If the texts may not be
redistributed, remove them:

```bash
cba strip-text runs/*.jsonl --out-dir shared_runs
```

This writes copies without `text` and `raw_response`, with the evidence quotes blanked
(evidence quotes are copied from the text by design. Their `verified` flags stay), and with the
excerpts of unreadable answers removed from `parse_notes`. It also clears unreadable chunks in
`.attempts.jsonl` files. The output folder must differ from the input folder, so the originals
are never overwritten. Ids,
predictions, reasoning and all provenance fields stay, so the evaluation still works with the
gold data. The model's `reasoning` can quote short pieces of the text, so check it too. You can
also avoid storing the text in the first place with `store_text: false` and
`store_raw_response: false` (then `parse_notes` holds no excerpt of the answer either), but
evidence quotes are still stored then. Use `cba strip-text`
before sharing runs made with `evidence: true`.

## Re-reading old runs

The `replay` provider takes the raw answers from an earlier run file and runs them through the
current answer parser, without calling a model:

```bash
cba run -c run.yaml --provider replay --replay-from runs/old_run.jsonl --run-id old_run_reparsed
```

If `run.yaml` sets `model_digest: auto`, the replay run records no digest and prints a note,
because only a running Ollama server can report one.
