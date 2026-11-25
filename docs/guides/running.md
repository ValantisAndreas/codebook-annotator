# Running annotations

`cba run` annotates a dataset with one codebook and one model. This guide covers config files,
sampling, resuming, speed and what is stored. Every setting is listed in the
[configuration reference](../reference/configuration.md).

## Config files and command-line options

A run config is a YAML (or JSON) file with run settings:

```yaml
run_id: first_run
codebook: codebook.yaml
dataset: items.jsonl
provider: ollama
model_tag: llama3.1:8b
model_digest: auto
temperature: 0.0
output_dir: runs
```

```bash
cba run -c run.yaml
```

Settings are combined in this order, and later sources win:

1. built-in defaults,
2. config files, in the order you give them with `-c` (you can give several),
3. options on the command line.

So a shared file plus a small per-run file works well:

```bash
cba run -c shared.yaml -c run_variant_b.yaml
cba run -c shared.yaml --codebook other.yaml --run-id other
```

On the command line, a setting is written with dashes (`sample_size` becomes `--sample-size`),
and true/false settings are switched with `--evidence` / `--no-evidence`. A misspelled setting is
an error, both in files and on the command line. Relative paths are read from the folder you run
`cba` in.

To see the final settings without running:

```bash
cba run -c run.yaml --print-config
```

## Check before you run

```bash
cba run -c run.yaml --dry-run
cba run -c run.yaml --dry-run --show-prompt
```

A dry run loads the codebook and the data, draws the sample and builds the prompt, then prints a
summary (number of items, provider, template, prompt condition, sample id, first item). It makes no
model calls. `--show-prompt` also prints the system prompt.

## Which texts are annotated

| Setting | Effect |
|---|---|
| `id_field`, `text_field` | Names of the id and text fields in your data (default `item_id` and `text`). |
| `split_value` | Keep only items whose `split_field` (default `split`) equals this value. If no item matches, all items are used and a warning is printed. |
| `sample_size` | Annotate only this many items. Default: all. |
| `sampling` | `first_n_by_id` (default): the first N items in id order. `random`: N items drawn with `sampling_seed` (default 42). |

Items are always processed in id order. The same settings always give the same sample.

## Output

The run writes `<output_dir>/<run_id>.jsonl`, one record per item, each written as soon as the
item is done. At the end it prints a summary: "Canonical parses" (answers read with the expected
`predictions` key), "Format drift / Fallback key activations" (answers read through another key),
"Non-boolean/score format warnings", and "Parse / Query errors" (items that are `partial` or
`failed`). If any item is not `ok`, `cba run` exits with status 1, so scripts notice.

Exit codes: `0` all items ok, `1` some items not ok, `2` the run could not start (bad config,
missing file, model not installed, resume refused), `130` interrupted with Ctrl+C.

## Interrupting and resuming

Stop a run at any time (Ctrl+C) and start the same command again. Items with an `ok` record are
skipped, the rest are annotated. Records that were not `ok` are moved to
`<run_id>.attempts.jsonl` before they are retried, so earlier failures stay visible. The run file
is read line by line, so a damaged line (for example from a crash) does not affect the records
after it. Damaged lines and duplicate records are also moved to the attempts file, and the file is
rewritten safely.

Resuming is refused when the codebook, the prompt, the sample, the model or a setting that affects
the answers has changed (see [Concepts](../concepts.md#resume-and-the-config-fingerprint)). The message names the settings that differ. Use a
new `run_id` for the new set-up. If you really want to add to a file made with other settings, set
`force_resume: true`.

## Speed

- `concurrency: 4` sends four items to the model at the same time. The output keeps the id order.
  This helps most with API providers. A local Ollama server processes requests one after another
  unless it is configured for parallel requests.
- `timeout` (seconds per request) and `max_retries` (attempts per item, API providers only) control
  how long the tool waits on a slow or busy API.

## What is stored

By default every record contains the item text and the model's full answer. For sensitive or
licensed texts:

| Setting | Effect |
|---|---|
| `store_text: false` | Leave the item text out of the records. |
| `store_raw_response: false` | Leave the model's full answer out (the parsed predictions and reasoning stay). |

You can also remove both afterwards with `cba strip-text`, see
[Reproducibility and sharing](reproducibility.md).
