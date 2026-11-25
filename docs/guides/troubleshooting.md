# Troubleshooting

## Ollama

**`Cannot reach Ollama at http://localhost:11434 (...). Is ollama serve running?`**
Ollama is not running, or runs at another address. Start the Ollama app (or `ollama serve` in a
terminal). If it runs elsewhere, set `ollama_host` or the `OLLAMA_HOST` environment variable.

**`Ollama model 'X' is not installed. Installed: ...`**
The model name in `model_tag` does not match an installed model. Check the exact name with
`ollama list` (including the part after the colon, such as `:8b`), or download the model with
`ollama pull X`.

**`Configured model_digest ... does not match the digest Ollama reports ...`**
The installed model is not the one recorded in your config, for example after an update. Either
use `model_digest: auto` to record the current one, or install the recorded version. Set
`strict_digest: false` to run anyway with a warning.

**Very slow runs or timeouts.** Large models on a laptop can take a minute per item. Try a smaller
model, lower `num_ctx`, or raise `timeout`. A timed-out item is recorded as failed and retried when
you resume.

## Configuration

**`Invalid run configuration:`** followed by a list of settings. Typical causes: a misspelled key
(`Extra inputs are not permitted`), a missing required setting (`Field required`: `run_id`,
`codebook`, `dataset`, `provider` and `model_tag` are required), or a value of the wrong kind.
`cba run ... --print-config` shows the combined settings.

**`Error: ANTHROPIC_API_KEY environment variable is not set.`** (or `MISTRAL_API_KEY`)
Set the key in your shell (`export ANTHROPIC_API_KEY=...`) before running, or name another
variable with `api_key_env`.

**`field 'text' not found in dataset records. Available: ...`**
Your data uses other field names. Set `id_field` and `text_field`.

**`few_shot is on but the codebook has no 'examples'`**
Add an `examples` list to the codebook, or turn `few_shot` off.

## Resuming

**`... contains N record(s) produced with a different configuration ...`**
The run file was made with another codebook, prompt, model, sample or setting. The message lists
what differs, for example `Differences: temperature: 0.0 in the file, 0.5 now.` This protects the
file from mixing set-ups. Use a new `run_id`, or move the old file away. If you really want to mix
them, set `force_resume: true`.

**`Interrupted. Finished items are saved. Run the same command again to resume.`**
You pressed Ctrl+C. Every finished item is already in the run file. The exit code is 130.

## Answers the tool could not read

Look at `parse_status` and `parse_notes` in the records, and at the summary at the end of the run.

| `parse_notes` | Meaning |
|---|---|
| `Harness query error: ...` | The model did not answer (connection problem, timeout, API error). |
| `Empty response from model` | The model answered with nothing. |
| `JSON parse error: ...` | The answer was not valid JSON, often extra text around it. |
| `Missing 'predictions' key in JSON object` | Valid JSON, but not in the requested shape. |
| `Missing category predictions in JSON: A, B` | Some categories were left out. They count as absent, status `partial`. |
| `FORMAT WARNING: Fallback key used: ...` | The model used another key name (for example `results`) instead of `predictions`. The answer was still read. |
| `FORMAT WARNING: non-boolean value for 'A' ...` | The model wrote, for example, `"false"` as text. It was read by its meaning. |
| `Invalid score for 'A': ...` | Ordinal output: the score was outside the scale or not a whole number. |

Failed and partial items are retried when you start the run again. If a model keeps failing on
the format, try a larger model, `few_shot` with one or two examples, or a lower `num_ctx` if the
prompt plus text is very long.

## Evaluation

**`This command needs '<package>', which comes with the eval extra.`** (for example numpy or pandas)
Install the evaluation packages. In the folder with the tool's source code: `pip install -e ".[eval]"`.

**A sheet is missing from the workbook.** Sheets without content are not written. Model Agreement
needs at least two runs with the same codebook and prompt condition. The Ordinal sheet needs
`--gold-scores-field`. The Evidence and Usage sheets need runs that contain evidence or usage data.

**Metrics are empty in a Sheet1 row.** The row has fewer items than `--min-cell-floor` (default 5).
Use a larger sample, or lower the floor with `--min-cell-floor 1` if you accept unstable values.
