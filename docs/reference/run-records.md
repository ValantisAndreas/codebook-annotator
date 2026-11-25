# Run record format (schema_version 2)

One JSON object per line, one line per item, written to `<output_dir>/<run_id>.jsonl`.

```json
{
  "item_id": "f01",
  "text": "…",                                  // omitted with store_text: false
  "run": {
    "dataset": "items", "dataset_revision": null,
    "codebook_id": "my_codebook", "codebook_version": "0.1",
    "model_tag": "llama3.1:8b", "model_digest": "sha256:46e0…", "provider": "ollama",
    "prompt_condition": "zero_shot", "sample_id": "items_n4_51294613",
    "context_window": 65536, "temperature": 0.0, "seed": 42, "timestamp": "2026-11-24T22:59:52Z",
    "tool_version": "0.1.0", "prompt_template": "default_v1",
    "codebook_sha256": "…", "system_prompt_sha256": "…", "config_sha256": "…", "sample_sha256": "…",
    "model_digest_observed": "sha256:46e0…"    // Ollama only
  },
  "parse_status": "ok",                         // ok | partial | failed
  "parse_notes": null,
  "predictions": {"Complaint": false, "Praise": true, "Suggestion": false},
  "raw_response": "…",                          // omitted with store_raw_response: false
  "reasoning": "…",
  "usage": {"input_tokens": 512, "output_tokens": 58, "latency_ms": 7802.6, "passes": 1},
  "schema_version": 2
}
```

Optional fields, present only when the matching feature is on:

| Field | Feature | Content |
|---|---|---|
| `scores` | codebook `output.type: ordinal` | `{category: int or null}`, and `predictions[c]` = score is not null |
| `evidence` | `evidence: true` | `{category: [{"quote": str, "verified": bool}]}`. Verified = the quote occurs in the text (whitespace and case normalised) |
| `evidence_enforced`, `predictions_before_evidence` | `evidence_policy: enforce` | categories set to absent for lack of a verified quote, and the predictions before that |
| `first_pass` | `verify: true` | the first answer (raw_response, predictions, parse fields, scores/evidence). The top-level fields hold the verified answer. If the verification call itself fails, the record is `failed` with every category absent, and `first_pass` still holds the first answer |

`run.prompt_condition` is `zero_shot` or `few_shot`, followed by `+evidence`, `+ordinal`, `+verify` as applicable.

Notes:
- `context_window` records `num_ctx`, which is only sent to Ollama.
- `parse_status` is `partial` when categories are missing or scores are invalid. Missing categories count as absent.
- Format oddities that were recovered are listed in `parse_notes` as `FORMAT WARNING: …`: alternative keys, non-boolean values, string scores. Several notes are separated by a semicolon and a space.
- For an unreadable answer, `parse_notes` holds the parse error and the first 200 characters of the answer. With `store_raw_response: false` the excerpt is left out (`Raw response not stored`), and `cba strip-text` replaces it with `Raw response: [removed]`.
- Records without `config_sha256` (for example from older tools) can still be resumed. Their codebook id, model tag, provider and temperature must match the current configuration.
