# Optional features

Four features change how the model is asked. All of them are **off by default**. Each one adds a
marker to the record's `prompt_condition`, so runs with and without a feature are always told
apart, and they are kept apart in the evaluation.

| Feature | Turn on with | `prompt_condition` | Extra model calls |
|---|---|---|---|
| Evidence quotes | `evidence: true` | `+evidence` | none, longer answers |
| Verification pass | `verify: true` | `+verify` | one more call per item |
| Scores instead of yes/no | codebook `output: {type: ordinal}` | `+ordinal` | none |
| Worked examples | `few_shot: true` and codebook `examples` | `few_shot` instead of `zero_shot` | none, longer prompt |

They can be combined, for example `few_shot+evidence+verify`.

## Evidence quotes

```yaml
evidence: true
evidence_policy: report      # or enforce
```

**What changes in the prompt.** The model is asked for a third key, `evidence`: for each category it
judged present, a list of short quotes copied exactly from the text.

**What the tool does.** Every quote is checked against the text. A quote counts as verified if it
occurs in the text, ignoring differences in spacing and capital letters, and ignoring quotation
marks or "..." around it. Everything else must match exactly.

**What is recorded.** `evidence: {category: [{"quote": "...", "verified": true}]}`. With
`evidence_policy: enforce`, a category that the model marked present without any verified quote is
set to absent (and, for scores, its score is removed). The record then also lists those categories in `evidence_enforced` and keeps the
original answer in `predictions_before_evidence`.

**When to use it.** When you want to read why the model chose a category, or want to reduce labels
the text does not support. The Evidence sheet of `cba evaluate` reports how many quotes were
verified.

## Verification pass

```yaml
verify: true
```

**What changes.** After the normal answer, the model gets the same system prompt again, with the
text, its own first answer, and the instruction to re-check that answer against the coding rules,
remove categories the text does not support and add clearly missed ones. Its second answer is the
final one.

**What is recorded.** The final answer at the top level, and the first answer under `first_pass`.
`usage.passes` is 2, and the token counts include both calls. If the first answer could not be
read, the second call is skipped and noted in `parse_notes`.

**When to use it.** Smaller models sometimes correct themselves when asked to check. It doubles the
number of calls, so compare a verified run with an unverified one before using it widely.

## Scores instead of yes/no

```yaml
# in the codebook
output:
  type: ordinal
  scale: {min: 1, max: 10}
  scale_description: "1 = barely expressed, 10 = strongly expressed."
```

**What changes in the prompt.** Instead of true/false, the model gives each category a whole number
on the scale, or `null` when the category is absent. `scale_description` is added to the instructions.
The task sentence of the template still speaks of "presence or absence", so set `prompt.task` in
the codebook to describe scoring, for example "Your task is to score how strongly each of the
following categories is expressed, based on the provided coding rules."

**What is recorded.** `scores: {category: number or null}`. `predictions` is still filled in:
a category counts as present when it has a score, so all yes/no metrics keep working. A score
outside the scale, or not a whole number, is recorded as missing and the item gets status `partial`.

**When to use it.** When strength or intensity matters. Compare with gold scores using
`cba evaluate --gold-scores-field` (mean absolute error and the share within one point, see
[Evaluating runs](evaluation.md)). The [ordinal example](../examples/ordinal.md) shows the full workflow.

## Worked examples (few-shot)

```yaml
# in the codebook
examples:
  - text: "Thanks, the repair was quick and free."
    labels: [Praise]
    reasoning: "Thanks the service."
    evidence: {Praise: ["Thanks, the repair was quick"]}   # used when evidence is on
```

```yaml
# in the run config
few_shot: true
```

**What changes in the prompt.** An `### Examples:` section is added after the categories, showing
each example text and the exact JSON answer expected for it.

**What is recorded.** `prompt_condition` starts with `few_shot`. The examples are part of the
codebook, so they are covered by `codebook_sha256` and `system_prompt_sha256`.

**When to use it.** When the model misreads the answer format or a category. Keep examples short
and never take them from the texts you evaluate on. A run with `few_shot: true` and a codebook
without examples stops with an error.

## Token cost

Every record stores `usage.input_tokens` and `usage.output_tokens` as reported by the provider,
and `usage.latency_ms`, the time the tool waited for the answer. The Usage sheet of `cba evaluate` adds them up per run, so you can compare the
cost of each feature directly.
