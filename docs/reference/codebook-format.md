# Codebook format

A codebook is a YAML file. Only `categories` is required.

```yaml
id: my_codebook                  # optional, default: the file name without .yaml
version: "1.0"                   # optional, default: "1.0" in run records
prompt:                          # optional: domain wording for the prompt
  role: "You are an expert annotator of customer feedback."
  task: "Your task is to predict the presence or absence of each of the following categories based on the provided coding rules."
  rules_heading: "### Feedback Coding Rules:"
  reasoning_example: "The writer thanks the staff..."
abstention_instruction: "If you are unsure whether a category applies, mark it as false."   # optional
render:                          # optional: which fields to show, in which order, with which label
  - {field: definition, label: Definition}
  - {field: indicators, label: Typical indicators}
categories:                      # required
  Praise:
    definition: The writer expresses satisfaction.
    indicators: [great, thank you]
  Complaint:
    definition: The writer reports a problem.
    indicators: [broken, late]
output:                          # optional, default {type: binary}
  type: ordinal
  scale: {min: 1, max: 10}
  scale_description: "1 = barely expressed, 10 = strongly expressed."
examples:                        # optional, only used when few_shot is on
  - text: "Thanks, the repair was quick."
    labels: [Praise]
    reasoning: "Thanks the service."
provenance:                      # optional, free-form notes, not shown to the model
  author: Jane Doe
```

## Keys

| Key | Type | Meaning |
|---|---|---|
| `categories` | mapping | **Required.** Category key → mapping of field → value. The category keys are the labels the model must return. They appear in the prompt in alphabetical order. |
| `render` | list of `{field, label}` | Which category fields are printed in the prompt, in this order, with this label. Default below. |
| `prompt` | mapping | Replaces wording from the prompt template. Allowed keys: `role`, `task`, `rules_heading`, `reasoning_example`. |
| `abstention_instruction` | text | One extra sentence placed right after the task sentence, for example to say what to do when unsure. |
| `output` | mapping | `type`: `binary` (default, true/false per category) or `ordinal` (a whole-number score per category, or null when absent). For ordinal output: `scale` with whole-number `min` (default 1) and `max` (default 10), and an optional `scale_description` added to the answer instructions. A score that is not a finite whole number within the scale is recorded as missing and the item becomes `partial`. |
| `examples` | list | Worked examples, shown only when the run setting `few_shot` is on. Each has `text` and, for binary output, `labels` (the categories present), or for ordinal output, `scores` (category → score). Optional: `reasoning`, and `evidence` (category → list of quotes) used when `evidence` is on. |
| `id` | text | Codebook id written into every record. Default: the file name without extension. A run can override it with `codebook_id`. |
| `version` | text | Codebook version written into every record. Default `1.0`. A run can override it with `codebook_version`. |
| `provenance` | mapping | Free-form notes about where the codebook comes from. Never sent to the model. |
| `schema_version` | number | Optional marker of the codebook format version. |

Other top-level keys are ignored with a warning.

## Field values

Inside a category, any field name can be used. How a value is printed:

| Value | Printed as |
|---|---|
| text | the text as written |
| list, e.g. `[great, thank you]` | `'great', 'thank you'` |
| mapping with a `words` list, e.g. `{words: [great]}` | `'great'` |
| empty text or `null` | the line is left out for that category |

## Default `render`

Without a `render` list, these fields are shown:

| Field | Label |
|---|---|
| `name` | Name |
| `definition` | Definition |
| `example_item` | Example Item |
| `example_words` | Illustrative Words |

## How the category block looks

For the codebook above, the category part of the system prompt is:

```text
- **Category**: Complaint
  * **Definition**: The writer reports a problem.
  * **Typical indicators**: 'broken', 'late'

- **Category**: Praise
  * **Definition**: The writer expresses satisfaction.
  * **Typical indicators**: 'great', 'thank you'
```

Use `cba render codebook.yaml` to see the complete prompt.
