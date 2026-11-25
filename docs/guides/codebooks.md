# Writing a codebook

This guide shows how to write a codebook, check it, and see what the model will receive. The
complete list of keys is in the [codebook format reference](../reference/codebook-format.md).

## Start from a small file

Always list your fields in `render`. Without a `render` list, the tool shows the fields `name`,
`definition`, `example_item` and `example_words`, and any other field (like `indicators` below)
never reaches the model. `cba validate` warns about this.

```yaml
id: feedback
version: "0.1"
prompt:
  role: "You are an expert annotator of customer feedback."
  rules_heading: "### Feedback Coding Rules:"
render:
  - {field: definition, label: Definition}
  - {field: indicators, label: Typical indicators}
  - {field: boundary_note, label: Boundary note}
categories:
  Praise:
    definition: The writer expresses satisfaction with the product, the service or the people involved.
    indicators: [great, thank you, recommend]
    boundary_note: Polite phrases at the end of a complaint are not Praise on their own.
  Complaint:
    definition: The writer reports a problem, a failure or dissatisfaction.
    indicators: [broken, late, refund]
    boundary_note: A neutral question is not a Complaint.
```

`cba init` writes a similar file that you can edit.

## Check it

```bash
cba validate codebook.yaml
```

The command lists errors (the codebook cannot be used) and warnings (it can be used, but probably
not as you intended):

| Message | What to do |
|---|---|
| `'categories' must be a non-empty mapping ...` | Add a `categories:` block with at least one category. |
| `category 'X' must be a mapping of field -> value` | Indent the fields of category X under its name. |
| `render[i] must have 'field' and 'label'` | Every `render` entry needs both keys. |
| `unknown prompt key 'k'` | Only `role`, `task`, `rules_heading` and `reasoning_example` are allowed under `prompt`. |
| `output.type must be 'binary' or 'ordinal'` | Check the spelling of `output.type`. |
| `output.scale.min must be smaller than output.scale.max` | Fix the scale. |
| `examples[i] refers to unknown category 'X'` | An example names a category that is not in `categories`. |
| warning: `render field 'f' is empty or missing in every category` | The field name in `render` does not match any category field, check the spelling. |
| warning: `render field 'f' is empty or missing in: X, Y` | Those categories have no value for the field. That line is left out for them. This is fine if intended. |
| warning: `unknown top-level key 'k' (ignored)` | A misspelled key, or a note that belongs under `provenance`. |

When the codebook has no `prompt:` block, `validate` also notes that the neutral wording of the
`default_v1` template will be used.

## See the prompt

```bash
cba render codebook.yaml
cba render codebook.yaml --text "Great staff, but the parcel was late."
```

The first command prints the system prompt. The second also prints the user prompt for a text.
Read the prompt as if you were the annotator: is every category clear from what is shown?

## Advice on writing categories

- **One idea per field.** A definition, typical words, and a boundary note are easier for the
  model (and for you, when comparing versions) than one long paragraph.
- **Say what a category is not.** Boundary notes that separate neighbouring categories often help
  more than longer definitions.
- **Keep keys short and stable.** The category keys are the labels in the output and in the
  evaluation. Change descriptions freely, but keep the keys the same as in your gold data.
- **Use `render` to experiment.** Keep extra fields in the file and switch them on and off in
  `render`, or build variants automatically with `cba ablate` (see
  [Comparing codebooks](comparing-codebooks.md)).
- **Adapt the wording.** Set `prompt.role` and `prompt.task` for your domain, for example
  "You are an expert annotator of political speeches."
- **Avoid hints in examples.** If you use worked examples, make sure they do not come from the texts
  you evaluate on.

## Writing the codebook in a spreadsheet

Many teams write codebooks in a spreadsheet. `cba convert` turns a table with one row per category
into a codebook. The first column holds the category names, and every other column becomes a field,
shown in column order.

| category | definition | indicators |
|---|---|---|
| Praise | The writer expresses satisfaction. | great, thank you |
| Complaint | The writer reports a problem. | broken, late |

```bash
cba convert codebook.csv -o codebook.yaml --list-column indicators --label indicators="Typical indicators"
```

Useful options (see the [command reference](../reference/cli.md#cba-convert)):

| Option | Effect |
|---|---|
| `--list-column COL` | Split the cells of this column into a list (on commas, or `--list-sep`). |
| `--label FIELD=LABEL` | Label to show in the prompt. Default: the column name with underscores as spaces and a capital first letter. |
| `--columns a,b,c` | Use only these columns, in this order. |
| `--categories A,B,C` | Check that exactly these categories are present (case-insensitive). |
| `--allow-empty` | Allow empty cells. Without it, an empty cell is an error. |
| `--add-name` | Add a `name` field equal to the category key, shown first. |
| `--sheet NAME` | Sheet to read from an `.xlsx` file. |

The command prints a rough token estimate per category (about 1.3 tokens per word), which helps
to keep long codebooks within the model's context window.

`cba convert` writes the `render` list, a `provenance` block with the source file and date, and the
categories. Add a `prompt:` block by hand afterwards if you need domain wording.
