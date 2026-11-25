# Preparing data

## Input files for `cba run`

The texts to annotate can be in any of these formats:

| Format | Notes |
|---|---|
| `.jsonl` | One JSON object per line. Recommended. |
| `.json` | A list of objects. |
| `.csv`, `.tsv` | First row is the header. UTF-8. |
| `.xlsx` | First row is the header. First sheet, or the sheet named in `dataset_sheet`. |

Each item needs an **id** and a **text**. By default these fields are called `item_id` and `text`.
For other names, set `id_field` and `text_field`:

```yaml
dataset: reviews.csv
id_field: review_id
text_field: body
```

Ids must be unique. They are read as text, so `7` and `"7"` are the same id, also in the gold file
used for evaluation. Items are processed in id order. When every id is a whole number, the order is
numeric (2, 9, 10). Otherwise ids are compared as text (`"a10"` comes before `"a2"`). Other fields are allowed and
ignored by `cba run`. They are useful for the evaluation.

## Gold data for evaluation

The evaluation commands read a JSONL file with one item per line:

```json
{"item_id": "f03", "text": "Great staff, but please open on Sundays.",
 "labels_per_annotator": {"a1": ["Praise", "Suggestion"], "a2": ["Suggestion"]},
 "label_majority": ["Praise", "Suggestion"], "channel": "web"}
```

| Field | Meaning |
|---|---|
| `item_id` | Must match the ids in the run files. |
| `label_majority` | The agreed labels. An empty list means "no category". |
| `labels_per_annotator` | Optional. Each annotator's labels. Needed for agreement with individual annotators and for out-of-range counts. |
| any other field | Can be used to slice the results, for example `--slice-by channel`. |

The same file can serve as input for `cba run` and as gold data for `cba evaluate`.

## From one row per annotation to one item per text

Annotation tools often export one row per (text, annotator):

| text | annotator | annotation |
|---|---|---|
| Great staff, but please open on Sundays. | a1 | Praise, Suggestion |
| Great staff, but please open on Sundays. | a2 | Suggestion |

`cba aggregate` groups such a table into items:

```bash
cba aggregate annotations.csv -o items.jsonl --labels "Praise,Complaint,Suggestion,None" --drop None
```

- Rows with the same text become one item (or rows with the same value in `--group-field`).
- Labels in one cell are split on commas (`--label-sep`).
- `--labels` lists the accepted labels. Others are kept as `UNKNOWN:<label>` with a warning, so you
  can spot typos. `--map RAW=LABEL` renames a raw label to an accepted one.
- `--drop` removes a label, for example a "none" label, so that it counts as "no category".
- A label is in `label_majority` when at least half of the item's annotators gave it. Change the
  share with `--threshold`.
- Item ids are made from the text: `--id-prefix` (default `item_`) plus 12 digits derived from the
  text. The same text always gets the same id.
- `--keep FIELD` copies an item-level column (from the first row of the item) into the item.
- Column names are set with `--text-field`, `--annotator-field` and `--labels-field`.

## Flagging word overlap with a codebook

Do the texts contain the words listed in the codebook? Models often do well on texts that use the
codebook's own words, and less well on texts that express a category in other words.
`cba overlap` adds a true/false field:

```bash
cba overlap items.jsonl -o items_flagged.jsonl --codebook codebook.yaml --field indicators
```

A text is flagged when it contains any word of the field (in any category), as a whole word and
regardless of capitals. Then compare the two groups:

```bash
cba evaluate --dataset items_flagged.jsonl --runs runs/llama_run.jsonl -o runs/results.xlsx \
    --slice-by lexical_overlap --bool-labels lexical_overlap=overlaps,no_overlap
```
