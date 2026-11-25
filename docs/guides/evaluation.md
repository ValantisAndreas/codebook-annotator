# Evaluating runs

The evaluation tools compare annotation runs (the JSONL files written by `cba run`)
with human annotations. They need the `eval` extra:

    pip install -e ".[eval]"         # in the folder with the tool's source code

| Command | Output |
|---|---|
| `cba evaluate` | Excel workbook: classification metrics, Krippendorff's alpha, slices, model and codebook agreement, token counts, optional ordinal and evidence sheets |
| `cba bootstrap` | Paired bootstrap confidence intervals and p-values against a baseline run, including out-of-range labels and coverage |
| `cba disagreements` | Item-level CSV of disagreements between runs and with the annotators |
| `cba annotator-disagreements` | Item × category CSV of where the human annotators disagreed |

The same commands also run without the main CLI: `python -m codebook_annotator.evaluate.cli <command> ...`.

## Gold data

The gold file is JSONL, one item per line. Three fields are read, and their names are options:

| Option | Default | Content |
|---|---|---|
| `--id-field` | `item_id` | Item id, must match `item_id` in the run records |
| `--majority-field` | `label_majority` | List of consensus labels (empty list = no label) |
| `--annotators-field` | `labels_per_annotator` | `{annotator_id: [labels]}` |

The per-annotator field is optional. Without it, everything that needs individual
annotators (model-to-annotator alpha, out-of-range labels, coverage, disagreement
vote strings) is left empty or skipped, and a note is printed. F1, Jaccard and
Hamming loss are still computed against the majority field.

Any other field (for example `region` or `lexical_overlap`) can be used for slicing.

## Categories

Category order matters only for row order and for the order in which per-category
values are averaged (which can change the last bits of a float). Resolution:

1. `--categories A B C`: explicit list and order.
2. `--codebook path.yaml`: the order of `categories` in that codebook file.
3. Otherwise the sorted union of the `predictions` keys in the runs
   (for `annotator-disagreements`: the sorted union of annotator labels).

A category missing from a record's `predictions` counts as absent.

## `cba evaluate`

    cba evaluate --dataset gold.jsonl --runs runs/*.jsonl -o results.xlsx \
        --codebook codebooks/my_codebook.yaml --slice-by region

Files named `<run_id>.attempts.jsonl` (earlier failed attempts kept by a resume) are skipped with a
note, so a `runs/*.jsonl` pattern is safe to use.

**Undefined F1.** When a category has no gold labels and no predictions in a row or slice, its
F1 is undefined and counts as 0 (in macro-F1, micro-F1 and the per-category rows). Small slices
can therefore show a low macro-F1 although the model made no mistake. Jaccard counts an item with
no gold labels and no predictions as full agreement (1).

Sheets:

- **Sheet1**: one row per run × category (`ALL` plus each category) × slice
  (`pooled`, and with `--sheet1-slice-field F` also one row per value of `F`). Columns:
  `n_items`, micro/macro F1, Jaccard, Hamming loss, model-to-annotator alpha (mean over
  categories for `ALL`), and model-to-model alpha (mean pairwise alpha between runs of the
  same arm, or the string `NA` when there is no second run).
  **Minimum-cell floor:** a Sheet1 row with fewer than `--min-cell-floor` items (default 5) states
  `n_items` and leaves the metrics empty, and a note is printed. The floor applies to Sheet1 only.
  The slice sheets and Category Slices always show their values, so read small slices with care.
- **`<Field> Slices`**: one sheet per `--slice-by FIELD`, metrics per value of the field.
- **Category Slices**: F1 and model-to-annotator alpha per category.
- **Model Agreement**: pairwise alpha between any two runs of the same arm, usually the same
  codebook on different models, or repeated runs. Only items in the gold file are used.
- **Codebook Agreement**: pairwise alpha between arms run on the same model, overall,
  per value of each `--agreement-slice-by` field (default: the `--slice-by` fields, and a
  value needs at least 2 items) and per category. Only items in the gold file are used. When one
  model has several runs of the same arm, the last one given is used.
- **Token Counts**: estimated rendered tokens per category (about 1.3 tokens per word) for
  the codebooks given with `--codebooks`, or else `<--codebooks-dir>/<codebook_id>.yaml`. Each
  codebook is named by its `id` key, or by its file name when it has none.
- **Ordinal**: only for runs made with ordinal scores (`+ordinal` in `prompt_condition`)
  and only when `--gold-scores-field F` names a gold field of the form
  `{category: score or null}`. Per category and `ALL`: `n_pairs` (items with a gold score),
  `n_pred_null` (of those, items where the model gave no score), `mae` (over pairs where
  both are scores), and `acc_within_1` (share of `n_pairs` with |gold − predicted| ≤ 1, where
  a missing prediction counts as a miss). Items with a predicted but no gold score are
  covered by the presence metrics.
- **Evidence**: only for runs made with evidence quotes (`+evidence`). Per category and
  `ALL`, over the items in the gold set: `n_present` (predictions marked present),
  `n_quotes`, `n_verified` (quotes found verbatim in the text), `share_verified`, and
  `share_present_supported` (present predictions with at least one verified quote).
- **Usage**: one row per run whose records carry usage data, over all records in the run file:
  `n_records`, `n_with_usage` (records with a `usage` field, whose token counts can still be empty,
  for example for `replay` runs), `input_tokens_total` and `_mean`, `output_tokens_total` and
  `_mean`, `latency_ms_mean`, `_median` and `_p95`, `latency_s_total` and `passes_mean`
  (2 for runs with the verification pass). Use it to compare the cost and speed of models,
  codebooks and optional features.

**Arms.** Runs are grouped into arms for the agreement measures. A run's arm is its
`codebook_id`. When the run used a non-default prompt condition (for example
`few_shot+evidence`), the arm is `<codebook_id> [<condition>]`, so runs that share a
codebook but differ in prompt condition are compared, not merged.

Records with `parse_status` other than `ok` are counted as they are and reported in an audit
line. A `partial` record keeps the categories it did report (missing ones count as absent). A
`failed` record has every category absent. Text-stripped runs work, since the text is never
read from the runs.

**Reading run and gold files.** Ids are matched as text, so a gold id `7` matches the run id
`"7"`. Files are read line by line: a damaged line is skipped with a note and does not affect the
records after it. When an id appears twice, the last record is used and a note is printed.

**Conventions worth knowing.**

- Each run is scored on its own items that are in the gold file, so two runs in Sheet1 can have a
  different `n_items`. `cba bootstrap` uses only the items common to all runs.
- Model-to-annotator alpha is Krippendorff's alpha for nominal data over all annotators plus the
  model, so it also reflects how much the annotators disagree among themselves.
- Alpha is 1 when nobody varies (for example a category nobody chose and the model never
  predicted), and 0 when no item has two or more ratings. These values enter the `ALL` average.
- In per-category rows, Jaccard is tp / (tp + fp + fn) for that category (1 when the category is
  absent everywhere), and micro and macro F1 are the same value. In `ALL` rows, Jaccard is the
  mean over items.
- The minimum-cell floor also applies to model-to-model alpha: pairs with fewer common items than
  the floor are skipped, and the cell shows `NA` when no pair is left.

Options for slicing:

- `--slice-by FIELD` (repeatable): one slice sheet per field. Values are rendered as strings,
  and a missing value is `(missing)`.
- `--bool-labels FIELD=TRUE_LABEL,FALSE_LABEL`: render a field by truthiness, with a missing
  value counting as false (for example `lexical_overlap=overlaps,no_overlap`).
- `--agreement-slice-by FIELD` (repeatable): slices for the Codebook Agreement sheet.
- `--sheet1-slice-field FIELD`: extra Sheet1 rows per value of one field.

`--merge` updates an existing workbook. Rows of the runs being evaluated replace their old
rows in Sheet1 and in every other per-run sheet that was computed again. A sheet of the old
workbook that was not computed again (for example a slice sheet for a field you did not pass
this time) is kept unchanged, and a note lists it. Text cells such as `NA` are kept as text. The Token Counts, Model Agreement and
Codebook Agreement sheets are kept from the old file only when nothing new was computed for them.

## `cba bootstrap`

    cba bootstrap --dataset sample.jsonl --runs baseline.jsonl armB.jsonl armC.jsonl \
        --save-npz boot.npz --output-csv boot.csv

The first run is the baseline. Items scored are those present in the gold file and in every
run. Metrics per run, with a 95% percentile interval from `--resamples` (default 5000)
paired resamples drawn with `--seed` (default 42):

- `OOR%`: share of items with at least one predicted label that **no** annotator chose
  (out of range of the human annotations). `OOR/item`: mean number of such labels. Items
  without annotator labels are left out of both, with a note.
- `coverage%`: share of annotator-chosen labels (union over annotators, pooled over items)
  that the model also predicted.
- `macro-F1`, `micro-F1`, `Jaccard` against the majority labels.

Only the evaluated categories count: gold labels outside the category list are ignored, as in
`cba evaluate`. When no item is in the gold file and in every run, the command stops with an
error. Two run files with the same name (for example `a/run.jsonl` and `b/run.jsonl`) are an error,
because each run is named after its file.

For each non-baseline run it prints the paired difference with its interval and a two-sided
p-value (paired bootstrap test: share of null-centred replicates at least as extreme as the
observed difference, with a floor of 1/B), and an efficiency ratio: OOR labels avoided per
point of coverage lost (reported only when coverage drops by more than 0.5 points and in over
90% of the replicates). Without per-annotator labels only the last three metrics are given.
In the `--output-csv` table, a p-value below 1/B is written as 1/B, and the column `p_at_floor`
is true for those rows.

An F1 that is undefined, because a category has no gold labels and no predictions in the items
scored, counts as 0 here and in `cba evaluate`. Jaccard counts an item with no gold labels and no
predictions as full agreement (1) in both commands.

## `cba disagreements`

    cba disagreements --dataset gold.jsonl --runs runs/*_mymodel.jsonl -o disagreements.csv --item-field region

Exports the items (present in all runs) where any two runs differ, any run differs from the
majority labels, or a run predicts a category no annotator chose or omits a category every
annotator chose. Columns: `item_id`, `text`, the `--item-field` fields, `gold_majority`,
`labels_per_annotator` (vote counts and each annotator's labels), then `pred_<key>` and
`reasoning_<key>` per run. `--key codebook_id` (default) labels runs by codebook (runs that
share a codebook id collapse to the last one given, and a warning names them), and `--key run_id`
uses file names. `--categories` limits the prediction columns and all comparisons to those
categories. The printed summary counts the items that are in the gold file and in every run.
Items where the runs disagree with each other are listed first. `--empty-label` sets the
display string for an empty label set (default `(none)`).

## `cba annotator-disagreements`

    cba annotator-disagreements --dataset gold.jsonl -o annotator_disagreements.csv --split all

For each item and category, `disagree_<cat>` is true if at least one annotator chose the
category and at least one annotator of the item did not. `votes_<cat>` lists the votes and
`any_disagreement` summarises. `--split` keeps items whose `--split-field` equals the value
(`all` keeps everything, default `test`, falling back to all items if none match). The file
starts with three `#` comment lines, the last one a generation timestamp.
