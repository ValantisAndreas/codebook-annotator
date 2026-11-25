# Comparing codebooks

A common question is which parts of a codebook help the model. The tool supports a clean
comparison: build variants that differ in exactly one thing, check that the prompts differ only in
the codebook, run every variant on the same texts with the same model, and compare.

A complete, runnable version of this guide is the [codebook comparison example](../examples/ablation.md).

## 1. Build variants with `cba ablate`

Start from a **base codebook** and a table with the **extra fields** you want to test, one row per
category:

| category | context | boundary_note |
|---|---|---|
| Praise | Usually at the start or the end of a review. | Polite closings are not Praise. |
| Complaint | Often names a product part or a date. | Questions are not Complaints. |

```bash
cba ablate --base base.yaml --extension extra_fields.csv --fields context,boundary_note \
    --out-dir variants --prefix fb --label context="Context of use" --label boundary_note="Boundary note"
```

This writes one codebook per added field and one with all of them:

| File | Content |
|---|---|
| `variants/fb_context.yaml` | base + `context` |
| `variants/fb_boundary_note.yaml` | base + `boundary_note` |
| `variants/fb_full.yaml` | base + both fields |

The extra fields must be new: they cannot already exist in the base codebook. The base fields are
copied verbatim, and the command checks this. The new fields are added at the
end of `render`. The base codebook's `prompt`, `abstention_instruction`, `output` and `examples` are
carried over unchanged. Name the variants with `--name FIELD=SHORT` (for example
`--name boundary_note=boundary`) and `--full-name`. `--no-single` builds only the all-fields
variant, `--no-full` only the single-field ones.

## 2. Check that only the codebook changed

```bash
cba check-invariance base.yaml variants/fb_context.yaml variants/fb_boundary_note.yaml variants/fb_full.yaml
```

The first file is the reference. For every other codebook the command checks that:

- the part of the prompt before the categories is identical,
- the answer instructions after the categories are identical,
- the category block is different.

It prints a diff when something outside the category block differs, for example when one codebook
has its own `prompt:` wording or an `abstention_instruction`. Pass `--template`, `--few-shot` or
`--evidence` to check the prompts as they will be used in your runs.

## 3. Run every variant the same way

Put the shared settings in one file and give each run only what differs:

```yaml
# shared.yaml
dataset: items.jsonl
provider: ollama
model_tag: llama3.1:8b
model_digest: auto
temperature: 0.0
output_dir: runs
```

```bash
cba run -c shared.yaml --codebook base.yaml --run-id base
cba run -c shared.yaml --codebook variants/fb_context.yaml --run-id fb_context
cba run -c shared.yaml --codebook variants/fb_full.yaml --run-id fb_full
```

All runs then use the same items, model and settings, and the records show which codebook each one
used (`codebook_id`, `codebook_sha256`).

## 4. Compare

```bash
cba evaluate --dataset items.jsonl --runs runs/base.jsonl runs/fb_context.jsonl runs/fb_full.jsonl -o runs/compare.xlsx
```

- **Sheet1** and **Category Slices** show each variant's F1 and agreement with the annotators.
- **Codebook Agreement** shows how similar the variants' labels are to each other on the same
  model (Krippendorff's alpha per pair, overall, per slice and per category).
- **Token Counts** shows how much longer each variant makes the prompt (pass the codebooks with
  `--codebooks`).

`cba disagreements` lists the texts where the variants disagree, which is the quickest way to see
*what* a field changes. For confidence intervals of the differences, use `cba bootstrap` with the
base run first. See [Evaluating runs](evaluation.md).
