# Example: comparing codebook variants

This example tests which extra fields help the model. It starts from a small base codebook written
in a spreadsheet, adds two more fields one at a time, checks that the prompts differ only in the
codebook, runs every variant on the same ten reviews, and compares the results.

The reviews of a coffee machine and their labels are invented for this example.

| File | Content |
|---|---|
| `base.csv` | The base codebook: one row per category with a definition and typical words |
| `extra_fields.csv` | Two extra fields per category: `context` and `boundary_note` |
| `items.jsonl` | Ten reviews, each labelled by three annotators, with a `platform` field |
| `shared.yaml` | Run settings shared by all variants (local Ollama model `llama3.1:8b`) |

Copy the folder out of the tool's source first, so the generated files do not land in it, and run
every command from inside the copy:

```bash
cp -R examples/ablation ~/cba-ablation
cd ~/cba-ablation
```

> **No Ollama?** Add `--provider stub --model-tag stub` to the four `cba run` commands to try every step offline.
> The stub answers "absent" for every category, so the numbers are not meaningful.

## 1. Turn the spreadsheet into a codebook

```bash
cba convert base.csv -o base.yaml --list-column indicators --label indicators="Typical words" --id reviews_base
cba validate base.yaml
```

`base.yaml` now holds three categories (Quality, Price, Service) with a `definition` and a list of
`indicators`, shown in the prompt as "Definition" and "Typical words".

## 2. Build the variants

```bash
cba ablate --base base.yaml --extension extra_fields.csv --fields context,boundary_note \
    --out-dir variants --prefix reviews --name boundary_note=boundary \
    --label context="Context of use" --label boundary_note="Boundary note"
```

This writes three codebooks into `variants/`:

| File | Fields shown to the model |
|---|---|
| `reviews_context.yaml` | definition, typical words, context of use |
| `reviews_boundary.yaml` | definition, typical words, boundary note |
| `reviews_full.yaml` | definition, typical words, context of use, boundary note |

## 3. Check that only the codebook differs

```bash
cba check-invariance base.yaml variants/reviews_context.yaml variants/reviews_boundary.yaml variants/reviews_full.yaml
```

Every line after the first ("Reference: base.yaml") should start with ✓ and the last line should read
"Prompts differ only in the codebook block".

## 4. Run every variant

```bash
cba run -c shared.yaml --codebook base.yaml --run-id base
cba run -c shared.yaml --codebook variants/reviews_context.yaml --run-id context
cba run -c shared.yaml --codebook variants/reviews_boundary.yaml --run-id boundary
cba run -c shared.yaml --codebook variants/reviews_full.yaml --run-id full
```

Each run writes `runs/<run-id>.jsonl`. With `llama3.1:8b` on a laptop, each run takes about a
minute.

## 5. Compare

```bash
cba evaluate --dataset items.jsonl --runs runs/base.jsonl runs/context.jsonl runs/boundary.jsonl runs/full.jsonl \
    -o runs/compare.xlsx --slice-by platform \
    --codebooks base.yaml variants/reviews_context.yaml variants/reviews_boundary.yaml variants/reviews_full.yaml
```

Open `runs/compare.xlsx`:

- **Sheet1**: F1 and agreement with the annotators for each variant, overall and per category.
- **Platform Slices**: the same per value of the `platform` field.
- **Codebook Agreement**: Krippendorff's alpha between every pair of variants on the same model.
  An alpha of 1 means the two variants gave exactly the same labels.
- **Token Counts**: how many tokens each variant adds to the prompt.
- **Usage**: tokens and response time per run.

To see the reviews where the variants disagree:

```bash
cba disagreements --dataset items.jsonl --runs runs/base.jsonl runs/context.jsonl runs/boundary.jsonl runs/full.jsonl -o runs/disagreements.csv
```

With only ten short reviews the variants often agree completely, and the differences between them
are small. On a real dataset, use a few hundred items and `cba bootstrap` (with the base run first)
to get confidence intervals for the differences.
