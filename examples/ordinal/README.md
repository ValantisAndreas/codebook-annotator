# Example: scores instead of yes/no

This example asks the model how strongly each category is present, on a scale from 1 to 10, and
compares the scores with gold scores. The headlines and their scores are invented for this example.

| File | Content |
|---|---|
| `frames.yaml` | A codebook with three news frames and `output: {type: ordinal}` |
| `headlines.jsonl` | Eight headlines with gold scores in the field `gold_scores` and the frames present in `label_majority` |
| `run.yaml` | Run settings (local Ollama model `llama3.1:8b`) |

Copy the folder out of the tool's source first, so the outputs do not land in it, and run every
command from inside the copy:

```bash
cp -R examples/ordinal ~/cba-ordinal
cd ~/cba-ordinal
```

> **No Ollama?** Add `--provider stub --model-tag stub` to the `cba run` command to try the steps offline. The stub
> answers "absent" for every category, so there are no scores to compare.

## 1. Look at the codebook and its prompt

```bash
cba validate frames.yaml
cba render frames.yaml
```

The answer instructions at the end of the prompt ask for a whole number from 1 to 10 for each
frame, or `null` when the frame is absent:

```text
1. "predictions": A dictionary mapping each category key to an integer score from 1 to 10
   indicating how strongly the category is expressed, or null if the category is absent. ...
```

## 2. Run

```bash
cba run -c run.yaml
```

Each record in `runs/frames_scores.jsonl` has a `scores` field, for example
`{"Conflict": 8, "Economy": 9, "Human interest": null}`, and `predictions` with true for every frame
that got a score. The record's `prompt_condition` is `zero_shot+ordinal`.

## 3. Compare with the gold scores

```bash
cba evaluate --dataset headlines.jsonl --runs runs/frames_scores.jsonl -o runs/results.xlsx --gold-scores-field gold_scores
```

The **Ordinal** sheet compares the scores per frame and overall:

| Column | Meaning |
|---|---|
| `n_pairs` | Headlines with a gold score for this frame |
| `n_pred_null` | Of those, headlines where the model gave no score |
| `mae` | Mean absolute difference between gold and model score, where both exist |
| `acc_within_1` | Share of `n_pairs` where the model score is at most one point away (a missing score counts as a miss) |

Sheet1 and Category Slices treat a frame as present when it has a score, so F1, Jaccard and
Hamming loss are there too, computed against `label_majority`. The items have no per-annotator
labels, so the model-to-annotator alpha columns stay empty.
