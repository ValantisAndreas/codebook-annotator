# Getting started

This page takes you from installation to your first evaluated run. It takes about 15 minutes,
most of it for downloading a model.

## 1. Install

You need Python 3.11 or newer. Check which versions you have:

```bash
python3 --version
```

If this shows an older version (macOS ships 3.9), install a newer Python from
[python.org](https://www.python.org/downloads/) or with your package manager, and call it by its
version, for example `python3.12`. Then, in the folder that contains the tool's source code:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[eval]"
```

The `source .venv/bin/activate` line must be repeated in every new terminal before using `cba`.

The `eval` extra adds the packages for the evaluation commands (numpy, pandas, openpyxl).
Without it you can still annotate, but `cba evaluate` and the other evaluation commands will ask
you to install it.

Check the installation:

```bash
cba --version
```

## 2. Set up a local model with Ollama

[Ollama](https://ollama.com) runs open models on your own computer, so your texts never leave
it. Install Ollama from its website and start it. Then download a model and check that it is
there:

```bash
ollama pull llama3.1:8b
ollama list
```

`llama3.1:8b` needs about 5 GB of disk space and works on most recent laptops. Larger models give
better annotations but need more memory. Any model that `ollama list` shows can be used.

!!! tip "No local model?"
    You can also use a model through an API. See [Providers](guides/providers.md). The rest of
    this page works the same, only the `provider` and `model_tag` settings change.

## 3. Create a starter project

Create your projects outside the tool's source folder, for example in your home folder:

```bash
cd ~
cba init my_project
cd my_project
```

This creates three files:

| File | Content |
|---|---|
| `codebook.yaml` | A codebook with three categories: Praise, Complaint, Suggestion |
| `items.jsonl` | Six short texts, each with labels from two human annotators |
| `run.yaml` | The run settings |

Have a look at the codebook, then check it and see the prompt it produces:

```bash
cba validate codebook.yaml
cba render codebook.yaml --text "Thank you, but please open on Sundays."
```

`cba render` prints the exact instructions the model will receive. Reading it is the best way to
see how your codebook comes across.

## 4. A first run without a model

`run.yaml` starts with the `stub` provider. It answers "absent" for every category without calling
any model, which is useful to check that files and settings are in order.

```bash
cba run -c run.yaml --dry-run
cba run -c run.yaml
```

The dry run checks the codebook, the data, the sample and the prompt, and prints a summary,
without annotating. It does not contact the model server, so it cannot check the model or an API
key. The first real run does that. The real run
writes `runs/first_run.jsonl`, one line (a **record**) per text.

## 5. A run with your model

Open `run.yaml` and change the model lines:

```yaml
provider: ollama
model_tag: llama3.1:8b
model_digest: auto
```

`model_digest: auto` records the exact version of the model that Ollama has installed. Give the
run a new name, so it does not mix with the stub run, and start it:

```bash
cba run -c run.yaml --run-id llama_run
```

Settings given on the command line, like `--run-id`, override the config file. The run prints its
progress and a summary at the end. If anything goes wrong for an item (no answer, unreadable
answer) it is recorded, counted in the summary, and retried the next time you start the same run.

## 6. Evaluate

```bash
cba evaluate --dataset items.jsonl --runs runs/llama_run.jsonl -o runs/results.xlsx
```

`runs/results.xlsx` holds F1 scores, agreement with the human annotators (Krippendorff's alpha),
per-category results and the token usage of the run. [Evaluating runs](guides/evaluation.md)
explains every sheet.

## Next steps

- Replace the starter codebook with your own: [Writing a codebook](guides/codebooks.md).
- Bring your own data: [Preparing data](guides/preparing-data.md).
- Compare codebook versions: [Comparing codebooks](guides/comparing-codebooks.md).
- Work through the [examples](examples/index.md).
