"""`cba init`: write a starter codebook, items file and run config into a folder."""

from __future__ import annotations

from pathlib import Path

CODEBOOK = """\
# Starter codebook. Replace the categories and their fields with your own.
# Check it with:   cba validate codebook.yaml
# See the prompt:  cba render codebook.yaml --text "Some example text"
id: my_codebook
version: "0.1"
prompt:
  role: "You are an expert annotator of customer feedback."
  task: "Your task is to predict the presence or absence of each of the following categories based on the provided coding rules."
  rules_heading: "### Feedback Coding Rules:"
  reasoning_example: "The writer thanks the staff and asks for longer opening hours..."
render:
  - {field: definition, label: Definition}
  - {field: indicators, label: Typical indicators}
  - {field: boundary_note, label: Boundary note}
categories:
  Praise:
    definition: The writer expresses satisfaction with the product, the service or the people involved.
    indicators: [great, thank you, loved, recommend]
    boundary_note: Polite phrases at the end of a complaint are not Praise on their own.
  Complaint:
    definition: The writer reports a problem, a failure or dissatisfaction.
    indicators: [broken, late, disappointed, refund]
    boundary_note: A neutral question about how something works is not a Complaint.
  Suggestion:
    definition: The writer proposes a change or an improvement.
    indicators: [could you, it would be better, please add]
    boundary_note: A complaint without a proposed change is not a Suggestion.
"""

ITEMS = """\
{"item_id": "f01", "text": "Thank you for the quick delivery, I will recommend you to friends.", "labels_per_annotator": {"a1": ["Praise"], "a2": ["Praise"]}, "label_majority": ["Praise"], "channel": "email"}
{"item_id": "f02", "text": "The parcel arrived two weeks late and the box was broken.", "labels_per_annotator": {"a1": ["Complaint"], "a2": ["Complaint"]}, "label_majority": ["Complaint"], "channel": "web"}
{"item_id": "f03", "text": "Great staff, but it would be better if you opened on Sundays.", "labels_per_annotator": {"a1": ["Praise", "Suggestion"], "a2": ["Suggestion"]}, "label_majority": ["Praise", "Suggestion"], "channel": "web"}
{"item_id": "f04", "text": "What are your opening hours?", "labels_per_annotator": {"a1": [], "a2": []}, "label_majority": [], "channel": "email"}
{"item_id": "f05", "text": "Could you please add a vegetarian option to the menu?", "labels_per_annotator": {"a1": ["Suggestion"], "a2": ["Suggestion"]}, "label_majority": ["Suggestion"], "channel": "web"}
{"item_id": "f06", "text": "I was charged twice and nobody answers the phone. Very disappointed.", "labels_per_annotator": {"a1": ["Complaint"], "a2": ["Complaint"]}, "label_majority": ["Complaint"], "channel": "email"}
"""

RUN = """\
# Run configuration. Every key can also be given on the command line, e.g. --sample-size 2.
# Start with the offline stub provider (answers "absent" for everything) to check the setup:
#   cba run -c run.yaml --dry-run
#   cba run -c run.yaml
# Then switch to a real model, for example a local Ollama model:
#   provider: ollama
#   model_tag: llama3.1:8b      # any model listed by `ollama list`
#   model_digest: auto
run_id: first_run
codebook: codebook.yaml
dataset: items.jsonl
provider: stub
model_tag: stub
temperature: 0.0
output_dir: runs
"""

FILES = {"codebook.yaml": CODEBOOK, "items.jsonl": ITEMS, "run.yaml": RUN}


def init_project(folder: str | Path, *, force: bool = False) -> list[Path]:
    """Write the starter files. Existing files are only overwritten with force=True."""
    folder = Path(folder)
    existing = [folder / n for n in FILES if (folder / n).exists()]
    if existing and not force:
        raise FileExistsError("already exists: " + ", ".join(str(p) for p in existing) + " (use --force to overwrite)")
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in FILES.items():
        (folder / name).write_text(content, encoding="utf-8")
        written.append(folder / name)
    return written
