# Contributing to codebook-annotator

Thank you for helping. This page explains how to set up a development copy, the rules that keep
results reproducible, and what a pull request needs.

## Development setup

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -e ".[dev]"

Run the checks before every pull request:

    pytest
    ruff check src tests scripts

The tests need no network, no API keys and no local model. Providers are tested against mocked
HTTP servers.

Continuous integration runs the same checks on Linux and macOS with Python 3.11, 3.12 and 3.13,
and builds the documentation in strict mode. A pull request needs all of them to pass.

## Documentation

The documentation is an MkDocs site in `docs/`, configured in `mkdocs.yml`.

    pip install -e ".[docs]"
    mkdocs serve                       # preview at http://127.0.0.1:8000
    mkdocs build --strict              # the check CI runs

The configuration reference and the CLI reference are generated from the code. After changing a
run setting, its description, or any command-line option, regenerate them and commit the result:

    python scripts/gen_reference.py

A test fails when the committed reference pages are out of date.

## Rules that protect reproducibility

Runs made with this tool are research data. Other people must be able to rerun them and get the
same prompts, the same parsing and the same metrics. Some parts of the code are therefore frozen.

1. **Golden tests are never edited to make a change pass.** `tests/test_golden.py` and the files in
   `tests/fixtures/` pin the exact prompt text and parser behaviour. If one of them fails, your change
   alters frozen behaviour. Either change your code, or add the new behaviour behind an option that
   is off by default.
2. **Prompt templates are never edited.** A template in `src/codebook_annotator/templates/` is
   recorded by name in every run. To change wording, add a new template file (for example
   `default_v2.yaml`) and leave the old one as it is.
3. **The run record format only grows.** New fields go after the existing ones, and existing fields
   keep their names and meaning. A change that removes or renames a field needs a new
   `schema_version` and a note in `CHANGELOG.md`.
4. **New features that change the prompt are off by default** and must add a marker to
   `run.prompt_condition` (like `+evidence`), so runs with and without the feature can always be
   told apart.
5. **Anything that affects outputs belongs in the config fingerprint.** If you add a run setting
   that changes what the model sees or how outputs are produced, add it to `_HASHED` in
   `config.py`. Resuming a run with a different value is then refused.

## Common contributions

**A new provider.** Subclass `Provider` in `providers/adapters.py`, implement `url`,
`build_payload`, `headers` and `extract` (or override `query`), register it in `PROVIDERS`, add the
name to `ProviderName` in `config.py`, and add a test with `httpx.MockTransport` that checks the
exact request payload and the usage fields.

**A new evaluation metric or sheet.** Add the computation to `evaluate/metrics.py`, the sheet to
`evaluate/report.py`, a hand-checked unit test on a tiny synthetic example, and a description in
`docs/guides/evaluation.md`.

**A new codebook field or option.** Extend the validation in `codebook.py`, describe the key in the
module docstring, and add tests for both a valid and an invalid codebook.

## Test data

Use synthetic data only: invented texts, ids and labels. Never commit real corpus text, real
annotations or real run outputs, even in small amounts. `runs/` folders are ignored by git.

## Writing style

- Plain, direct English. Short sentences.
- No semicolons in documentation, comments, docstrings, help texts or messages.
- Describe what the tool does on its own terms.

## Pull requests

- One topic per pull request.
- Describe what changed and why, and how you tested it.
- Add an entry under "Unreleased" in `CHANGELOG.md` for anything a user would notice.
- Update the documentation for anything a user would notice, and regenerate the reference pages.
- Tests, ruff and the strict documentation build must pass.
