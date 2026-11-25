## What and why

<!-- What does this change, and why is it needed? Link the issue if there is one. -->

## How it was tested

<!-- Commands run, new tests added. -->

## Checklist

- [ ] `pytest` and `ruff check src tests` pass
- [ ] Golden tests and fixtures are unchanged, or the change is behind an option that is off by default
- [ ] No prompt template was edited (new wording goes in a new template file)
- [ ] Run record fields were only added, not renamed or removed
- [ ] New output-affecting settings are part of the config fingerprint (`_HASHED` in `config.py`)
- [ ] Only synthetic data in tests and examples
- [ ] `CHANGELOG.md` updated under "Unreleased" for user-visible changes
