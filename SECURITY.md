# Security

## API keys

- Keys are read only from environment variables (`ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`,
  `OPENAI_API_KEY`, or the variable named by `api_key_env`). Never put a key in a config file,
  a codebook or a command line that ends up in your shell history or in a repository.
- Keys are never written to run records, logs or evaluation outputs.

## Data sent to providers

With an API provider, every item text and the full codebook are sent to that provider. Check the
provider's data-use terms and your data agreements before running sensitive or licensed text
through an external API. Local providers (`ollama`, or `openai_compatible` pointing at a local
server) keep the data on your machine.

## Sharing runs

Run records contain the item text and the raw model response by default, and runs made with
`evidence: true` contain verbatim quotes from the text. Use `cba strip-text` before sharing runs
of text you may not redistribute. It removes the text and raw responses and blanks the evidence
quotes. The model's `reasoning` can also quote short pieces of the text, so check it.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use the repository's private
vulnerability reporting ("Security" tab, "Report a vulnerability"), or contact the maintainers
directly. We will acknowledge the report and agree on a fix and disclosure timeline with you.
