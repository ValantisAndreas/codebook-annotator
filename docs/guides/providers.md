# Providers

The `provider` setting chooses where the model runs. `model_tag` is the model's name as that
provider knows it.

| Provider | Model runs | Needs |
|---|---|---|
| `ollama` | on your computer, with [Ollama](https://ollama.com) | Ollama running, the model pulled |
| `openai_compatible` | any server with an OpenAI-style chat API: OpenAI, OpenRouter, vLLM, LM Studio, llama.cpp server, Ollama's `/v1` endpoint | `api_base`, and a key if the server needs one |
| `claude_api` | Anthropic API | `ANTHROPIC_API_KEY` |
| `mistral_api` | Mistral API | `MISTRAL_API_KEY` |
| `stub` | nowhere, answers "absent" for every category | nothing |
| `replay` | nowhere, returns the answers stored in an earlier run file | `replay_from` |

## Ollama

```yaml
provider: ollama
model_tag: llama3.1:8b
model_digest: auto
```

- The tool connects to `http://localhost:11434`, or to the address in the `OLLAMA_HOST`
  environment variable, or to `ollama_host` if you set it.
- Before the first item it checks that Ollama is reachable and that the model is installed. If
  not, it stops with a message that lists the installed models.
- `model_digest: auto` records the full digest of the installed model. If you write a digest
  yourself and Ollama reports a different one, the run stops, because you would not be running the
  model you think. Set `strict_digest: false` to only warn.
- `num_ctx` (default 65536) sets the context window. Lower it if the model runs out of memory, but
  keep it larger than your system prompt plus the longest text.
- Ollama is asked for JSON output. Each item gets one attempt. A failed item is recorded and
  retried when you resume the run.

## API providers

API keys are read from environment variables, never from config files:

```bash
export ANTHROPIC_API_KEY="..."      # for claude_api
export MISTRAL_API_KEY="..."        # for mistral_api
export OPENAI_API_KEY="..."         # for openai_compatible with api.openai.com
```

To read the key from another variable, set `api_key_env: MY_KEY_VARIABLE`. For
`openai_compatible`, `OPENAI_API_KEY` is only sent when `api_base` points at `api.openai.com`.
For any other server (a local server, OpenRouter, DeepSeek, ...), the tool sends a key only if
`api_key_env` names the variable that holds it, so your OpenAI key never reaches another server.
Error messages leave out passwords and query strings that are part of `api_base`.

```yaml
provider: claude_api
model_tag: "<model id>"            # a pinned model id from the provider's model list
```

- API providers retry an item up to `max_retries` times (default 5), with growing waits, when the
  API is busy (status 429) or fails.
- `max_tokens` limits the answer length. `claude_api` uses 1024 when it is not set. Mistral and
  OpenAI-compatible servers get no limit unless you set one. The `ollama` provider does not use
  this setting.
- `seed` is sent to Ollama, Mistral and OpenAI-compatible servers. The Anthropic API takes no seed.
- `api_base` replaces the default address of the Anthropic or Mistral API, for example for a proxy.

Prefer pinned model names (with a date or version) over names like `latest`, so that later
readers know which model produced the labels. The name you give is stored in every record.

## OpenAI-compatible servers

```yaml
provider: openai_compatible
api_base: http://localhost:8000/v1   # vLLM. LM Studio: http://localhost:1234/v1
model_tag: my-model
```

`api_base` is required and must point to the folder that contains `/chat/completions`. The tool
asks for a JSON answer (`response_format: json_object`). If a server does not support this, set
`json_mode: false`.

Ollama also offers this kind of endpoint at `http://localhost:11434/v1`. With `ollama` as the
provider you get the model digest and the installed-model check, so prefer it for Ollama.

## Stub and replay

`stub` answers every item with all categories absent, instantly and offline. Use it to test files
and settings before spending time or money on a real model.

`replay` reads the raw answers from an earlier run file and returns them again, by item id. Use it
to re-read a run with a newer version of the tool, or to rebuild a run file, without calling a
model:

```yaml
provider: replay
replay_from: runs/llama_run.jsonl
model_tag: llama3.1:8b
```

Items without a stored answer are recorded as failed.
