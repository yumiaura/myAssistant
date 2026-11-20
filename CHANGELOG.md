# Changelog

## Unreleased

### Added
- `OLLAMA_API_KEY` setting (`feat/ollama-api-key`): a Bearer token for an
  Ollama behind an authenticating gateway, passed to pydantic-ai's
  `OllamaProvider`. The Spark gateway (`http://10.16.69.251:11435`) admits
  callers by token only since 2026-11-20. Empty keeps today's behaviour for a
  local Ollama. Documented in `env_template.txt` and the README settings table.
