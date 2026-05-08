# Changelog

## 0.1.0 — initial release

- Static `audit()` primitive across six backends:
  - `bedrock-converse` — Bedrock Converse API
  - `bedrock-invoke-anthropic` — Bedrock InvokeModel with raw Anthropic body
  - `anthropic-bedrock` — `anthropic.AnthropicBedrock` client
  - `litellm` — LiteLLM `completion()` with bedrock model id
  - `strands` — Strands `BedrockModel` call shape
  - `pydantic-ai-bedrock` — pydantic-ai with `model_settings.extra_body`
- `AuditReport.assert_caches()` for use in pytest / CI.
- `bedrockcache audit FILE.json --backend X --strict` CLI.
- 21 tests, no runtime dependencies.
