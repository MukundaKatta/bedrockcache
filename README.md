# bedrockcache

[![ci](https://github.com/MukundaKatta/bedrockcache/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/bedrockcache/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/bedrockcache.svg)](https://pypi.org/project/bedrockcache/)
[![python](https://img.shields.io/pypi/pyversions/bedrockcache.svg)](https://pypi.org/project/bedrockcache/)

Audit and fix Anthropic prompt caching when calling Claude on AWS Bedrock through any abstraction stack.

## Why

Anthropic prompt caching cuts cost by ~90% on stable prefixes. On the native Anthropic API the SDK applies it automatically. But if you call Claude through Bedrock — directly via `boto3`, indirectly via LiteLLM, via `AnthropicBedrock`, as a Strands or pydantic-ai backend — caching breaks silently at one of the abstraction boundaries, and you find out via the bill.

A real Bedrock+LiteLLM stack ran up [$37,901 in 9 days](https://news.ycombinator.com/item?id=47933355) on uncached input tokens because each layer "nominally supported" caching but caching was not actually applying. `bedrockcache` catches that class of bug at request time.

## Install

```bash
pip install bedrockcache
```

## Audit a request before sending

```python
from bedrockcache import audit, Backend

request = {
    "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
    "messages": [{"role": "user", "content": "hi"}],
}
report = audit(request, Backend.LITELLM)
print(report.will_cache)        # False
print(report.errors)            # ['no `cache_control` on any message; LiteLLM has nothing to translate']
print(report.recommendations)   # ['add cache_control={...} either on the message dict or …']
```

## Lock it in CI

```python
def test_my_rag_caches():
    request = build_my_rag_request(query="...")
    audit(request, Backend.BEDROCK_CONVERSE).assert_caches()
```

`assert_caches()` raises `AssertionError` with the specific reason if any layer is dropping the directive.

## CLI

```bash
$ bedrockcache audit request.json --backend litellm --strict
backend:           litellm
will_cache:        False
breakpoint_count:  0
reasons:
  [error] no `cache_control` on any message; LiteLLM has nothing to translate
recommendations:
  - add `cache_control={'type': 'ephemeral'}` either on the message dict or on the last content sub-item of stable prefixes.
$ echo $?
1
```

## Supported backends

| Backend | Identifier | What it audits |
|---|---|---|
| Bedrock Converse API | `Backend.BEDROCK_CONVERSE` | `cachePoint` blocks across system / messages / toolConfig, ≥1024 token segment rule, 4-breakpoint cap |
| Bedrock InvokeModel + Anthropic body | `Backend.BEDROCK_INVOKE_ANTHROPIC` | `cache_control` blocks, missing `anthropic-beta: prompt-caching-2024-07-31` for older Claude families |
| `anthropic.AnthropicBedrock` | `Backend.ANTHROPIC_BEDROCK` | same as InvokeModel |
| LiteLLM `completion()` | `Backend.LITELLM` | `cache_control` placement (message dict vs content), Bedrock-only model check, dual-placement warning |
| Strands `BedrockModel` | `Backend.STRANDS` | `cache_points` parameter present; rejects misuse of Anthropic-shape `cache_control` |
| pydantic-ai bedrock | `Backend.PYDANTIC_AI_BEDROCK` | `model_settings.extra_body` passthrough; recursive audit of inner Anthropic body |

## What it explicitly is not

- Not a router. Not a proxy. Not a generic LLM cost tracker.
- Bedrock-only by design.
- Not a replacement for Phoenix or Langfuse.
- Not coupled to any framework — zero runtime dependencies.

## Roadmap

- v0.2: real Anthropic tokenizer in place of the char-based heuristic.
- v0.3: `Reporter.from_cloudwatch(log_group=...)` — sample CloudWatch InvokeModel logs and dollar-ize the cache miss rate.
- v0.4: `suggest_breakpoints(system_prompt, model=...)` — recommend optimal placement for long prefixes.

## Develop

```bash
pip install -e ".[dev]"
pytest -v
```

## License

MIT
