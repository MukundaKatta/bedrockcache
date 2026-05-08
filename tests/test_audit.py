"""Tests for the audit() primitive across all six backends."""

import pytest

from bedrockcache import Backend, audit


# ---------- Bedrock Converse ----------


def test_converse_no_cachepoint_blocks_caching():
    request = {
        "modelId": "anthropic.claude-sonnet-4-5-v1:0",
        "system": [{"text": "You are a helpful assistant. " * 200}],
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    report = audit(request, Backend.BEDROCK_CONVERSE)
    assert not report.will_cache
    assert any("no cachePoint" in m for _, m in report.reasons)
    assert report.recommendations


def test_converse_system_cachepoint_caches():
    long_system = "You are a helpful assistant. " * 400  # ~1200 tokens
    request = {
        "modelId": "anthropic.claude-sonnet-4-5-v1:0",
        "system": [{"text": long_system}, {"cachePoint": {"type": "default"}}],
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    report = audit(request, Backend.BEDROCK_CONVERSE)
    assert report.will_cache
    assert report.breakpoint_count == 1


def test_converse_short_system_under_min_warns():
    short_system = "You are a helpful assistant."
    request = {
        "modelId": "anthropic.claude-sonnet-4-5-v1:0",
        "system": [{"text": short_system}, {"cachePoint": {"type": "default"}}],
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    report = audit(request, Backend.BEDROCK_CONVERSE)
    assert report.will_cache
    assert any(sev == "warn" and "1024" in msg for sev, msg in report.reasons)


def test_converse_too_many_breakpoints_blocks_caching():
    request = {
        "modelId": "anthropic.claude-sonnet-4-5-v1:0",
        "system": [{"text": "x" * 5000}, {"cachePoint": {"type": "default"}}],
        "messages": [
            {"role": "user", "content": [{"text": "a"}, {"cachePoint": {"type": "default"}}]},
            {"role": "user", "content": [{"text": "b"}, {"cachePoint": {"type": "default"}}]},
            {"role": "user", "content": [{"text": "c"}, {"cachePoint": {"type": "default"}}]},
            {"role": "user", "content": [{"text": "d"}, {"cachePoint": {"type": "default"}}]},
        ],
        "toolConfig": {"tools": [{"cachePoint": {"type": "default"}}]},
    }
    report = audit(request, Backend.BEDROCK_CONVERSE)
    assert not report.will_cache
    assert any("exceeds" in m for _, m in report.reasons)


# ---------- Bedrock InvokeModel (Anthropic body shape) ----------


def test_invoke_anthropic_no_cache_control_blocks():
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": [{"type": "text", "text": "x" * 5000}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    report = audit({"body": body, "modelId": "anthropic.claude-sonnet-4-5-v1:0"},
                   Backend.BEDROCK_INVOKE_ANTHROPIC)
    assert not report.will_cache
    assert any("no `cache_control" in m for _, m in report.reasons)


def test_invoke_anthropic_with_cache_control():
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": [
            {"type": "text", "text": "x" * 5000, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    report = audit({"body": body, "modelId": "anthropic.claude-sonnet-4-5-v1:0"},
                   Backend.BEDROCK_INVOKE_ANTHROPIC)
    assert report.will_cache
    assert report.breakpoint_count == 1


def test_invoke_anthropic_old_family_warns_without_beta_header():
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": [{"type": "text", "text": "x" * 5000, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    report = audit(
        {"body": body, "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
        Backend.BEDROCK_INVOKE_ANTHROPIC,
    )
    assert any("anthropic-beta" in m for _, m in report.reasons)


# ---------- AnthropicBedrock ----------


def test_anthropic_bedrock_passes_through_invoke_logic():
    body = {
        "system": [{"type": "text", "text": "x" * 5000, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    report = audit({"body": body}, Backend.ANTHROPIC_BEDROCK)
    assert report.will_cache
    assert report.backend is Backend.ANTHROPIC_BEDROCK


# ---------- LiteLLM ----------


def test_litellm_no_cache_control_blocks():
    request = {
        "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
    }
    report = audit(request, Backend.LITELLM)
    assert not report.will_cache
    assert any("no `cache_control" in m for _, m in report.reasons)


def test_litellm_with_cache_control_on_message():
    request = {
        "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [
            {
                "role": "system",
                "content": "x" * 5000,
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "hi"},
        ],
    }
    report = audit(request, Backend.LITELLM)
    assert report.will_cache
    assert report.breakpoint_count == 1


def test_litellm_dual_cache_control_warns():
    request = {
        "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "x" * 5000, "cache_control": {"type": "ephemeral"}}
                ],
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "hi"},
        ],
    }
    report = audit(request, Backend.LITELLM)
    assert report.will_cache
    assert any("both on message dict AND on content" in m for _, m in report.reasons)


def test_litellm_non_bedrock_model_warns():
    request = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": "x" * 5000, "cache_control": {"type": "ephemeral"}}
        ],
    }
    report = audit(request, Backend.LITELLM)
    assert any("Bedrock-specific" in m for _, m in report.reasons)


# ---------- Strands ----------


def test_strands_no_cache_points_blocks():
    request = {
        "model_id": "anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
    }
    report = audit(request, Backend.STRANDS)
    assert not report.will_cache
    assert any("no `cache_points`" in m for _, m in report.reasons)


def test_strands_with_cache_points():
    request = {
        "model_id": "anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
        "cache_points": [{"segment": "system"}, {"segment": "tools"}],
    }
    report = audit(request, Backend.STRANDS)
    assert report.will_cache
    assert report.breakpoint_count == 2


def test_strands_anthropic_shape_caught():
    request = {
        "model_id": "anthropic.claude-sonnet-4-5-v1:0",
        "messages": [
            {"role": "user", "content": "hi", "cache_control": {"type": "ephemeral"}}
        ],
    }
    report = audit(request, Backend.STRANDS)
    assert not report.will_cache
    assert any("Strands ignores this" in m for _, m in report.reasons)


# ---------- pydantic-ai ----------


def test_pydantic_ai_default_does_not_cache():
    request = {
        "model": "bedrock:anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
    }
    report = audit(request, Backend.PYDANTIC_AI_BEDROCK)
    assert not report.will_cache
    assert any("does not pass through" in m for _, m in report.reasons)


def test_pydantic_ai_with_extra_body_can_cache():
    extra_body = {
        "system": [{"type": "text", "text": "x" * 5000, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    request = {
        "model": "bedrock:anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
        "model_settings": {"extra_body": extra_body},
    }
    report = audit(request, Backend.PYDANTIC_AI_BEDROCK)
    assert report.will_cache
    assert report.backend is Backend.PYDANTIC_AI_BEDROCK


# ---------- assert_caches() ----------


def test_assert_caches_passes_when_caching():
    body = {
        "system": [{"type": "text", "text": "x" * 5000, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    audit({"body": body, "modelId": "anthropic.claude-sonnet-4-5-v1:0"},
          Backend.BEDROCK_INVOKE_ANTHROPIC).assert_caches()


def test_assert_caches_raises_when_not_caching():
    request = {
        "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
    }
    with pytest.raises(AssertionError) as exc:
        audit(request, Backend.LITELLM).assert_caches()
    assert "bedrockcache" in str(exc.value)
    assert "no `cache_control" in str(exc.value)


# ---------- properties ----------


def test_audit_report_errors_warnings_split():
    request = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }
    report = audit(request, Backend.LITELLM)
    assert report.errors
    assert report.warnings  # the non-bedrock model warning


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        audit({}, "totally-made-up-backend")
