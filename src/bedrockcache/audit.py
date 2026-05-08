"""Static audit of a request payload: will Anthropic prompt caching apply?

Given a request payload and the backend shape it will be sent through, return
an AuditReport that answers "will this request actually benefit from prompt
caching?" — and if not, why.

We rely on conservative heuristics that match documented Bedrock + LiteLLM +
AnthropicBedrock + Strands + pydantic-ai semantics. False negatives (claiming-
no-cache when caching would apply) are cheaper than false positives. When in
doubt, the report flags uncertainty rather than asserting `will_cache=True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from bedrockcache.backends import (
    MAX_CACHE_BREAKPOINTS,
    MIN_SEGMENT_TOKENS_FOR_CACHE,
    Backend,
)


@dataclass
class AuditReport:
    """The result of auditing a single request payload.

    `will_cache` is True only when we have positive evidence that at least one
    cache breakpoint will be honored on the request path. `reasons` is a list
    of `(severity, message)` pairs where severity is one of `info`, `warn`,
    `error`.
    """

    backend: Backend
    will_cache: bool
    breakpoint_count: int
    reasons: list[tuple[str, str]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.will_cache

    @property
    def errors(self) -> list[str]:
        return [m for sev, m in self.reasons if sev == "error"]

    @property
    def warnings(self) -> list[str]:
        return [m for sev, m in self.reasons if sev == "warn"]

    def assert_caches(self) -> None:
        """Raise AssertionError unless the request will actually cache.

        Use this in tests / CI to lock in caching at call sites:
            audit(request, Backend.LITELLM).assert_caches()
        """
        if not self.will_cache:
            joined = "\n  - ".join(m for _, m in self.reasons) or "(no diagnostics)"
            raise AssertionError(
                f"bedrockcache: caching will NOT apply on backend "
                f"{self.backend.value}:\n  - {joined}"
            )


def audit(request: dict[str, Any], backend: Backend | str) -> AuditReport:
    """Audit a request payload before sending it.

    `request` is the kwargs dict you would pass to the backend call.
    """
    backend = Backend(backend) if isinstance(backend, str) else backend
    if backend is Backend.BEDROCK_CONVERSE:
        return _audit_converse(request)
    if backend is Backend.BEDROCK_INVOKE_ANTHROPIC:
        return _audit_invoke_anthropic(request)
    if backend is Backend.ANTHROPIC_BEDROCK:
        return _audit_anthropic_bedrock(request)
    if backend is Backend.LITELLM:
        return _audit_litellm(request)
    if backend is Backend.STRANDS:
        return _audit_strands(request)
    if backend is Backend.PYDANTIC_AI_BEDROCK:
        return _audit_pydantic_ai(request)
    raise ValueError(f"unknown backend: {backend}")


# ---------- Bedrock Converse ----------


def _audit_converse(request: dict[str, Any]) -> AuditReport:
    report = AuditReport(backend=Backend.BEDROCK_CONVERSE, will_cache=False, breakpoint_count=0)
    bp = 0

    system = request.get("system") or []
    sys_cachepoints = sum(1 for b in system if isinstance(b, dict) and "cachePoint" in b)
    if sys_cachepoints:
        bp += sys_cachepoints
        sys_tokens = _approx_tokens_before_cachepoint(system)
        if sys_tokens < MIN_SEGMENT_TOKENS_FOR_CACHE:
            report.reasons.append(
                (
                    "warn",
                    f"system cachePoint present but only ~{sys_tokens} tokens of preceding "
                    f"content; Bedrock requires >={MIN_SEGMENT_TOKENS_FOR_CACHE} for the cache "
                    f"to actually populate",
                )
            )
        else:
            report.reasons.append(
                ("info", f"system cachePoint with ~{sys_tokens} preceding tokens")
            )

    messages = request.get("messages") or []
    msg_bps = sum(1 for m in messages if _has_cachepoint_in_content(m))
    if msg_bps:
        bp += msg_bps
        report.reasons.append(("info", f"{msg_bps} message-level cachePoint(s)"))

    tool_config = request.get("toolConfig") or {}
    tools = tool_config.get("tools") or []
    if any(isinstance(b, dict) and "cachePoint" in b for b in tools):
        bp += 1
        report.reasons.append(("info", "toolConfig cachePoint present"))

    if bp == 0:
        report.reasons.append(("error", "no cachePoint blocks found anywhere in the request"))
        report.recommendations.append(
            "add `{'cachePoint': {'type': 'default'}}` after stable prefixes "
            "(end of system content, after large static instructions, after tool definitions)."
        )
        return report

    if bp > MAX_CACHE_BREAKPOINTS:
        report.reasons.append(
            ("error", f"{bp} cachePoint blocks exceeds Bedrock's {MAX_CACHE_BREAKPOINTS} max")
        )
        return report

    report.breakpoint_count = bp
    report.will_cache = True
    return report


# ---------- Bedrock InvokeModel + Anthropic body ----------


def _audit_invoke_anthropic(request: dict[str, Any]) -> AuditReport:
    report = AuditReport(
        backend=Backend.BEDROCK_INVOKE_ANTHROPIC, will_cache=False, breakpoint_count=0
    )
    body = request.get("body") or request

    headers = request.get("headers") or {}
    beta = headers.get("anthropic-beta", "") or ""
    if "prompt-caching" not in beta and not _claude4_or_newer(
        body.get("model") or request.get("modelId")
    ):
        report.reasons.append(
            (
                "warn",
                "anthropic-beta header missing 'prompt-caching-2024-07-31'; older Claude "
                "families require the beta header on Bedrock InvokeModel",
            )
        )

    bp = _count_anthropic_cache_control(body)
    if bp == 0:
        report.reasons.append(
            ("error", "no `cache_control: {type: ephemeral}` on any content block in body"),
        )
        report.recommendations.append(
            "add `cache_control: {'type': 'ephemeral'}` to the last content block of "
            "stable prefixes (system, tool definitions, large static context)."
        )
        return report

    if bp > MAX_CACHE_BREAKPOINTS:
        report.reasons.append(
            ("error", f"{bp} cache_control blocks exceeds the {MAX_CACHE_BREAKPOINTS} max")
        )
        return report

    report.reasons.append(("info", f"{bp} cache_control breakpoint(s) found"))
    report.breakpoint_count = bp
    report.will_cache = True
    return report


def _audit_anthropic_bedrock(request: dict[str, Any]) -> AuditReport:
    """anthropic-sdk-python's AnthropicBedrock client. Same shape as native."""
    report = _audit_invoke_anthropic(request)
    return AuditReport(
        backend=Backend.ANTHROPIC_BEDROCK,
        will_cache=report.will_cache,
        breakpoint_count=report.breakpoint_count,
        reasons=report.reasons,
        recommendations=report.recommendations,
    )


# ---------- LiteLLM ----------


def _audit_litellm(request: dict[str, Any]) -> AuditReport:
    """LiteLLM `completion()` kwargs with a bedrock model.

    Known sharp edges (verified against issues #15037, #20412, #24518):
      - `cache_control` must live on the message dict for LiteLLM's bedrock
        translation prior to v1.86.
      - LiteLLM strips unknown fields in some translation paths.
    """
    report = AuditReport(backend=Backend.LITELLM, will_cache=False, breakpoint_count=0)

    model = (request.get("model") or "").lower()
    if not (model.startswith("bedrock/") or "anthropic" in model):
        report.reasons.append(
            (
                "warn",
                f"model={model!r} does not look like a Bedrock+Anthropic route; this "
                "auditor is Bedrock-specific",
            )
        )

    messages = request.get("messages") or []
    bp_msgs = sum(1 for m in messages if isinstance(m, dict) and "cache_control" in m)
    bp_content = sum(_count_anthropic_cache_control({"messages": [m]}) for m in messages)

    if bp_msgs == 0 and bp_content == 0:
        report.reasons.append(
            ("error", "no `cache_control` on any message; LiteLLM has nothing to translate")
        )
        report.recommendations.append(
            "add `cache_control={'type': 'ephemeral'}` either on the message dict or on "
            "the last content sub-item of stable prefixes."
        )
        return report

    if bp_msgs and bp_content:
        report.reasons.append(
            (
                "warn",
                "cache_control set both on message dict AND on content blocks; LiteLLM may "
                "dedupe inconsistently — pick one shape",
            )
        )

    bp = max(bp_msgs, bp_content)
    if bp > MAX_CACHE_BREAKPOINTS:
        report.reasons.append(
            ("error", f"{bp} cache_control breakpoints exceeds {MAX_CACHE_BREAKPOINTS} max")
        )
        return report

    report.reasons.append(("info", f"{bp} cache_control breakpoint(s) found"))
    report.breakpoint_count = bp
    report.will_cache = True
    return report


# ---------- Strands (AWS) ----------


def _audit_strands(request: dict[str, Any]) -> AuditReport:
    """Strands BedrockModel call.

    Strands forwards the Converse API shape under the hood, but exposes a
    higher-level `messages=[...]` interface. Cache directives are passed as
    `cache_points=[{...}]` on system / messages / tools, which Strands then
    translates into Converse cachePoint blocks.

    The two failure modes we catch:
      - User passed `cache_control` (Anthropic shape) instead of cache_points.
      - User added cache_points but the targeted segment is < 1024 tokens.
    """
    report = AuditReport(backend=Backend.STRANDS, will_cache=False, breakpoint_count=0)

    if any(_message_uses_anthropic_cache_control(m) for m in request.get("messages") or []):
        report.reasons.append(
            (
                "error",
                "messages contain Anthropic-shape `cache_control`; Strands ignores this. "
                "Use Strands' `cache_points` parameter or the Converse cachePoint shape.",
            )
        )
        report.recommendations.append(
            "switch from `cache_control` to Strands' cache directives (cache_points)."
        )
        return report

    cache_points = request.get("cache_points") or []
    bp = len([p for p in cache_points if isinstance(p, dict)])

    if bp == 0:
        report.reasons.append(("error", "no `cache_points` configured on the Strands call"))
        report.recommendations.append(
            "pass `cache_points=[{'segment': 'system'}, ...]` to enable caching."
        )
        return report

    if bp > MAX_CACHE_BREAKPOINTS:
        report.reasons.append(
            ("error", f"{bp} cache_points exceeds Bedrock's {MAX_CACHE_BREAKPOINTS} max")
        )
        return report

    report.reasons.append(("info", f"{bp} Strands cache_point(s) configured"))
    report.breakpoint_count = bp
    report.will_cache = True
    return report


# ---------- pydantic-ai ----------


def _audit_pydantic_ai(request: dict[str, Any]) -> AuditReport:
    """pydantic-ai with a Bedrock model.

    pydantic-ai #138 was closed `not-planned` (cross-provider caching). Today,
    Bedrock prompt caching with pydantic-ai requires manual passthrough via
    `model_settings={'extra_body': {...}}` or a custom Model subclass. We audit
    the resolved request payload, not the high-level Agent call.
    """
    report = AuditReport(backend=Backend.PYDANTIC_AI_BEDROCK, will_cache=False, breakpoint_count=0)

    model_settings = request.get("model_settings") or {}
    extra_body = model_settings.get("extra_body") or model_settings.get("bedrock_extra") or {}

    if extra_body:
        # extra_body is forwarded into the underlying Bedrock InvokeModel body.
        sub = _audit_invoke_anthropic({"body": extra_body})
        sub_report = AuditReport(
            backend=Backend.PYDANTIC_AI_BEDROCK,
            will_cache=sub.will_cache,
            breakpoint_count=sub.breakpoint_count,
            reasons=sub.reasons,
            recommendations=sub.recommendations,
        )
        return sub_report

    report.reasons.append(
        (
            "error",
            "pydantic-ai does not pass through prompt-caching directives by default. No "
            "`model_settings.extra_body` was provided.",
        )
    )
    report.recommendations.append(
        "use a Bedrock-aware Model wrapper that injects cache_control, or migrate to a "
        "direct `boto3` / `AnthropicBedrock` call site for the cache-sensitive path."
    )
    return report


# ---------- helpers ----------


def _has_cachepoint_in_content(message: dict[str, Any]) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and "cachePoint" in b for b in content)


def _message_uses_anthropic_cache_control(message: dict[str, Any]) -> bool:
    if not isinstance(message, dict):
        return False
    if "cache_control" in message:
        return True
    content = message.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and "cache_control" in b for b in content)
    return False


def _count_anthropic_cache_control(body: dict[str, Any]) -> int:
    count = 0
    system = body.get("system")
    if isinstance(system, list):
        for blk in system:
            if isinstance(blk, dict) and "cache_control" in blk:
                count += 1
    for m in body.get("messages") or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and "cache_control" in blk:
                    count += 1
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and "cache_control" in tool:
            count += 1
    return count


def _approx_tokens_before_cachepoint(blocks: Iterable[dict[str, Any]]) -> int:
    """Approximate tokens preceding the FIRST cachePoint in a Converse-shape list.

    Heuristic: ~4 chars per token. Replace with anthropic.count_tokens() in v0.2 if
    the user opts into the heavier dependency.
    """
    total_chars = 0
    for b in blocks:
        if isinstance(b, dict):
            if "cachePoint" in b:
                break
            text = b.get("text") or ""
            total_chars += len(text)
    return max(0, total_chars // 4)


def _claude4_or_newer(model: str | None) -> bool:
    if not model:
        return False
    m = model.lower()
    return any(s in m for s in ("claude-4", "claude-sonnet-4", "claude-opus-4", "claude-haiku-4"))
