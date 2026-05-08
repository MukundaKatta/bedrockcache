"""Backend identifiers and shape detectors.

A "backend" is the call-site shape we are inspecting. The audit logic differs
per shape because each one transports the cache directive differently.
"""

from enum import Enum


class Backend(str, Enum):
    BEDROCK_CONVERSE = "bedrock-converse"
    BEDROCK_INVOKE_ANTHROPIC = "bedrock-invoke-anthropic"
    ANTHROPIC_BEDROCK = "anthropic-bedrock"
    LITELLM = "litellm"
    STRANDS = "strands"
    PYDANTIC_AI_BEDROCK = "pydantic-ai-bedrock"


# Bedrock prompt caching constraints (verified May 2026):
#   - Up to 4 cachePoint blocks per request, distributed across system,
#     messages, and toolConfig.
#   - System and messages each need at least 1024 tokens of preceding content
#     for caching to actually apply (Claude Sonnet/Opus 4.x).
MIN_SEGMENT_TOKENS_FOR_CACHE = 1024
MAX_CACHE_BREAKPOINTS = 4
