"""Tests for the bedrockcache CLI."""

import json
import subprocess
import sys
from pathlib import Path


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "request.json"
    p.write_text(json.dumps(payload))
    return p


def test_cli_audit_litellm_no_cache(tmp_path: Path):
    request = {
        "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
    }
    file = _write(tmp_path, request)
    res = subprocess.run(
        [sys.executable, "-m", "bedrockcache.cli", "audit", str(file), "--backend", "litellm"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    assert "will_cache:        False" in res.stdout
    assert "no `cache_control" in res.stdout


def test_cli_audit_strict_exits_nonzero(tmp_path: Path):
    request = {
        "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
    }
    file = _write(tmp_path, request)
    res = subprocess.run(
        [sys.executable, "-m", "bedrockcache.cli", "audit", str(file),
         "--backend", "litellm", "--strict"],
        capture_output=True, text=True,
    )
    assert res.returncode == 1


def test_cli_audit_passing_request(tmp_path: Path):
    request = {
        "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [
            {"role": "system", "content": "x" * 5000, "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "hi"},
        ],
    }
    file = _write(tmp_path, request)
    res = subprocess.run(
        [sys.executable, "-m", "bedrockcache.cli", "audit", str(file),
         "--backend", "litellm", "--strict"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    assert "will_cache:        True" in res.stdout
