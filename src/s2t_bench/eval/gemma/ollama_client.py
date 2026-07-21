"""Minimal Ollama client using the stdlib (no extra deps).

Talks to the local Ollama server (default http://localhost:11434). Supports
`format` for JSON / JSON-schema-constrained output and a system prompt.

Pull models first, e.g.:  ollama pull gemma3:1b   /   ollama pull gemma3:4b
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


def generate(
    model: str,
    prompt: str,
    system: str | None = None,
    fmt: dict | str | None = None,
    host: str = DEFAULT_HOST,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    """Single-shot generation. `fmt` may be "json" or a JSON-schema dict."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    if fmt is not None:
        payload["format"] = fmt

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(f"Ollama request failed ({exc}). Is `ollama serve` running?")
    return body.get("response", "")


def list_models(host: str = DEFAULT_HOST, timeout: float = 10.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(f"Cannot reach Ollama ({exc}). Is `ollama serve` running?")
    return [m.get("name", "") for m in body.get("models", [])]
