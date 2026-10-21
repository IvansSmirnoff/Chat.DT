"""Minimal OpenAI-compatible chat client (OpenRouter, Gemini's compat endpoint, vLLM).

``httpx`` only -- it is already a base dependency and the same code runs in the
container, on a laptop and in Colab. Every agent step and every judge call goes
through :meth:`ChatClient.complete`, which returns the text plus token usage
and, on OpenRouter, the billed cost and the provider that served the call (the
plan pins providers, so drift has to be visible in the logs).

A model is named by a spec ``<provider>:<model>``:

    openrouter:z-ai/glm-4.5-air      OPENROUTER_API_KEY
    gemini:gemini-2.5-flash          GEMINI_API_KEY, else LLM_API_KEY
    vllm:Qwen/Qwen2.5-14B-Instruct   VLLM_BASE_URL (+ optional VLLM_API_KEY)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": ("OPENROUTER_API_KEY",),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": ("GEMINI_API_KEY", "LLM_API_KEY"),
    },
    "vllm": {
        "base_url_env": "VLLM_BASE_URL",
        "key_env": ("VLLM_API_KEY",),
        "key_optional": True,
    },
}

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Thinking is a run factor. ``default`` sends nothing (the model's own default:
# on for every GLM/MiniMax backbone used here); ``on``/``off`` force it through
# OpenRouter's unified ``reasoning`` parameter. Effort levels are not offered:
# GLM-4.5-air's providers ignore them (measured, results/ifcbench/probe/).
REASONING_MODES = ("default", "on", "off")


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_s: float = 0.0
    finish_reason: Optional[str] = None

    @property
    def truncated(self) -> bool:
        """The reply hit max_tokens. With thinking on the reasoning counts
        against the same budget, so the visible text is usually empty."""
        return self.finish_reason == "length"


@dataclass
class ChatClient:
    model: str
    base_url: str
    api_key: Optional[str] = None
    # Hellin's setting for every backbone and the judge (cobbie clients.baml).
    temperature: float = 0.1
    # GLM-4.5-air / GLM-4.7 are reasoning models: reasoning tokens count
    # against max_tokens, and a short budget returns empty content. 8192 is
    # Hellin's client setting (cobbie clients.baml).
    max_tokens: int = 8192
    timeout_s: float = 180.0
    max_retries: int = 5
    extra_body: Dict[str, Any] = field(default_factory=dict)
    spec: str = ""
    reasoning: str = "default"

    def __post_init__(self) -> None:
        if self.reasoning not in REASONING_MODES:
            raise ValueError(f"reasoning must be one of {REASONING_MODES}: {self.reasoning!r}")

    def complete(self, messages: List[Dict[str, str]], **overrides: Any) -> Completion:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": overrides.pop("temperature", self.temperature),
            "max_tokens": overrides.pop("max_tokens", self.max_tokens),
            **self.extra_body,
            **overrides,
        }
        if self.reasoning != "default":
            body.setdefault("reasoning", {"enabled": self.reasoning == "on"})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        delay = 2.0
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            start = time.monotonic()
            try:
                resp = httpx.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    json=body, headers=headers, timeout=self.timeout_s,
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json(), time.monotonic() - start)
                last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                if resp.status_code not in RETRY_STATUS:
                    raise LLMError(f"{self.spec or self.model}: {last_error}")
            if attempt < self.max_retries:
                logger.warning("%s attempt %d failed (%s); retrying in %.0fs",
                               self.spec or self.model, attempt, last_error[:120], delay)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise LLMError(f"{self.spec or self.model}: gave up after {self.max_retries} attempts: {last_error}")

    def _parse(self, data: Dict[str, Any], latency: float) -> Completion:
        if data.get("error"):
            raise LLMError(f"{self.spec or self.model}: {data['error']}")
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.spec or self.model}: response has no choices: {str(data)[:300]}")
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        return Completion(
            text=message.get("content") or "",
            prompt_tokens=usage.get("prompt_tokens") or 0,
            completion_tokens=usage.get("completion_tokens") or 0,
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            cost_usd=usage.get("cost"),
            provider=data.get("provider"),
            model=data.get("model"),
            latency_s=latency,
            finish_reason=choices[0].get("finish_reason"),
        )


def client_from_spec(spec: str, provider_order: Optional[List[str]] = None,
                     **kwargs: Any) -> ChatClient:
    """``openrouter:z-ai/glm-4.5-air`` -> a configured :class:`ChatClient`.

    ``provider_order`` pins the OpenRouter provider(s), e.g. ``["z-ai"]``.
    ``allow_fallbacks: false`` alone does not pin anything: unpinned calls were
    measured landing on three providers within one probe.
    """
    provider, sep, model = spec.partition(":")
    if not sep or provider not in PROVIDERS:
        raise ValueError(f"Model spec must be <provider>:<model> with provider in {sorted(PROVIDERS)}: {spec!r}")
    conf = PROVIDERS[provider]
    base_url = conf.get("base_url") or os.environ.get(conf.get("base_url_env", ""), "")
    if not base_url:
        raise ValueError(f"{spec}: set {conf['base_url_env']}")
    api_key = next((os.environ[k] for k in conf["key_env"] if os.environ.get(k)), None)
    if not api_key and not conf.get("key_optional"):
        raise ValueError(f"{spec}: set one of {', '.join(conf['key_env'])}")

    extra: Dict[str, Any] = dict(kwargs.pop("extra_body", {}) or {})
    if provider == "openrouter":
        # Billed cost in the response, and no silent fallback to another
        # provider mid-run (different quantisations answer differently).
        extra.setdefault("usage", {"include": True})
        extra.setdefault("provider", {"allow_fallbacks": False})
        if provider_order:
            extra["provider"] = {**extra["provider"], "order": list(provider_order)}
    elif provider_order or kwargs.get("reasoning", "default") != "default":
        raise ValueError(f"{spec}: provider pinning and reasoning on/off are OpenRouter options")
    return ChatClient(model=model, base_url=base_url, api_key=api_key,
                      extra_body=extra, spec=spec, **kwargs)
