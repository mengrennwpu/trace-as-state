from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import ModelConfig
from .prompts import GRAPHWALKS_SYSTEM


@dataclass
class GenerationResult:
    text: str
    reasoning: str
    raw: dict[str, Any]
    elapsed_s: float
    usage: dict[str, Any]


class OpenAICompatibleClient:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.client = httpx.Client(timeout=cfg.timeout_seconds)

    def close(self) -> None:
        self.client.close()

    def _extract_reasoning(self, message: dict[str, Any], raw: dict[str, Any]) -> str:
        candidates = [
            message.get(self.cfg.reasoning_field),
            message.get("reasoning_content"),
            message.get("reasoning"),
            raw.get("reasoning_content"),
        ]
        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip()
        # Content blocks used by some OpenAI-compatible endpoints.
        content = message.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                typ = str(block.get("type", ""))
                txt = block.get("text") or block.get("content") or ""
                if "reason" in typ.lower() and isinstance(txt, str):
                    parts.append(txt)
            if parts:
                return "".join(parts).strip()
        return ""

    def generate(self, user_prompt: str, *, temperature: float | None = None, extra_messages: list[dict[str, str]] | None = None, retry=3) -> GenerationResult:
        messages = [{"role": "system", "content": GRAPHWALKS_SYSTEM}]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_prompt})

        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_completion_tokens": self.cfg.max_output_tokens,
            "top_p": self.cfg.top_p
        }
        if self.cfg.extra_body:
            payload.update(self.cfg.extra_body)

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        for i in range(retry):
            try:
                resp = self.client.post(f"{self.cfg.base_url}/chat/completions", headers=headers, json=payload)
                break
            except httpx.RequestError as e:
                print(f"Request error: {e}. Retrying {i + 1}/{retry}...")
                time.sleep(10)
        elapsed = time.perf_counter() - start
        if resp.status_code >= 400:
            raise RuntimeError(f"Model API error {resp.status_code}: {resp.text[:4000]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Model API returned no choices: {str(data)[:2000]}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):
            text = "".join(
                str(block.get("text") or block.get("content") or "")
                for block in text
                if isinstance(block, dict)
            )
        reasoning = self._extract_reasoning(message, data)
        return GenerationResult(
            text=str(text),
            reasoning=reasoning,
            raw=data,
            elapsed_s=elapsed,
            usage=data.get("usage", {}) or {},
        )
