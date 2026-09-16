from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    reasoning_field: str = "reasoning_content"
    temperature: float = 1.0
    top_p: float | None = 0.95
    max_output_tokens: int | None = None
    timeout_seconds: float = 1200.0
    extra_body: dict[str, Any] | None = None


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip().lower() in {"", "null", "none"}:
        return None
    return int(value)


def load_model_config(cfg: dict[str, Any]) -> ModelConfig:
    load_dotenv()
    m = cfg.get("model", {})
    base_url = os.getenv("API_BASE_URL", str(m.get("base_url", ""))).rstrip("/")
    api_key = os.getenv("API_KEY", str(m.get("api_key", "")))
    model = os.getenv("MODEL", str(m.get("name", "")))
    reasoning_field = os.getenv("REASONING_FIELD", str(m.get("reasoning_field", "reasoning_content")))
    temperature = float(os.getenv("TEMPERATURE", m.get("temperature", 1.0)))
    top_p_raw = os.getenv("TOP_P", str(m.get("top_p", 0.95)))
    top_p = None if top_p_raw.lower() in {"none", "null", ""} else float(top_p_raw)
    max_output_tokens = _optional_int(os.getenv("MAX_OUTPUT_TOKENS", m.get("max_output_tokens")))
    timeout_seconds = float(os.getenv("TIMEOUT_SECONDS", m.get("timeout_seconds", 1200)))
    extra_body = m.get("extra_body") or {}

    if not base_url or not api_key or not model:
        raise RuntimeError("Missing API_BASE_URL/API_KEY/MODEL. Configure .env first.")

    return ModelConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_field=reasoning_field,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        extra_body=extra_body,
    )
