from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "off"
    model: str = ""
    base_url: str = "http://127.0.0.1:11434"
    api_key_env: str = ""
    system_prompt: str = (
        "You are the language layer of Digital Brain Cognitive OS. "
        "Be useful, safe, concise, and preserve the agent's memory-grounded cognition."
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        return cls(
            provider=str(data.get("provider", "off")),
            model=str(data.get("model", "")),
            base_url=str(data.get("base_url", "http://127.0.0.1:11434")).rstrip("/"),
            api_key_env=str(data.get("api_key_env", "")),
            system_prompt=str(data.get("system_prompt", cls.system_prompt)),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "system_prompt": self.system_prompt,
        }


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    text: str
    provider: str
    model: str
    error: str = ""


class OptionalLLM:
    def complete(
        self,
        config: LLMConfig,
        user_message: str,
        deterministic_reply: str,
        memory_context: list[dict[str, Any]],
    ) -> LLMResult:
        if config.provider == "off":
            return LLMResult(True, deterministic_reply, "off", "deterministic")
        if config.provider == "ollama":
            return self._ollama(config, user_message, deterministic_reply, memory_context)
        if config.provider == "openai-compatible":
            return self._openai_compatible(config, user_message, deterministic_reply, memory_context)
        return LLMResult(False, deterministic_reply, config.provider, config.model, "Unsupported provider")

    def list_models(self, config: LLMConfig) -> list[str]:
        if config.provider != "ollama":
            return []
        try:
            request = urllib.request.Request(f"{config.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return [str(model["name"]) for model in payload.get("models", []) if "name" in model]
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return []

    def _messages(
        self,
        config: LLMConfig,
        user_message: str,
        deterministic_reply: str,
        memory_context: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        context = "\n".join(item.get("text", "")[:500] for item in memory_context[:4])
        return [
            {"role": "system", "content": config.system_prompt},
            {
                "role": "system",
                "content": (
                    "Core cognition already produced this grounded response:\n"
                    f"{deterministic_reply}\n\nRelevant memory:\n{context or 'none'}"
                ),
            },
            {"role": "user", "content": user_message},
        ]

    def _ollama(
        self,
        config: LLMConfig,
        user_message: str,
        deterministic_reply: str,
        memory_context: list[dict[str, Any]],
    ) -> LLMResult:
        model = config.model or "llama3.1"
        payload = {
            "model": model,
            "messages": self._messages(config, user_message, deterministic_reply, memory_context),
            "stream": False,
        }
        return self._post_json(f"{config.base_url}/api/chat", payload, {}, "ollama", model, deterministic_reply)

    def _openai_compatible(
        self,
        config: LLMConfig,
        user_message: str,
        deterministic_reply: str,
        memory_context: list[dict[str, Any]],
    ) -> LLMResult:
        model = config.model or "local-model"
        headers = {}
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env, "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": self._messages(config, user_message, deterministic_reply, memory_context),
            "temperature": 0.4,
        }
        return self._post_json(
            f"{config.base_url}/v1/chat/completions",
            payload,
            headers,
            "openai-compatible",
            model,
            deterministic_reply,
        )

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        provider: str,
        model: str,
        fallback: str,
    ) -> LLMResult:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            return LLMResult(False, fallback, provider, model, str(error))

        if provider == "ollama":
            text = str(data.get("message", {}).get("content", "")).strip()
        else:
            choices = data.get("choices", [])
            text = str(choices[0].get("message", {}).get("content", "")).strip() if choices else ""
        if not text:
            return LLMResult(False, fallback, provider, model, "Provider returned an empty response")
        return LLMResult(True, text, provider, model)
