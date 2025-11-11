"""Registry and routing for LLM providers."""

from __future__ import annotations

from typing import Dict

from .base import LLMError, LLMProvider, LLMResult, MediaEvent


class LLMClient:
    """Maintains a registry of providers and invokes them by model name."""

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        key = provider.model_name.lower()
        self._providers[key] = provider

    def get(self, model_name: str) -> LLMProvider:
        provider = self._providers.get(model_name.lower())
        if not provider:
            raise LLMError(f"No provider registered for model '{model_name}'.")
        return provider

    def caption_event(self, model_name: str, event: MediaEvent) -> LLMResult:
        provider = self.get(model_name)
        return provider.caption_event(event)


__all__ = ["LLMClient"]


