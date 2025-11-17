"""Registry and routing for LLM providers."""

from __future__ import annotations

from typing import Dict

from .base import LLMError, LLMProvider, LLMResult, MediaEvent


class LLMClient:
    """Maintains a registry of providers and invokes them by model name."""

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        primary = self._normalize(provider.model_name)
        self._providers[primary] = provider
        # Keep a secondary alias without the models/ prefix to match typical CLI usage.
        alias = self._strip_prefix(primary)
        self._providers.setdefault(alias, provider)

    def get(self, model_name: str) -> LLMProvider:
        provider = self._providers.get(self._normalize(model_name))
        if not provider:
            raise LLMError(f"No provider registered for model '{model_name}'.")
        return provider

    def caption_event(self, model_name: str, event: MediaEvent) -> LLMResult:
        provider = self.get(model_name)
        return provider.caption_event(event)

    @staticmethod
    def _normalize(model_name: str) -> str:
        return model_name.strip().lower()

    @staticmethod
    def _strip_prefix(model_name: str) -> str:
        if "/" in model_name:
            return model_name.split("/", 1)[-1]
        return model_name


__all__ = ["LLMClient"]


