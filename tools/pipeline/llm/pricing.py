"""Pricing utilities for LLM token accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping


@dataclass
class PricingRule:
    """Per-model pricing expressed in USD per million tokens."""

    input_cost_per_million: float
    output_cost_per_million: float
    fixed_overhead_usd: float = 0.0

    def cost(self, *, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_million
        completion_cost = (completion_tokens / 1_000_000) * self.output_cost_per_million
        return prompt_cost + completion_cost + self.fixed_overhead_usd


class PricingTable:
    """Holds pricing rules keyed by model name."""

    def __init__(self, rules: Mapping[str, PricingRule] | None = None) -> None:
        self._rules: MutableMapping[str, PricingRule] = {}
        if rules:
            for model, rule in rules.items():
                self._rules[self._normalize(model)] = rule

    def register(self, model_name: str, rule: PricingRule) -> None:
        self._rules[self._normalize(model_name)] = rule

    def get(self, model_name: str) -> PricingRule | None:
        return self._rules.get(self._normalize(model_name))

    def cost_for_tokens(self, model_name: str, *, prompt_tokens: int, completion_tokens: int) -> float | None:
        rule = self.get(model_name)
        if not rule:
            return None
        return rule.cost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    @staticmethod
    def _normalize(model_name: str) -> str:
        value = model_name.strip().lower()
        if value.startswith("models/"):
            return value.split("/", 1)[1]
        return value


__all__ = ["PricingRule", "PricingTable"]


