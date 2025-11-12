"""Pricing utilities for LLM token accounting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _PROJECT_ROOT / "config"
DEFAULT_PRICING_PATH = _CONFIG_DIR / "llm_pricing.json"


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
                self.register(model, rule)

    def register(self, model_name: str, rule: PricingRule) -> None:
        self._rules[self._normalize(model_name)] = rule

    def register_aliases(self, aliases: Iterable[str], rule: PricingRule) -> None:
        for alias in aliases:
            self.register(alias, rule)

    def get(self, model_name: str) -> PricingRule | None:
        return self._rules.get(self._normalize(model_name))

    def cost_for_tokens(self, model_name: str, *, prompt_tokens: int, completion_tokens: int) -> float | None:
        rule = self.get(model_name)
        if not rule:
            return None
        return rule.cost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    @classmethod
    def load_from_file(cls, path: Path | str) -> PricingTable:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            LOGGER.warning("LLM pricing file not found: %s", resolved)
            return cls()

        try:
            with resolved.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # pragma: no cover - configuration error
            LOGGER.warning("Failed to load pricing data from %s: %s", resolved, exc)
            return cls()

        table = cls()
        if not isinstance(data, Mapping):
            LOGGER.warning("Pricing data in %s is not a mapping; ignoring.", resolved)
            return table

        for model_name, payload in data.items():
            if not isinstance(payload, Mapping):
                LOGGER.warning("Pricing rule for %s is not a mapping; skipping.", model_name)
                continue
            try:
                rule = PricingRule(
                    input_cost_per_million=float(payload["input_cost_per_million"]),
                    output_cost_per_million=float(payload["output_cost_per_million"]),
                    fixed_overhead_usd=float(payload.get("fixed_overhead_usd", 0.0)),
                )
            except Exception as exc:  # pragma: no cover - config error
                LOGGER.warning("Invalid pricing rule for %s: %s", model_name, exc)
                continue

            aliases = payload.get("aliases") or []
            if not isinstance(aliases, Iterable):
                aliases = []

            table.register(model_name, rule)
            table.register_aliases(aliases, rule)

        return table

    @classmethod
    def load_default(cls) -> PricingTable:
        return cls.load_from_file(DEFAULT_PRICING_PATH)

    @staticmethod
    def _normalize(model_name: str) -> str:
        value = model_name.strip().lower()
        if value.startswith("models/"):
            return value.split("/", 1)[1]
        return value


__all__ = ["DEFAULT_PRICING_PATH", "PricingRule", "PricingTable"]


