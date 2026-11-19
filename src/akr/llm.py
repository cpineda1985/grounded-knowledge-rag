"""LLM client interface, the Anthropic implementation, and cost calculation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> Completion: ...


class AnthropicClient:
    """Calls the Claude Messages API. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self, model: str, max_tokens: int = 600):
        import anthropic  # imported here so retrieval-only use needs no API key

        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, user: str) -> Completion:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=time.perf_counter() - start,
        )


def cost_usd(completion: Completion, pricing: dict) -> float | None:
    """Cost from the configured price table, or None if the model has no price entry."""
    price = pricing.get(completion.model)
    if not price:
        return None
    return (
        completion.input_tokens * price["input_per_mtok"] + completion.output_tokens * price["output_per_mtok"]
    ) / 1_000_000
