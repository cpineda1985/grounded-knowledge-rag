from pathlib import Path

import pytest

from akr.answer import KnowledgeBase
from akr.llm import Completion

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    """Returns a canned reply and fixed token counts, so tests never call the API."""

    model = "claude-haiku-5-5"

    def __init__(self, reply: str):
        self.reply, self.calls = reply, 0

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        return Completion(self.reply, self.model, input_tokens=1000, output_tokens=100, latency_s=0.01)


@pytest.fixture
def make_kb():
    def _make(reply: str = "") -> tuple[KnowledgeBase, FakeLLM]:
        llm = FakeLLM(reply)
        return KnowledgeBase.from_config(ROOT / "config.toml", llm=llm), llm

    return _make
