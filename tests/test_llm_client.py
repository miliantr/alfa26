import pytest

from agent.llm_client import LLMClient, extract_json

SCHEMA = {"type": "object", "properties": {"hypotheses": {"type": "array"}}, "required": ["hypotheses"]}


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('<think>long reasoning {not json}</think>\n{"a": 2}', {"a": 2}),
    ('Here you go:\n```json\n{"a": [1, 2,],}\n```\nthanks', {"a": [1, 2]}),
    ("{'a': True, 'b': None}", {"a": True, "b": None}),
    ('{"a": {"b": [1, 2', {"a": {"b": [1, 2]}}),     # truncated output gets closed
    ('[{"x": 1}]', [{"x": 1}]),
    ("no json at all", None),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_off_and_missing_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = LLMClient(mode="real")
    assert c.mode == "off" and not c.available()
    assert c.chat_json("s", "u") is None


def test_mock_good_and_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_LLM_MOCK", "good")
    c = LLMClient(mode="mock", run_dir=tmp_path, max_calls=2)
    r = c.chat_json("sys", "columns: \"x1:numeric nun=5\"", schema=SCHEMA, purpose="hypotheses")
    assert r and r["hypotheses"]
    c.chat_json("sys", "other prompt", schema=SCHEMA, purpose="hypotheses")
    assert c.calls == 2 and not c.available()
    assert c.chat_json("sys", "third", schema=SCHEMA, purpose="hypotheses") is None  # budget exhausted
    assert (tmp_path / "llm_log.jsonl").exists()


def test_cache_avoids_repeat_calls(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_MOCK", "good")
    c = LLMClient(mode="mock", max_calls=5)
    a = c.chat_json("s", "same", schema=SCHEMA, purpose="hypotheses")
    b = c.chat_json("s", "same", schema=SCHEMA, purpose="hypotheses")
    assert a == b and c.calls == 1


@pytest.mark.parametrize("behaviour", ["garbage", "error"])
def test_failures_disable_client(monkeypatch, behaviour):
    monkeypatch.setenv("AGENT_LLM_MOCK", behaviour)
    c = LLMClient(mode="mock", max_calls=10, max_failures=3)
    for i in range(5):
        assert c.chat_json("s", f"u{i}", schema=SCHEMA, purpose="hypotheses") is None
    assert c.disabled_reason and c.calls == 3


def test_schema_violation_is_failure(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_MOCK", "good")
    c = LLMClient(mode="mock")
    strict = {"type": "object", "required": ["something_else"]}
    assert c.chat_json("s", "u", schema=strict, purpose="other") is None
    assert c.failures == 1
