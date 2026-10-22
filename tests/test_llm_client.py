import pytest

from agent_llm_client import LLMClient, extract_json

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


def test_hanging_endpoint_is_given_up_quickly(monkeypatch):
    """An endpoint that accepts connections and never answers must cost about two
    timeouts in total, not retries x calls (it used to be 9 timeouts + backoff)."""
    import socket
    import threading
    import time

    import openai  # noqa: F401 - a cold SDK import must not count towards the timing

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(20)
    held = []

    def accept():
        while True:
            try:
                conn, _ = srv.accept()
                held.append(conn)
            except OSError:
                return
    threading.Thread(target=accept, daemon=True).start()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "qwen-test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.getsockname()[1]}/v1")
    c = LLMClient(mode="real", timeout=1.0, max_calls=30)
    t0 = time.time()
    while c.available():
        assert c.chat_json("s", f"u{c.calls}", purpose="hypotheses") is None
    elapsed = time.time() - t0
    srv.close()
    assert c.disabled_reason == "2 timeouts" and c.calls == 2
    assert elapsed < 6.0, elapsed


def _completion(content, finish="stop", reasoning=None, usage=(120, 30)):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return 200, {"id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "qwen",
                 "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
                 "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}


def _stub_server(respond):
    """Local OpenAI-compatible /v1/chat/completions stub standing in for the organizers'
    Qwen proxy; ``respond(body) -> (status, payload)`` decides every answer."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            seen.append(body)
            code, payload = respond(body)
            data = _json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def _real_client(monkeypatch, srv, model="qwen3.8-27b", **kw):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    if model:
        monkeypatch.setenv("OPENAI_MODEL", model)
    else:
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
    return LLMClient(mode="real", timeout=10, **kw)


GOOD = '<think>let me think</think>\n```json\n{"hypotheses": [{"action": "tune", "family": "lgbm"}]}\n```'


def test_real_path_against_openai_compatible_stub(monkeypatch):
    """A server that rejects both thinking-off options: each 400 drops one set, and the
    working set is remembered for later calls."""
    def respond(body):
        for field in ("reasoning_effort", "chat_template_kwargs"):
            if field in body:
                return 400, {"error": {"message": f"unknown field {field}"}}
        return _completion(GOOD)
    srv, seen = _stub_server(respond)
    c = _real_client(monkeypatch, srv)
    out = c.chat_json("sys", "state 1", schema=SCHEMA, purpose="hypotheses")
    assert out == {"hypotheses": [{"action": "tune", "family": "lgbm"}]}
    assert seen[0]["reasoning_effort"] == "none" and "chat_template_kwargs" in seen[0]
    assert "reasoning_effort" not in seen[1] and "chat_template_kwargs" in seen[1]
    assert "reasoning_effort" not in seen[2] and "chat_template_kwargs" not in seen[2]
    assert seen[2]["model"] == "qwen3.8-27b" and seen[2]["messages"][-1]["content"].endswith("/no_think")
    assert c.prompt_tokens == 120 and c.completion_tokens == 30 and c.failures == 0
    out2 = c.chat_json("sys", "state 2", schema=SCHEMA, purpose="hypotheses")
    assert out2 and len(seen) == 4 and "chat_template_kwargs" not in seen[3]
    srv.shutdown()


def test_vllm_style_server_keeps_chat_template_kwargs(monkeypatch):
    def respond(body):
        if "reasoning_effort" in body:
            return 400, {"error": {"message": "reasoning_effort: input should be 'low', 'medium' or 'high'"}}
        return _completion(GOOD)
    srv, seen = _stub_server(respond)
    c = _real_client(monkeypatch, srv)
    assert c.chat_json("sys", "s", schema=SCHEMA, purpose="hypotheses")
    assert c.chat_json("sys", "s2", schema=SCHEMA, purpose="hypotheses")
    assert len(seen) == 3 and all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in seen[1:])
    srv.shutdown()


def test_reasoning_cut_off_is_not_taken_as_the_answer(monkeypatch):
    """Live Qwen behaviour: with thinking on, the reasoning (which quotes JSON-like bits
    of the prompt) can use the whole max_tokens. That must not be parsed as the answer;
    one retry with a larger budget is allowed."""
    cut = 'We need answer... maybe ["mae"]? Schema says {"metric": str}. Need decide'

    def respond(body):
        if body["max_tokens"] < 1000:
            return _completion("", finish="length", reasoning=cut, usage=(100, body["max_tokens"]))
        return _completion(GOOD, usage=(100, 900))
    srv, seen = _stub_server(respond)
    c = _real_client(monkeypatch, srv)
    out = c.chat_json("sys", "s", schema=SCHEMA, max_tokens=800, purpose="hypotheses")
    assert out == {"hypotheses": [{"action": "tune", "family": "lgbm"}]}
    assert [b["max_tokens"] for b in seen] == [800, 2400]
    assert c.completion_tokens == 800 + 900 and c.failures == 0
    srv.shutdown()

    srv, seen = _stub_server(lambda body: _completion("", finish="length", reasoning=cut,
                                                      usage=(100, body["max_tokens"])))
    c = _real_client(monkeypatch, srv)
    assert c.chat_json("sys", "s", schema=SCHEMA, max_tokens=800, purpose="hypotheses") is None
    assert len(seen) == 2 and c.failures == 1 and c.completion_tokens == 800 + 2400
    srv.shutdown()


def test_reply_cut_off_in_the_answer_is_flagged(monkeypatch):
    """Live case: the third hypothesis' code was cut off at max_tokens. JSON repair keeps
    the complete items; the flag lets the caller drop the broken last one."""
    cut = '{"hypotheses": [{"action": "tune", "family": "lgbm"}, {"action": "feature_engineering", "code": "def transform(df):\\n    out = '
    replies = iter([_completion(cut, finish="length"), _completion(GOOD)])
    srv, _ = _stub_server(lambda body: next(replies))
    c = _real_client(monkeypatch, srv)
    out = c.chat_json("sys", "s", schema=SCHEMA, purpose="hypotheses")
    assert out and out["hypotheses"][0] == {"action": "tune", "family": "lgbm"} and c.last_truncated
    assert c.chat_json("sys", "s2", schema=SCHEMA, purpose="hypotheses") and not c.last_truncated
    srv.shutdown()


def test_finished_answer_only_in_reasoning_field_is_used(monkeypatch):
    srv, _ = _stub_server(lambda body: _completion("", reasoning='{"hypotheses": []}'))
    c = _real_client(monkeypatch, srv)
    assert c.chat_json("sys", "s", schema=SCHEMA, purpose="hypotheses") == {"hypotheses": []}
    srv.shutdown()


def test_exhausted_key_budget_disables_at_once(monkeypatch):
    srv, seen = _stub_server(lambda body: (400, {"error": {"message": "Budget has been exceeded! Current cost: 600"}}))
    c = _real_client(monkeypatch, srv)
    assert c.chat_json("sys", "s", schema=SCHEMA, purpose="hypotheses") is None
    assert c.disabled_reason == "key budget exceeded" and len(seen) == 1 and not c.available()
    srv.shutdown()


def test_model_name_is_optional_but_base_url_is_not(monkeypatch):
    """The platform sets OPENAI_API_KEY and OPENAI_BASE_URL; the proxy picks the model."""
    from agent_config import load_settings

    srv, seen = _stub_server(lambda body: _completion(GOOD))
    c = _real_client(monkeypatch, srv, model=None)
    assert c.mode == "real" and c.chat_json("sys", "s", schema=SCHEMA, purpose="hypotheses")
    assert seen[0]["model"] == "qwen/qwen3.8-27b"
    monkeypatch.delenv("AGENT_LLM_MODE", raising=False)
    assert load_settings().llm_mode == "real"
    srv.shutdown()

    monkeypatch.delenv("OPENAI_BASE_URL")
    assert LLMClient(mode="real").mode == "off"
    assert load_settings().llm_mode == "off"
