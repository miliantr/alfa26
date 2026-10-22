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


def test_hanging_endpoint_is_given_up_quickly(monkeypatch):
    """An endpoint that accepts connections and never answers must cost about two
    timeouts in total, not retries x calls (it used to be 9 timeouts + backoff)."""
    import socket
    import threading
    import time

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


def _stub_server(reply_text, reject_extra_body=True):
    """Local OpenAI-compatible /v1/chat/completions stub (stands in for the organizers'
    Qwen endpoint): optional 400 on vLLM-only fields, a Qwen-style reply with usage."""
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
            if reject_extra_body and "chat_template_kwargs" in body:
                code, payload = 400, {"error": {"message": "unknown field chat_template_kwargs"}}
            else:
                code, payload = 200, {
                    "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": body.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": reply_text}}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}}
            data = _json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_real_path_against_openai_compatible_stub(monkeypatch):
    reply = '<think>let me think</think>\n```json\n{"hypotheses": [{"action": "tune", "family": "lgbm"}]}\n```'
    srv, seen = _stub_server(reply)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "qwen3.8-27b")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    c = LLMClient(mode="real", timeout=10)
    out = c.chat_json("sys", "state 1", schema=SCHEMA, purpose="hypotheses")
    assert out == {"hypotheses": [{"action": "tune", "family": "lgbm"}]}
    # first request carried the vLLM option, got 400, the retry dropped it
    assert "chat_template_kwargs" in seen[0] and "chat_template_kwargs" not in seen[1]
    assert seen[1]["model"] == "qwen3.8-27b" and seen[1]["messages"][-1]["content"].endswith("/no_think")
    assert c.prompt_tokens == 120 and c.completion_tokens == 30 and c.failures == 0
    out2 = c.chat_json("sys", "state 2", schema=SCHEMA, purpose="hypotheses")
    assert out2 and len(seen) == 3 and "chat_template_kwargs" not in seen[2]
    srv.shutdown()
