"""LLM access with hard budgets, caching, retries and strict JSON parsing.

Only the organizers' OpenAI-compatible endpoint from ``OPENAI_BASE_URL`` /
``OPENAI_API_KEY`` / ``OPENAI_MODEL`` is ever contacted. Every failure mode (no
credentials, timeouts, HTTP errors, garbage output, exhausted budget) returns ``None``
and the caller falls back to deterministic logic. After ``max_failures`` consecutive
failures the client disables itself for the rest of the run.

``mode="mock"`` is a deterministic offline stub for tests and CI; its behaviour can be
switched with ``AGENT_LLM_MOCK`` = good | garbage | error | slow | mixed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from . import log
from .io_utils import append_jsonl, dumps

L = log.get("llm")

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


# ----------------------------------------------------------------------- JSON utils
def extract_json(text: str):
    """Best-effort extraction of the first JSON object/array in an LLM reply."""
    if text is None:
        return None
    t = _THINK.sub("", text).strip()
    if "<think>" in t.lower():  # unterminated reasoning block
        t = t[t.lower().rfind("</think>") + 8:] if "</think>" in t.lower() else t.split("<think>")[0]
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S | re.I)
    candidates = [fence.group(1)] if fence else []
    candidates.append(t)
    for cand in candidates:
        pairs = [(cand.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if cand.find(o) != -1]
        for start, opener, closer in sorted(pairs):  # outermost structure first
            frag = _balanced(cand[start:], opener, closer)
            if frag is None:
                continue
            for fixer in (lambda s: s, _repair):
                try:
                    return json.loads(fixer(frag))
                except Exception:
                    continue
    return None


def _balanced(s: str, opener: str, closer: str) -> str | None:
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[: i + 1]
    return s if depth > 0 else None  # truncated: let _repair try to close it


def _repair(s: str) -> str:
    s = re.sub(r"//[^\n]*", "", s)                     # line comments
    s = re.sub(r",\s*([}\]])", r"\1", s)              # trailing commas
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    if '"' not in s and "'" in s:                      # python-style quotes
        s = s.replace("'", '"')
    # close truncated structures
    opens = []
    in_str = esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            opens.append(ch)
        elif ch in "}]" and opens:
            opens.pop()
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s)
    s += "".join("}" if o == "{" else "]" for o in reversed(opens))
    return s


def validate(obj, schema: dict | None) -> list[str]:
    if schema is None:
        return []
    try:
        import jsonschema

        v = jsonschema.Draft7Validator(schema)
        return [e.message for e in v.iter_errors(obj)][:5]
    except Exception as e:  # jsonschema missing or schema invalid
        return [] if "No module" in str(e) else [str(e)]


# ---------------------------------------------------------------------------- client
class LLMClient:
    def __init__(self, mode: str = "off", run_dir: Path | None = None, max_calls: int = 30,
                 max_tokens: int = 160_000, timeout: float = 120.0, max_failures: int = 3):
        self.mode = mode
        self.run_dir = Path(run_dir) if run_dir else None
        self.max_calls, self.max_tokens, self.timeout = max_calls, max_tokens, timeout
        self.max_failures = max_failures
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.timeouts = 0
        self.disabled_reason = ""
        self._cache: dict[str, str] = {}
        self._client = None
        self._extra_body_ok = True
        self.model = os.environ.get("OPENAI_MODEL", "")
        if mode == "real" and not (self.model and os.environ.get("OPENAI_API_KEY")):
            self.mode, self.disabled_reason = "off", "missing OPENAI_MODEL/OPENAI_API_KEY"
        self._mock_calls = 0

    # ---------------------------------------------------------------- status
    @property
    def tokens_used(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def available(self, reserve_tokens: int = 2000) -> bool:
        if self.mode == "off" or self.disabled_reason:
            return False
        return self.calls < self.max_calls and self.tokens_used + reserve_tokens <= self.max_tokens

    def usage(self) -> dict:
        return {"mode": self.mode, "calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "failures": self.failures,
                "disabled": self.disabled_reason or None, "model": self.model if self.mode == "real" else self.mode}

    # ----------------------------------------------------------------- calls
    def chat_json(self, system: str, user: str, schema: dict | None = None, max_tokens: int = 1200,
                  temperature: float = 0.2, purpose: str = "", timeout: float | None = None):
        """Return parsed+validated JSON or None. Never raises."""
        try:
            return self._chat_json(system, user, schema, max_tokens, temperature, purpose, timeout)
        except Exception as e:  # noqa: BLE001
            self._fail(f"unexpected: {e!r}")
            return None

    def _chat_json(self, system, user, schema, max_tokens, temperature, purpose, timeout):
        est_prompt = (len(system) + len(user)) // 3
        if not self.available(reserve_tokens=est_prompt + max_tokens):
            return None
        key = hashlib.sha1(f"{system}\n{user}\n{max_tokens}".encode()).hexdigest()
        text = self._cache.get(key)
        if text is None:
            text = self._complete(system, user, max_tokens, temperature, purpose, timeout)
            if text is None:
                return None
        obj = extract_json(text)
        errs = ["no JSON found"] if obj is None else validate(obj, schema)
        if errs:
            self._log(purpose, user, text, error=errs)
            self._fail(f"invalid JSON for {purpose}: {errs[:2]}")
            return None
        self._cache[key] = text
        self.consecutive_failures = 0
        return obj

    def _complete(self, system, user, max_tokens, temperature, purpose, timeout) -> str | None:
        self.calls += 1
        t0 = time.time()
        if self.mode == "mock":
            text = self._mock(system, user, purpose)
            pt, ct = (len(system) + len(user)) // 4, len(text or "") // 4
        else:
            text, pt, ct = self._real(system, user, max_tokens, temperature, timeout)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self._log(purpose, user, text, seconds=round(time.time() - t0, 2), tokens=[pt, ct])
        if text is None:
            self._fail(f"no response for {purpose}")
        return text

    def _real(self, system, user, max_tokens, temperature, timeout):
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user + "\n/no_think"}]
        delay = 2.0
        for attempt in range(3):
            try:
                kw = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                          timeout=min(self.timeout, timeout or self.timeout))
                if self._extra_body_ok:
                    kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                resp = self._client.chat.completions.create(**kw)
                self.timeouts = 0
                msg = resp.choices[0].message
                text = msg.content or ""
                if not text.strip() and getattr(msg, "reasoning_content", None):
                    text = msg.reasoning_content
                u = getattr(resp, "usage", None)
                pt = getattr(u, "prompt_tokens", None) or (len(system) + len(user)) // 3
                ct = getattr(u, "completion_tokens", None) or len(text) // 3
                return text, int(pt), int(ct)
            except Exception as e:  # noqa: BLE001
                name = type(e).__name__
                status = getattr(e, "status_code", None)
                if self._extra_body_ok and status == 400:
                    self._extra_body_ok = False  # server rejects vLLM-specific options
                    continue
                if status in (401, 403, 404):
                    self.disabled_reason = f"{name} {status}"
                    L.warning("LLM disabled: %s", self.disabled_reason)
                    return None, 0, 0
                if "Timeout" in name:
                    # A server that let the request time out will not answer a retry
                    # either; two timeouts mean it is hanging, so stop using it.
                    self.timeouts += 1
                    L.warning("LLM call timed out (%d so far)", self.timeouts)
                    if self.timeouts >= 2:
                        self.disabled_reason = f"{self.timeouts} timeouts"
                        L.warning("LLM disabled for the rest of the run: %s", self.disabled_reason)
                    return None, 0, 0
                L.warning("LLM call failed (%s %s), attempt %d", name, status, attempt + 1)
                if attempt < 2:
                    time.sleep(delay)
                    delay *= 2
        return None, 0, 0

    def _fail(self, why: str) -> None:
        self.failures += 1
        self.consecutive_failures += 1
        L.warning("LLM failure %d: %s", self.consecutive_failures, why)
        if self.consecutive_failures >= self.max_failures:
            self.disabled_reason = f"{self.consecutive_failures} consecutive failures"
            L.warning("LLM disabled for the rest of the run: %s", self.disabled_reason)

    def _log(self, purpose, user, text, **kw) -> None:
        if not self.run_dir:
            return
        try:
            append_jsonl(self.run_dir / "llm_log.jsonl", {
                "t": time.strftime("%H:%M:%S"), "purpose": purpose, "prompt_chars": len(user),
                "prompt_head": user[:600], "response": (text or "")[:4000], **kw})
        except Exception:
            pass

    # ------------------------------------------------------------------ mock
    def _mock(self, system: str, user: str, purpose: str) -> str | None:
        behaviour = os.environ.get("AGENT_LLM_MOCK", "good")
        self._mock_calls += 1
        n = self._mock_calls
        if behaviour == "error" or (behaviour == "mixed" and n % 3 == 0):
            return None
        if behaviour == "garbage" or (behaviour == "mixed" and n % 3 == 1 and n > 1):
            return "Sure! Here are some ideas: try more trees {not json"
        if behaviour == "slow":
            time.sleep(float(os.environ.get("AGENT_LLM_MOCK_DELAY", "5")))
        if purpose == "task_parse":
            return self._mock_task(user)
        if purpose == "hypotheses":
            return self._mock_hypotheses(user, n)
        return json.dumps({"ok": True})

    @staticmethod
    def _mock_task(user: str) -> str:
        from .metrics import extract_primary_metric

        metric, mentioned = extract_primary_metric(user)
        tt = "regression" if metric in ("rmse", "mae", "mse", "rmsle", "r2", "mape") else "multiclass"
        tgt = re.findall(r"target candidates: \[([^\]]*)\]", user)
        targets = [t.strip(" '\"") for t in tgt[0].split(",")] if tgt else []
        return "<think>reasoning...</think>\n```json\n" + dumps({
            "task_type": tt, "target_columns": targets, "metric": metric or "rmse",
            "secondary_metrics": [m for m in mentioned if m != metric], "output_format": "value" if tt == "regression" else "label",
            "time_column": None, "group_column": None, "special_requirements": [], "domain_notes": "mock"}) + "\n```"

    @staticmethod
    def _mock_hypotheses(user: str, n: int) -> str:
        cols = re.findall(r'"([A-Za-z_][\w]*):(?:numeric|bool)', user)
        a = cols[0] if cols else "x0"
        b = cols[1] if len(cols) > 1 else a
        bank = [
            {"action": "feature_engineering", "recipe": "freq_encoding", "rationale": "category frequency",
             "expected_gain": "small", "cost": "low"},
            {"action": "feature_engineering", "name": "ratio_ab", "rationale": "interaction of top features",
             "code": f"def transform(df):\n    out = pd.DataFrame(index=df.index)\n"
                     f"    out['ratio_{a}_{b}'] = df['{a}'] / (df['{b}'].abs() + 1e-6)\n    return out\n",
             "expected_gain": "small", "cost": "low"},
            {"action": "model", "family": "lgbm", "params": {"num_leaves": 127, "learning_rate": 0.03,
                                                             "min_child_samples": 40},
             "rationale": "bigger trees, lower lr", "expected_gain": "small", "cost": "medium"},
            {"action": "feature_engineering", "recipe": "row_stats", "rationale": "group summaries",
             "expected_gain": "small", "cost": "low"},
            {"action": "drop_features", "cols": ["__zero_importance__"], "rationale": "remove noise",
             "expected_gain": "tiny", "cost": "low"},
            {"action": "model", "family": "xgb", "params": {"max_depth": 9, "min_child_weight": 3},
             "rationale": "deeper xgb", "expected_gain": "small", "cost": "medium"},
            {"action": "feature_engineering", "name": "evil", "rationale": "sandbox must block this",
             "code": "import os\ndef transform(df):\n    os.system('echo hacked')\n    return df\n",
             "expected_gain": "none", "cost": "low"},
            {"action": "tune", "family": "lgbm", "rationale": "TPE trial", "expected_gain": "small", "cost": "medium"},
        ]
        picks = [bank[(2 * n + i) % len(bank)] for i in range(2)]
        return "```json\n" + dumps({"hypotheses": picks}) + "\n```"


def make_client(settings, run_dir: Path | None) -> LLMClient:
    return LLMClient(mode=settings.llm_mode, run_dir=run_dir, max_calls=settings.llm_max_calls,
                     max_tokens=settings.llm_max_tokens, timeout=settings.llm_timeout)
