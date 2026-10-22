"""AutoResearch loop: hypothesis -> experiment -> evaluation -> keep/discard -> next.

Roles (one process, several "agents"):
  * Researcher  — the LLM (or a deterministic planner) proposes structured hypotheses;
  * Engineer    — hypotheses become experiment specs; LLM feature code runs in the sandbox;
  * Evaluator   — paired comparison on identical folds decides keep/discard;
  * Ensembler   — every successful experiment joins the pool; the submission is refreshed.

The champion (default: LightGBM, the fastest strong learner) is the reference for
feature hypotheses; accepted feature changes become the new base feature set and are
periodically propagated to the other model families so the ensemble benefits too.
"""
from __future__ import annotations

import copy
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import agent_features as features
import agent_log as log
import agent_metrics as metrics
import agent_postprocess as postprocess
from agent_io_utils import append_jsonl, dumps, write_json
from agent_llm_client import LLMClient
from agent_models import FAMILIES
from agent_sandbox import run_feature_code

L = log.get("research")

ACTIONS = ("feature_engineering", "drop_features", "target_encoding", "model", "tune")

HYP_SCHEMA = {
    "type": "object",
    "properties": {"hypotheses": {"type": "array", "maxItems": 6, "items": {
        "type": "object",
        "properties": {"action": {"type": "string"}, "rationale": {"type": "string"}},
        "required": ["action"]}}},
    "required": ["hypotheses"],
}

# Allowed hyper-parameters per family with clamping ranges (anything else is dropped).
PARAM_SPACE = {
    "lgbm": {"num_leaves": (2, 1024, int), "learning_rate": (0.005, 0.3, float), "min_child_samples": (1, 2000, int),
             "feature_fraction": (0.05, 1.0, float), "bagging_fraction": (0.1, 1.0, float), "bagging_freq": (0, 10, int),
             "lambda_l1": (0.0, 100.0, float), "lambda_l2": (0.0, 100.0, float), "max_depth": (-1, 64, int),
             "min_split_gain": (0.0, 10.0, float), "max_bin": (15, 1023, int), "extra_trees": (0, 1, bool),
             "path_smooth": (0.0, 100.0, float), "cat_smooth": (0.0, 100.0, float), "cat_l2": (0.0, 100.0, float),
             "min_data_per_group": (1, 1000, int), "max_cat_threshold": (4, 256, int),
             "n_estimators": (50, 20000, int), "early_stopping": (20, 500, int)},
    "xgb": {"max_depth": (1, 16, int), "min_child_weight": (0.0, 100.0, float), "subsample": (0.1, 1.0, float),
            "colsample_bytree": (0.05, 1.0, float), "colsample_bylevel": (0.05, 1.0, float),
            "reg_lambda": (0.0, 100.0, float), "reg_alpha": (0.0, 100.0, float), "gamma": (0.0, 20.0, float),
            "learning_rate": (0.005, 0.3, float), "max_bin": (16, 1024, int), "n_estimators": (50, 20000, int),
            "max_cat_to_onehot": (1, 64, int), "max_leaves": (0, 1024, int), "early_stopping": (20, 500, int)},
    "catboost": {"depth": (2, 10, int), "l2_leaf_reg": (0.0, 100.0, float), "learning_rate": (0.005, 0.3, float),
                 "random_strength": (0.0, 20.0, float), "bagging_temperature": (0.0, 10.0, float),
                 "border_count": (16, 255, int), "n_estimators": (50, 10000, int), "one_hot_max_size": (1, 255, int),
                 "min_data_in_leaf": (1, 1000, int), "max_ctr_complexity": (1, 4, int), "early_stopping": (20, 500, int)},
    "linear": {"C": (1e-4, 1e3, float), "alpha": (1e-4, 1e4, float)},
    "extratrees": {"n_estimators": (50, 800, int), "min_samples_leaf": (1, 100, int), "max_features": (0.05, 1.0, float)},
    "randomforest": {"n_estimators": (50, 800, int), "min_samples_leaf": (1, 100, int), "max_features": (0.05, 1.0, float)},
}


def clamp_params(family: str, params: dict) -> tuple[dict, list[str]]:
    space = PARAM_SPACE.get(family, {})
    out, dropped = {}, []
    for k, v in (params or {}).items():
        if k not in space:
            dropped.append(k)
            continue
        lo, hi, typ = space[k]
        try:
            if typ is bool:
                out[k] = bool(v)
            else:
                out[k] = typ(min(max(typ(v), lo), hi))
        except (TypeError, ValueError):
            dropped.append(k)
    return out, dropped


# --------------------------------------------------------------------- Optuna tuning
def _suggest(trial, family: str, n_rows: int) -> dict:
    # Ranges shrink with the training size so that no trial is degenerate (e.g. a leaf
    # minimum larger than a tenth of the training fold gives constant predictions).
    leaf_max = int(max(10, min(300, n_rows // 40)))
    leaves_max = int(max(16, min(255, n_rows // 20)))
    if family == "lgbm":
        return {"num_leaves": trial.suggest_int("num_leaves", 8, leaves_max, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 3, leaf_max, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                "extra_trees": trial.suggest_categorical("extra_trees", [False, True])}
    if family == "xgb":
        return {"max_depth": trial.suggest_int("max_depth", 3, 12 if n_rows > 5000 else 8),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, min(30.0, max(2.0, n_rows / 100)), log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True)}
    if family == "catboost":
        return {"depth": trial.suggest_int("depth", 4, 9),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.15, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0)}
    raise ValueError(family)


class Tuner:
    """Optuna TPE driven by the research loop via ask/tell (one study per family)."""

    def __init__(self, seed: int, direction: str):
        self.seed, self.direction = seed, direction
        self.studies: dict = {}
        self.pending: dict = {}

    def ask(self, family: str, n_rows: int) -> dict | None:
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            if family not in self.studies:
                self.studies[family] = optuna.create_study(
                    direction=self.direction, sampler=optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=4))
            trial = self.studies[family].ask()
            params = _suggest(trial, family, n_rows)
            key = json.dumps(params, sort_keys=True, default=str)
            self.pending[key] = (family, trial)
            return params
        except Exception as e:  # noqa: BLE001
            L.warning("tuner ask failed: %s", e)
            return None

    def tell(self, params: dict, value: float | None) -> None:
        key = json.dumps(params, sort_keys=True, default=str)
        item = self.pending.pop(key, None)
        if not item:
            return
        family, trial = item
        try:
            import optuna

            if value is None or not np.isfinite(value):
                self.studies[family].tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                self.studies[family].tell(trial, value)
        except Exception:
            pass


# --------------------------------------------------------------------------- prompts
SYSTEM_PROMPT = """You are the research module of an autonomous AutoML agent working on a tabular task.
Propose hypotheses that can improve the cross-validated metric. Each hypothesis is executed as one
experiment on FIXED folds and kept only if the improvement is significant across folds, so prefer
ideas with a clear mechanism. Do not repeat experiments from the journal. Never use the target in
features, never use the test set for selection. Reply with JSON only, no prose:
{"hypotheses": [ {<hypothesis>}, ... ]}   (1-3 hypotheses, best first)

Hypothesis forms (pick fields per action):
- {"action":"feature_engineering","recipe":"<name>","args":{...},"rationale":"..."}  — use a library recipe
- {"action":"feature_engineering","name":"<short_id>","code":"<python>","rationale":"..."} — custom features.
  The code must define transform(df) -> pandas.DataFrame with ONLY the new columns (same index), using
  df's original column names; numpy as np and pandas as pd are available; allowed imports: numpy, pandas,
  math, re, datetime. No file/network access. Called separately on train and test.
- {"action":"drop_features","cols":["..."] or ["__zero_importance__"],"rationale":"..."}
- {"action":"target_encoding","cols":["<categorical or key column>"],"rationale":"..."}  (out-of-fold)
- {"action":"model","family":"lgbm|xgb|catboost|linear|extratrees","params":{...},
   "class_weight":null|"balanced"|"sqrt","objective":"l2|l1|huber|mape","target_transform":"none|log1p|sqrt","rationale":"..."}
- {"action":"tune","family":"lgbm|xgb|catboost","rationale":"..."}  — one Bayesian-optimisation trial
Optional on any hypothesis: "expected_gain":"tiny|small|medium|large", "cost":"low|medium|high".
Be brief: each rationale one sentence (under 25 words), custom code under 25 lines.
"""


def _excerpt(text: str, n: int = 1500) -> str:
    t = re.sub(r"\|[^\n]*\|", "", text)  # drop markdown tables (column lists come separately)
    t = re.sub(r"\n{2,}", "\n", t)
    return t[:n]


# ----------------------------------------------------------------------------- loop
class ResearchLoop:
    def __init__(self, pipe, spec, prof, settings, deadline, run_dir: Path, llm: LLMClient | None = None,
                 raw_frames: tuple[pd.DataFrame, pd.DataFrame] | None = None, base_X: pd.DataFrame | None = None,
                 domain_notes: str = ""):
        self.pipe, self.spec, self.prof = pipe, spec, prof
        self.settings, self.deadline, self.run_dir = settings, deadline, Path(run_dir)
        self.llm = llm
        self.raw_frames = raw_frames
        self.base_X = base_X
        self.domain_notes = domain_notes
        self.m = metrics.REGISTRY[spec.metric]
        self.tuner = Tuner(settings.seed, "maximize" if self.m.greater_is_better else "minimize")
        self.journal_path = self.run_dir / "research_journal.jsonl"
        self.history: list[dict] = []
        self.tried: set[str] = set()
        self.queue: list[dict] = self._deterministic_queue()
        self.llm_queue: list[dict] = []
        self.champ = None
        self.since_propagation = 0
        self.code_runs = 0

    # ------------------------------------------------------------- champion
    def _pick_champion(self):
        for fam in ("lgbm", "xgb", "catboost"):
            e = self.pipe.best(family=fam)
            if e is not None and e.feature_ops == self.pipe.base_ops:
                best_any = self.pipe.best()
                # keep LightGBM as proxy unless another family is clearly better
                if fam == "lgbm" and best_any is not None and best_any.model["family"] != "lgbm":
                    cmp = self.pipe.compare(best_any, e)
                    if cmp["significant"] and cmp["rel_delta"] > 0.02 and best_any.seconds < 4 * e.seconds:
                        return best_any
                return e
        return self.pipe.best()

    # ------------------------------------------------------------ planning
    def _deterministic_queue(self) -> list[dict]:
        p, spec = self.prof, self.spec
        q: list[dict] = []
        has_cats = bool(p.cat_features)
        has_missing = any(p.columns[c]["missing"] > 0 for c in p.features)
        groups = features._column_groups(p.num_features)
        q.append({"action": "tune", "family": "lgbm", "rationale": "Bayesian optimisation trial"})
        if has_cats:
            q.append({"action": "feature_engineering", "recipe": "freq_encoding", "rationale": "category frequencies"})
        q.append({"action": "tune", "family": "lgbm", "rationale": "Bayesian optimisation trial"})
        if groups:
            q.append({"action": "feature_engineering", "recipe": "row_stats", "rationale": "summaries of related column groups"})
        q.append({"action": "feature_engineering", "recipe": "num_pairs", "args": {"top": 6},
                  "rationale": "ratios/differences of the strongest numeric features"})
        if has_missing:
            q.append({"action": "feature_engineering", "recipe": "missing_count", "rationale": "missingness patterns"})
        q.append({"action": "tune", "family": "lgbm", "rationale": "Bayesian optimisation trial"})
        if has_cats:
            q.append({"action": "feature_engineering", "recipe": "group_stats", "rationale": "deviation from category means"})
        if len(p.cat_features) >= 2:
            q.append({"action": "feature_engineering", "recipe": "cat_combo", "args": {"top": 3},
                      "rationale": "interactions of top categoricals"})
        hc = [c for c in p.cat_features if p.columns[c]["high_card"]]
        if hc:
            q.append({"action": "target_encoding", "cols": hc[:5], "rationale": "OOF target encoding of high-cardinality categoricals"})
        q.append({"action": "drop_features", "cols": ["__zero_importance__"], "rationale": "remove unused features"})
        q.append({"action": "tune", "family": "xgb", "rationale": "Bayesian optimisation trial"})
        if spec.task_type in metrics.CLASSIFICATION and (p.target.get("imbalance_ratio") or 1) > 3:
            q.append({"action": "model", "family": "lgbm", "class_weight": "balanced", "rationale": "rebalance rare classes"})
        if spec.task_type == metrics.REGRESSION:
            if p.target.get("nonnegative") and abs(p.target.get("skew") or 0) > 1:
                q.append({"action": "model", "family": "lgbm", "target_transform": "sqrt", "rationale": "variance-stabilising target"})
            q.append({"action": "model", "family": "lgbm", "objective": "huber", "rationale": "robust loss against outliers"})
        q.append({"action": "model", "family": "lgbm", "params": {"num_leaves": 255, "min_child_samples": 10,
                                                                    "learning_rate": 0.03},
                  "rationale": "higher capacity"})
        q.append({"action": "tune", "family": "catboost", "rationale": "Bayesian optimisation trial"})
        return q

    def _state(self) -> str:
        pipe, spec, p = self.pipe, self.spec, self.prof
        champ = self.champ
        best = pipe.best()
        # Token budget: full detail (kind, cardinality, missingness, a short dictionary
        # description) only for the 20 most important columns; the rest by name. Sending
        # every column with its Russian description made each call ~4-5k tokens.
        cols = []
        dd = {str(r.get("feature")): r for r in spec.dictionary}
        imp = [n for n, _ in (champ.importance if champ else [])]
        ordered = [c for c in imp if c in p.columns] + [c for c in p.features if c not in imp]
        for c in ordered[:20]:
            ci = p.columns[c]
            d = dd.get(c, {})
            desc = (str(d.get("description_ru") or "")[:40] + (f" [{d.get('unit')}]" if d.get("unit") else "")).strip()
            cols.append(f"{c}:{ci['kind']} nun={ci['n_unique']} na={ci['missing']:.2f}" + (f" | {desc}" if desc else ""))
        rest = [c for c in ordered[20:] if c in p.columns]
        if rest:
            cols.append(("other columns: " + ", ".join(rest))[:1200])
        journal = []
        for h in self.history[-10:]:
            journal.append(f"{h['exp_id']} {h['summary']} -> {h.get('score')} "
                           f"(delta {h.get('delta'):+.5f}, {'KEPT' if h.get('accepted') else 'rejected'}"
                           f"{', ' + h['note'] if h.get('note') else ''})" if h.get("delta") is not None else
                           f"{h['exp_id']} {h['summary']} -> FAILED: {h.get('note', '')[:120]}")
        state = {
            "task": {"title": spec.title, "type": spec.task_type, "metric": spec.metric,
                     "direction": "higher is better" if self.m.greater_is_better else "lower is better",
                     "rows": {"train": p.n_train, "test": p.n_test}},
            "time_left_min": round(self.deadline.research_remaining() / 60, 1),
            "llm_calls_left": (self.llm.max_calls - self.llm.calls) if self.llm else 0,
            "description_excerpt": _excerpt(spec.text, 800),
            "domain_notes": self.domain_notes[:400],
            "columns": cols,
            "validation": {k: v for k, v in p.validation.items() if k != "eval_mask"},
            "train_test_shift": {"adversarial_auc": p.adversarial.get("auc"),
                                 "top_drift_features": p.adversarial.get("top_features", [])[:5]},
            "target": p.target,
            "champion": {"exp": champ.id if champ else None, "family": champ.model["family"] if champ else None,
                         "params": champ.model.get("params", {}) if champ else {},
                         "score": round(champ.score, 6) if champ else None},
            "best_overall_score": round(best.score, 6) if best else None,
            "ensemble_score": round(pipe.sub_score, 6) if pipe.sub_score is not None else None,
            "accepted_feature_ops": [op.get("name") or op.get("op") for op in pipe.base_ops],
            "top_features_gain_share": [f"{n}:{v:.3f}" for n, v in (champ.importance[:15] if champ else [])],
            "error_analysis": self._error_analysis(),
            "journal": journal,
            "recipes": features.RECIPE_DOCS,
        }
        return dumps(state)

    def _error_analysis(self) -> dict:
        best = self.pipe.best()
        if best is None:
            return {}
        try:
            oof, _ = self.pipe.preds(best)
            y = self.pipe._y_eval()
            mk = self.pipe.eval_mask
            if self.spec.task_type == metrics.REGRESSION:
                pred = postprocess.apply(best.decision, oof)
                res = pred - y
                out = {"bias": round(float(res[mk].mean()), 4)}
                q = pd.qcut(pd.Series(y[mk]).rank(method="first"), 5, labels=False)
                out["bias_by_target_quintile"] = [round(float(v), 3) for v in pd.Series(res[mk]).groupby(q.values).mean()]
                if self.base_X is not None:
                    worst = []
                    for c in self.prof.cat_features[:4]:
                        g = pd.DataFrame({"g": self.base_X[c].astype(str).values[mk], "e": res[mk] ** 2})
                        agg = g.groupby("g")["e"].agg(["mean", "size"])
                        agg = agg[agg["size"] >= 30].sort_values("mean", ascending=False).head(3)
                        worst += [f"{c}={k}: rmse={math.sqrt(v['mean']):.2f} n={int(v['size'])}" for k, v in agg.iterrows()]
                    out["worst_groups"] = worst[:8]
                return out
            lab = postprocess.labels_from(best.decision, oof)
            k = len(self.spec.classes)
            cm = np.bincount(y[mk] * k + lab[mk], minlength=k * k).reshape(k, k)
            tp = np.diag(cm).astype(float)
            prec = tp / np.maximum(cm.sum(0), 1)
            rec = tp / np.maximum(cm.sum(1), 1)
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
            per_class = {self.spec.classes[i]: {"f1": round(float(f1[i]), 3), "n": int(cm[i].sum())} for i in range(k)}
            off = [(cm[i, j], i, j) for i in range(k) for j in range(k) if i != j]
            off.sort(reverse=True)
            conf = [f"{self.spec.classes[i]}->{self.spec.classes[j]}: {int(n)}" for n, i, j in off[:6]]
            return {"per_class": per_class, "top_confusions": conf}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:100]}

    def _ask_llm(self) -> list[dict]:
        if not self.llm or not self.llm.available(reserve_tokens=6000):
            return []
        # Live Qwen answers 1-2k tokens in 20-45 s; a short timeout late in the window would
        # cut off a healthy server (and two timeouts disable the client).
        remaining = self.deadline.research_remaining()
        timeout = min(self.settings.llm_timeout, max(60.0, remaining * 0.15), max(20.0, remaining * 0.5))
        resp = self.llm.chat_json(SYSTEM_PROMPT, "STATE:\n" + self._state() + "\nPropose up to 3 hypotheses.",
                                  schema=HYP_SCHEMA, max_tokens=2500, purpose="hypotheses", timeout=timeout)
        if not resp:
            return []
        hyps = [h for h in resp.get("hypotheses", []) if isinstance(h, dict) and h.get("action") in ACTIONS]
        if self.llm.last_truncated:
            hyps = hyps[:-1]  # the reply hit max_tokens: its last hypothesis is cut off
        for h in hyps:
            h["source"] = "llm"
        return hyps[:3]

    def _llm_paced(self) -> bool:
        """Spread LLM calls over the research window instead of spending them early."""
        if self.llm is None or not self.llm.available():
            return False
        span = max(1.0, self.deadline.research_end_at - self.t_start)
        frac = min(1.0, (self.deadline.now() - self.t_start) / span)
        allowed = 3 + int(self.llm.max_calls * frac)
        return self.llm.calls < allowed

    def _next(self) -> dict | None:
        """Interleave LLM ideas with the deterministic plan; tuning fills idle time."""
        if not self.llm_queue and self._llm_paced():
            self.llm_queue = self._ask_llm()
        use_llm = self.llm_queue and (len(self.history) % 3 != 2 or not self.queue)
        if use_llm:
            return self.llm_queue.pop(0)
        if self.queue:
            h = self.queue.pop(0)
            h.setdefault("source", "planner")
            return h
        if self.llm_queue:
            return self.llm_queue.pop(0)
        fam = "lgbm" if len(self.history) % 3 else ("xgb" if self.pipe.best("xgb") else "lgbm")
        return {"action": "tune", "family": fam, "source": "planner", "rationale": "idle-time optimisation"}

    # ----------------------------------------------------------- materialise
    def _materialize(self, h: dict) -> tuple[dict | None, str]:
        """Hypothesis -> experiment kwargs for Pipeline.run (or None + reason)."""
        champ = self.champ
        base_model = copy.deepcopy(champ.model) if champ else {"family": "lgbm"}
        action = h.get("action")
        ops = list(self.pipe.base_ops)
        drop = list(champ.drop_cols) if champ else []
        if action == "feature_engineering":
            if h.get("code"):
                return self._code_feature(h, base_model, ops, drop)
            recipe = h.get("recipe")
            if recipe not in features.RECIPES or recipe == "cached_columns":
                return None, f"unknown recipe {recipe}"
            args = h.get("args") if isinstance(h.get("args"), dict) else {}
            te = bool(args.pop("te", False)) if recipe == "make_key" else False
            op = {"op": recipe, "args": args, "name": recipe}
            model = base_model
            if te:
                model = {**base_model, "te_cols": list(base_model.get("te_cols") or [])}
                key_name = args.get("name") or "key__" + "__".join(features._safe_name(c) for c in args.get("cols", []))[:60]
                model["te_cols"].append(key_name)
            return {"model": model, "feature_ops": ops + [op], "drop_cols": drop}, ""
        if action == "drop_features":
            cols = h.get("cols") or []
            if cols == ["__zero_importance__"] or "__zero_importance__" in cols:
                cols = list(champ.zero_importance) if champ else []
            cols = [c for c in cols if isinstance(c, str)]
            if not cols:
                return None, "nothing to drop"
            return {"model": base_model, "feature_ops": ops, "drop_cols": sorted(set(drop + cols))}, ""
        if action == "target_encoding":
            cols = [c for c in (h.get("cols") or []) if isinstance(c, str)]
            if not cols:
                return None, "no columns"
            model = {**base_model, "te_cols": sorted(set((base_model.get("te_cols") or []) + cols))}
            return {"model": model, "feature_ops": ops, "drop_cols": drop}, ""
        if action in ("model", "tune"):
            fam = h.get("family") or base_model["family"]
            if fam not in FAMILIES:
                return None, f"unknown family {fam}"
            if not self._worth_it(fam):
                return None, f"{fam} carries no blend weight and costs >10% of the research time left"
            ref = self.pipe.best(family=fam)
            model = copy.deepcopy(ref.model) if ref is not None and ref.feature_ops == ops else {"family": fam}
            if fam == base_model["family"]:
                model = copy.deepcopy(base_model)
            if action == "tune":
                if fam not in ("lgbm", "xgb", "catboost"):
                    return None, "tuning only for GBDTs"
                params = self.tuner.ask(fam, self.prof.n_train)
                if params is None:
                    return None, "tuner unavailable"
                model["params"] = {**{k: v for k, v in model.get("params", {}).items() if k in ("n_estimators", "early_stopping")}, **params}
                h["_tune_params"] = params
            else:
                params, dropped = clamp_params(fam, h.get("params") or {})
                if dropped:
                    h["_dropped_params"] = dropped
                model["params"] = {**model.get("params", {}), **params}
                for k in ("class_weight", "objective", "target_transform"):
                    if k in h and h[k] is not None:
                        model[k] = h[k]
                if self.spec.task_type != metrics.REGRESSION:
                    model.pop("objective", None)
                    model.pop("target_transform", None)
                elif model.get("class_weight"):
                    model.pop("class_weight")
            model["family"] = fam
            return {"model": model, "feature_ops": ops, "drop_cols": drop}, ""
        return None, f"unsupported action {action}"

    def _worth_it(self, fam: str) -> bool:
        """Spend research time on a model family only if it matters for the blend or is
        cheap: on task 2 a CatBoost tuning trial (no blend weight, ~18 min per run) would
        otherwise take a large share of the window."""
        if self.champ is not None and fam == self.champ.model["family"]:
            return True
        desc = self.pipe.sub_desc or {}
        weights = dict(zip(desc.get("members", []), desc.get("weights", [])))
        in_blend = any(e.model["family"] == fam and weights.get(e.id, 0) > 0 for e in self.pipe.ok)
        est = self.pipe.estimate_seconds(fam)
        return in_blend or est <= 0.10 * max(1.0, self.deadline.research_remaining())

    def _code_feature(self, h, base_model, ops, drop):
        if self.raw_frames is None:
            return None, "no raw frames for sandbox"
        if self.code_runs >= 12:
            return None, "code budget exhausted"
        self.code_runs += 1
        name = re.sub(r"[^A-Za-z0-9_]", "_", str(h.get("name") or f"c{self.code_runs}"))[:24] or f"c{self.code_runs}"
        name = f"{name}_{self.code_runs}"
        tr, te = self.raw_frames
        res = run_feature_code(h["code"], tr, te, self.run_dir / "sandbox", name,
                               timeout=min(120.0, max(20.0, self.deadline.research_remaining() * 0.05)))
        if res.get("status") != "ok":
            return None, f"sandbox {res.get('status')}: {str(res.get('error'))[:300]}"
        op = {"op": "cached_columns", "args": {"path": res["path"]}, "name": f"code:{name}",
              "columns": res["columns"][:20]}
        return {"model": base_model, "feature_ops": ops + [op], "drop_cols": drop}, ""

    # ----------------------------------------------------------------- judge
    def _judge(self, h: dict, exp) -> dict:
        action = h["action"]
        rec = {"exp_id": exp.id, "action": action, "source": h.get("source"), "summary": self._summary(h),
               "rationale": str(h.get("rationale", ""))[:300], "status": exp.status, "seconds": exp.seconds}
        if "_tune_params" in h:
            self.tuner.tell(h["_tune_params"], exp.score if exp.status == "ok" else None)
        if exp.status != "ok":
            rec.update(accepted=False, note=exp.error[:200])
            return rec
        fam = exp.model["family"]
        if action in ("feature_engineering", "drop_features", "target_encoding"):
            ref = self.champ
        else:
            same = [e for e in self.pipe.ok if e.model["family"] == fam and e.feature_ops == exp.feature_ops and e is not exp]
            ref = max(same, key=lambda e: self.m.sign() * e.score) if same else self.champ
            if self.champ is not None and self.m.better(self.champ.score, ref.score):
                ref = self.champ  # a weak same-family run (e.g. a failed tuning trial) is no reference
        cmp = self.pipe.compare(exp, ref) if ref is not None else {"delta": 0.0, "significant": True}
        accepted = bool(cmp["significant"])
        rec.update(score=round(exp.score, 6), ref=ref.id if ref else None, delta=cmp["delta"], accepted=accepted,
                   note=f"{cmp.get('fold_wins')}/{cmp.get('folds')} folds better")
        exp.parent, exp.delta, exp.accepted = rec["ref"], cmp["delta"], accepted
        if accepted and action in ("feature_engineering", "drop_features", "target_encoding"):
            self.pipe.base_ops = exp.feature_ops
            self.champ = exp
            self.since_propagation += 1
            L.info("ACCEPTED %s -> new base feature set (%d ops)", rec["summary"], len(exp.feature_ops))
        elif accepted and self.champ is not None and fam == self.champ.model["family"] and self.m.better(exp.score, self.champ.score):
            self.champ = exp
            L.info("ACCEPTED %s -> new champion params", rec["summary"])
        return rec

    @staticmethod
    def _summary(h: dict) -> str:
        a = h.get("action")
        if a == "feature_engineering":
            return f"FE {h.get('recipe') or 'code:' + str(h.get('name'))} {json.dumps(h.get('args', {}), default=str)[:80] if h.get('args') else ''}".strip()
        if a == "drop_features":
            return f"drop {h.get('cols')}"[:100]
        if a == "target_encoding":
            return f"TE {h.get('cols')}"[:100]
        if a == "tune":
            return f"tune {h.get('family')}"
        extras = {k: h[k] for k in ("class_weight", "objective", "target_transform") if h.get(k)}
        return f"model {h.get('family')} {json.dumps(h.get('params', {}))[:80]} {extras if extras else ''}".strip()

    # ------------------------------------------------------------- propagate
    def _propagate(self) -> None:
        """Re-run the other families that matter for the blend on the current base
        feature set — only when that is cheap relative to the research time left."""
        members = set(self.pipe.sub_desc.get("members", []))
        weights = dict(zip(self.pipe.sub_desc.get("members", []), self.pipe.sub_desc.get("weights", [])))
        in_blend = {e.model["family"] for e in self.pipe.ok if e.id in members and weights.get(e.id, 0) > 0}
        for fam in ("lgbm", "xgb", "catboost"):
            if self.champ is not None and fam == self.champ.model["family"]:
                continue
            ref = self.pipe.best(family=fam)
            if ref is None or ref.feature_ops == self.pipe.base_ops or fam not in in_blend:
                continue
            est = self.pipe.estimate_seconds(fam) * 1.2
            if est > 0.25 * self.deadline.research_remaining():
                continue
            e = self.pipe.run(f"{fam}_on_base{len(self.pipe.base_ops)}", ref.model, feature_ops=self.pipe.base_ops,
                              drop_cols=self.champ.drop_cols if self.champ else [], source="propagate")
            if e.status == "ok":
                self.pipe.refresh_submission("propagate")
        self.since_propagation = 0

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        self.t_start = self.deadline.now()
        self.champ = self._pick_champion()
        if self.champ is None:
            L.warning("no champion experiment; research loop skipped")
            return
        L.info("research loop: champion %s (%s %.5f), %.0fs left, llm=%s", self.champ.id,
               self.champ.model["family"], self.champ.score, self.deadline.research_remaining(),
               self.llm.mode if self.llm else "none")
        it = idle = 0
        while idle < 25:
            est = self.pipe.estimate_seconds(self.champ.model["family"], self.champ.n_features)
            if self.deadline.research_remaining() < max(20.0, est * 1.1):
                break
            h = self._next()
            if h is None:
                break
            it += 1
            idle += 1  # reset only when an experiment actually runs
            key = json.dumps({k: v for k, v in h.items() if k not in ("rationale", "source", "expected_gain", "cost")},
                             sort_keys=True, default=str)
            if key in self.tried and h.get("action") != "tune":
                continue
            self.tried.add(key)
            kwargs, why = self._materialize(h)
            if kwargs is None:
                rec = {"exp_id": "-", "action": h.get("action"), "source": h.get("source"), "summary": self._summary(h),
                       "status": "invalid", "accepted": False, "note": why}
                self._record(rec, h)
                continue
            fam = kwargs["model"]["family"]
            est_f = self.pipe.estimate_seconds(fam, self.champ.n_features)
            if not self.deadline.can_start(est_f * 1.1):
                self._record({"exp_id": "-", "action": h.get("action"), "source": h.get("source"),
                              "summary": self._summary(h), "status": "skipped", "accepted": False,
                              "note": f"needs ~{est_f:.0f}s"}, h)
                continue
            idle = 0
            exp = self.pipe.run(f"r{it:02d}_{h['action'][:5]}_{fam}", kwargs["model"], feature_ops=kwargs["feature_ops"],
                                drop_cols=kwargs["drop_cols"], source=h.get("source", "planner"),
                                hypothesis=self._summary(h),
                                # new hyper-parameters (lower learning rate, bigger trees) can
                                # cost several times the reference run
                                timeout_factor=5.0 if h["action"] in ("tune", "model") else 3.0,
                                max_share=0.25)
            rec = self._judge(h, exp)
            self._record(rec, h)
            if exp.status == "ok":
                self.pipe.refresh_submission(f"research:{exp.id}")
            if self.since_propagation >= 1 and it % 4 == 0:
                self._propagate()
        if self.since_propagation:
            self._propagate()
        write_json(self.run_dir / "llm_usage.json", self.llm.usage() if self.llm else {})
        L.info("research loop done: %d hypotheses, %d accepted, base ops=%s", len(self.history),
               sum(1 for r in self.history if r.get("accepted")), [o.get("name") for o in self.pipe.base_ops])

    def _record(self, rec: dict, h: dict) -> None:
        rec["elapsed"] = round(self.deadline.elapsed(), 1)
        rec["hypothesis"] = {k: v for k, v in h.items() if not k.startswith("_")}
        self.history.append(rec)
        try:
            append_jsonl(self.journal_path, rec)
        except Exception:
            pass
        verdict = "KEPT" if rec.get("accepted") else ("rejected" if rec.get("status") == "ok" else rec.get("status"))
        L.info("hypothesis [%s] %s: %s%s", rec.get("source"), rec["summary"], verdict,
               f" ({rec.get('note')})" if rec.get("note") else "")
