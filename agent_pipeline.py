"""Experiment pool, cost model, journal and submission refresh."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import agent_ensemble as ensemble
import agent_log as log
import agent_metrics as metrics
import agent_postprocess as postprocess
from agent_executor import run_job
from agent_io_utils import append_jsonl, write_json

L = log.get("pipeline")

# Relative cost vs default LightGBM, used before a family has been timed.
FAMILY_COST = {"lgbm_fast": 0.3, "lgbm": 1.0, "catboost": 3.0, "xgb": 1.8, "linear": 0.4,
               "extratrees": 1.2, "randomforest": 1.5}


@dataclass
class Experiment:
    id: str
    name: str
    source: str
    model: dict
    feature_ops: list = field(default_factory=list)
    drop_cols: list = field(default_factory=list)
    hypothesis: str = ""
    status: str = "pending"
    score: float | None = None
    raw_score: float | None = None
    fold_scores: list = field(default_factory=list)
    decision: dict = field(default_factory=dict)
    seconds: float = 0.0
    n_features: int = 0
    importance: list = field(default_factory=list)
    zero_importance: list = field(default_factory=list)
    error: str = ""
    fs_id: str = ""
    parent: str | None = None
    delta: float | None = None
    accepted: bool | None = None
    peak_rss_gb: float = 0.0
    started_at: float = 0.0
    seed: int = 42

    def record(self) -> dict:
        d = asdict(self)
        d["importance"] = self.importance[:15]
        d.pop("zero_importance", None)
        return d


class Pipeline:
    def __init__(self, *, run_dir: Path, base_path: Path, y: np.ndarray, folds: np.ndarray, meta: dict,
                 metric: str, task: str, settings, deadline, template, spec, n_jobs: int, mem_limit: int,
                 eval_mask: np.ndarray | None = None):
        self.run_dir, self.base_path = Path(run_dir), Path(base_path)
        self.y, self.folds, self.meta = y, folds, meta
        self.metric, self.task, self.k = metric, task, int(meta.get("n_classes", 0))
        self.m = metrics.REGISTRY[metric]
        self.settings, self.deadline = settings, deadline
        self.template, self.spec = template, spec
        self.n_jobs, self.mem_limit = n_jobs, mem_limit
        self.experiments: list[Experiment] = []
        self.base_ops: list[dict] = []
        self.importance: list[str] = []
        self.sub_score: float | None = None
        self.sub_desc: dict = {}
        self.timings: dict[str, list[tuple[int, float]]] = {}
        self.journal = self.run_dir / "journal.jsonl"
        self.y_range = tuple(meta["y_range"]) if meta.get("y_range") else None
        self.eval_mask = np.ones(len(y), dtype=bool) if eval_mask is None else np.asarray(eval_mask, bool)
        self._n = 0

    # ---------------------------------------------------------------- bookkeeping
    @property
    def ok(self) -> list[Experiment]:
        return [e for e in self.experiments if e.status == "ok" and e.score is not None and np.isfinite(e.score)]

    def best(self, family: str | None = None, source: str | None = None) -> Experiment | None:
        pool = [e for e in self.ok if (family is None or e.model["family"] == family)
                and (source is None or e.source == source)]
        if not pool:
            return None
        return max(pool, key=lambda e: self.m.sign() * e.score)

    def estimate_seconds(self, family: str, n_features: int | None = None) -> float:
        n_features = n_features or self.meta.get("n_features", 10)
        hist = self.timings.get(family)
        if hist:
            # Median of the last runs, each rescaled to this feature count: one quick
            # trial (e.g. a high learning rate) must not shrink the next job's timeout.
            scaled = [s * max(0.5, n_features / max(nf, 1)) for nf, s in hist[-5:]]
            return float(np.median(scaled))
        ref = self.timings.get("lgbm") or self.timings.get("lgbm_fast")
        if ref:
            nf, s = ref[-1]
            base = s / FAMILY_COST["lgbm_fast" if "lgbm" not in self.timings else "lgbm"]
            return base * FAMILY_COST.get(family, 2.0) * max(0.5, n_features / max(nf, 1))
        return 60.0 * FAMILY_COST.get(family, 2.0)

    # ------------------------------------------------------------------- running
    def run(self, name: str, model: dict, *, feature_ops: list | None = None, drop_cols: list | None = None,
            source: str = "zoo", hypothesis: str = "", timeout: float | None = None,
            parent: str | None = None, seed: int | None = None, phase: str = "research",
            timeout_factor: float = 3.0, max_share: float | None = None) -> Experiment:
        self._n += 1
        ops = self.base_ops if feature_ops is None else feature_ops
        exp = Experiment(id=f"e{self._n:03d}", name=name, source=source, model=model, feature_ops=ops,
                         drop_cols=list(drop_cols or []), hypothesis=hypothesis, parent=parent,
                         seed=self.settings.seed if seed is None else seed, started_at=round(self.deadline.elapsed(), 1))
        est = self.estimate_seconds(model["family"])
        if timeout is None:
            if model["family"] not in self.timings:
                timeout_factor = max(timeout_factor, 5.0)  # first run of a family: estimate is a guess
            timeout = self.deadline.job_timeout(est, factor=timeout_factor, phase=phase, minimum=30.0)
            if max_share is not None:
                # One research job may not swallow the window: at most max_share of what is
                # left (but never less than 1.5x its own estimate).
                timeout = min(timeout, max(est * 1.5, max_share * self.deadline.research_remaining()))
        if timeout < 10:
            exp.status, exp.error = "skipped", "no time left"
            self._log(exp)
            return exp
        job = {"base_path": str(self.base_path), "fs_cache_dir": str(self.run_dir / "fs"),
               "feature_ops": ops, "drop_cols": exp.drop_cols, "model": model, "metric": self.metric,
               "seed": exp.seed, "importance": self.importance[:40], "time_limit": timeout * 0.9}
        L.info("[%s] %s (%s) est=%.0fs timeout=%.0fs", exp.id, name, model["family"], est, timeout)
        res = run_job(job, self.run_dir / "jobs" / exp.id, timeout=timeout, mem_limit=self.mem_limit,
                      n_jobs=self.n_jobs)
        exp.status = res.get("status", "error")
        exp.seconds = res.get("wall_seconds", 0.0)
        exp.peak_rss_gb = res.get("peak_rss_gb", 0.0)
        if exp.status == "ok":
            exp.score = float(res["score"])
            exp.raw_score = res.get("raw_score")
            exp.fold_scores = res.get("fold_scores", [])
            exp.decision = res.get("decision", {})
            exp.n_features = res.get("n_features", 0)
            exp.importance = res.get("importance", [])
            exp.zero_importance = res.get("zero_importance", [])
            exp.fs_id = res.get("fs_id", "")
            self.timings.setdefault(model["family"], []).append((exp.n_features, exp.seconds))
            if model["family"] in ("lgbm", "lgbm_fast", "catboost", "xgb") and exp.importance:
                if self.best() is exp or not self.importance:
                    self.importance = [n for n, _ in exp.importance]
            L.info("[%s] %s: %s=%.5f (raw %.5f) folds=%s in %.0fs", exp.id, name, self.metric, exp.score,
                   exp.raw_score if exp.raw_score is not None else float("nan"),
                   [round(s, 4) for s in exp.fold_scores], exp.seconds)
        else:
            exp.error = res.get("error", "")[:2000]
            L.warning("[%s] %s failed: %s", exp.id, name, exp.error[:300])
        self.experiments.append(exp)
        self._log(exp)
        return exp

    def _log(self, exp: Experiment) -> None:
        try:
            append_jsonl(self.journal, {**exp.record(), "elapsed": round(self.deadline.elapsed(), 1)})
        except Exception:
            pass

    def preds(self, exp: Experiment) -> tuple[np.ndarray, np.ndarray]:
        d = self.run_dir / "jobs" / exp.id
        return np.load(d / "oof.npy").astype(np.float64), np.load(d / "test.npy").astype(np.float64)

    # ------------------------------------------------------------ comparison rule
    def compare(self, new: Experiment, ref: Experiment) -> dict:
        """Paired comparison on identical folds (all repeats). Significant = better overall,
        better on at least 60% of fold pairs, and a mean fold gain above half a standard
        error — conservative on purpose: many trials on noisy folds produce lucky winners."""
        s = self.m.sign()
        delta = s * (new.score - ref.score)
        fd = s * (np.asarray(new.fold_scores, float) - np.asarray(ref.fold_scores, float))
        fd = fd[np.isfinite(fd)]
        k = len(fd)
        wins = int((fd > 0).sum())
        se = float(fd.std(ddof=1) / np.sqrt(k)) if k > 1 else float("inf")
        scale = max(abs(ref.score), 1e-9)
        min_delta = 1e-4 * scale
        if k < 2:
            significant = bool(delta > min_delta)
        else:
            significant = bool(delta > min_delta and wins >= int(np.ceil(0.6 * k)) and fd.mean() > 0.5 * se)
        return {"delta": delta, "rel_delta": delta / scale, "fold_wins": wins, "folds": k,
                "fold_se": se, "significant": significant}

    # ---------------------------------------------------------------- submission
    def candidates(self, max_n: int = 12) -> list[Experiment]:
        ranked = sorted(self.ok, key=lambda e: -self.m.sign() * e.score)
        out, per_family = [], {}
        for e in ranked:
            f = e.model["family"]
            if per_family.get(f, 0) >= 4:
                continue
            per_family[f] = per_family.get(f, 0) + 1
            out.append(e)
            if len(out) >= max_n:
                break
        return out

    def refresh_submission(self, tag: str = "") -> bool:
        """Blend the best experiments; write the submission if it beats what is on disk."""
        cands = self.candidates()
        if not cands:
            return False
        key = tuple(e.id for e in cands)
        if key == getattr(self, "_last_cands", None):
            return False  # the newest experiment did not enter the candidate pool
        self._last_cands = key
        try:
            loaded = [(e, *self.preds(e)) for e in cands]
        except Exception as e:
            L.warning("cannot load predictions: %s", e)
            return False
        best_single = loaded[0]
        choice = {"kind": "single", "members": [best_single[0].id], "weights": [1.0],
                  "decision": best_single[0].decision, "score": best_single[0].score}
        test = best_single[2]
        if len(loaded) > 1:
            try:
                mk = self.eval_mask
                ens = ensemble.build(self.metric, self.task, self._y_eval()[mk], [o[mk] for _, o, _ in loaded],
                                     self.k, y_range=self.y_range, groups=np.asarray(self.folds)[mk])
                if self.m.better(ens["score"], best_single[0].score):
                    w = np.asarray(ens["weights"])
                    test = ensemble.blend([t for _, _, t in loaded], w)
                    choice = {"kind": "blend", "members": [e.id for e, _, _ in loaded],
                              "weights": w.tolist(), "decision": ens["decision"], "score": ens["score"]}
            except Exception as e:
                L.warning("ensemble failed: %s", e)
        if self.sub_score is not None and not self.m.better(choice["score"], self.sub_score):
            return False
        df = self.to_frame(test, choice["decision"])
        if df is None or not self.template.write(df, tag or choice["kind"]):
            return False
        self.sub_score, self.sub_desc = choice["score"], choice
        L.info("submission <- %s %s=%.5f members=%s", choice["kind"], self.metric, choice["score"],
               [f"{m}:{w:.2f}" for m, w in zip(choice["members"], choice["weights"]) if w > 0])
        write_json(self.run_dir / "submission_choice.json",
                   {**{k: v for k, v in choice.items()}, "elapsed": round(self.deadline.elapsed(), 1)})
        return True

    def _y_eval(self) -> np.ndarray:
        return self.y.astype(float) if self.task == metrics.REGRESSION else self.y.astype(int)

    def to_frame(self, test_raw: np.ndarray, decision: dict):
        spec = self.spec
        try:
            if self.task == metrics.REGRESSION:
                vals = postprocess.apply(decision, test_raw)
                return self.template.from_predictions(vals)
            out_kind = spec.output_kind
            if out_kind == "label":
                idx = postprocess.labels_from(decision, test_raw)
                return self.template.from_predictions([spec.classes[i] for i in idx])
            P = postprocess.apply(decision if decision.get("kind") == "temperature" else {"kind": "proba"}, test_raw)
            P = np.clip(P, 0.0, 1.0)
            if out_kind == "proba":
                pos = spec.classes.index(spec.positive_label) if spec.positive_label in spec.classes else 1
                return self.template.from_predictions(P[:, pos])
            return self.template.from_predictions(P)
        except Exception as e:
            L.error("cannot build submission frame: %s", e)
            return None
