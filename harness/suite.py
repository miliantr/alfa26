"""Generalization suite: synthetic "unseen" tasks in the platform layout.

Each task is built from the public data or sklearn's bundled (offline) datasets, gets a
Russian task.md in the organizers' style, a hidden-label file, and is run through
solution.py exactly like the platform would. The agent must finish every task within the
budget with a valid submission; scores are compared with a simple baseline proxy.

    python -m harness.suite --budget 240            # build + run all tasks
    python -m harness.suite --only wine_tiny,wide_binary --budget 120
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.synth import write_task  # noqa: E402

DATA = REPO / "data"


def _split(df: pd.DataFrame, frac: float = 0.3, seed: int = 0, by: str | None = None):
    rng = np.random.default_rng(seed)
    if by is not None:
        last = df[by].max()
        te = df[by] == last
    else:
        te = pd.Series(rng.random(len(df)) < frac, index=df.index)
    return df[~te].reset_index(drop=True), df[te].reset_index(drop=True)


def _energy(n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA / "task1_energy/regression/participants/train.csv")
    return df.sample(n, random_state=0).reset_index(drop=True) if n and n < len(df) else df


def _attacks(n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA / "task2_attacks/multiclass/participants/train.csv")
    return df.sample(n, random_state=0).reset_index(drop=True) if n and n < len(df) else df


# --------------------------------------------------------------------------- tasks
def t_energy_star_mae(root: Path):
    df = _energy(30000).drop(columns=["site_eui"])
    tr, te = _split(df)
    tr = tr  # rows with missing target stay in train on purpose: the agent must drop them
    te = te.dropna(subset=["energy_star_rating"]).reset_index(drop=True)
    labels = te[["id", "energy_star_rating"]]
    md = write_task(root, tr, te.drop(columns="energy_star_rating"), "energy_star_rating",
                    "Метрика качества — **MAE** (средняя абсолютная ошибка). Чем меньше, тем лучше.",
                    title="Рейтинг энергоэффективности", description="Оцените рейтинг ENERGY STAR здания. "
                    "В обучающей выборке у части зданий рейтинг не указан.", output_kind="value")
    return md, labels


def t_building_class_auc(root: Path):
    df = _energy(25000).drop(columns=["site_eui"])
    tr, te = _split(df, seed=1)
    labels = te[["id", "building_class"]]
    md = write_task(root, tr, te.drop(columns="building_class"), "building_class",
                    "Метрика — **ROC-AUC**. Для каждой строки нужно вернуть вероятность того, что здание "
                    "коммерческое (Commercial).", title="Тип здания", output_kind="proba",
                    description="Определите, является ли здание коммерческим.",
                    format_note="Значение — вероятность класса Commercial в диапазоне [0, 1].")
    return md, labels


def t_attack_binary_logloss(root: Path):
    df = _attacks(40000)
    df["is_attack"] = (df["attack_cat"] != "Normal").astype(int)
    df = df.drop(columns="attack_cat")
    tr, te = _split(df, seed=2)
    labels = te[["id", "is_attack"]]
    md = write_task(root, tr, te.drop(columns="is_attack"), "is_attack",
                    "Метрика — **LogLoss** (логарифмическая функция потерь) по вероятности атаки.",
                    title="Обнаружение атак", output_kind="proba",
                    description="По характеристикам сетевого потока оцените вероятность атаки (1) против нормального трафика (0).")
    return md, labels


def t_attack_weighted_f1_renamed(root: Path):
    df = _attacks(30000)
    cls = sorted(df["attack_cat"].unique())
    df["target"] = df["attack_cat"].map({c: i for i, c in enumerate(cls)})
    df = df.drop(columns="attack_cat")
    feats = [c for c in df.columns if c not in ("id", "target")]
    df = df.rename(columns={c: f"f_{i:02d}" for i, c in enumerate(feats)}).rename(columns={"id": "row_id"})
    tr, te = _split(df, seed=3)
    labels = te[["row_id", "target"]]
    md = write_task(root, tr, te.drop(columns="target"), "target",
                    "Оценка по метрике **Weighted F1** (`f1_score(average=\"weighted\")`).",
                    title="Анонимизированная классификация", id_col="row_id", unit=False, output_kind="label",
                    description="Признаки анонимизированы. Классы закодированы целыми числами.")
    return md, labels


def t_attack_proba_wide(root: Path):
    df = _attacks(20000)
    df = df[df["attack_cat"].isin(["Normal", "Generic", "Exploits", "Fuzzers", "DoS"])].reset_index(drop=True)
    tr, te = _split(df, seed=4)
    labels = te[["id", "attack_cat"]]
    md = write_task(root, tr, te.drop(columns="attack_cat"), "attack_cat",
                    "Метрика — **LogLoss**. Для каждого класса нужно вернуть вероятность в отдельном столбце.",
                    title="Вероятности типов трафика", output_kind="proba_wide",
                    format_note="Сумма вероятностей по строке должна быть равна 1.")
    return md, labels


def t_energy_missing_rmsle(root: Path):
    df = _energy(30000)
    rng = np.random.default_rng(5)
    for c in [c for c in df.columns if c not in ("id", "site_eui", "Year_Factor")]:
        df.loc[rng.random(len(df)) < 0.2, c] = np.nan
    df = df.rename(columns={c: c.upper() for c in df.columns if c not in ("id", "site_eui")})
    tr, te = _split(df, by="YEAR_FACTOR")
    labels = te[["id", "site_eui"]]
    md = write_task(root, tr, te.drop(columns="site_eui"), "site_eui",
                    "Метрика — **RMSLE** (корень из среднеквадратичной логарифмической ошибки).",
                    title="Энергопотребление с пропусками", output_kind="value",
                    description="Данные содержат много пропусков. Тест относится к более позднему периоду.")
    return md, labels


def t_energy_qwk(root: Path):
    df = _energy(30000).drop(columns=["site_eui"]).dropna(subset=["energy_star_rating"])
    df["grade"] = pd.cut(df["energy_star_rating"], [0, 20, 40, 60, 80, 101], labels=False, include_lowest=True).astype(int)
    df = df.drop(columns="energy_star_rating").reset_index(drop=True)
    tr, te = _split(df, seed=6)
    labels = te[["id", "grade"]]
    md = write_task(root, tr, te.drop(columns="grade"), "grade",
                    "Метрика — **Quadratic Weighted Kappa** (`cohen_kappa_score(weights=\"quadratic\")`).",
                    title="Класс энергоэффективности", output_kind="label",
                    description="Предскажите порядковый класс энергоэффективности (0 — худший, 4 — лучший).")
    return md, labels


def _sk(loader, target_name="target"):
    d = loader(as_frame=True)
    df = d.frame.copy()
    if "target" in df.columns and target_name != "target":
        df = df.rename(columns={"target": target_name})
    df.insert(0, "id", np.arange(len(df)))
    return df


def t_diabetes_r2_tiny(root: Path):
    from sklearn.datasets import load_diabetes

    df = _sk(load_diabetes, "progression")
    tr, te = _split(df, seed=7)
    labels = te[["id", "progression"]]
    md = write_task(root, tr, te.drop(columns="progression"), "progression",
                    "Метрика — **R²** (коэффициент детерминации), больше — лучше.", title="Прогрессия диабета",
                    output_kind="value", description="Очень маленькая выборка (442 пациента).")
    return md, labels


def t_breast_cancer_f1(root: Path):
    from sklearn.datasets import load_breast_cancer

    df = _sk(load_breast_cancer, "diagnosis")
    df["diagnosis"] = df["diagnosis"].map({0: "malignant", 1: "benign"})
    tr, te = _split(df, seed=8)
    labels = te[["id", "diagnosis"]]
    md = write_task(root, tr, te.drop(columns="diagnosis"), "diagnosis",
                    "Метрика — **F1** для положительного класса `malignant` (злокачественная опухоль).",
                    title="Диагностика опухоли", output_kind="label", positive_label="malignant")
    return md, labels


def t_wine_tiny(root: Path):
    from sklearn.datasets import load_wine

    df = _sk(load_wine, "cultivar")
    tr, te = _split(df, seed=9)
    labels = te[["id", "cultivar"]]
    md = write_task(root, tr, te.drop(columns="cultivar"), "cultivar",
                    "Метрика — **Accuracy** (доля правильных ответов).", title="Сорт вина", output_kind="label",
                    dictionary=False)
    return md, labels


def t_digits_macro_f1(root: Path):
    from sklearn.datasets import load_digits

    df = _sk(load_digits, "digit")
    tr, te = _split(df, seed=10)
    labels = te[["id", "digit"]]
    md = write_task(root, tr, te.drop(columns="digit"), "digit",
                    "Основная метрика — **Macro F1**.", title="Рукописные цифры", output_kind="label")
    return md, labels


def t_wide_binary(root: Path):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=3000, n_features=400, n_informative=25, n_redundant=50,
                               weights=[0.85, 0.15], random_state=11)
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(X.shape[1])])
    df.insert(0, "id", np.arange(len(df)))
    df["label"] = y
    tr, te = _split(df, seed=11)
    labels = te[["id", "label"]]
    md = write_task(root, tr, te.drop(columns="label"), "label", "Метрика — **ROC-AUC**.",
                    title="Широкая таблица", output_kind="proba", description="400 признаков, 3000 объектов.")
    return md, labels


def t_datetime_text_mae(root: Path):
    rng = np.random.default_rng(12)
    n = 6000
    dates = pd.Timestamp("2021-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 900, n)), unit="D")
    words = np.array(["быстро", "доставка", "качество", "плохо", "отлично", "цена", "возврат", "сервис"])
    text = [" ".join(rng.choice(words, rng.integers(3, 9))) for _ in range(n)]
    cat = rng.choice(["север", "юг", "центр"], n)
    x = rng.normal(size=n)
    y = 10 + 3 * (dates.dayofweek >= 5) + 2 * np.sin(dates.month / 12 * 2 * np.pi) + 4 * x \
        + np.array([t.count("плохо") * -2.0 + t.count("отлично") * 2.0 for t in text]) + (cat == "юг") * 3 \
        + rng.normal(0, 1, n)
    df = pd.DataFrame({"id": np.arange(n), "order_date": dates.strftime("%Y-%m-%d"), "review": text,
                       "region": cat, "x": x, "amount": y})
    cut = int(n * 0.75)
    tr, te = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    labels = te[["id", "amount"]]
    md = write_task(root, tr, te.drop(columns="amount"), "amount", "Метрика — **MAE**.",
                    title="Сумма заказа", output_kind="value",
                    description="Есть дата заказа и текст отзыва. Тест — более поздние заказы.")
    return md, labels


def t_multi_target(root: Path):
    df = _energy(20000).dropna(subset=["energy_star_rating"]).reset_index(drop=True)
    tr, te = _split(df, seed=13)
    labels = te[["id", "site_eui", "energy_star_rating"]]
    md = write_task(root, tr, te.drop(columns=["site_eui", "energy_star_rating"]), ["site_eui", "energy_star_rating"],
                    "Метрика — **RMSE**, усреднённая по двум целевым переменным.", title="Два таргета",
                    output_kind="value", description="Предскажите одновременно site_eui и energy_star_rating.")
    return md, labels


TASKS = {
    "multi_target": t_multi_target,
    "energy_star_mae": t_energy_star_mae,
    "building_class_auc": t_building_class_auc,
    "attack_binary_logloss": t_attack_binary_logloss,
    "attack_weighted_f1_renamed": t_attack_weighted_f1_renamed,
    "attack_proba_wide": t_attack_proba_wide,
    "energy_missing_rmsle": t_energy_missing_rmsle,
    "energy_qwk": t_energy_qwk,
    "diabetes_r2_tiny": t_diabetes_r2_tiny,
    "breast_cancer_f1": t_breast_cancer_f1,
    "wine_tiny": t_wine_tiny,
    "digits_macro_f1": t_digits_macro_f1,
    "wide_binary": t_wide_binary,
    "datetime_text_mae": t_datetime_text_mae,
}


# -------------------------------------------------------------------------- runner
def _score(md: Path, labels: pd.DataFrame, labels_path: Path) -> dict:
    from agent.submission import SubmissionTemplate, check_file
    from agent.task_parser import parse_task
    from harness.scoring import score_submission

    spec = parse_task(md)
    out = Path(spec.output_paths[0])
    if not out.exists():
        return {"valid": False, "errors": ["missing"], "spec": spec.summary()}
    if len(spec.target_cols) > 1:
        from agent.task_parser import sub_spec

        sub = pd.read_csv(out, dtype={spec.id_col: str})
        lab = pd.read_csv(labels_path, dtype={spec.id_col: str})
        errs = [] if list(sub.columns) == spec.submission_cols and len(sub) == len(lab) else ["bad shape"]
        errs += [f"{t}: non-finite" for t in spec.target_cols if not np.isfinite(pd.to_numeric(sub[t], errors="coerce")).all()]
        res = {"valid": not errs, "errors": errs, "spec": spec.summary()}
        if not errs:
            m = lab.merge(sub, on=spec.id_col, suffixes=("_true", ""))
            res["score"] = {t: round(float(np.sqrt(np.mean((m[t] - m[f"{t}_true"]) ** 2))), 4) for t in spec.target_cols}
        return res
    errs = check_file(out, SubmissionTemplate.from_spec(spec))
    res = {"valid": not errs, "errors": errs, "spec": spec.summary()}
    if not errs:
        res["score"] = score_submission(out, labels_path, spec)["value"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=240)
    ap.add_argument("--only", default="")
    ap.add_argument("--llm", default="mock", choices=["off", "mock", "real"])
    ap.add_argument("--out", default=str(REPO / "runs" / f"suite_{time.strftime('%Y%m%d_%H%M%S')}"))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    names = [n for n in (args.only.split(",") if args.only else TASKS) if n]
    results = {}
    for name in names:
        root = out / name / "task"
        md, labels = TASKS[name](root)
        labels_path = out / name / "labels.csv"
        labels.to_csv(labels_path, index=False)
        env = {**{k: v for k, v in __import__("os").environ.items() if not k.startswith(("OPENAI_", "AGENT_"))},
               "AGENT_TIME_BUDGET": str(args.budget), "AGENT_LLM_MODE": args.llm, "PYTHONIOENCODING": "utf-8"}
        t0 = time.time()
        with open(out / name / "stdout.log", "w", encoding="utf-8") as f:
            try:
                rc = subprocess.run([sys.executable, str(REPO / "solution.py"), str(md)], cwd=str(out / name),
                                    env=env, stdout=f, stderr=subprocess.STDOUT, timeout=args.budget + 30).returncode
            except subprocess.TimeoutExpired:
                rc = "killed"
        wall = round(time.time() - t0, 1)
        r = {"exit": rc, "wall_s": wall, "within_budget": wall <= args.budget + 5, **_score(md, labels, labels_path)}
        results[name] = r
        print(f"{name:30s} exit={rc} wall={wall:6.1f}s valid={r['valid']} score={r.get('score')} | {r['spec']}",
              flush=True)
    (out / "suite_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    ok = all(r["valid"] and r["within_budget"] for r in results.values())
    print(f"\n{sum(r['valid'] for r in results.values())}/{len(results)} valid; all within budget: "
          f"{all(r['within_budget'] for r in results.values())}; results: {out / 'suite_results.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
