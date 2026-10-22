import numpy as np
import pytest
from sklearn import metrics as skm

import agent_metrics as M

rng = np.random.default_rng(0)
Y_REG = rng.normal(10, 3, 200).clip(0.1)
P_REG = Y_REG + rng.normal(0, 1, 200)
Y_CLS = rng.integers(0, 3, 300)
P_CLS = rng.dirichlet([1, 1, 1], 300)
Y_BIN = rng.integers(0, 2, 300)
P_BIN = np.column_stack([1 - (p := rng.random(300)), p])


@pytest.mark.parametrize("name,ref", [
    ("rmse", lambda: np.sqrt(skm.mean_squared_error(Y_REG, P_REG))),
    ("mse", lambda: skm.mean_squared_error(Y_REG, P_REG)),
    ("mae", lambda: skm.mean_absolute_error(Y_REG, P_REG)),
    ("r2", lambda: skm.r2_score(Y_REG, P_REG)),
    ("mape", lambda: skm.mean_absolute_percentage_error(Y_REG, P_REG)),
    ("rmsle", lambda: np.sqrt(skm.mean_squared_log_error(Y_REG, P_REG.clip(0)))),
    ("medae", lambda: skm.median_absolute_error(Y_REG, P_REG)),
])
def test_regression_matches_sklearn(name, ref):
    assert M.evaluate(name, Y_REG, P_REG) == pytest.approx(ref(), rel=1e-9)


@pytest.mark.parametrize("name,ref", [
    ("f1_macro", lambda lab: skm.f1_score(Y_CLS, lab, average="macro")),
    ("f1_weighted", lambda lab: skm.f1_score(Y_CLS, lab, average="weighted")),
    ("f1_micro", lambda lab: skm.f1_score(Y_CLS, lab, average="micro")),
    ("accuracy", lambda lab: skm.accuracy_score(Y_CLS, lab)),
    ("balanced_accuracy", lambda lab: skm.balanced_accuracy_score(Y_CLS, lab)),
    ("mcc", lambda lab: skm.matthews_corrcoef(Y_CLS, lab)),
    ("qwk", lambda lab: skm.cohen_kappa_score(Y_CLS, lab, weights="quadratic")),
])
def test_label_metrics_match_sklearn(name, ref):
    lab = P_CLS.argmax(1)
    assert M.evaluate(name, Y_CLS, P_CLS) == pytest.approx(ref(lab))


def test_proba_metrics():
    assert M.evaluate("logloss", Y_CLS, P_CLS) == pytest.approx(skm.log_loss(Y_CLS, P_CLS))
    assert M.evaluate("roc_auc", Y_BIN, P_BIN) == pytest.approx(skm.roc_auc_score(Y_BIN, P_BIN[:, 1]))
    assert M.evaluate("gini", Y_BIN, P_BIN) == pytest.approx(2 * skm.roc_auc_score(Y_BIN, P_BIN[:, 1]) - 1)
    assert M.evaluate("pr_auc", Y_BIN, P_BIN) == pytest.approx(skm.average_precision_score(Y_BIN, P_BIN[:, 1]))
    assert M.evaluate("roc_auc", Y_CLS, P_CLS) == pytest.approx(
        skm.roc_auc_score(Y_CLS, P_CLS, multi_class="ovr", average="macro"))


def test_binary_f1_uses_threshold():
    lab = (P_BIN[:, 1] >= 0.3).astype(int)
    got = M.evaluate("f1", Y_BIN, P_BIN, decision={"threshold": 0.3})
    assert got == pytest.approx(skm.f1_score(Y_BIN, lab))


def test_class_weights_change_decision():
    p = np.array([[0.6, 0.4], [0.55, 0.45]])
    assert list(M.apply_decision(p)) == [0, 0]
    assert list(M.apply_decision(p, weights=np.array([1.0, 2.0]))) == [1, 1]


def test_direction_and_better():
    assert M.get("rmse").better(1.0, 2.0)
    assert M.get("f1_macro").better(0.9, 0.8)
    assert not M.get("logloss").better(0.9, 0.5)
    assert M.get("auc").name == "roc_auc"


@pytest.mark.parametrize("text,expected", [
    ("Метрика оценки — **RMSE** на исходной шкале таргета", "rmse"),
    ("Основная метрика — **Macro F1**. Дополнительно рассчитываются **Accuracy** и **Weighted F1**.", "f1_macro"),
    ("Качество оценивается по MAE; RMSE приводится справочно.", "mae"),
    ('f1_score(y_true, y_pred, average="weighted")', "f1_weighted"),
    ("Метрика: ROC-AUC", "roc_auc"),
    ("Основная метрика — RMSLE", "rmsle"),
    ("Метрика: LogLoss (кросс-энтропия)", "logloss"),
    ("Метрика качества — коэффициент детерминации R²", "r2"),
    ("Оценка по Quadratic Weighted Kappa", "qwk"),
    ("Используется MAPE", "mape"),
    ("Метрика — Gini = 2·AUC − 1", "gini"),
    ("Метрика — сбалансированная точность (balanced accuracy)", "balanced_accuracy"),
])
def test_extract_primary_metric(text, expected):
    assert M.extract_primary_metric(text)[0] == expected


def test_extract_none():
    assert M.extract_primary_metric("Постройте хорошую модель.")[0] is None


def test_normalized_score_formulas():
    assert M.normalized_score("rmse", 40.0, 50.0) == pytest.approx(20.0)
    assert M.normalized_score("rmse", 60.0, 50.0) == 0.0
    assert M.normalized_score("f1_macro", 0.7, 0.4) == pytest.approx(50.0)
    assert M.normalized_score("f1_macro", 0.3, 0.4) == 0.0
    assert M.normalized_score("logloss", 0.5, 1.0) == pytest.approx(50.0)


def test_every_metric_has_default_and_runs():
    for name, m in M.REGISTRY.items():
        if M.REGRESSION in m.task_types:
            v = M.evaluate(name, Y_REG, P_REG)
        elif M.MULTICLASS in m.task_types:
            v = M.evaluate(name, Y_CLS, P_CLS)
        else:
            v = M.evaluate(name, Y_BIN, P_BIN)
        assert np.isfinite(v), name
    for t in (M.REGRESSION, M.BINARY, M.MULTICLASS):
        assert M.compatible(M.default_metric(t), t)
