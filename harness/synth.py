"""Write synthetic task directories in the platform layout (task.md + CSVs).

Used by unit tests and by the generalization suite to emulate unseen private tasks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TASK_MD = """# {title}

## Задача

{description}

## Данные

Пути указаны относительно каталога этого Markdown-файла:

- `{sub}/train.csv` — {n_train} строк с признаками и таргетом.
- `{sub}/test.csv` — {n_test} строк с признаками без таргета.
- `{sub}/data_dictionary.csv` — описание признаков.
- `{sub}/sample_submission.csv` — шаблон ответа; значения в нём не являются правильными ответами.

CSV используют кодировку UTF-8, запятую как разделитель и точку как десятичный
разделитель. Пустая ячейка означает пропуск.

## Что предсказать

Для каждой строки `test.csv` необходимо предсказать `{target}`.

{metric_text}

## Формат результата

Файл `submission.csv` в каталоге этого Markdown-файла: CSV в UTF-8,
разделитель — запятая, без дополнительного индекса, ровно столбцы:

```csv
{header}
```

{format_note}
Порядок строк должен совпадать с `test.csv`.
"""


def _dtype_name(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "category"


def write_task(root: Path, train: pd.DataFrame, test: pd.DataFrame, target: str | list[str],
               metric_text: str, *, title: str = "Синтетическая задача",
               description: str = "Постройте модель по табличным данным.",
               id_col: str | None = "id", sub: str = "data/participants",
               output_kind: str = "auto", classes: list[str] | None = None,
               dictionary: bool = True, unit: bool = True, format_note: str = "",
               positive_label: str | None = None) -> Path:
    """Create ``root/task.md`` and ``root/<sub>/*.csv``; returns the task.md path."""
    root = Path(root)
    targets = [target] if isinstance(target, str) else list(target)
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    train.to_csv(d / "train.csv", index=False)
    test.to_csv(d / "test.csv", index=False)
    if dictionary:
        rows = []
        for c in train.columns:
            role = "id" if c == id_col else ("target" if c in targets else "feature")
            r = {"feature": c, "role": role, "dtype": _dtype_name(train[c])}
            if unit:
                r["unit"] = "—"
            r["description_ru"] = f"Признак {c}"
            rows.append(r)
        pd.DataFrame(rows).to_csv(d / "data_dictionary.csv", index=False)
    n = len(test)
    t0 = targets[0]
    if output_kind == "proba_wide":
        cls = classes or sorted(map(str, train[t0].dropna().unique()))
        sample = pd.DataFrame({c: np.full(n, 1.0 / len(cls)) for c in cls})
    elif output_kind == "proba":
        sample = pd.DataFrame({t0: np.full(n, 0.5)})
    elif output_kind == "label":
        sample = pd.DataFrame({t: train[t].dropna().iloc[0] for t in targets}, index=range(n))
    else:  # value
        sample = pd.DataFrame({t: np.zeros(n) for t in targets})
    if id_col:
        sample.insert(0, id_col, test[id_col].values)
    sample.to_csv(d / "sample_submission.csv", index=False)
    md = TASK_MD.format(title=title, description=description, sub=sub, n_train=len(train),
                        n_test=len(test), target=", ".join(targets), metric_text=metric_text,
                        header=",".join(sample.columns), format_note=format_note)
    (root / "task.md").write_text(md, encoding="utf-8")
    return root / "task.md"


def make_frame(n: int, seed: int = 0, n_num: int = 6, n_cat: int = 2, missing: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(n_num)})
    for j in range(n_cat):
        df[f"c{j}"] = rng.choice(["a", "b", "c", "d"], size=n)
    if missing:
        for c in list(df.columns)[: max(1, n_num // 2)]:
            df.loc[rng.random(n) < missing, c] = np.nan
    return df
