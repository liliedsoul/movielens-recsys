"""Обучение и сохранение итоговой модели (ПР № 3).

    uv run python -m src.models.train

Берёт алгоритм и гиперпараметры из reports/model_selection.json (результат src.models.experiments),
обучает пайплайн «признаки + предобработка + модель» на всей обучающей части (первые 80 % исходной
части по времени) и сохраняет models/model_<version>.joblib и описание models/model_<version>.json.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features.pipeline import (PARAMS, ROOT, SEED, clean, feature_names, load_catalog, make_pipeline,
                                   target, temporal_split)

ALGORITHMS = {
    "LogisticRegression": lambda p: (LogisticRegression(max_iter=3000, random_state=SEED, **p), True),
    "HistGradientBoostingClassifier": lambda p: (HistGradientBoostingClassifier(random_state=SEED, **p), False),
    "RandomForestClassifier": lambda p: (RandomForestClassifier(n_jobs=-1, random_state=SEED, **p), False),
}


def main() -> None:
    sel = json.loads((ROOT / "reports" / "model_selection.json").read_text(encoding="utf-8"))
    version = PARAMS["model"]["version"]
    catalog = load_catalog()
    base_path = ROOT / PARAMS["batches"]["base_file"]
    df, _ = clean(pd.read_csv(base_path), catalog)
    train, _ = temporal_split(df, PARAMS["model"]["test_share"])

    model, scale = ALGORITHMS[sel["algorithm"]](sel["params"])
    pipe = make_pipeline(model, catalog, features="extended", scale=scale)
    pipe.fit(train, target(train))

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(pipe, out / f"model_{version}.joblib")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = "n/a"
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    info = {
        "version": version,
        "trained": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": f"P(rating >= {PARAMS['relevance_threshold']}) для пары пользователь–фильм; ранжирование top-{PARAMS['top_k']}",
        "algorithm": f"{sel['algorithm']}({', '.join(f'{k}={v}' for k, v in sel['params'].items())})",
        "selection_rule": sel["rule"],
        "data": {
            "file": base_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "train_rows": len(train),
            "train_period": [str(pd.to_datetime(train.timestamp.min(), unit="s")),
                             str(pd.to_datetime(train.timestamp.max(), unit="s"))],
            "split": f"первые {int((1 - PARAMS['model']['test_share']) * 100)} % исходной части по времени",
        },
        "cv": {"scheme": f"TimeSeriesSplit(n_splits={PARAMS['model']['cv_splits']})",
               "roc_auc_mean": sel["cv_roc_auc_mean"], "roc_auc_std": sel["cv_roc_auc_std"],
               "pr_auc_mean": sel["cv_pr_auc_mean"]},
        "features": names,
        "seed": SEED,
        "git_commit": commit,
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "usage": "joblib.load(...).predict_proba(df[['userId','movieId','timestamp']])[:, 1] после src.features.pipeline.clean",
    }
    (out / f"model_{version}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))
    assert names == feature_names(), "состав признаков отличается от ожидаемого"


if __name__ == "__main__":
    main()
