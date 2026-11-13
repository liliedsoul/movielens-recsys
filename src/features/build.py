"""Превращает любой файл с оценками (исходную часть или порцию) в данные, готовые для модели.

    uv run python -m src.features.build data/incoming/batch_01.csv

Шаги: чтение → очистка по требованиям DQ (clean) → признаки строки из каталога (add_row_features)
→ если итоговая модель уже обучена (models/model_<version>.joblib) — признаки с параметрами,
оценёнными на обучающей выборке (все шаги пайплайна, кроме самой модели).
Результат: data/processed/<имя>_features.csv и отчёт очистки data/processed/<имя>_clean_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.pipeline import PARAMS, ROOT, add_row_features, clean, load_catalog


def build(path: Path, out_dir: Path | None = None) -> Path:
    catalog = load_catalog()
    df, report = clean(pd.read_csv(path), catalog)
    model_path = ROOT / "models" / f"model_{PARAMS['model']['version']}.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        feats = model[:-1].transform(df)                     # параметры не переоцениваются
        cols = list(model.named_steps["prep"].get_feature_names_out())
        out = pd.DataFrame(feats, columns=cols)
        out.insert(0, "movieId", df["movieId"].to_numpy()); out.insert(0, "userId", df["userId"].to_numpy())
        out["rating"] = df["rating"].to_numpy()
        report["mode"] = f"признаки по {model_path.name}"
    else:
        out = add_row_features(df, catalog)
        report["mode"] = "очистка и признаки строки (модель ещё не обучена)"
    out_dir = out_dir or ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{path.stem}_features.csv"
    out.to_csv(target, index=False, lineterminator="\n", float_format="%.10g")
    report["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (out_dir / f"{path.stem}_clean_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{path} → {target.relative_to(ROOT).as_posix()}: {report}")
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    for f in ap.parse_args().files:
        build(f if f.is_absolute() else ROOT / f)
