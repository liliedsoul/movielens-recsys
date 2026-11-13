"""Разбиение журнала оценок MovieLens на исходную часть и порции «новых» данных.

Оценки упорядочиваются по времени (timestamp). Первые `base_share` записей образуют
исходную часть data/base/base_60.csv, остаток делится на `n_batches` порций примерно
равного объёма: data/incoming/batch_01.csv, batch_02.csv, ... Границы проходят по
временным меткам, поэтому оценки с одинаковым timestamp попадают в одну порцию.
Исходный файл data/raw/<dataset>/ratings.csv не изменяется.

Контрольные суммы SHA-256 исходного файла и всех производных CSV записываются
в data/metadata/data_versions.csv.

Запуск из корня репозитория:
    uv run python -m src.data.split_batches
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path, part: pd.DataFrame, role: str, source: str) -> dict:
    dt = pd.to_datetime(part["timestamp"], unit="s")
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "role": role,
        "rows": len(part),
        "users": part["userId"].nunique(),
        "movies": part["movieId"].nunique(),
        "period_start": dt.min().strftime("%Y-%m-%d %H:%M:%S"),
        "period_end": dt.max().strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": sha256(path),
        "source": source,
    }


def save(part: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # единый символ конца строки — одинаковые контрольные суммы в Windows и Linux
    part.to_csv(path, index=False, lineterminator="\n")


def main() -> None:
    params = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))
    cfg = params["batches"]
    np.random.seed(params["seed"])

    raw_path = ROOT / "data" / "raw" / cfg.get("dataset", params["dataset"]) / "ratings.csv"
    ratings = pd.read_csv(raw_path)
    ratings = ratings.sort_values(["timestamp", "userId", "movieId"], kind="mergesort").reset_index(drop=True)

    n_batches = int(cfg["n_batches"])
    shares = np.linspace(cfg["base_share"], 1.0, n_batches + 1)
    ts = ratings["timestamp"].to_numpy()
    edges = [ts[int(len(ts) * s)] for s in shares[:-1]] + [ts[-1] + 1]

    raw_rel = raw_path.relative_to(ROOT).as_posix()
    rows = [{
        "file": raw_rel, "role": "исходный набор (не изменяется)", "rows": len(ratings),
        "users": ratings["userId"].nunique(), "movies": ratings["movieId"].nunique(),
        "period_start": pd.to_datetime(ts[0], unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "period_end": pd.to_datetime(ts[-1], unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": sha256(raw_path), "source": "GroupLens, ml-latest-small (DVC)",
    }]

    base = ratings[ratings["timestamp"] < edges[0]]
    base_path = ROOT / cfg["base_file"]
    save(base, base_path)
    rows.append(describe(base_path, base, "исходная часть для EDA и обучения", raw_rel))

    for i in range(n_batches):
        part = ratings[(ratings["timestamp"] >= edges[i]) & (ratings["timestamp"] < edges[i + 1])]
        path = ROOT / cfg["incoming_dir"] / f"batch_{i + 1:02d}.csv"
        save(part, path)
        rows.append(describe(path, part, f"порция новых данных № {i + 1}", raw_rel))

    assert sum(r["rows"] for r in rows[1:]) == len(ratings), "потеряны или задублированы записи"

    versions = pd.DataFrame(rows)
    out = ROOT / cfg["versions_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    versions.to_csv(out, index=False, lineterminator="\n")

    for r in rows[1:]:
        share = r["rows"] / len(ratings)
        print(f"{r['file']:28s} {r['rows']:>7,} строк ({share:6.2%})  {r['period_start'][:10]} — {r['period_end'][:10]}")
    print(f"Контрольные суммы: {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
