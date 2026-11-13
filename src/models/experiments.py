"""Эксперименты ПР № 3: базовые решения, линейные модели, методы классификации, ансамбли, подбор.

    uv run python -m src.models.experiments

Все сравнения — по кросс-валидации TimeSeriesSplit на обучающей части (первые 80 % исходной
части по времени). Отложенная тестовая часть здесь не используется (ПР № 4).
Результаты: reports/experiments.csv, графики reports/figures_pr3/, выбор модели
reports/model_selection.json. Если установлен MLflow, каждый эксперимент пишется и в mlruns/.
"""

from __future__ import annotations

import json
import subprocess
import time
import warnings
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.features.pipeline import (PARAMS, ROOT, SEED, clean, load_catalog, make_pipeline, target,
                                   temporal_split)

warnings.filterwarnings("ignore")
FIG = ROOT / "reports" / "figures_pr3"
OUT = ROOT / "reports" / "experiments.csv"
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "recall": "recall"}
BLUE, ORANGE = "#1f77b4", "#ff7f0e"


class RuleScorer(ClassifierMixin, BaseEstimator):
    """Базовое правило без обучения: оценка = значение одного признака (уже в шкале [0, 1] или любой)."""

    def __init__(self, column: int = 0, threshold: float = 0.5):
        self.column = column
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, self.column].astype(float)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (np.asarray(X)[:, self.column] >= self.threshold).astype(int)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "n/a"


class Runner:
    def __init__(self, X, y, cv, catalog, data_sha):
        self.X, self.y, self.cv, self.catalog = X, y, cv, catalog
        self.rows, self.data_sha, self.commit = [], data_sha, git_commit()
        try:
            import mlflow
            mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
            mlflow.set_experiment("pr3-model-comparison")
            self.mlflow = mlflow
        except Exception:
            self.mlflow = None

    def run(self, name, family, model, features="extended", scale=True, params=None, train_score=False):
        pipe = make_pipeline(model, self.catalog, features=features, scale=scale)
        t = time.time()
        res = cross_validate(pipe, self.X, self.y, cv=self.cv, scoring=SCORING, n_jobs=1, error_score="raise")
        if train_score:
            # качество на обучающих блоках считается на тех же признаках, на которых модель обучалась
            # (fit_transform: признаки «на момент прогноза»), иначе в признак попала бы сама оценка
            from sklearn.base import clone
            from sklearn.metrics import roc_auc_score
            tr_scores = []
            for a, _ in self.cv.split(self.X):
                p = clone(pipe)
                Xa, ya = self.X.iloc[a], self.y[a]
                F = p[:-1].fit_transform(Xa, ya)
                p[-1].fit(F, ya)
                tr_scores.append(roc_auc_score(ya, p[-1].predict_proba(F)[:, 1]))
            res["train_roc_auc"] = np.array(tr_scores)
        row = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "experiment": name, "family": family,
               "features": features, "params": json.dumps(params or {}, ensure_ascii=False),
               "fit_time_s": round(float(np.mean(res["fit_time"])), 3), "total_time_s": round(time.time() - t, 1),
               "git_commit": self.commit, "data_sha256": self.data_sha[:16]}
        for k in SCORING:
            row[f"{k}_mean"] = round(float(np.nanmean(res[f"test_{k}"])), 4)
            row[f"{k}_std"] = round(float(np.nanstd(res[f"test_{k}"])), 4)
        row["roc_auc_folds"] = ";".join(f"{v:.4f}" for v in res["test_roc_auc"])
        if train_score:
            row["train_roc_auc_mean"] = round(float(np.mean(res["train_roc_auc"])), 4)
        self.rows.append(row)
        if self.mlflow:
            with self.mlflow.start_run(run_name=name):
                self.mlflow.log_params({"family": family, "features": features, **(params or {})})
                self.mlflow.log_metrics({k: v for k, v in row.items() if k.endswith(("_mean", "_std", "time_s"))})
                self.mlflow.set_tags({"git_commit": self.commit, "data_sha256": self.data_sha})
        print(f"{name:45s} ROC-AUC {row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}  "
              f"PR-AUC {row['pr_auc_mean']:.3f}  F1 {row['f1_mean']:.3f}  {row['fit_time_s']:.2f} c")
        return row

    def save(self):
        pd.DataFrame(self.rows).to_csv(OUT, index=False, lineterminator="\n")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    catalog = load_catalog()
    base_path = ROOT / PARAMS["batches"]["base_file"]
    df, _ = clean(pd.read_csv(base_path), catalog)
    train, test = temporal_split(df, PARAMS["model"]["test_share"])
    X, y = train, target(train)
    split_info = {
        "base_rows": len(df), "train_rows": len(train), "test_rows": len(test),
        "train_period": [str(pd.to_datetime(train.timestamp.min(), unit="s")), str(pd.to_datetime(train.timestamp.max(), unit="s"))],
        "test_period": [str(pd.to_datetime(test.timestamp.min(), unit="s")), str(pd.to_datetime(test.timestamp.max(), unit="s"))],
        "train_positive_share": round(float(y.mean()), 4), "test_positive_share": round(float(target(test).mean()), 4),
        "test_new_users_share_of_ratings": round(float((~test.userId.isin(train.userId)).mean()), 4),
    }
    (ROOT / "reports" / "split_info.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(split_info)

    cv = TimeSeriesSplit(n_splits=PARAMS["model"]["cv_splits"])
    folds = [{"fold": i + 1, "train": len(a), "valid": len(b),
              "new_users_share": round(float((~X.iloc[b].userId.isin(X.iloc[a].userId)).mean()), 3)}
             for i, (a, b) in enumerate(cv.split(X))]
    (ROOT / "reports" / "cv_folds.json").write_text(json.dumps(folds, ensure_ascii=False, indent=1), encoding="utf-8")
    import hashlib
    R = Runner(X, y, cv, catalog, hashlib.sha256(base_path.read_bytes()).hexdigest())

    # ---------------- 1. базовые решения
    R.run("Наиболее частый класс", "baseline", DummyClassifier(strategy="most_frequent"), features="base")
    # индексы признаков после ColumnTransformer (scale=False): см. pipeline.make_preprocessor
    from src.features.pipeline import BASE_NUM, NEW_NUM
    num = BASE_NUM + NEW_NUM
    R.run("Правило: доля высоких оценок фильма ≥ 0,5", "baseline",
          RuleScorer(column=num.index("movie_bayes_mean"), threshold=0.5), scale=False,
          params={"rule": "movie_bayes_mean >= 0.5"})
    R.run("Правило: популярность фильма", "baseline",
          RuleScorer(column=num.index("movie_pop_log"), threshold=np.log1p(20)), scale=False,
          params={"rule": "число оценок фильма >= 20"})

    # ---------------- 2. линейные модели: влияние регуляризации
    Cs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    reg = []
    full = {}
    for pen in ["l2", "l1"]:
        for C in Cs:
            m = LogisticRegression(C=C, l1_ratio=1.0 if pen == "l1" else 0.0, solver="saga" if pen == "l1" else "lbfgs",
                                   max_iter=3000, random_state=SEED)
            row = R.run(f"Логистическая регрессия {pen.upper()}, C = {C}", "linear", m,
                        params={"penalty": pen, "C": C})
            p = make_pipeline(m, catalog).fit(X, y)
            nz = int((np.abs(p[-1].coef_) > 1e-6).sum())
            reg.append({"penalty": pen, "C": C, "roc_auc": row["roc_auc_mean"], "nonzero": nz,
                        "n_features": p[-1].coef_.size})
            full[(pen, C)] = p
    reg = pd.DataFrame(reg)
    reg.to_csv(ROOT / "reports" / "regularization.csv", index=False, lineterminator="\n")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for pen, c in [("l2", BLUE), ("l1", ORANGE)]:
        d = reg[reg.penalty == pen]
        ax[0].plot(d.C, d.roc_auc, "o-", color=c, label=pen.upper())
        ax[1].plot(d.C, d.nonzero, "o-", color=c, label=pen.upper())
    for a, t in zip(ax, ["ROC-AUC (кросс-валидация)", "Число ненулевых коэффициентов"]):
        a.set_xscale("log"); a.set_xlabel("C (меньше — сильнее регуляризация)"); a.set_title(t); a.legend()
    ax[0].set_ylabel("ROC-AUC"); ax[1].set_ylabel("коэффициентов")
    fig.tight_layout(); fig.savefig(FIG / "fig_3_1_regularization.png", dpi=160); plt.close(fig)

    p = full[("l2", 1)]
    coef = pd.Series(p[-1].coef_.ravel(), index=p.named_steps["prep"].get_feature_names_out()).sort_values(key=np.abs, ascending=False)
    coef.round(4).to_csv(ROOT / "reports" / "logreg_l2_coefficients.csv", header=["coef"], lineterminator="\n")
    fig, ax = plt.subplots(figsize=(8, 5))
    top = coef.head(15)[::-1]
    ax.barh(top.index, top.values, color=[BLUE if v > 0 else ORANGE for v in top.values])
    ax.set_title("ЛР L2 (C = 1): 15 наибольших по модулю коэффициентов")
    ax.set_xlabel("коэффициент (признаки стандартизованы)")
    fig.tight_layout(); fig.savefig(FIG / "fig_3_2_coefficients.png", dpi=160); plt.close(fig)

    # ---------------- 3. методы классификации
    R.run("k ближайших соседей (k = 50)", "knn", KNeighborsClassifier(n_neighbors=50, n_jobs=-1), params={"k": 50})
    R.run("Наивный байесовский классификатор", "nb", GaussianNB())
    depth_rows = []
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, None]:
        row = R.run(f"Дерево решений (глубина {d if d else 'без ограничения'})", "tree",
                    DecisionTreeClassifier(max_depth=d, random_state=SEED), scale=False,
                    params={"max_depth": d}, train_score=True)
        depth_rows.append({"depth": d if d else 30, "label": str(d) if d else "∞",
                           "train": row["train_roc_auc_mean"], "valid": row["roc_auc_mean"]})
    dr = pd.DataFrame(depth_rows)
    dr.to_csv(ROOT / "reports" / "tree_depth.csv", index=False, lineterminator="\n")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(dr)), dr.train, "o-", color=BLUE, label="обучающие блоки")
    ax.plot(range(len(dr)), dr.valid, "o-", color=ORANGE, label="проверочные блоки")
    ax.set_xticks(range(len(dr))); ax.set_xticklabels(dr.label)
    ax.set_xlabel("Максимальная глубина дерева"); ax.set_ylabel("ROC-AUC"); ax.legend()
    ax.set_title("Дерево решений: качество в зависимости от глубины")
    fig.tight_layout(); fig.savefig(FIG / "fig_3_3_tree_depth.png", dpi=160); plt.close(fig)

    # ---------------- 4. ансамбли
    R.run("Случайный лес (300 деревьев)", "rf",
          RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED), scale=False,
          params={"n_estimators": 300, "min_samples_leaf": 5})
    R.run("Градиентный бустинг (по умолчанию)", "hgb", HistGradientBoostingClassifier(random_state=SEED), scale=False)

    # ---------------- 5. польза новых признаков
    for name, m, sc in [("Логистическая регрессия L2, C = 1", LogisticRegression(C=1, max_iter=3000), True),
                        ("Градиентный бустинг (по умолчанию)", HistGradientBoostingClassifier(random_state=SEED), False)]:
        R.run(f"{name} — только признаки каталога", m.__class__.__name__, m, features="base", scale=sc)

    # ---------------- 6. подбор гиперпараметров
    from itertools import product
    grid_hgb = []
    for lr, depth, it, leaf in product([0.03, 0.05, 0.1], [3, 5, None], [150, 300], [20, 50]):
        m = HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth, max_iter=it, min_samples_leaf=leaf,
                                           random_state=SEED)
        row = R.run(f"ГБ подбор lr={lr}, depth={depth}, iter={it}, leaf={leaf}", "hgb_tuning", m, scale=False,
                    params={"learning_rate": lr, "max_depth": depth, "max_iter": it, "min_samples_leaf": leaf})
        grid_hgb.append(row)
    grid_lr = []
    for C, cw in product([0.03, 0.1, 0.3, 1, 3], [None, "balanced"]):
        m = LogisticRegression(C=C, class_weight=cw, max_iter=3000)
        row = R.run(f"ЛР подбор C={C}, class_weight={cw}", "lr_tuning", m, params={"C": C, "class_weight": cw})
        grid_lr.append(row)
    R.save()

    # ---------------- 7. сводка и выбор модели
    ex = pd.DataFrame(R.rows)
    best_hgb = max(grid_hgb, key=lambda r: r["roc_auc_mean"])
    best_lr = max(grid_lr, key=lambda r: r["roc_auc_mean"])
    rf_row = ex[ex.family == "rf"].iloc[0]
    knn_row = ex[ex.family == "knn"].iloc[0]
    nb_row = ex[ex.family == "nb"].iloc[0]
    tree_best = ex[ex.family == "tree"].sort_values("roc_auc_mean").iloc[-1]
    hgb_def = ex[(ex.family == "hgb") & (ex.features == "extended")].iloc[0]
    candidates = {
        "logreg": {"row": best_lr, "complexity": 1, "algorithm": "LogisticRegression"},
        "hgb": {"row": best_hgb, "complexity": 3, "algorithm": "HistGradientBoostingClassifier"},
        "rf": {"row": rf_row.to_dict(), "complexity": 4, "algorithm": "RandomForestClassifier"},
    }
    top = max(c["row"]["roc_auc_mean"] for c in candidates.values())
    near = {k: c for k, c in candidates.items() if top - c["row"]["roc_auc_mean"] < 0.005}
    chosen = min(near, key=lambda k: candidates[k]["complexity"])
    sel = {"rule": "максимальный средний ROC-AUC; при разнице < 0,005 — более простая и быстрая модель",
           "chosen": chosen, "algorithm": candidates[chosen]["algorithm"],
           "params": json.loads(candidates[chosen]["row"]["params"]),
           "cv_roc_auc_mean": candidates[chosen]["row"]["roc_auc_mean"],
           "cv_roc_auc_std": candidates[chosen]["row"]["roc_auc_std"],
           "cv_pr_auc_mean": candidates[chosen]["row"]["pr_auc_mean"],
           "candidates": {k: {"roc_auc": c["row"]["roc_auc_mean"], "params": c["row"]["params"]} for k, c in candidates.items()}}
    (ROOT / "reports" / "model_selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = [ex[ex.experiment == "Наиболее частый класс"].iloc[0],
               ex[ex.experiment.str.startswith("Правило: доля")].iloc[0],
               ex[ex.experiment.str.startswith("Правило: попул")].iloc[0],
               ex[ex.experiment == "Логистическая регрессия L2, C = 1"].iloc[0],
               ex[ex.experiment == "Логистическая регрессия L1, C = 0.1"].iloc[0],
               knn_row, nb_row, tree_best,
               ex[ex.experiment == "Дерево решений (глубина без ограничения)"].iloc[0],
               rf_row, hgb_def, pd.Series(best_hgb), pd.Series(best_lr)]
    sm = pd.DataFrame(summary)[["experiment", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "f1_mean",
                                "recall_mean", "fit_time_s"]]
    sm.to_csv(ROOT / "reports" / "experiments_summary.csv", index=False, lineterminator="\n")
    print(sm.to_string(index=False))
    print("Выбор:", sel)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    s2 = sm.iloc[::-1]
    ax.barh(range(len(s2)), s2.roc_auc_mean, xerr=s2.roc_auc_std, color=BLUE, capsize=3)
    ax.set_yticks(range(len(s2)))
    def short(e):
        if e.startswith("ГБ подбор"):
            return "ГБ после подбора (" + e.split("подбор ")[1].replace(", leaf", ",\nleaf") + ")"
        if e.startswith("ЛР подбор"):
            return "ЛР после подбора (" + e.split("подбор ")[1].split(",")[0] + ")"
        return e.replace("Логистическая регрессия", "ЛР").replace("Градиентный бустинг", "ГБ")
    ax.set_yticklabels([short(e) for e in s2.experiment], fontsize=8)
    ax.set_xlim(0.45, max(0.8, s2.roc_auc_mean.max() + 0.03)); ax.axvline(0.5, color="grey", lw=1, ls="--")
    ax.set_xlabel("ROC-AUC (среднее ± ст. откл. по 5 блокам TimeSeriesSplit)")
    ax.set_title("Сравнение моделей по кросс-валидации")
    fig.tight_layout(); fig.savefig(FIG / "fig_3_4_models.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
