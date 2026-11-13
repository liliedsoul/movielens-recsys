"""Предметный разведочный анализ исходной части данных (ПР № 2).

Анализ выполняется только на data/base/base_60.csv. Скрипт строит графики для отчёта
(reports/figures_pr2/) и сохраняет числовые результаты, проверку качества и проверку
гипотез в reports/eda_pr2_summary.json.

Запуск из корня репозитория:
    uv run python -m src.data.eda_pr2
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures_pr2"
BLUE = "#1f77b4"


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    params = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))
    thr = params["relevance_threshold"]
    np.random.seed(params["seed"])
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    raw = ROOT / "data" / "raw" / params["batches"].get("dataset", params["dataset"])
    r = pd.read_csv(ROOT / params["batches"]["base_file"])
    movies = pd.read_csv(raw / "movies.csv")
    links = pd.read_csv(raw / "links.csv")
    tags = pd.read_csv(raw / "tags.csv")
    tags = tags[tags["timestamp"] <= r["timestamp"].max()]          # теги «из будущего» не используются

    r["datetime"] = pd.to_datetime(r["timestamp"], unit="s")
    movies["year"] = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    movies["n_genres"] = np.where(movies["genres"] == "(no genres listed)", 0, movies["genres"].str.count(r"\|") + 1)
    S: dict = {}

    # ---------- структура и цель ----------
    S["structure"] = {
        "rows": len(r), "users": int(r.userId.nunique()), "movies_rated": int(r.movieId.nunique()),
        "catalog": len(movies), "tags_before_boundary": len(tags),
        "period": [str(r.datetime.min()), str(r.datetime.max())],
        "density_pct": round(100 * len(r) / (r.userId.nunique() * r.movieId.nunique()), 2),
    }
    r["relevant"] = (r.rating >= thr).astype(int)
    S["target"] = {
        "mean": round(r.rating.mean(), 2), "median": float(r.rating.median()),
        "share_integer_pct": round(100 * (r.rating % 1 == 0).mean(), 1),
        "share_relevant_pct": round(100 * r.relevant.mean(), 1),
    }
    user_mean = r.groupby("userId").rating.mean()
    S["target"]["user_mean_range"] = [round(user_mean.min(), 2), round(user_mean.max(), 2)]
    S["target"]["user_mean_iqr"] = [round(user_mean.quantile(.25), 2), round(user_mean.quantile(.75), 2)]

    fig, ax = plt.subplots(figsize=(8, 4))
    c = r.rating.value_counts().sort_index()
    ax.bar(c.index.astype(str), c.values, color=BLUE)
    ax.set_title("Распределение оценок в исходной части (60 %)")
    ax.set_xlabel("Оценка rating"); ax.set_ylabel("Число оценок")
    savefig(fig, "fig_2_1_ratings")

    # ---------- качество ----------
    merged = r.merge(movies[["movieId", "year"]], on="movieId")
    q = {
        "Пропуски в userId, movieId, rating, timestamp": int(r[["userId", "movieId", "rating", "timestamp"]].isna().sum().sum()),
        "Полные дубликаты строк": int(r.duplicated().sum()),
        "Повторные оценки пары userId–movieId": int(r.duplicated(["userId", "movieId"]).sum()),
        "Оценки вне 0,5–5,0 или не кратные 0,5": int((~r.rating.between(0.5, 5) | ((r.rating * 2) % 1 != 0)).sum()),
        "movieId отсутствует в каталоге": int((~r.movieId.isin(movies.movieId)).sum()),
        "Оценка раньше года выпуска фильма": int((merged.datetime.dt.year < merged.year - 1).sum()),
        "Фильмы каталога без жанров": int((movies.n_genres == 0).sum()),
        "Фильмы без года в названии": int(movies.year.isna().sum()),
        "Повторяющиеся названия фильмов": int(movies.title.duplicated(keep=False).sum()),
        "Пропуски tmdbId": int(links.tmdbId.isna().sum()),
    }
    dup = movies[movies.title.duplicated(keep=False)].merge(links, on="movieId")
    q["Пары одноимённых фильмов с разными imdbId"] = int(dup.groupby("title").imdbId.nunique().gt(1).sum())
    S["quality"] = q

    user_cnt = r.groupby("userId").size()
    movie_cnt = r.groupby("movieId").size()
    q1, q3 = np.percentile(user_cnt, [25, 75]); hi = q3 + 1.5 * (q3 - q1)
    per_min = r.assign(m=r.timestamp // 60).groupby(["userId", "m"]).size()
    S["activity"] = {
        "user_median": float(user_cnt.median()), "user_max": int(user_cnt.max()), "iqr_upper": round(hi),
        "users_above_iqr": int((user_cnt > hi).sum()),
        "share_ratings_above_iqr_pct": round(100 * user_cnt[user_cnt > hi].sum() / len(r), 1),
        "max_per_minute": int(per_min.max()),
        "movie_median": float(movie_cnt.median()), "movie_max": int(movie_cnt.max()),
        "movies_single_rating_pct": round(100 * (movie_cnt == 1).mean(), 1),
    }
    srt = np.sort(movie_cnt.to_numpy())[::-1]
    S["activity"]["top20_movies_share_pct"] = round(100 * srt[: int(len(srt) * 0.2)].sum() / srt.sum(), 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(user_cnt, bins=50, color=BLUE, log=True)
    ax.axvline(hi, color="red", ls="--", lw=1, label=f"граница IQR = {hi:.0f}")
    ax.set_title("Число оценок на пользователя (логарифмическая шкала)")
    ax.set_xlabel("Оценок у пользователя"); ax.set_ylabel("Пользователей"); ax.legend()
    savefig(fig, "fig_2_2_user_activity")

    # ---------- признаки пары и связь с целью ----------
    r = merged.merge(movies[["movieId", "n_genres", "genres"]], on="movieId")
    r["movie_age"] = r.datetime.dt.year - r.year
    r["user_n"] = r.userId.map(user_cnt); r["movie_n"] = r.movieId.map(movie_cnt)
    r = r.sort_values(["userId", "timestamp"])
    r["order_pct"] = r.groupby("userId").cumcount() / (r.user_n - 1).clip(lower=1)
    feat = r[["user_n", "movie_n", "movie_age", "year", "n_genres", "order_pct", "rating"]].copy()
    feat[["user_n", "movie_n"]] = np.log1p(feat[["user_n", "movie_n"]])
    S["spearman"] = feat.corr(method="spearman").round(2).to_dict()

    g = r.assign(genre=r.genres.str.split("|")).explode("genre")
    g = g[g.genre != "(no genres listed)"]
    gt = g.groupby("genre").agg(n=("rating", "size"), mean=("rating", "mean"), rel=("relevant", "mean")).sort_values("rel")
    S["genres"] = gt.round(3).to_dict(orient="index")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(gt.index, gt.rel * 100, color=BLUE)
    ax.axvline(r.relevant.mean() * 100, color="grey", ls="--", lw=1, label="в среднем")
    ax.set_title("Доля релевантных оценок (≥ 4) по жанрам")
    ax.set_xlabel("%"); ax.legend(loc="lower right")
    savefig(fig, "fig_2_4_genres")

    # ---------- утечка: среднее по всем данным против среднего по прошлому ----------
    rs = r.sort_values("timestamp")
    cut = rs.timestamp.quantile(0.8)
    past, fut = rs[rs.timestamp < cut], rs[rs.timestamp >= cut].copy()
    glob = past.rating.mean()
    hm = past.groupby("movieId").rating.agg(["mean", "count"])
    hm = (hm["mean"] * hm["count"] + glob * 5) / (hm["count"] + 5)
    fut["leaky"] = fut.movieId.map(rs.groupby("movieId").rating.mean())
    fut["honest"] = fut.movieId.map(hm).fillna(glob)
    S["leakage_auc"] = {"leaky": round(roc_auc_score(fut.relevant, fut.leaky), 3),
                        "honest": round(roc_auc_score(fut.relevant, fut.honest), 3)}

    # ---------- гипотезы ----------
    H = []
    m = r.groupby("movieId").agg(n=("rating", "size"), mean=("rating", "mean"))
    m5 = m[m.n >= 5]
    rho, p = stats.spearmanr(m5.n, m5["mean"])
    H.append({"id": "H1", "rho": round(rho, 3), "p": p, "n": len(m5)})
    u = r.groupby("userId").agg(n=("rating", "size"), mean=("rating", "mean"))
    rho, p = stats.spearmanr(u.n, u["mean"])
    H.append({"id": "H2", "rho": round(rho, 3), "p": p, "n": len(u)})
    old, new = r.loc[r.movie_age > 20, "rating"], r.loc[r.movie_age <= 2, "rating"]
    U, p = stats.mannwhitneyu(old, new, alternative="greater")
    H.append({"id": "H3", "mean_old": round(old.mean(), 2), "mean_new": round(new.mean(), 2), "p": p,
              "n_old": len(old), "n_new": len(new)})
    Hs, p = stats.kruskal(*[s.values for _, s in g.groupby("genre")["rating"]])
    H.append({"id": "H4", "H": round(Hs, 1), "p": p, "groups": int(g.genre.nunique())})
    first = r[r.order_pct <= 0.2].groupby("userId").rating.mean()
    last = r[r.order_pct >= 0.8].groupby("userId").rating.mean()
    both = pd.concat([first, last], axis=1, keys=["f", "l"]).dropna()
    W, p = stats.wilcoxon(both.f, both.l, alternative="greater")
    H.append({"id": "H5", "mean_first": round(both.f.mean(), 2), "mean_last": round(both.l.mean(), 2), "p": p, "n": len(both)})
    S["hypotheses"] = H

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(m5.n, m5["mean"], s=6, alpha=0.35, color=BLUE)
    bins = pd.qcut(m5.n, 8, duplicates="drop")
    gm = m5.groupby(bins, observed=True).agg(x=("n", "median"), y=("mean", "mean"))
    axes[0].plot(gm.x, gm.y, "r-o", ms=4, label="среднее по группам")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Число оценок фильма (лог.)"); axes[0].set_ylabel("Средняя оценка")
    axes[0].set_title(f"H1: популярность и оценка (ρ = {H[0]['rho']:.2f})"); axes[0].legend()
    age_bins = pd.cut(r.movie_age, [-2, 2, 10, 20, 40, 120], labels=["≤ 2", "3–10", "11–20", "21–40", "> 40"])
    ag = r.groupby(age_bins, observed=True).rating.agg(["mean", "sem"])
    axes[1].bar(ag.index.astype(str), ag["mean"], yerr=1.96 * ag["sem"], color=BLUE, capsize=3)
    axes[1].set_ylim(3, 4); axes[1].set_xlabel("Возраст фильма в момент оценки, лет"); axes[1].set_ylabel("Средняя оценка")
    axes[1].set_title("H3: возраст фильма и оценка")
    savefig(fig, "fig_2_3_h1_h3")

    # ---------- порции: дрейф и холодный старт ----------
    known_u, known_m = set(r.userId), set(r.movieId)
    rows = []
    for f in sorted((ROOT / params["batches"]["incoming_dir"]).glob("batch_*.csv")):
        b = pd.read_csv(f)
        rows.append({"batch": f.stem, "rows": len(b), "users": int(b.userId.nunique()),
                     "new_users_pct": round(100 * (~pd.Series(b.userId.unique()).isin(known_u)).mean(), 1),
                     "new_movie_ratings_pct": round(100 * (~b.movieId.isin(known_m)).mean(), 1),
                     "mean": round(b.rating.mean(), 2), "relevant_pct": round(100 * (b.rating >= thr).mean(), 1),
                     "users_gt_500": int((b.groupby("userId").size() > 500).sum()),
                     "max_per_minute": int(b.assign(m=b.timestamp // 60).groupby(["userId", "m"]).size().max())})
        known_u |= set(b.userId); known_m |= set(b.movieId)
    S["batches"] = rows

    out = ROOT / "reports" / "eda_pr2_summary.json"
    out.write_text(json.dumps(S, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: S[k] for k in ["structure", "target", "activity", "leakage_auc"]}, ensure_ascii=False, indent=1))
    for h in H:
        print(h)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Графики: {FIG.relative_to(ROOT).as_posix()}, сводка: {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
