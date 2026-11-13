"""Пайплайн подготовки данных и признаков (ПР № 3).

Реализует проект из ПР № 2:
* clean()            — очистка по требованиям DQ без параметров (одинаково для обучения и порций);
* add_row_features() — признаки одной строки из каталога фильмов (жанры, год, возраст фильма);
* PointInTimeStats   — история пользователя и фильма на момент прогноза: популярность, сглаженные
                       средние, вкус к жанрам; оценивается только по обучающим данным;
* make_pipeline()    — объединяет признаки, предобработку и модель в один объект Pipeline.

Все параметры, зависящие от данных (средние, частоты, медианы, кодировки), оцениваются только
в fit на обучающих данных. Поэтому при кросс-валидации они заново считаются на каждом обучающем
блоке, а на новой порции только применяются.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))
SEED = PARAMS["seed"]
THRESHOLD = PARAMS["relevance_threshold"]
RAW = ROOT / "data" / "raw" / PARAMS["batches"].get("dataset", PARAMS["dataset"])

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]
GENRE_COLS = [f"g_{g}" for g in GENRES]
ACTIVITY_BINS = [-1, 0, 19, 500, np.inf]
ACTIVITY_LABELS = ["new", "low", "active", "heavy"]          # группы по DQ-07 и плану признаков

BASE_NUM = ["year", "n_genres"]                              # исходные признаки каталога
NEW_NUM = ["movie_age", "movie_pop_log", "movie_bayes_mean", "user_n_log", "user_bias", "user_genre_affinity"]
CAT_COLS = ["user_activity_group"]


# ------------------------------------------------------------------ загрузка и очистка
def load_catalog() -> pd.DataFrame:
    movies = pd.read_csv(RAW / "movies.csv")
    movies["year"] = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    g = movies["genres"].str.get_dummies(sep="|")
    for name in GENRES:
        movies[f"g_{name}"] = g[name].astype(np.int8) if name in g else np.int8(0)
    movies["n_genres"] = movies[GENRE_COLS].sum(axis=1).astype(np.int8)
    return movies[["movieId", "year", "n_genres"] + GENRE_COLS]


def clean(df: pd.DataFrame, catalog: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Очистка без параметров: требования DQ-01–DQ-06 из ПР № 2. Возвращает данные и отчёт."""
    report = {"rows_in": len(df)}
    need = {"userId", "movieId", "rating", "timestamp"}
    missing = need - set(df.columns)
    if missing:                                                            # DQ-01
        raise ValueError(f"DQ-01: нет столбцов {sorted(missing)}")
    df = df.dropna(subset=["userId", "movieId", "timestamp"]).copy()       # DQ-02
    df = df[(df["userId"] > 0) & (df["movieId"] > 0)]
    ok = df["rating"].between(0.5, 5.0) & ((df["rating"] * 2) % 1 == 0)    # DQ-03
    report["bad_rating"] = int((~ok).sum())
    df = df[ok]
    if catalog is not None:                                                # DQ-05
        known = df["movieId"].isin(catalog["movieId"])
        report["unknown_movie"] = int((~known).sum())
        df = df[known]
    df = (df.sort_values(["timestamp", "userId", "movieId"], kind="mergesort")   # DQ-04: последняя оценка пары
            .drop_duplicates(["userId", "movieId"], keep="last"))
    df = df.astype({"userId": "int64", "movieId": "int64", "timestamp": "int64", "rating": "float64"})
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def add_row_features(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Признаки одной записи: жанры, год выпуска, возраст фильма в момент оценки (гипотезы H3, H4)."""
    out = df.drop(columns=[c for c in ["year", "n_genres", "movie_age", *GENRE_COLS] if c in df], errors="ignore")
    out = out.merge(catalog, on="movieId", how="left")
    when = pd.to_datetime(out["timestamp"], unit="s").dt.year
    out["movie_age"] = (when - out["year"]).clip(lower=0)
    out[GENRE_COLS] = out[GENRE_COLS].fillna(0).astype(np.int8)
    out["n_genres"] = out["n_genres"].fillna(0)
    return out


def target(df: pd.DataFrame) -> np.ndarray:
    return (df["rating"] >= THRESHOLD).astype(int).to_numpy()


# ------------------------------------------------------------------ трансформер с параметрами
class PointInTimeStats(BaseEstimator, TransformerMixin):
    """Признаки истории пользователя и фильма «на момент прогноза» (план признаков ПР № 2).

    * movie_pop_log       — log(1 + число оценок фильма), гипотеза H1;
    * movie_bayes_mean    — сглаженная доля высоких оценок фильма: (s + m·p) / (n + m), H1 и анализ утечки;
    * user_n_log          — log(1 + число оценок пользователя);
    * user_bias           — сглаженная доля высоких оценок пользователя минус общая доля («щедрость»);
    * user_genre_affinity — насколько пользователю нравятся жанры фильма сильнее его среднего, H4;
    * user_activity_group — new / low / active / heavy (DQ-07, холодный старт).

    fit_transform (обучение): для каждой строки статистики считаются только по оценкам с более
    ранним временем (point-in-time), поэтому собственная и будущие оценки в признак не попадают,
    а первые оценки пользователя выглядят как холодный старт — так же, как новые пользователи в порциях.
    transform (валидация, тест, порции): используются итоговые статистики по всей обучающей выборке.
    """

    def __init__(self, m: float = 5.0):
        self.m = m

    # --- агрегаты «до момента t» по ключу: число и сумма y по строго более ранним моментам
    @staticmethod
    def _prior(keys: pd.DataFrame, ts: pd.Series, y: np.ndarray, key_cols: list[str]):
        d = keys.copy()
        d["ts"] = ts.to_numpy(); d["y"] = y
        g = d.groupby(key_cols + ["ts"], sort=True)["y"].agg(["sum", "count"]).reset_index()
        g[["cs", "cn"]] = g.groupby(key_cols)[["sum", "count"]].cumsum().to_numpy()
        g["ps"] = g["cs"] - g["sum"]; g["pn"] = g["cn"] - g["count"]          # исключаем тот же момент
        out = d.merge(g[key_cols + ["ts", "ps", "pn"]], on=key_cols + ["ts"], how="left")
        return out["ps"].to_numpy(float), out["pn"].to_numpy(float)

    def _smooth(self, s, n, prior):
        return (s + self.m * prior) / (n + self.m)

    def _genre_long(self, X):
        d = X[["userId", *GENRE_COLS]].reset_index(drop=True).reset_index()
        long = d.melt(id_vars=["index", "userId"], value_vars=GENRE_COLS, var_name="g", value_name="has")
        return long[long["has"] == 1].drop(columns="has")

    def _assemble(self, X, us, un, ms, mn, aff_row):
        X = X.copy()
        p = self.p_
        X["movie_pop_log"] = np.log1p(mn)
        X["movie_bayes_mean"] = self._smooth(ms, mn, p)
        X["user_n_log"] = np.log1p(un)
        X["user_bias"] = self._smooth(us, un, p) - p
        X["user_genre_affinity"] = aff_row
        X["user_activity_group"] = pd.cut(pd.Series(un), ACTIVITY_BINS, labels=ACTIVITY_LABELS).astype(str).to_numpy()
        return X

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.p_ = float(y.mean())
        self.user_ = pd.DataFrame({"userId": X["userId"].to_numpy(), "y": y}).groupby("userId")["y"].agg(["sum", "count"])
        self.movie_ = pd.DataFrame({"movieId": X["movieId"].to_numpy(), "y": y}).groupby("movieId")["y"].agg(["sum", "count"])
        long = self._genre_long(X)
        long["y"] = y[long["index"].to_numpy()]
        self.ug_ = long.groupby(["userId", "g"])["y"].agg(["sum", "count"])
        return self

    def _affinity(self, long, gs, gn, us, un):
        um = self._smooth(us, un, self.p_)                                    # сглаженная доля пользователя
        long = long.assign(aff=self._smooth(gs, gn, um[long["index"].to_numpy()]) - um[long["index"].to_numpy()])
        return long.groupby("index")["aff"].mean()

    def transform(self, X):
        X = X.reset_index(drop=True)
        u = self.user_.reindex(X["userId"]); m = self.movie_.reindex(X["movieId"])
        us, un = u["sum"].fillna(0).to_numpy(), u["count"].fillna(0).to_numpy()
        ms, mn = m["sum"].fillna(0).to_numpy(), m["count"].fillna(0).to_numpy()
        long = self._genre_long(X)
        g = self.ug_.reindex(pd.MultiIndex.from_arrays([long["userId"], long["g"]]))
        aff = self._affinity(long, g["sum"].fillna(0).to_numpy(), g["count"].fillna(0).to_numpy(), us, un)
        return self._assemble(X, us, un, ms, mn, aff.reindex(range(len(X))).fillna(0.0).to_numpy())

    def fit_transform(self, X, y):
        self.fit(X, y)
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        ts = X["timestamp"]
        us, un = self._prior(X[["userId"]], ts, y, ["userId"])
        ms, mn = self._prior(X[["movieId"]], ts, y, ["movieId"])
        long = self._genre_long(X)
        gs, gn = self._prior(long[["userId", "g"]], ts.iloc[long["index"]].reset_index(drop=True),
                             y[long["index"].to_numpy()], ["userId", "g"])
        aff = self._affinity(long.reset_index(drop=True), gs, gn, us, un)
        return self._assemble(X, us, un, ms, mn, aff.reindex(range(len(X))).fillna(0.0).to_numpy())


# ------------------------------------------------------------------ сборка пайплайна
def make_preprocessor(features: str = "extended", scale: bool = True) -> ColumnTransformer:
    """features='base' — только признаки каталога; 'extended' — плюс новые признаки из плана ПР № 2."""
    num = BASE_NUM + (NEW_NUM if features == "extended" else [])
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler() if scale else "passthrough")])
    parts = [("num", numeric, num), ("genres", "passthrough", GENRE_COLS)]
    if features == "extended":
        parts += [("cat", OneHotEncoder(categories=[ACTIVITY_LABELS], handle_unknown="ignore",
                                        sparse_output=False), CAT_COLS)]
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def make_pipeline(model, catalog: pd.DataFrame, features: str = "extended", scale: bool = True) -> Pipeline:
    steps = [("row", FunctionTransformer(add_row_features, kw_args={"catalog": catalog}))]
    if features == "extended":
        steps += [("history", PointInTimeStats())]
    steps += [("prep", make_preprocessor(features, scale)), ("model", model)]
    return Pipeline(steps)


def feature_names() -> list[str]:
    return BASE_NUM + NEW_NUM + GENRE_COLS + [f"user_activity_group_{g}" for g in ACTIVITY_LABELS]


def temporal_split(df: pd.DataFrame, test_share: float):
    """Обучающая и отложенная тестовая части по времени (тест — последние test_share записей)."""
    df = df.sort_values(["timestamp", "userId", "movieId"], kind="mergesort").reset_index(drop=True)
    cut = df["timestamp"].iloc[int(len(df) * (1 - test_share))]
    return df[df["timestamp"] < cut].reset_index(drop=True), df[df["timestamp"] >= cut].reset_index(drop=True)
