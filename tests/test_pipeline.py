"""Тесты пайплайна признаков (ПР № 3): отсутствие утечки и повторяемость."""

import numpy as np
import pandas as pd

from src.features.pipeline import GENRE_COLS, PointInTimeStats, clean


def toy():
    X = pd.DataFrame({"userId": [1, 1, 1, 2, 1], "movieId": [10, 11, 10, 10, 12], "timestamp": [1, 2, 2, 3, 4]})
    for c in GENRE_COLS:
        X[c] = 0
    X["g_Drama"] = [1, 1, 0, 1, 1]
    X["g_Comedy"] = [0, 0, 1, 0, 1]
    return X, np.array([1, 0, 1, 1, 0])


def test_point_in_time_uses_only_earlier_ratings():
    X, y = toy()
    Z = PointInTimeStats(m=1e-9).fit_transform(X, y)
    # первая оценка пользователя — холодный старт
    assert Z.loc[0, "user_n_log"] == 0 and Z.loc[0, "user_activity_group"] == "new"
    # строки 1 и 2 в один момент: видят только строку 0
    assert np.isclose(Z.loc[1, "user_n_log"], np.log1p(1)) and np.isclose(Z.loc[2, "user_n_log"], np.log1p(1))
    # фильм 10 в момент 3: две более ранние оценки, обе высокие
    assert np.isclose(Z.loc[3, "movie_pop_log"], np.log1p(2)) and np.isclose(Z.loc[3, "movie_bayes_mean"], 1.0)
    # пользователь 1 в момент 4: три более ранние оценки, доля высоких 2/3
    assert np.isclose(Z.loc[4, "user_bias"], 2 / 3 - y.mean(), atol=1e-6)


def test_transform_uses_full_training_statistics():
    X, y = toy()
    st = PointInTimeStats(m=1e-9).fit(X, y)
    new = X.iloc[[0]].assign(userId=99)            # неизвестный пользователь
    Z = st.transform(new)
    assert Z.loc[0, "user_activity_group"] == "new" and Z.loc[0, "user_bias"] == 0


def test_clean_is_deterministic_and_applies_dq():
    df = pd.DataFrame({"userId": [1, 1, 2, 3], "movieId": [5, 5, 6, 7],
                       "rating": [4.0, 3.5, 7.0, 2.0], "timestamp": [1, 2, 3, 4]})
    a, rep = clean(df)
    b, _ = clean(df.sample(frac=1, random_state=0))
    assert rep["bad_rating"] == 1                  # DQ-03
    assert len(a) == 2 and a.loc[a.userId == 1, "rating"].item() == 3.5   # DQ-04: последняя оценка пары
    pd.testing.assert_frame_equal(a, b)
