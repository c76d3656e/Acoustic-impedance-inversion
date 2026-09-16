import numpy as np

from validation import r2_score, rmse, mae, summary


def test_perfect_prediction():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(r2_score(a, a), 1.0)
    assert np.isclose(rmse(a, a), 0.0)
    assert np.isclose(mae(a, a), 0.0)


def test_summary_keys():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.1, 1.9, 3.2])
    s = summary(a, b)
    assert set(s) == {"R2", "RMSE", "MAE"}
