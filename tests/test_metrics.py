import numpy as np
import pytest

from biobot.modeling.metrics import regression_metrics


def test_regression_metrics_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    result = regression_metrics(y, y)
    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_regression_metrics_constant_prediction_gives_zero_r2():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    result = regression_metrics(y_true, y_pred)
    assert result["r2"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(2 / 3)


def test_regression_metrics_returns_mae_rmse_r2_keys():
    y = np.array([0.0, 1.0])
    result = regression_metrics(y, y)
    assert set(result.keys()) == {"mae", "rmse", "r2"}
