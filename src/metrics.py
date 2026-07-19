import numpy as np
from scipy.stats import pearsonr


def _to_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _to_1d(y_true), _to_1d(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _to_1d(y_true), _to_1d(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _to_1d(y_true), _to_1d(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pearson_r(y_true, y_pred) -> float:
    y_true, y_pred = _to_1d(y_true), _to_1d(y_pred)
    if y_true.size < 2 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return all metrics as a dict: MAE, RMSE, R2, Pearson r."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Pearson_r": pearson_r(y_true, y_pred),
    }
