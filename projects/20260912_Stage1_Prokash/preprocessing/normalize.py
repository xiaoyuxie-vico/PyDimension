"""
Data normalization utility.

Supports three normalization methods:
    "standard" : zero mean, unit std per column
    "minmax"   : scale each column to [0, 1]
    "robust"   : subtract median, divide by IQR per column
"""

import numpy as np


class _StandardScaler:
    def fit(self, arr):
        self.mean_ = arr.mean(axis=0)
        self.std_  = arr.std(axis=0)
        self.std_  = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, arr):
        return (arr - self.mean_) / self.std_

    def inverse_transform(self, arr):
        return arr * self.std_ + self.mean_


class _MinMaxScaler:
    def fit(self, arr):
        self.min_ = arr.min(axis=0)
        self.max_ = arr.max(axis=0)
        self.range_ = np.where(self.max_ == self.min_, 1.0, self.max_ - self.min_)
        return self

    def transform(self, arr):
        return (arr - self.min_) / self.range_

    def inverse_transform(self, arr):
        return arr * self.range_ + self.min_


class _RobustScaler:
    def fit(self, arr):
        self.median_ = np.median(arr, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        iqr = q75 - q25
        self.iqr_ = np.where(iqr == 0, 1.0, iqr)
        return self

    def transform(self, arr):
        return (arr - self.median_) / self.iqr_

    def inverse_transform(self, arr):
        return arr * self.iqr_ + self.median_


_SCALERS = {
    "standard": _StandardScaler,
    "minmax":   _MinMaxScaler,
    "robust":   _RobustScaler,
}


def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "standard",
) -> dict:
    """
    Normalize input features X and target y.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_inputs)
    y : np.ndarray, shape (n_samples,)
    method : str
        One of "standard", "minmax", "robust".

    Returns
    -------
    dict with keys:
        X_normalized : (n_samples, n_inputs)
        y_normalized : (n_samples,)
        scaler_X     : fitted scaler with .inverse_transform()
        scaler_y     : fitted scaler with .inverse_transform()
    """
    if method not in _SCALERS:
        raise ValueError(f"method must be one of {list(_SCALERS)}, got '{method}'")

    scaler_X = _SCALERS[method]().fit(X)
    scaler_y = _SCALERS[method]().fit(y.reshape(-1, 1))

    X_norm = scaler_X.transform(X)
    y_norm = scaler_y.transform(y.reshape(-1, 1)).ravel()

    return {
        "X_normalized": X_norm,
        "y_normalized": y_norm,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }
