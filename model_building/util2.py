
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.lower_ = {}
        self.upper_ = {}
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_[col] = q1 - self.factor * iqr
            self.upper_[col] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = np.clip(X[col], self.lower_[col], self.upper_[col])
        return X

    def set_output(self, transform=None):
        return self
