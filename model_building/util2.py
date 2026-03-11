
from sklearn.base import BaseEstimator, TransformerMixin
from huggingface_hub import HfApi
import pandas as pd
import os

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

## train test split from HF ##
def get_train_test_split():
    HF_REPO = os.getenv("HF_REPO")
    hfApi = HfApi(token=os.getenv("HF_TOKEN"))

    # Checking train/test splits are present or not
    files = ["X_train", "y_train", "X_test", "y_test"]
    for f in files:
        path = f"hf://datasets/{HF_REPO}/{f}.csv"
        try:
            pd.read_csv(path, nrows=1)
        except FileNotFoundError:
            raise RuntimeError(f"{f}.csv missing @HF Dataset")
        except Exception as e:
            raise RuntimeError(f"Error Checking Path: {path} | Err: {e}")

    ## Load the train and test data from the Hugging Face dataset space ##
    X_train = pd.read_csv(f"hf://datasets/{HF_REPO}/X_train.csv")
    yTrain = pd.read_csv(f"hf://datasets/{HF_REPO}/y_train.csv")
    X_test = pd.read_csv(f"hf://datasets/{HF_REPO}/X_test.csv")
    yTest = pd.read_csv(f"hf://datasets/{HF_REPO}/y_test.csv")

    print(X_train.info())

    return X_train, y_train, X_test, y_test
