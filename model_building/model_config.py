
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
)

from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

MODEL_CONFIG = {
    "BaggingClassifier": {
        "estimator": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            random_state=42
        ),
        "grid_params": {
            "baggingclassifier__bootstrap": [True, False],
            "baggingclassifier__n_estimators": randint(50, 200),
            "baggingclassifier__max_samples": uniform(0.5, 0.5),
            "baggingclassifier__max_features": uniform(0.5, 0.5),
            "baggingclassifier__estimator__max_depth": randint(3, 20),
            "baggingclassifier__estimator__max_leaf_nodes": randint(5, 25),
            "baggingclassifier__estimator__min_samples_leaf": randint(1, 20),
            "baggingclassifier__estimator__min_samples_split": randint(2, 20),
            "baggingclassifier__estimator__class_weight": [None, "balanced"],
            "baggingclassifier__estimator__criterion": ["gini", "entropy"],
        }
    },
    "AdaBoostClassifier": {
        "estimator": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=42), random_state=42
        ),
        "grid_params": {
            "adaboostclassifier__n_estimators": randint(50, 200),
            "adaboostclassifier__learning_rate": uniform(0.01, 1.0),
            "adaboostclassifier__estimator__max_depth": randint(3, 20),
            "adaboostclassifier__estimator__max_leaf_nodes": randint(5, 20),
            "adaboostclassifier__estimator__min_samples_leaf": randint(1, 20),
            "adaboostclassifier__estimator__min_samples_split": randint(2, 20),
            "adaboostclassifier__estimator__class_weight": [None, "balanced"],
            "adaboostclassifier__estimator__criterion": ["gini", "entropy"],
        }
    },
    "XGBoostClassifier": {
        "estimator": XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
        ),
        "grid_params": {
            "xgbclassifier__reg_alpha": uniform(0, 1),
            "xgbclassifier__max_depth": randint(3, 20),
            "xgbclassifier__max_leaves": randint(5, 50),
            "xgbclassifier__reg_lambda": uniform(0.5, 2),
            "xgbclassifier__subsample": uniform(0.6, 0.3),
            "xgbclassifier__n_estimators": randint(50, 200),
            "xgbclassifier__min_child_weight": randint(1, 10),
            "xgbclassifier__learning_rate": uniform(0.01, 0.5),
            "xgbclassifier__scale_pos_weight": uniform(0.5, 5),
            "xgbclassifier__colsample_bytree": uniform(0.6, 0.3),
        }
    }
}
