
from xgboost import XGBClassifier
import time
import os

## model complexity & performance ##
def extract_model_structure(model):
    """
    Extracts structural indicators from the final estimator.
    Supports:
        - sklearn tree models
        - sklearn ensemble models
        - XGBClassifier / XGBRegressor
        - linear models (coefficients)
    """

    # unwrap sklearn Pipeline
    if hasattr(model, "steps"):  
        estimator = model.steps[-1][1]
    else:
        estimator = model

    structure = {
        "model_family": estimator.__class__.__name__,
        "n_estimators": None,
        "total_tree_nodes": None,
        "n_coefficients": None,
    }

    # XGBoost Models
    if isinstance(estimator, (XGBClassifier)):
        try:
            booster = estimator.get_booster()

            structure["n_estimators"] = booster.num_boosted_rounds()

            df = booster.trees_to_dataframe()
            structure["total_tree_nodes"] = df.shape[0]

        except Exception:
            pass

        return structure

    # sklearn Tree Ensembles
    if hasattr(estimator, "estimators_"):
        try:
            structure["n_estimators"] = len(estimator.estimators_)

            structure["total_tree_nodes"] = sum(
                est.tree_.node_count
                for est in estimator.estimators_
                if hasattr(est, "tree_")
            )
        except Exception:
            pass

        return structure

    # single Decision Tree
    if hasattr(estimator, "tree_"):
        try:
            structure["total_tree_nodes"] = estimator.tree_.node_count
        except Exception:
            pass

        return structure

    # Linear Models
    if hasattr(estimator, "coef_"):
        try:
            structure["n_coefficients"] = estimator.coef_.size
        except Exception:
            pass

    return structure

def classify_model_complexity(structure):
    """
    Converts structural indicators into complexity classes
    """

    model_family = structure["model_family"]
    total_nodes = structure.get("total_tree_nodes")
    n_estimators = structure.get("n_estimators") or 1
    max_depth = structure.get("max_depth")

    # Tree-based ensembles
    if model_family in ["RandomForestClassifier", "BaggingClassifier", "AdaBoostClassifier"]:
        if n_estimators <= 100 and total_nodes and total_nodes <= 50_000:
            return "medium"
        return "high"

    # XGBoost models
    if model_family == "XGBClassifier":
        # shallow boosted trees
        if n_estimators <= 100 and (max_depth is None or max_depth <= 6):
            return "medium"

        # large boosted ensembles
        if n_estimators > 300 or (total_nodes and total_nodes > 100_000):
            return "high"

        return "medium"

    # Single trees
    if model_family == "DecisionTreeClassifier":
        if total_nodes and total_nodes <= 5_000:
            return "low"
        return "medium"

    return "unknown"

def measure_inference_latency(model, X, n_runs=100):
    X_sample = X.iloc[:1]

    # warm-up
    model.predict(X_sample)

    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(X_sample)
    end = time.perf_counter()

    return round((end - start) / n_runs * 1000, 4)
