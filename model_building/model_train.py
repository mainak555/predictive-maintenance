
from performance_evaluators import extract_model_structure, classify_model_complexity, measure_inference_latency
from util2 import IQRCapper

from sklearn.compose import make_column_transformer, make_column_selector
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    recall_score,
    f1_score, 
)

import pandas as pd
import numpy as np
import mlflow
import time
import copy
import os

def evaluate(
    PIPELINE_RUN_ID: str,
    pipeline_job: str,
    MODEL_CONFIG: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,   
    tags: dict = {},
) -> dict:
    """
    PIPELINE_RUN_ID: string => github run id
    pipeline_job: string => job type
    returns: dict => {
        model_name: {
            estimator: estimator => model,
            mlflow_run_id: str => mlflow run id
        }
    }
    """

    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not MLFLOW_EXPERIMENT_NAME:
        raise RuntimeError("MLFLOW_EXPERIMENT_NAME not found")

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if not MLFLOW_TRACKING_URI:
        raise RuntimeError("MLFLOW_TRACKING_URI not found")

    # column processors
    num_pipeline = Pipeline([
        ("outlier_cap", IQRCapper(factor=1.5)),
        ("imputer", SimpleImputer(strategy="median"))
    ])

    col_processor = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        remainder="drop"
    )
    col_processor.set_output(transform="pandas")

    ## experiment ##
    GIT_SHA = os.getenv("GITHUB_SHA")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    output: dict = {}
    for model_name, cfg in MODEL_CONFIG.items():
        run_name = f"{model_name}_{PIPELINE_RUN_ID}"
        print(f"mlFlow Run: {run_name}")

        with mlflow.start_run(run_name=run_name) as run:     
            tags["model_name"] = model_name
            tags["git_commit_sha"] = GIT_SHA
            tags["pipeline_job"] = pipeline_job
            tags["pipeline_run_id"] = PIPELINE_RUN_ID
            tags["run_at"] = f"{time.strftime('%Y-%m-%dT%H:%M')}"
            mlflow.set_tags(tags)

            pipeline = make_pipeline(col_processor, cfg["estimator"])
            search = RandomizedSearchCV(
                param_distributions=cfg["grid_params"],
                estimator=pipeline,
                random_state=42,
                scoring="f1",
                n_iter=20,
                n_jobs=-1,
                verbose=0,
                cv=8,
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            # calibrate
            # best_model = CalibratedClassifierCV(
            #     estimator=search.best_estimator_,
            #     method="isotonic", cv=5
            # )
            # best_model.fit(X_train, y_train)

            # returns
            output[model_name] = {
                "estimator": copy.deepcopy(best_model),
                "mlflow_run_id": run.info.run_id
            }

            # predictions
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] # type: ignore

            # threshold optimization
            TARGET_RECALL = float(os.getenv("TARGET_RECALL", 0.9))
            DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", 0.5))

            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            valid_idxs = np.where(recall >= TARGET_RECALL)[0]

            if len(valid_idxs) > 0:
                best_idx = valid_idxs[np.argmax(precision[valid_idxs])]
                best_threshold = thresholds[best_idx]
            else:
                best_threshold = DECISION_THRESHOLD

            y_pred = (y_pred_proba >= best_threshold).astype(int)

            # metrics
            test_aps = average_precision_score(y_test, y_pred_proba)
            test_auc = roc_auc_score(y_test, y_pred_proba)
            test_precision = precision_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)

            # log metrics
            mlflow.log_metric("decision_threshold", float(best_threshold))
            mlflow.log_metric("test_avg_precision_score", float(test_aps))
            mlflow.log_metric("test_precision", float(test_precision))
            mlflow.log_metric("cv_best_score", search.best_score_)
            mlflow.log_metric("test_recall", float(test_recall))
            mlflow.log_metric("test_roc_auc", float(test_auc))
            mlflow.log_metric("test_f1", float(test_f1))

            # log hyper-parameters
            mlflow.log_params(search.best_params_)

            # feature importance        
            perm = permutation_importance(
                best_model,
                X_test,
                y_test,
                scoring="f1",
                random_state=42,
                n_repeats=5,
                n_jobs=-1
            )

            df_feature_importance = pd.DataFrame({
                "importance": perm.importances_mean, # type: ignore
                "feature": X_test.columns,
            }).sort_values("importance", ascending=False).reset_index(drop=True)

            df_feature_importance["importance_norm"] = df_feature_importance["importance"] / df_feature_importance["importance"].sum()
            df_feature_importance["cum_importance"] = df_feature_importance["importance_norm"].cumsum()

            # top k features
            MAX_FEATURES = 6
            COVERAGE_THRESHOLD = 0.95

            # first index where cumulative importance >= threshold
            cutoff_idx = df_feature_importance[df_feature_importance["cum_importance"] >= COVERAGE_THRESHOLD].index[0]
            top_k_df = df_feature_importance.loc[:cutoff_idx].head(MAX_FEATURES)
            top_k_features_artifact = {
                "top_k": len(top_k_df),
                "coverage_threshold": COVERAGE_THRESHOLD,
                "coverage_pct": round(top_k_df["importance_norm"].sum() * 100, 2),
                "features": [
                    {
                        "name": row["feature"],
                        "importance": round(row["importance_norm"], 6),
                        "cumulative_importance": round(row["cum_importance"], 6)
                    }
                    for _, row in top_k_df.iterrows()
                ]
            }

            mlflow.log_dict(
                top_k_features_artifact,
                artifact_file="feature_analysis/top_k_features.json"
            )
            mlflow.set_tag("feature_importance_method", "permutation_importance")
            mlflow.log_metric("top_k_features_count", top_k_features_artifact["top_k"])
            mlflow.log_metric("feature_importance_coverage_pct", top_k_features_artifact["coverage_pct"])

            # feature details
            raw_features = X_train.columns.to_list()
            raw_features_artifact = {
                "total_raw_features": len(raw_features),
                "features": raw_features
            }

            mlflow.log_dict(
                raw_features_artifact,
                artifact_file="feature_schema/raw_features.json"
            )
            mlflow.set_tag("total_raw_features", len(raw_features))

            # model performance
            inference_latency_ms = measure_inference_latency(best_model, X_test)
            mlflow.log_metric("inference_latency_ms", inference_latency_ms)

            # model complexity
            # structure = extract_model_structure(best_model)
            # model_complexity = classify_model_complexity(structure)
            # mlflow.log_param("model_complexity", model_complexity)            
        #eo: with
    #eo: for
    return output
