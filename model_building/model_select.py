
from agents.model_selector_agent.run import run_model_selector
from mlflow.tracking import MlflowClient
from datetime import datetime
from pprint import pprint
import asyncio
import mlflow
import os

## calling agent ##
async def get_selection():
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not MLFLOW_EXPERIMENT_NAME:
        raise RuntimeError("MLFLOW_EXPERIMENT_NAME not found")

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if not MLFLOW_TRACKING_URI:
        raise RuntimeError("MLFLOW_TRACKING_URI not found")

    PIPELINE_RUN_ID = os.getenv("GITHUB_RUN_ID")
    if not PIPELINE_RUN_ID:
        raise RuntimeError("PIPELINE_RUN_ID not found")

    PIPELINE_RUN_ID = '23046450942'
    ## agent payload ##
    agent_payload = {
        "objective": {
            "business_goal": "Identify engines requiring maintenance (Engine Condition = 1) to reduce unplanned breakdowns and operational downtime",
            "decision_type": "binary_classification",
            "primary_metric": "f1",
            "constraints": {
                "min_f1": 0.77,
                "min_recall": 0.9,
                "min_precision": 0.66
            }
        },
        "pipeline_context": {
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "pipeline_run_id": PIPELINE_RUN_ID,
        },
        "candidates": []
    }

    # get mlflow runs
    client = MlflowClient()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], # type: ignore
        filter_string=f"tags.pipeline_run_id = '{PIPELINE_RUN_ID}'"
    )
    if not runs:
        raise Exception(f"No MLflow runs found for pipeline_run_id={PIPELINE_RUN_ID}")

    for run in runs:
        data = run.data
        tags = data.tags
        agent_payload['candidates'].append({
            "model_name": tags.get("model_name"),
            "mlflow_run_id": run.info.run_id,
            "metrics": {
                "test_f1": data.metrics.get("test_f1"),
                "test_roc_auc": data.metrics.get("test_roc_auc"),
                "test_recall": data.metrics.get("test_recall"),
                "test_precision": data.metrics.get("test_precision"),
            },
            "complexity": {
                "model_complexity": data.params.get("model_complexity"),
                "inference_latency_ms": data.metrics.get("inference_latency_ms"),
            },
            "feature_profile": {
                "top_k_features_count": data.metrics.get("top_k_features_count"),
                "feature_importance_method": tags.get("feature_importance_method"),
                "coverage_pct": data.metrics.get("feature_importance_coverage_pct"),
            },
        })

    try:
        decision = await run_model_selector(agent_payload)
        pprint(decision)

        ## tagging selected model ##
        client.set_tag(decision["selected_mlflow_run_id"], "selected_for_deployment", "true")
        client.set_tag(decision["selected_mlflow_run_id"], "selection_timestamp", datetime.now().isoformat())
        client.set_tag(decision["selected_mlflow_run_id"], "selection_justification", decision["justification"])
    except Exception as e:
        print(f"Agent failed: {e}")

asyncio.run(get_selection())
