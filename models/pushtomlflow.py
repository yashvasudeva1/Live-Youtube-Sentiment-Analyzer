"""
Run this LOCALLY (not Colab) to push your saved model files into MLflow as
artifacts on an existing (or new) run.

Steps:
  1. pip install mlflow   (if not already installed)
  2. Put all your saved files in one folder, or list their exact paths below.
  3. If you have the mlflow.db you downloaded earlier from Colab, place it in
     the same folder as this script (or point TRACKING_URI at wherever it is).
  4. Run: python push_artifacts_to_mlflow_local.py
"""

import os
import mlflow

# ---------------------------------------------------------------------------
# 1. Point MLflow at your local tracking store
#    - If you downloaded mlflow.db from Colab, put it next to this script
#      (or change the path below to wherever it actually is).
#    - If you never had one, this will create a fresh mlflow.db here.
# ---------------------------------------------------------------------------
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "youtube-comment-sentiment"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------------------------------------------------------
# 2. List the exact local file paths you want to push as artifacts.
#    Edit this list to match where your files actually are on disk.
# ---------------------------------------------------------------------------
FILES_TO_PUSH = [
    "lgbm_sentiment_model.joblib",
    "tfidf_word_vectorizer.joblib",
    "tfidf_char_vectorizer.joblib",
    "label_encoder.joblib",
    "train_vs_test_metrics.csv",
    "confusion_matrix.png",
    "feature_importance.csv",
]

existing_files = [f for f in FILES_TO_PUSH if os.path.exists(f)]
missing_files = [f for f in FILES_TO_PUSH if not os.path.exists(f)]

print(f"Found {len(existing_files)} file(s): {existing_files}")
if missing_files:
    print(f"NOT found (edit the path or check the folder): {missing_files}")

if not existing_files:
    raise SystemExit("No files found. Fix the paths in FILES_TO_PUSH and rerun.")

# ---------------------------------------------------------------------------
# 3. Show existing runs so you can choose where these artifacts go
# ---------------------------------------------------------------------------
runs_df = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])

if len(runs_df) == 0:
    print("\nNo existing runs found in this experiment.")
else:
    print("\nExisting runs:")
    print(runs_df[["run_id", "start_time", "tags.mlflow.runName"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 4. SET THIS: paste a run_id from the printed table above to attach
#    artifacts to that exact run, or leave as None to create a new run.
# ---------------------------------------------------------------------------
TARGET_RUN_ID = None  # <-- e.g. "a1b2c3d4e5f6..."

# ---------------------------------------------------------------------------
# 5. Push the files into MLflow as artifacts
# ---------------------------------------------------------------------------
if TARGET_RUN_ID:
    with mlflow.start_run(run_id=TARGET_RUN_ID):
        for fname in existing_files:
            mlflow.log_artifact(fname)
        run_id = TARGET_RUN_ID
        print(f"\nPushed {len(existing_files)} artifact(s) to existing run: {run_id}")
else:
    with mlflow.start_run(run_name="artifacts_recovered"):
        for fname in existing_files:
            mlflow.log_artifact(fname)
        run_id = mlflow.active_run().info.run_id
        print(f"\nPushed {len(existing_files)} artifact(s) to new run: {run_id}")

# ---------------------------------------------------------------------------
# 6. Verify — list what actually landed in that run's artifact store
# ---------------------------------------------------------------------------
client = mlflow.tracking.MlflowClient()
artifacts = client.list_artifacts(run_id)
print(f"\nArtifacts now stored in run {run_id}:")
for a in artifacts:
    print(f"  {a.path}/ (dir)" if a.is_dir else f"  {a.path}  ({a.file_size} bytes)")

print(f"\nTo browse this in the MLflow UI, run:")
print(f"  mlflow ui --backend-store-uri {TRACKING_URI}")
print("Then open http://127.0.0.1:5000 in your browser.")