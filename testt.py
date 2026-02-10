import mlflow

mlflow.set_tracking_uri("https://dagshub.com/Nikhiy/imbd_sentiment_analysis.mlflow")

client = mlflow.tracking.MlflowClient()

model_name = "my_model"

# list versions
versions = client.search_model_versions(f"name='{model_name}'")

for v in versions:
    print("Version:", v.version, "Stage:", v.current_stage)
