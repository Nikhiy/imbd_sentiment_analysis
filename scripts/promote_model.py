import os
import mlflow
def promote_model():
    #set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    dagshub_url = "https://dagshub.com"
    repo_owner = "Nikhiy"
    repo_name = "imbd_sentiment_analysis"
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    client = mlflow.MlflowClient()
    model_name="my_model"
    # Get the latest version of the model from staging
    latest_version_staging=client.get_latest_versions(model_name,stages=["Staging"])[0]

    #archive the current production model if it exists
    prod_verions=client.get_latest_versions(model_name,stages=["Production"])
    for version in prod_verions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    #promote the staging model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"model version {latest_version_staging.version} promoted to production")
if __name__=="__main__":
    promote_model()