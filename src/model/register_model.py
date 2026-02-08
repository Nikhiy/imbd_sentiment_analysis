import json
import mlflow
from src.logger import logging
from src.exception import MyException
import os
import dagshub
import warnings
import sys
warnings.simplefilter("ignore",UserWarning)
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os

load_dotenv()


dagshub_token=os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("environment varaible CAPSTONE_TEST not set")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Nikhiy"
repo_name = "imbd_sentiment_analysis"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def load_model_info(file_path:str)->dict:
    """loads the model info from json file"""
    try:
        with open(file_path,'r') as file:
            model_info=json.load(file)
        logging.info(f"model info loaded form {file_path}")
        return model_info
    except Exception as e:
        raise MyException(e,sys)
    
def register_model(model_name:str,model_info:dict):
    """registers the model name adn info to mlflow model registory"""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version=mlflow.register_model(model_uri,model_name)

        # Transition the model to "Staging" stage
        client=mlflow.tracking.MlflowClient()
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        raise MyException(e,sys)
    

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()