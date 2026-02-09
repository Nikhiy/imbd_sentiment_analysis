import numpy as np
import pandas as pd
import yaml
import sys
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
from src.logger import logging
from src.exception import MyException
import mlflow
import mlflow.sklearn
import dagshub
import os

mlflow.set_tracking_uri("https://dagshub.com/Nikhiy/imbd_sentiment_analysis.mlflow")


# Detect if running in GitHub Actions
running_in_ci = os.getenv("GITHUB_ACTIONS") == "true"

if not running_in_ci:
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        dagshub.init(repo_owner="Nikhiy", repo_name="imbd_sentiment_analysis", mlflow=True)
    else:
        print("Local run without DagsHub token")
else:
    print("Skipping DagsHub init in CI")



def load_model(file_path:str):
    """loads the mel from the file path"""
    try:
        with open(file_path,'rb') as file:
            model=pickle.load(file)
        logging.info("model loaded succesfull")
        return model
    except Exception as e:
        raise MyException(e,sys)

def load_data(file_path:str)->pd.DataFrame:
    """loads the data from the file path"""
    try:
        df=pd.read_csv(file_path)
        logging.info(f"data loaded succesfully from {file_path}")
        return df
    except Exception as e:
        raise MyException(e,sys)
    
def evaluate_model(model,x_test:np.ndarray,y_test:np.ndarray)->dict:
    """evaluate the model and return the evaluation metrics"""
    try:
        y_pred=model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        accuracy=accuracy_score(y_test,y_pred)
        precission=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_proba)

        metrics_dict={
            'accuracy':accuracy,
            'precission':precission,
            'recall':recall,
            'auc':auc
        }

        logging.info("model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        raise MyException(e,sys)
    
def save_metrics(metrics:dict,file_path:str)->None:
    """saves the evaluation metics to a file path"""
    try:
        os.makedirs("reports", exist_ok=True)
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logging.info(f"metrics saved to {file_path}")
    except Exception as e:
        raise MyException(e,sys)
    
def save_model_info(run_id:str,model_path:str,file_path:str)->None:
    """saves the model id and the path to a json file"""
    try:
        model_info={'run_id':run_id,'model_path':model_path}
        with open(file_path,'w') as file:
            json.dump(model_info,file,indent=4)
        logging.info(f"model info saved to {file_path}")
    except Exception as e:
        raise MyException(e,sys)
    
def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            model=load_model('./models/model.pkl')
            test_data=load_data('./data/processed/test_bow.csv')
            x_test=test_data.iloc[:,:-1].values
            y_test=test_data.iloc[:,-1].values
            metrics=evaluate_model(model=model,x_test=x_test,y_test=y_test)
            save_metrics(metrics,'reports/metrics.json')

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            mlflow.sklearn.log_model(sk_model=model,artifact_path="model",registered_model_name="my_model")
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            raise MyException(e,sys)
        
if __name__ =='__main__':
    main()
