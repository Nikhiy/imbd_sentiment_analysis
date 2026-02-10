from flask import Flask,render_template,request
from src.logger import logging
from src.exception import MyException
import sys
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
from prometheus_client import Counter,Histogram,generate_latest,CollectorRegistry,CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """lemmatize the texxt"""
    try:
        lemmatizer=WordNetLemmatizer()
        text=text.split()
        text=[lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)
    except Exception as e:
        raise MyException(e,sys)
    
def remove_stop_words(text):
    """remove stop words from the text"""
    stop_words=set(stopwords.words("english"))
    text=[word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """removes numbers from text"""
    text=''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """convert text to lower case"""
    text=text.split()
    text=[word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """converts text to lower case"""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """removes sentence smaller than 3 words"""
    for i in range(len(df)):
        if len(df.text.iloc[i].split())<3:
            df.text.iloc[i]=np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


#local code 
#----------------
# mlflow.set_tracking_uri('https://dagshub.com/Nikhiy/imbd_sentiment_analysis.mlflow')
# dagshub.init(repo_owner='Nikhiy', repo_name='imbd_sentiment_analysis', mlflow=True)

#deployment code
#----------
# Set up DagsHub credentials for MLflow tracking
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

app=Flask(__name__)
#create a c1stom registry
registry=CollectorRegistry()

#define your custom metrics using this registry
REQUEST_COUNT=Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

#model adn vectorizer setup
model_name="my_model"
from mlflow.tracking import MlflowClient

def get_latest_version(model_name):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(int(v.version) for v in versions)
    return latest

model_version = get_latest_version("my_model")
model_uri = f"models:/my_model/{model_version}"
print(f"fetching model from {model_uri}")
model=mlflow.pyfunc.load_model(model_uri)
vectorizer=pickle.load(open('models/vectorizer.pkl','rb'))

#Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method='GET',endpoint="/").inc()
    start_time=time.time()
    response=render_template("index.html",result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time()-start_time)
    return response

@app.route("/predict",methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST",endpoint="/predict").inc()
    start_time=time.time()
    text=request.form["text"]
    #clean data
    text=normalize_text(text)
    #covert to features
    features=vectorizer.transform([text])
    features_df=pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
    #predict
    result=model.predict(features_df)
    prediction=result[0]

    #increment prediction count metric
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    #measure the latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time()-start_time)
    return render_template("index.html",result=prediction)

@app.route("/metric",methods=['GET'])
def metrics():
    """expose only custom prometheus metrics"""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__=="__main__":
    #debug=true is for local run
    print("Model expects:", model._model_impl.sklearn_model.n_features_in_)
    print("Vectorizer gives:", vectorizer.transform(["test"]).shape[1])
    print(f"fetching model from {model_uri}")
    app.run(debug=True,host="0.0.0.0",port=5000)