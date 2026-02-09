import numpy as np
import pandas as pd
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
from src.exception import MyException
import pickle

def load_params(params_path:str)->dict:
    """loads params from a yaml file"""
    try:
        with open(params_path ,'r') as file:
            params=yaml.safe_load(file)
        logging.info(f"parameters loaded succesfully from {params_path}")
        return params
    except Exception as e:
        raise MyException(e,sys)

def load_data(file_path:str)->pd.DataFrame:
    """load data from a csv file"""
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logging.info(f"loaded data succesfully from {file_path}")
        return df
    except Exception as e:
        raise MyException(e,sys)
    
def apply_bow(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->tuple:
    """apply count vectorizer to the data"""
    try:
        logging.info("applying BOW.......")
        vectorizer=CountVectorizer(max_features=max_features)

        x_train=train_data['review'].values
        y_train=train_data['sentiment'].values

        x_test=test_data['review'].values
        y_test=test_data['sentiment'].values

        x_train_bow=vectorizer.fit_transform(x_train)
        x_test_bow=vectorizer.transform(x_test)

        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df['label']=y_train

        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df['label']=y_test

        os.makedirs("models", exist_ok=True)
        pickle.dump(vectorizer,open('models/vectorizer.pkl','wb'))
        logging.info("bag of words applied and data transformed")
        return train_df,test_df
    except Exception as e:
        raise MyException(e,sys)
    
def save_data(df:pd.DataFrame,file_path:str)->None:
    """save the dataframe to csv file"""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logging.info(f"data saved to {file_path}")
    except Exception as e:
        raise MyException(e,sys)
    
def main():
    try:
        params=load_params('params.yaml')
        max_features=params['feature_engineering']['max_features']
        # max_features=20

        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data('./data/interim/test_processed.csv')

        train_df,test_df=apply_bow(train_data=train_data,test_data=test_data,max_features=max_features)

        save_data(train_df,os.path.join("./data","processed","train_bow.csv"))
        save_data(test_df,os.path.join("./data","processed","test_bow.csv"))
    except Exception as e:
        raise MyException(e,sys)
    
if __name__=="__main__":
    main()

