import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
import sys
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
from src.logger import logging
from src.exception import MyException
from src.connections import s3_connection

def load_params(params_path:str)->dict:
    """loads params file from a yaml file"""
    try:
        with open(params_path ,'r') as file:
            params=yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        raise MyException(e,sys)

def load_data(data_url:str)->pd.DataFrame:
    """load data from csv file"""
    try:
        df=pd.read_csv(data_url)
        logging.info(f"data succesfully loaded from {data_url}")
        return df
    except Exception as e:
        raise MyException(e,sys)

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """preprocesses the data"""
    try:
        logging.info("pre processing data...............")
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
        logging.info('Data preprocessing completed')
        return final_df
    except Exception as e:
        raise MyException(e,sys)

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    """save the train adn test dataset"""
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logging.info(f"train adn test data saved to {raw_data_path}")
    except Exception as e:
        raise MyException(e,sys)
    
def main():
    try:
        params=load_params(params_path="params.yaml")
        test_size=params['data_ingestion']['test_size']
        # test_size=0.2
        df = load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')
        # s3 = s3_connection.s3_operations("bucket-name", "access-key", "secret-access-key")#fill these details to get data from cloud
        # df = s3.fetch_file_from_s3("data.csv")

        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data=train_data,test_data=test_data,data_path='./data')
    except Exception as e:
        raise MyException(e,sys)
    


if __name__ == '__main__':
    main()
