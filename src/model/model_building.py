import numpy as np
import pandas as pd
import pickle 
import sys
from src.exception import MyException
from src.logger import logging
import yaml
from sklearn.linear_model import LogisticRegression

def load_data(file_path:str)->pd.DataFrame:
    """load data from a csv file"""
    try:
        df=pd.read_csv(file_path)
        logging.info("data loaded succesfully")
        return df
    except Exception as e:
        raise MyException(e,sys)
    
def train_model(x_train:np.ndarray,y_train:np.ndarray)->LogisticRegression:
    """train the logistic regression model"""
    try:
        model=LogisticRegression(C=1,solver='liblinear',penalty='l1')
        model.fit(x_train,y_train)
        logging.info("model training complete")
        return model
    except Exception as e:
        raise MyException(e,sys)
    
def save_model(model,file_path:str)->None:
    """saves the trained moel to a file"""
    try:
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logging.info(f"model saved to {file_path}")
    except Exception as e:
        raise MyException(e,sys)
    
def main():
    try:
        train_data=load_data('./data/processed/train_bow.csv')
        x_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values

        model=train_model(x_train,y_train)
        save_model(model,'models/model.pkl')
    except Exception as e:
        raise MyException(e,sys)
    
if __name__=="__main__":
    main()