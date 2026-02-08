import boto3
import pandas as pd
from src.exception import MyException
from src.logger import logging
from io import StringIO
import sys

class s3_operations:
    def __init__(self,bucket_name,aws_access_key,aws_secret_key,region_name="us-east-1"):
        self.bucket_name=bucket_name
        self.s3_client=boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        logging.info("data ingestion from s3 is initialized")

    def fetch_file_from_s3(self,file_key):
        """
        Fetches a CSV file from the S3 bucket and returns it as a Pandas DataFrame.
        :param file_key: S3 file path (e.g., 'data/data.csv')
        :return: Pandas DataFrame
        """
        try:
            logging.info(f"fetching file from {file_key} from s3 bucket {self.bucket_name}")
            obj=self.s3_client.get_object(Bucket=self.bucket_name,Key=file_key)
            df=pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"Successfully fetched and loaded '{file_key}' from S3 that has {len(df)} records.")
            return df
        except Exception as e:
            raise MyException(e,sys)
            