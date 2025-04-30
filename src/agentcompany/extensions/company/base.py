import redis
import boto3
import os
import json
import abc
from botocore.exceptions import ClientError
from typing import List


class AgentCompany:
    """
    AgentCompany is a class that contains a list of agents
    It is used to run a company of agents on a given task
    """
    
    def __init__(self, interface_id: str, session_id: str, **kwargs):
        self.interface_id = interface_id
        self.session_id = session_id

        # Initialize Redis client
        self.redis_client = redis.Redis.from_url(
            os.environ["REDIS_URL"],
            decode_responses=True
        )

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['DATALAKE_AWS_ENDPOINT_URL'],
            aws_access_key_id=os.environ.get('DATALAKE_AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('DATALAKE_AWS_SECRET_ACCESS_KEY')
        )
        for key, value in kwargs.items():
            setattr(self, key, value)


    def check_s3_file_exists(self, bucket, key):
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False


    @abc.abstractmethod
    def setup_agent(self): 
        raise NotImplementedError("Subclasses must implement this method")
    

    def save_to_s3(self, bucket, key, content):
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content.encode('utf-8'),
                ContentType='text/markdown'
            )
            return True
        except Exception as e:
            print(f"Error saving to S3: {e}")
            return False
        
    def main(self):
        raise NotImplementedError("Subclasses must implement this method")
