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
        self.bucket = os.environ['AGENT_COMPANY_CONTEXT_BUCKET']
        self.prefix = os.environ['AGENT_COMPANY_CONTEXT_PREFIX']
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


    def read_from_s3(self, key):
        try:
            key = f"{self.prefix}/{key}"
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        except Exception as e:
            print(f"Error reading from S3: {e}")
            return None
        
    def check_s3_file_exists(self, key):
        try:
            key = f"{self.prefix}/{key}"
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


    @abc.abstractmethod
    def setup_agent(self): 
        raise NotImplementedError("Subclasses must implement this method")
    

    def save_to_s3(self, bucket, key, content):
        try:
            key = f"{self.prefix}/{key}"
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
