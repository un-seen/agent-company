import redis
import boto3
import os
import json
import abc
from botocore.exceptions import ClientError
from botocore.client import Config
from typing import List
from urllib.parse import urlparse
import psycopg2

class AgentCompany:
    """
    AgentCompany is a class that contains a list of agents
    It is used to run a company of agents on a given task
    """
    presigned_url_expiration = 3600

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
        # Initialize postgres client
        postgres_url = os.environ['POSTGRES_URL']
        self.postgres_client = psycopg2.connect(postgres_url)
        # Initialize S3 client

        cfg = Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"}   # gives bucket-name host style
        )
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['DATALAKE_AWS_ENDPOINT_URL'],
            aws_access_key_id=os.environ.get('DATALAKE_AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('DATALAKE_AWS_SECRET_ACCESS_KEY'),
            config=cfg
        )
        for key, value in kwargs.items():
            setattr(self, key, value)

    def generate_presigned_url(self, key):
        return self.s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=self.presigned_url_expiration
        )

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
            if len(self.prefix) > 0 and not key.startswith(self.prefix):
                key = f"{self.prefix}/{key}"
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


    @abc.abstractmethod
    def setup_agent(self): 
        raise NotImplementedError("Subclasses must implement this method")
    

    def save_to_s3(self, key, content):
        try:
            key = f"{self.prefix}/{key}"
            self.s3_client.put_object(
                Bucket=self.bucket,
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
