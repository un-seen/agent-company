import os
import boto3
import logging
import requests
import PIL
import io
import time
import json
import PIL.Image
from typing import List, Union, Any
from botocore.client import Config
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

S3_RESOURCE = None
BUCKET = os.environ.get("DATALAKE_AWS_BUCKET")
DATALAKE_LOCAL_DIR = os.environ.get("DATALAKE_LOCAL_DIR")

def init_s3():
    global S3_RESOURCE
    if not S3_RESOURCE:
        S3_ENDPOINT_URL = os.environ.get("DATALAKE_AWS_ENDPOINT_URL")
        AWS_ACCESS_KEY_ID = os.environ.get("DATALAKE_AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.environ.get("DATALAKE_AWS_SECRET_ACCESS_KEY")
        # Initialize the S3 client
        S3_RESOURCE = boto3.resource(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )


def check_key_exists(bucket: str, key: str) -> bool:
    """
    Check if a specific key exists in an S3 bucket.

    :param bucket_name: The name of the S3 bucket.
    :param key: The key to check for existence in the S3 bucket.
    :return: True if the key exists, False otherwise.
    """
    global S3_RESOURCE
    try:
        return S3_RESOURCE.Bucket(bucket).Object(key).content_length > 0
    except Exception as e:
        return False

def upload_file(bucket, local_file_path, s3_key):
    global S3_RESOURCE
    S3_RESOURCE.Bucket(bucket).upload_file(local_file_path, s3_key)
    logger.info(f"Uploaded {local_file_path} to s3://{bucket}/{s3_key}")

def delete_file(bucket, key):
    global S3_RESOURCE
    try:
        S3_RESOURCE.Bucket(bucket).Object(key).delete()
        logger.info(f"Deleted {key} from s3://{bucket}")
    except Exception as e:
        logger.error(f"Error deleting {key} from s3://{bucket}: {e}")

def upload_directory(bucket, local_directory, s3_prefix="", max_workers=8):
    global S3_RESOURCE
    upload_tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_key = os.path.relpath(local_file_path, local_directory).replace(
                    os.path.sep, "/"
                )
                # Append optional prefix
                s3_key = os.path.join(s3_prefix, s3_key) if s3_prefix else s3_key

                future = executor.submit(
                    upload_file, bucket, local_file_path, s3_key
                )
                upload_tasks.append(future)

        # Wait for all upload tasks to complete
        for future in upload_tasks:
            result = future.result()
            logger.info(f"LocalS3Executor: upload_tasks[result]: {result}")


def store_zip(bucket: str, key: str, zip_file_path: str):
    """
    Uploads a zip file from the specified file path to the S3 bucket.

    :param bucket: S3 bucket name
    :param key: S3 key (the path to store the file in the bucket)
    :param zip_file_path: Local file path of the zip file to upload
    """
    global S3_RESOURCE
    try:
        # Check if the file exists before attempting to upload
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f"File {zip_file_path} not found")

        # Open the zip file in binary mode and upload to S3
        s3_bucket = S3_RESOURCE.Bucket(bucket)
        with open(zip_file_path, 'rb') as zip_file:
            logger.info(f"StoreZip: {key} in {bucket}")
            s3_bucket.put_object(Key=key, Body=zip_file)
        logger.info(f"Successfully stored zip file {key} in {bucket}")
        
    except Exception as e:
        logger.error(f"StoreZipError {key}: {e}")
        
        

        
def store_image(bucket: str, key: str, image: PIL.Image.Image, timeout: int = 120) -> bool:
    '''
    Returns True if the image is stored in the bucket, False otherwise.
    '''
    global S3_RESOURCE
    try:
        s3_bucket = S3_RESOURCE.Bucket(bucket)
        file_content = io.BytesIO()
        image.save(file_content, format="webp", lossless=True)
        file_content.seek(0)
        logger.info(f"StoreImage: {key} in {bucket}")
        s3_bucket.put_object(Key=key, Body=file_content)
        # WaitToCheckKeyExists
        clock = 0 
        while not check_key_exists(bucket, key) and clock < timeout:
            time.sleep(5)
            clock += 5
        return True
    except Exception as e:
        logger.error(f"StoreContentError {key}: {e}")

def file_path_to_content(file_path):
    """
    Convert file path to file content for uploading to S3.
    
    Parameters:
    file_path (str): The path to the file.
    
    Returns:
    bytes: The content of the file as bytes.
    """
    try:
        with open(file_path, 'rb') as file:  # Open in binary mode
            content = file.read()
        return content
    except Exception as e:
        raise e
    
def store_file(bucket: str, key: str, file_path: str):
    global S3_RESOURCE
    try:
        s3_bucket = S3_RESOURCE.Bucket(bucket)
        file_content = file_path_to_content(file_path)
        s3_bucket.put_object(Key=key, Body=file_content)
    except Exception as e:
        logger.error(f"Error storing file {file_path} in {key} : {e}")


def store_json(bucket: str, key: str, data: Any):
    global S3_RESOURCE
    try:
        s3_bucket = S3_RESOURCE.Bucket(bucket)
        bytes_data = json.dumps(data).encode("utf-8")
        s3_bucket.put_object(Key=key, Body=bytes_data)
    except Exception as e:
        logger.error(f"Error storing data {str(data)} in {key} : {e}")    
    
def copy_file_to_datalake(bucket: str, key: str):
    """
    Stores a file in an S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket.
        file_path (str): The local path to the file to be stored.

    Returns:
        str: The S3 object key (path/filename) if successful, or None on error.
    """
    global S3_RESOURCE
    try:
        s3_bucket = S3_RESOURCE.Bucket(bucket)
        # Multipart upload for larger files (adjust as needed)
        config = TransferConfig(
            multipart_threshold=1024 * 25,  # 25MB threshold
            max_concurrency=10,
            multipart_chunksize=1024 * 25,
            use_threads=True,
        )

        s3_bucket.upload_file(f"{DATALAKE_LOCAL_DIR}/{key}", key, Config=config)
        return True

    except (ClientError, FileNotFoundError) as e:
        logger.error(f"Error storing file {key}: {e}")
        return False


def list_files(bucket: str, path) -> List[str]:
    global S3_RESOURCE
    s3_bucket = S3_RESOURCE.Bucket(bucket)
    files = []
    for obj in s3_bucket.objects.filter(Prefix=path):
        files.append(obj.key)
    return files

def generate_presigned_url(bucket: str, object_key, expiration=3600) -> str:
    """
    Generate a pre-signed URL for a file in an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_key (str): The key of the object (file) in the bucket.
        expiration (int): The URL expiration time in seconds (default: 1 hour).

    Returns:
        str: The pre-signed URL for downloading the object.
    """
    global S3_RESOURCE
    try:
        response = S3_RESOURCE.meta.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_key},
            ExpiresIn=expiration,
        )
        return response
    except Exception as e:
        raise e

def get_json_from_key(bucket: str, key: str):
    global S3_RESOURCE
    try:
        obj = S3_RESOURCE.Object(bucket, key)
        data = obj.get()["Body"].read().decode("utf-8")
        return json.loads(data)
    except Exception as e:
        logger.error(f"Error getting string from key {key}: {e}")
        return None

def download_file_from_s3_presigned_url(presigned_url, output_file_path):
    if os.path.exists(output_file_path):
        return
    logger.info(f"DownloadingUrl {presigned_url} to {output_file_path}")
    create_directories_for_file(output_file_path)
    # Send a GET request to the presigned URL to download the file
    response = requests.get(presigned_url, stream=True)
    if response.status_code == 200:
        # Open a file in binary write mode and save the downloaded content
        with open(output_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        logger.info(
            f"File downloaded successfully and saved as: {output_file_path}"
        )
    else:
        logger.info(f"Failed to download file. Status code: {response.status_code}")


def create_directories_for_file(file_path):
    # Extract the directory path from the file path
    directory = os.path.dirname(file_path)
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)