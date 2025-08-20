import os
from urllib.parse import urlparse

from kdcube_ai_app.tools.datasource import DataElement
from kdcube_ai_app.tools.content_type import fetch_url_with_content_type

import logging
logger = logging.getLogger("FetchTool")

class ContentFetcher:
    @staticmethod
    def fetch(data_element: DataElement) -> bytes:
        """Fetch content from external source and return raw bytes."""
        if data_element.type == "url":
            return ContentFetcher._fetch_url(data_element.url)
        elif data_element.type == "file":
            return ContentFetcher._fetch_file(data_element.path)
        elif data_element.type == "raw_text":
            return data_element.text.encode('utf-8')
        else:
            raise ValueError(f"Unsupported data element type: {data_element.type}")

    @staticmethod
    def _fetch_url(url: str) -> bytes:
        """Download content from URL, supporting HTTP(S) and S3."""
        parsed_url = urlparse(url)

        if parsed_url.scheme.lower() in ["http", "https"]:
            content_bytes, content_type, filename = fetch_url_with_content_type(url)
            return content_bytes

        elif parsed_url.scheme.lower() == "s3":
            return ContentFetcher._fetch_s3(parsed_url)

        else:
            raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}")

    @staticmethod
    def _fetch_s3(parsed_url) -> bytes:
        """Download content from S3 URL directly to memory."""
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
        except ImportError:
            raise ImportError("boto3 is required for S3 URLs. Install with: pip install boto3")

        # Extract bucket and key from URL
        bucket_name = parsed_url.netloc
        s3_key = parsed_url.path.lstrip('/')  # Remove leading slash

        if not bucket_name or not s3_key:
            raise ValueError(f"Invalid S3 URL format: {parsed_url.geturl()}")

        try:
            s3_client = boto3.client('s3')

            # Download object directly to memory
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            content = response['Body'].read()

            logger.info(f"Downloaded {len(content)} bytes from s3://{bucket_name}/{s3_key}")
            return content

        except NoCredentialsError:
            raise ConnectionError("AWS credentials not found. Please configure AWS credentials.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise FileNotFoundError(f"S3 bucket '{bucket_name}' does not exist")
            elif error_code == 'NoSuchKey':
                raise FileNotFoundError(f"S3 object '{s3_key}' not found in bucket '{bucket_name}'")
            elif error_code == 'AccessDenied':
                raise PermissionError(f"Access denied to s3://{bucket_name}/{s3_key}")
            else:
                raise ConnectionError(f"S3 error: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to download from S3: {e}")

    @staticmethod
    def _fetch_file(file_path: str) -> bytes:
        """Read file as raw bytes, supporting local and network paths."""
        parsed_path = urlparse(file_path)

        # Handle file:// URLs
        if parsed_path.scheme == "file":
            file_path = parsed_path.path

        # Handle local file paths
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            logger.info(f"Read {len(content)} bytes from {file_path}")
            return content

        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {file_path}")
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")
