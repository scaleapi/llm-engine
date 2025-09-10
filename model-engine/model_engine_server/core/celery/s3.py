"""s3 result store backend.
copied from https://github.com/celery/celery/blob/81df81acf8605ba3802810c7901be7d905c5200b/celery/backends/s3.py
"""

import threading
import base64
import hashlib

import tenacity
from celery.backends.base import KeyValueStoreBackend
from celery.exceptions import ImproperlyConfigured
from kombu.utils.encoding import bytes_to_str
from model_engine_server.core.config import infra_config

try:
    import botocore
except ImportError:
    botocore = None

__all__ = ("S3Backend",)


class S3Backend(KeyValueStoreBackend):
    """An S3 task result store.

    Raises:
        celery.exceptions.ImproperlyConfigured:
            if module :pypi:`boto3` is not available,
            if the :setting:`aws_access_key_id` or
            setting:`aws_secret_access_key` are not set,
            or it the :setting:`bucket` is not set.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        conf = self.app.conf

        self.boto3_session = conf.get("s3_boto3_session", None)
        if not self.boto3_session:
            raise ImproperlyConfigured("Missing boto3 session")
        if self._get_credentials() is None:
            raise ImproperlyConfigured("Missing aws s3 creds")

        # For backwards compatibility with the mainline version of this class -
        # this gets dereferenced later.
        self.endpoint_url = conf.get("s3_endpoint_url", None)

        self.bucket_name = conf.get("s3_bucket", None)
        if not self.bucket_name:
            raise ImproperlyConfigured("Missing bucket name")

        self.base_path = conf.get("s3_base_path", None)

        self._s3_resource_per_thread = {}  # thread identifier: s3 resource
        self._s3_resource_dict_lock = threading.Lock()  # might not be necessary but it's insurance

    def _get_s3_object(self, key):
        current_thread = threading.get_ident()
        key_bucket_path = self.base_path + key if self.base_path else key
        with self._s3_resource_dict_lock:
            if current_thread not in self._s3_resource_per_thread:
                self._s3_resource_per_thread[current_thread] = self._connect_to_s3()
            s3_resource = self._s3_resource_per_thread[current_thread]
        return s3_resource.Object(self.bucket_name, key_bucket_path)

    def get(self, key):
        key = bytes_to_str(key)
        s3_object = self._get_s3_object(key)
        try:
            s3_object.load()
            data = s3_object.get()["Body"].read()
            return data if self.content_encoding == "binary" else data.decode("utf-8")
        except botocore.exceptions.ClientError as error:
            # A 403 is returned if the object does not exist and we don't have ListBucket permissions
            # such as in Hosted Model Inference. If we do have ListBucket permissions, we get 404.
            if error.response["Error"]["Code"] in ["403", "404"]:
                return None
            raise error

    def set(self, key, value):
        key = bytes_to_str(key)
        s3_object = self._get_s3_object(key)

        # Check if celery_enable_sha256 mode is enabled via config to use sha256 hash instead of md5
        if infra_config().celery_enable_sha256:
            # Ensure value is bytes for hashing
            if isinstance(value, str):
                value_bytes = value.encode("utf-8")
            else:
                value_bytes = value

            sha256_hash = hashlib.sha256(value_bytes).digest()
            checksum_sha256 = base64.b64encode(sha256_hash).decode("utf-8")
            s3_object.put(Body=value, ChecksumAlgorithm="SHA256", ChecksumSHA256=checksum_sha256)
        else:
            s3_object.put(Body=value)

    def delete(self, key):
        key = bytes_to_str(key)
        s3_object = self._get_s3_object(key)
        s3_object.delete()

    # session.resource is not threadsafe, so suggested fix is to retry.
    # https://github.com/boto/boto3/issues/801#issuecomment-358195444
    @tenacity.retry(stop=tenacity.stop_after_attempt(10), reraise=True)
    def _connect_to_s3(self):
        session = self.boto3_session
        return session.resource("s3", endpoint_url=self.endpoint_url)

    # Same issue as above
    @tenacity.retry(stop=tenacity.stop_after_attempt(20), reraise=True)
    def _get_credentials(self):
        return self.boto3_session.get_credentials()

    def add_to_chord(self, chord_id, result):
        raise NotImplementedError

    def incr(self, key):
        raise NotImplementedError

    def mget(self, keys):
        raise NotImplementedError
