import re
from typing import Tuple

from google.auth import default
from google.cloud import storage


def get_gcs_sync_client() -> storage.Client:
    """Create a synchronous Google Cloud Storage client using application default credentials."""
    credentials, project = default()
    return storage.Client(credentials=credentials, project=project)


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """Parse a GCS URI into (bucket_name, blob_name).

    Supports both gs://bucket/key and https://storage.googleapis.com/bucket/key formats.

    Raises:
        ValueError: If the URI format is not recognized.
    """
    match = re.match(r"^gs://([^/]+)/(.+)$", uri)
    if not match:
        match = re.match(r"^https://storage\.googleapis\.com/([^/]+)/(.+)$", uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return match.group(1), match.group(2)
