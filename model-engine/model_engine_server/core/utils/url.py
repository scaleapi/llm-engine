"""URL-based utility functions."""
import re
from typing import NamedTuple, Optional


class ParsedURL(NamedTuple):
    protocol: str
    bucket: str
    key: str
    region: Optional[str]
    account: Optional[str] = None

    def canonical_url(self) -> str:
        """Packs the parsed URL information into a standard form of
        `<protocol>://<bucket>/<key>`.
        """
        return f"{self.protocol}://{self.bucket}/{self.key}"

    @staticmethod
    def s3(bucket: str, key: str, region: Optional[str] = None) -> "ParsedURL":
        return ParsedURL(protocol="s3", bucket=bucket, key=key, region=region)

    @staticmethod
    def gs(bucket: str, key: str, region: Optional[str] = None) -> "ParsedURL":
        return ParsedURL(protocol="gs", bucket=bucket, key=key, region=region)

    @staticmethod
    def azure(bucket: str, key: str, account: Optional[str] = None) -> "ParsedURL":
        return ParsedURL(protocol="azure", bucket=bucket, key=key, account=account)

    @staticmethod
    def cds(bucket: str, key: str, region: Optional[str] = None) -> "ParsedURL":
        return ParsedURL(protocol="scale-cds", bucket=bucket, key=key, region=region)


class InvalidAttachmentUrl(ValueError):
    pass


def parse_attachment_url(url: str, clean_key: bool = True) -> ParsedURL:
    """Extracts protocol, bucket, region, and key from the :param:`url`.

    :raises: InvalidAttachmentUrl Iff the input `url` is not a valid AWS S3 or GCS url.
    """
    # returns dict of protocol, bucket, region, key
    protocol = "s3"
    bucket = None
    region = None
    key = None
    account = None

    # s3://bucket/key1/key2
    match = re.search("^s3://([^/]+)/(.*?)$", url)
    if match:
        bucket, key = match.group(1), match.group(2)

    # gs://bucket/key1/key2
    match = re.search("^gs://([^/]+)/(.*?)$", url)
    if match:
        protocol = "gs"
        bucket, key = match.group(1), match.group(2)

    # azure://bucket/key1/key2
    # for Azure Blob Storage, bucket refers to an ABS container
    match = re.search("^azure://([^/]+)/(.*?)$", url)
    if match:
        protocol = "azure"
        bucket, key = match.group(1), match.group(2)

    # http://bucket.s3.amazonaws.com/key1/key2
    match = re.search("^https?://(.+).s3.amazonaws.com(.*?)$", url)
    if match:
        bucket, key = match.group(1), match.group(2)

    # http://bucket.s3-aws-region.amazonaws.com/key1/key2
    match = re.search("^https?://(.+).s3[.-](dualstack\\.)?([^.]+).amazonaws.com(.*?)$", url)
    if match:
        bucket, region, key = match.group(1), match.group(3), match.group(4)

    # http://s3.amazonaws.com/bucket/key1/key2
    match = re.search("^https?://s3.amazonaws.com/([^/]+)(.*?)$", url)
    if match:
        bucket, key = match.group(1), match.group(2)

    # http://s3-aws-region.amazonaws.com/bucket/key1/key2
    match = re.search("^https?://s3[.-](dualstack\\.)?([^.]+).amazonaws.com/([^/]+)(.*?)$", url)
    if match:
        bucket, region, key = match.group(3), match.group(2), match.group(4)

    # https://storage.cloud.google.com/bucket/this/is/a/key
    match = re.search("^https?://storage.cloud.google.com/([^/]+)(.*?)$", url)
    if match:
        protocol = "gs"
        bucket, key = match.group(1), match.group(2)

    # http://s3.amazonaws.com/bucket/key1/key2
    match = re.search("^https?://s3.amazonaws.com/([^/]+)(.*?)$", url)
    if match:
        bucket, key = match.group(1), match.group(2)

    # https://account.blob.core.windows.net/bucket/key1/key2
    # for Azure Blob Storage, bucket refers to an ABS container
    match = re.search("^https?://([^/]+).blob.core.windows.net/([^/]+)(.*?)$", url)
    if match:
        protocol = "azure"
        account, bucket, key = match.group(1), match.group(2), match.group(3)

    match = re.search("scale-cds://(\\w+)/([\\-\\w\\/]+)", url)
    if match:
        bucket, key = match.group(1), match.group(2)
        protocol = "scale-cds"

    if bucket is None or key is None:
        raise InvalidAttachmentUrl(
            "Invalid attachment URL: no bucket or key specified: \n" f"'{url}'"
        )

    def clean(v):
        return v and v.strip("/")

    return ParsedURL(
        protocol=clean(protocol),
        bucket=clean(bucket),
        region=clean(region),
        key=clean(key) if clean_key else key,
        account=clean(account),
    )
