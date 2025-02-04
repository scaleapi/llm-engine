import argparse
import shutil

from model_engine_server.common.serialization_utils import b64_to_str
from model_engine_server.core.aws import storage_client as aws_storage_client

# Top-level imports for remote storage clients with aliases.
from model_engine_server.core.gcp import storage_client as gcp_storage_client
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.utils.url import parse_attachment_url

logger = make_logger(logger_name())


def main(input_local: str, local_file: str, remote_file: str, file_contents_b64encoded: str):
    if input_local:
        logger.info("Writing file from local")
        with open(local_file, "w") as fw:
            # Have to encode as b64 to ensure it gets passed to k8s correctly
            fw.write(b64_to_str(file_contents_b64encoded))
        return
    else:
        logger.info("Copying file from remote")
        parsed_remote = parse_attachment_url(remote_file)
        # Conditional logic to support GCS file URLs without breaking S3 behavior.
        if remote_file.startswith("gs://"):
            # Use the GCP storage client.
            file_exists = gcp_storage_client.gcs_fileobj_exists(
                bucket=parsed_remote.bucket, key=parsed_remote.key
            )
            storage_open = gcp_storage_client.open
            file_label = "GCS"
        else:
            # Use the AWS storage client for backward compatibility.
            file_exists = aws_storage_client.s3_fileobj_exists(
                bucket=parsed_remote.bucket, key=parsed_remote.key
            )
            storage_open = aws_storage_client.open
            file_label = "S3"
        if not file_exists:
            logger.warning(f"{file_label} file doesn't exist, aborting")
            raise ValueError  # TODO: propagate error to the gateway
        # Open the remote file (using the appropriate storage client) and copy its contents locally.
        with storage_open(remote_file, "rb") as fr, open(local_file, "wb") as fw2:
            shutil.copyfileobj(fr, fw2)


# Usage: <script> --input-location local --local-file <file location to write stuff>
#   --file-contents-b64encoded <b64encoding of file contents, written as a string>
# or <script --input-location s3 --local-file <same as above>
#   --remote-file <s3 url of file to copy into here>
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-local", dest="input_local", action="store_true")
    parser.add_argument("--input-remote", dest="input_local", action="store_false")
    parser.set_defaults(input_local=True)
    parser.add_argument("--local-file", type=str, required=True)
    parser.add_argument("--remote-file", type=str)
    parser.add_argument("--file-contents-b64encoded", type=str)
    args = parser.parse_args()
    main(args.input_local, args.local_file, args.remote_file, args.file_contents_b64encoded)
