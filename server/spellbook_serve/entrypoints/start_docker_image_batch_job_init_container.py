import argparse
import shutil

import spellbook_serve.core.aws.storage_client as storage_client
from spellbook_serve.common.serialization_utils import b64_to_str
from spellbook_serve.core.aws.storage_client import s3_fileobj_exists
from spellbook_serve.core.loggers import filename_wo_ext, make_logger
from spellbook_serve.core.utils.url import parse_attachment_url

logger = make_logger(filename_wo_ext(__file__))


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
        if not s3_fileobj_exists(bucket=parsed_remote.bucket, key=parsed_remote.key):
            logger.warning("S3 file doesn't exist, aborting")
            raise ValueError  # TODO propagate error to the gateway
        # TODO if we need we can s5cmd this
        with storage_client.open(remote_file, "rb") as fr, open(local_file, "wb") as fw2:
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
