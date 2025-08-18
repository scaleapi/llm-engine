import os
import random
import string

try:
    from typing import List, NamedTuple, Optional, Sequence, TypedDict, TypeVar
except ImportError:
    from typing import List, NamedTuple, Optional, Sequence, TypeVar
    from typing_extensions import TypedDict

import boto3
from boto3 import Session, client
from botocore.client import BaseClient
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.config import infra_config

logger = make_logger(logger_name())

__all__: Sequence[str] = (
    "AwsCredentialsDict",
    "AwsCredentials",
    "assume_role",
    "ArnData",
    "parse_arn_string",
    "session",
)

SessionT = TypeVar("SessionT", bound=Session)


class ArnData(NamedTuple):
    """An AWS ARN string, parsed into a structured object. Able to re-create ARN string."""

    role: str
    account: int
    user: Optional[str]
    is_assumed: bool

    def as_arn_string(self) -> str:
        if self.is_assumed:
            kind = "sts"
            source = "assumed-role"
        else:
            kind = "iam"
            source = "role"

        maybe_user = f"/{self.user}" if self.user is not None else ""

        arn = f"arn:aws:{kind}::{self.account}:{source}/{self.role}{maybe_user}"

        return arn


class AwsCredentialsDict(TypedDict):
    """Dictionary form of an :class:`AwsCredentials` instance.
    Produced by that class's :func:`as_dict` method.
    """

    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str


class AwsCredentials(NamedTuple):
    """A complete set of authorized AWS credentials for a particular role.
    Produced by the `assume_role` function.
    """

    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str

    def client(self, client_type: str, region_name: str = "us-west-2") -> BaseClient:
        """Creates the specified Boto3 :param:`client_type` using the AWS credentials.

        The :param:`client_type` parameter is any valid value for `boto3.client` (e.g. `"s3"`).
        """
        return boto3.client(
            client_type,
            region_name=region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

    def as_dict(self) -> AwsCredentialsDict:
        return dict(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )


def assume_role(role_arn: str, role_session_name: Optional[str] = None) -> AwsCredentials:
    """Uses the currently active AWS profile to assume the role specified by :param:`role_arn`.

    If :param:`role_session_name` is not specified, this function will create a unique identifier
    by prefixing with "ml-infra-services"`, using the current active `USER` env var value, and a
    random 10-character long identifier.
    """
    if role_session_name is None:
        random_10_letters = "".join(random.choices(string.ascii_letters, k=10))
        username = os.environ.get("USER", "no_user")
        role_session_name = f"ml-infra-services--{username}--{random_10_letters}"
    sts_client = boto3.client("sts")
    response = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=role_session_name,
    )
    credentials = response["Credentials"]
    return AwsCredentials(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def session(role: Optional[str], session_type: SessionT = Session) -> Optional[SessionT]:
    """Obtain an AWS session using an arbitrary caller-specified role.

    :param:`session_type` defines the type of session to return. Most users will use
    the default boto3 type. Some users required a special type (e.g aioboto3 session).
    """
    # Check if AWS is disabled via config
    if infra_config().disable_aws:
        logger.warning(f"AWS disabled - skipping role assumption (ignoring: {role})")
        return None
    
    # Do not assume roles in CIRCLECI
    if os.getenv("CIRCLECI"):
        logger.warning(f"In circleci, not assuming role (ignoring: {role})")
        role = None
    sesh: SessionT = session_type(profile_name=role)
    return sesh


def _session_aws_okta(
    session_type: SessionT,
    arn: str,
) -> Session:
    current_arn: Optional[str] = boto3.client("sts").get_caller_identity().get("Arn")
    if current_arn is None:
        logger.error(
            "Could not get current identity from STS to check. This is unexpected! "
            "Is aws configuration setup correctly?"
        )
        creds = assume_role(arn)
    else:
        current_role = parse_arn_string(current_arn)
        desired_role = parse_arn_string(arn)

        if current_role.account == desired_role.account:
            logger.warning(
                f"Current user {current_role} is the same as desired {desired_role} -- "
                f"**NOT** assuming desired role with STS as this will be an error! "
                f"Using environment variables to create {session_type}"
            )
            try:
                creds = AwsCredentials(
                    aws_session_token=os.environ["AWS_SESSION_TOKEN"],
                    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                )
            except KeyError as err:
                raise EnvironmentError(
                    "Cannot find all 3 environment variables required for AWS authentication. "
                    "Did you run aws-okta to get credentials?"
                ) from err
        else:
            creds = assume_role(arn)
    sesh = session_type(**creds.as_dict())
    return sesh


def get_current_user() -> str:
    """Uses AWS sts to obtain the profile name of the currently authenticated AWS account."""
    arn = client("sts").get_caller_identity().get("Arn")
    if arn is None:
        raise ValueError("Failed to get identity from STS")
    user = parse_arn_string(arn).user
    if user is None:
        raise ValueError(f"No user identified from STS! arn={arn}")
    return user.split("@")[0]


def parse_arn_string(arn: str) -> ArnData:
    """Parses an AWS ARN string and converts it to structured data in the form of an `ArnData` class."""
    bits: List[str] = arn.split("/")
    if not 2 <= len(bits) <= 3:
        raise ValueError(
            f"Invalid format for AWS ARN string: {arn} -- "
            f"Expecting either 2 or 3 parts seperated by '/'"
        )

    account_and_source: List[str] = bits[0].split("::")
    if len(account_and_source) != 2:
        raise ValueError(
            f"Expecting ARN string to have 2 parts in the first '/' part, "
            f"seperated by '::'. Instead found {account_and_source} from "
            f"arn={arn}"
        )

    account_bits: List[str] = account_and_source[1].split(":")

    if not 1 <= len(account_bits) <= 2:
        raise ValueError(
            f"Expecting ARN string to have 1 or 2 parts in the first '/' part "
            f"of the second '::' part. Instead found {len(account_bits)}: "
            f"{account_bits} for arn={arn}"
        )

    account_str: str = account_bits[0]

    if len(account_bits) == 1:
        first_bits: List[str] = account_and_source[0].split(":")
        if len(first_bits) != 3:
            raise ValueError(
                f"Expecting to find 3 parts in the first part '/' and first part "
                f"of '::'. Instead found {len(first_bits)}: {first_bits} in arn={arn}"
            )
        is_assumed: bool = first_bits[2] == "sts"
    else:
        is_assumed = account_bits[1] == "assumed-role"

    try:
        account: int = int(account_str)
    except ValueError as err:
        raise ValueError(
            "ARN format invalid: expecting account ID to appear as 2nd to last "
            "value seperated by ':' within the first value seperated by '/' and "
            "second value seperated by '::' -- "
            f"arn={arn} and expecting {account_str} to be account ID"
        ) from err

    role: str = bits[1]

    user: Optional[str] = None if len(bits) == 2 else bits[2]

    return ArnData(role, account, user, is_assumed)
