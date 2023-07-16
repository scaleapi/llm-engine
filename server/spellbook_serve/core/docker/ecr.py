from typing import Dict, List, Optional

import boto3

from spellbook_serve.core.config import ml_infra_config
from spellbook_serve.core.utils.git import tag

DEFAULT_FILTER = {"tagStatus": "TAGGED"}


def repository_exists(repository_name: str):
    ecr = boto3.client("ecr", region_name=ml_infra_config().default_region)
    try:
        response = ecr.describe_repositories(
            registryId=ml_infra_config().ml_account_id, repositoryNames=[repository_name]
        )
        if response.get("repositories"):
            return True
    except ecr.exceptions.RepositoryNotFoundException:
        return False
    return False


def batch_image_exists(
    *,
    region_name: str = ml_infra_config().default_region,
    repository_name: str,
    image_tags: Optional[List[str]] = None,
    image_digests: Optional[List[str]] = None,
    filter: Optional[Dict[str, str]] = None,  # pylint:disable=redefined-builtin
    aws_profile: Optional[str] = None,
) -> bool:
    """Because the boto3 api raises an exception once it can't find a tag, this can only check that
    all the image tags exist
    """
    image_digests = [] if image_digests is None else image_digests
    image_tags = [] if image_tags is None else image_tags
    if filter is None:
        filter = DEFAULT_FILTER

    if aws_profile is None:
        client = boto3.client("ecr", region_name=region_name)
    else:
        session = boto3.Session(profile_name=aws_profile)
        client = session.client("ecr", region_name=region_name)
    try:
        client.describe_images(
            registryId=ml_infra_config().ml_account_id,
            repositoryName=repository_name,
            imageIds=[
                *[{"imageTag": t} for t in image_tags],
                *[{"imageDigest": d} for d in image_digests],
            ],
            filter=filter,
        )
    except client.exceptions.ImageNotFoundException:
        return False

    return True


def image_exists(
    *,
    region_name: str = ml_infra_config().default_region,
    repository_name: str,
    image_name: Optional[str] = None,
    image_tag: Optional[str] = None,
    image_digest: Optional[str] = None,
    filter: Optional[Dict[str, str]] = None,  # pylint:disable=redefined-builtin
    aws_profile: Optional[str] = None,
) -> bool:
    assert (bool(image_tag) + bool(image_digest) + bool(image_name)) == 1

    image_digests = None if image_digest is None else [image_digest]
    image_tags = [image_tag] if image_tag else [image_name.split(":")[1]] if image_name else None

    return batch_image_exists(
        region_name=region_name,
        repository_name=repository_name,
        image_tags=image_tags,
        image_digests=image_digests,
        filter=filter,
        aws_profile=aws_profile,
    )


def ecr_exists_for_repo(repo_name: str, image_tag: Optional[str] = None):
    """Check if image exists in ECR"""
    if image_tag is None:
        image_tag = tag()
    ecr = boto3.client("ecr", region_name=ml_infra_config().default_region)
    try:
        ecr.describe_images(
            registryId=ml_infra_config().ml_account_id,
            repositoryName=repo_name,
            imageIds=[{"imageTag": image_tag}],
        )
        return True
    except ecr.exceptions.ImageNotFoundException:
        return False
