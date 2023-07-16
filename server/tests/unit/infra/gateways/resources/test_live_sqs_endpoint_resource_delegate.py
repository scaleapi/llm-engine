import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import botocore.exceptions
import pytest
from llm_engine_server.common.dtos.endpoint_builder import BuildEndpointRequest
from llm_engine_server.domain.entities import ModelEndpointRecord
from llm_engine_server.domain.exceptions import EndpointResourceInfraException
from llm_engine_server.infra.gateways.resources.live_sqs_endpoint_resource_delegate import (
    LiveSQSEndpointResourceDelegate,
)

MODULE_PATH = "llm_engine_server.infra.gateways.resources.live_sqs_endpoint_resource_delegate"

EXPECTED_QUEUE_POLICY = """
{
  "Version": "2012-10-17",
  "Id": "__default_policy_ID",
  "Statement": [
    {
      "Sid": "__owner_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::000000000000:root"
      },
      "Action": "sqs:*",
      "Resource": "arn:aws:sqs:us-west-2:000000000000:llm-engine-endpoint-id-test_model_endpoint_id_3"
    },
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::000000000000:role/default"
      },
      "Action": "sqs:*",
      "Resource": "arn:aws:sqs:us-west-2:000000000000:llm-engine-endpoint-id-test_model_endpoint_id_3"
    },
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::000000000000:role/ml_llm_engine"
      },
      "Action": "sqs:*",
      "Resource": "arn:aws:sqs:us-west-2:000000000000:llm-engine-endpoint-id-test_model_endpoint_id_3"
    }
  ]
}
"""

EXPECTED_QUEUE_TAGS = {
    "infra.scale.com/product": "MLInfraLLMEngineSQS",
    "infra.scale.com/team": "test_team",
    "infra.scale.com/contact": "yi.xu@scale.com",
    "infra.scale.com/customer": "AllCustomers",
    "infra.scale.com/financialOwner": "yi.xu@scale.com",
    "Spellbook-Serve-Endpoint-Id": "test_model_endpoint_id_3",
    "Spellbook-Serve-Endpoint-Name": "test_model_endpoint_name_3",
    "Spellbook-Serve-Endpoint-Created-By": "test_user_id",
}


def _get_fake_botocore_exception():
    # https://github.com/boto/boto3/issues/2485#issuecomment-752454137
    return botocore.exceptions.ClientError(
        error_response={
            "Error": {
                "Code": "foo",
                "Message": "bar",
            }
        },
        operation_name="foobar",
    )


@pytest.fixture
def mock_create_async_sqs_client_create_queue():
    create_queue_response = {
        "QueueUrl": "https://us-west-2.queue.amazonaws.com/000000000000/llm-engine-endpoint-id-test_model_endpoint_id_3",
        "ResponseMetadata": {
            "RequestId": "9c05b1cc-d806-5cbd-bd4a-ea339c90e25f",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "x-amzn-requestid": "9c05b1cc-d806-5cbd-bd4a-ea339c90e25f",
                "date": "Mon, 28 Nov 2022 23:20:54 GMT",
                "content-type": "text/xml",
                "content-length": "348",
            },
            "RetryAttempts": 0,
        },
    }

    mock_sqs_client_session_val = AsyncMock()
    mock_sqs_client_session_val.get_queue_url = AsyncMock(
        side_effect=_get_fake_botocore_exception()
    )
    mock_sqs_client_session_val.create_queue = AsyncMock(return_value=create_queue_response)

    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.fixture
def mock_create_async_sqs_client_get_queue_url():
    get_queue_response = {
        "QueueUrl": "https://us-west-2.queue.amazonaws.com/000000000000/llm-engine-endpoint-id-test_model_endpoint_id_3",
    }

    mock_sqs_client_session_val = AsyncMock()
    mock_sqs_client_session_val.get_queue_url = AsyncMock(return_value=get_queue_response)
    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.fixture
def mock_create_async_sqs_client_create_queue_throws_exception():
    mock_sqs_client_session_val = AsyncMock()

    mock_sqs_client_session_val.get_queue_url = AsyncMock(
        side_effect=_get_fake_botocore_exception()
    )
    mock_sqs_client_session_val.create_queue = AsyncMock(side_effect=_get_fake_botocore_exception())
    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.fixture
def mock_create_async_sqs_client_create_queue_returns_non_200():
    create_queue_response = {
        "ResponseMetadata": {
            "RequestId": "9c05b1cc-d806-5cbd-bd4a-ea339c90e25f",
            "HTTPStatusCode": 400,
            "HTTPHeaders": {
                "x-amzn-requestid": "9c05b1cc-d806-5cbd-bd4a-ea339c90e25f",
                "date": "Mon, 28 Nov 2022 23:20:54 GMT",
                "content-type": "text/xml",
                "content-length": "348",
            },
            "RetryAttempts": 0,
        }
    }

    mock_sqs_client_session_val = AsyncMock()
    mock_sqs_client_session_val.get_queue_url = AsyncMock(
        side_effect=_get_fake_botocore_exception()
    )
    mock_sqs_client_session_val.create_queue = AsyncMock(return_value=create_queue_response)

    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.fixture()
def mock_create_async_sqs_client_delete_queue():
    mock_sqs_client_session_val = AsyncMock()

    mock_sqs_client_session_val.get_queue_url = AsyncMock()
    mock_sqs_client_session_val.get_queue_url.return_value = {
        "QueueUrl": "https://us-west-2.queue.amazonaws.com/000000000000/llm-engine-endpoint-id-model_endpoint_id_1"
    }

    delete_response = {
        "ResponseMetadata": {
            "RequestId": "3e120372-72f9-5e50-bc54-588d361c4499",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "x-amzn-requestid": "3e120372-72f9-5e50-bc54-588d361c4499",
                "date": "Mon, 28 Nov 2022 23:22:13 GMT",
                "content-type": "text/xml",
                "content-length": "211",
            },
            "RetryAttempts": 0,
        }
    }
    mock_sqs_client_session_val.delete_queue = AsyncMock(return_value=delete_response)

    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.fixture()
def mock_create_async_sqs_client_delete_queue_returns_non_200():
    mock_sqs_client_session_val = AsyncMock()

    mock_sqs_client_session_val.get_queue_url = AsyncMock()
    mock_sqs_client_session_val.get_queue_url.return_value = {
        "QueueUrl": "https://us-west-2.queue.amazonaws.com/000000000000/llm-engine-endpoint-id-model_endpoint_id_1"
    }

    delete_response = {
        "ResponseMetadata": {
            "RequestId": "3e120372-72f9-5e50-bc54-588d361c4499",
            "HTTPStatusCode": 400,
            "HTTPHeaders": {
                "x-amzn-requestid": "3e120372-72f9-5e50-bc54-588d361c4499",
                "date": "Mon, 28 Nov 2022 23:22:13 GMT",
                "content-type": "text/xml",
                "content-length": "211",
            },
            "RetryAttempts": 0,
        }
    }
    mock_sqs_client_session_val.delete_queue = AsyncMock(return_value=delete_response)

    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.fixture()
def mock_create_async_sqs_client_delete_queue_throws_exception():
    mock_sqs_client_session_val = AsyncMock()

    mock_sqs_client_session_val.get_queue_url = AsyncMock()
    mock_sqs_client_session_val.get_queue_url.return_value = {
        "QueueUrl": "https://us-west-2.queue.amazonaws.com/000000000000/llm-engine-endpoint-id-model_endpoint_id_1"
    }

    mock_sqs_client_session_val.delete_queue = AsyncMock(side_effect=_get_fake_botocore_exception())

    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.fixture()
def mock_create_async_sqs_client_get_queue_attributes():
    mock_sqs_client_session_val = AsyncMock()

    mock_sqs_client_session_val.get_queue_url = AsyncMock()
    mock_sqs_client_session_val.get_queue_url.return_value = {
        "QueueUrl": "https://us-west-2.queue.amazonaws.com/000000000000/llm-engine-endpoint-id-model_endpoint_id_1"
    }

    get_queue_attributes_response = {
        "Attributes": {
            "QueueArn": "arn:aws:sqs:us-west-2:000000000000:llm-engine-endpoint-id-model_endpoint_id_1",
            "ApproximateNumberOfMessages": "0",
            "ApproximateNumberOfMessagesNotVisible": "0",
            "ApproximateNumberOfMessagesDelayed": "0",
            "CreatedTimestamp": "1675897120",
            "LastModifiedTimestamp": "1675897120",
            "VisibilityTimeout": "3600",
            "MaximumMessageSize": "262144",
            "MessageRetentionPeriod": "345600",
            "DelaySeconds": "0",
            "Policy": "",
            "ReceiveMessageWaitTimeSeconds": "0",
            "SqsManagedSseEnabled": "true",
        }
    }
    mock_sqs_client_session_val.get_queue_attributes = AsyncMock(
        return_value=get_queue_attributes_response
    )

    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.fixture()
def mock_create_async_sqs_client_get_queue_attributes_queue_not_found():
    mock_sqs_client_session_val = AsyncMock()

    mock_sqs_client_session_val.get_queue_url = AsyncMock(
        side_effect=_get_fake_botocore_exception()
    )

    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.fixture()
def mock_create_async_sqs_client_get_queue_attributes_queue_throws_exception():
    mock_sqs_client_session_val = AsyncMock()

    mock_sqs_client_session_val.get_queue_url = AsyncMock()
    mock_sqs_client_session_val.get_queue_url.return_value = {
        "QueueUrl": "https://us-west-2.queue.amazonaws.com/000000000000/llm-engine-endpoint-id-model_endpoint_id_1"
    }

    mock_sqs_client_session_val.get_queue_attributes = AsyncMock(
        side_effect=_get_fake_botocore_exception()
    )

    mock_create_async_sqs_client = AsyncMock()
    mock_create_async_sqs_client.__aenter__ = AsyncMock(return_value=mock_sqs_client_session_val)

    with patch(
        f"{MODULE_PATH}._create_async_sqs_client",
        return_value=mock_create_async_sqs_client,
    ):
        yield mock_create_async_sqs_client


@pytest.mark.asyncio
async def test_sqs_create_or_update_resources_endpoint_exists(
    build_endpoint_request_async_custom: BuildEndpointRequest,
    mock_create_async_sqs_client_get_queue_url,
):
    delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
    endpoint_record: ModelEndpointRecord = build_endpoint_request_async_custom.model_endpoint_record
    queue_name, queue_url = await delegate.create_queue_if_not_exists(
        endpoint_id=endpoint_record.id,
        endpoint_name=endpoint_record.name,
        endpoint_created_by=endpoint_record.created_by,
        endpoint_labels=build_endpoint_request_async_custom.labels,
    )

    mock_create_async_sqs_client_get_queue_url.__aenter__.assert_called_once()

    expected_get_queue_url_args: Dict[str, Any] = {
        "QueueName": "llm-engine-endpoint-id-test_model_endpoint_id_3",
    }
    actual_get_queue_kwargs = (
        mock_create_async_sqs_client_get_queue_url.__aenter__.return_value.get_queue_url.call_args.kwargs
    )

    assert actual_get_queue_kwargs["QueueName"] == expected_get_queue_url_args["QueueName"]
    assert queue_name == actual_get_queue_kwargs["QueueName"]
    assert queue_url.endswith(actual_get_queue_kwargs["QueueName"])


@pytest.mark.asyncio
async def test_sqs_create_or_update_resources(
    build_endpoint_request_async_custom: BuildEndpointRequest,
    mock_create_async_sqs_client_create_queue,
):
    delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
    endpoint_record: ModelEndpointRecord = build_endpoint_request_async_custom.model_endpoint_record
    queue_name, queue_url = await delegate.create_queue_if_not_exists(
        endpoint_id=endpoint_record.id,
        endpoint_name=endpoint_record.name,
        endpoint_created_by=endpoint_record.created_by,
        endpoint_labels=build_endpoint_request_async_custom.labels,
    )

    mock_create_async_sqs_client_create_queue.__aenter__.assert_called_once()

    expected_create_queue_args: Dict[str, Any] = {
        "QueueName": "llm-engine-endpoint-id-test_model_endpoint_id_3",
        "Attributes": {
            "VisibilityTimeout": "3600",
            "Policy": EXPECTED_QUEUE_POLICY,
        },
        "tags": EXPECTED_QUEUE_TAGS,
    }
    actual_create_queue_kwargs = (
        mock_create_async_sqs_client_create_queue.__aenter__.return_value.create_queue.call_args.kwargs
    )

    assert actual_create_queue_kwargs["QueueName"] == expected_create_queue_args["QueueName"]
    actual_policy_json = json.loads(actual_create_queue_kwargs["Attributes"]["Policy"])
    expected_policy_json = json.loads(expected_create_queue_args["Attributes"]["Policy"])
    assert actual_policy_json == expected_policy_json
    assert actual_create_queue_kwargs["tags"] == expected_create_queue_args["tags"]

    # We're not testing that the test fixture's QueueUrl happens to be a particular value, since we've already specified
    # it. Instead, we're testing the behavior that whatever we pass to create_queue is also returned in the delegate class.
    assert queue_name == actual_create_queue_kwargs["QueueName"]
    assert queue_url.endswith(actual_create_queue_kwargs["QueueName"])


@pytest.mark.asyncio
async def test_sqs_create_or_update_resources_throws_exception(
    build_endpoint_request_async_custom: BuildEndpointRequest,
    mock_create_async_sqs_client_create_queue_throws_exception,
):
    delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
    endpoint_record: ModelEndpointRecord = build_endpoint_request_async_custom.model_endpoint_record
    with pytest.raises(EndpointResourceInfraException):
        await delegate.create_queue_if_not_exists(
            endpoint_id=endpoint_record.id,
            endpoint_name=endpoint_record.name,
            endpoint_created_by=endpoint_record.created_by,
            endpoint_labels=build_endpoint_request_async_custom.labels,
        )


@pytest.mark.asyncio
async def test_sqs_create_or_update_resources_non_200(
    build_endpoint_request_async_custom: BuildEndpointRequest,
    mock_create_async_sqs_client_create_queue_returns_non_200,
):
    delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
    endpoint_record: ModelEndpointRecord = build_endpoint_request_async_custom.model_endpoint_record
    with pytest.raises(EndpointResourceInfraException):
        await delegate.create_queue_if_not_exists(
            endpoint_id=endpoint_record.id,
            endpoint_name=endpoint_record.name,
            endpoint_created_by=endpoint_record.created_by,
            endpoint_labels=build_endpoint_request_async_custom.labels,
        )


@pytest.mark.asyncio
async def test_sqs_delete_resources(mock_create_async_sqs_client_delete_queue):
    delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
    await delegate.delete_queue(endpoint_id="model_endpoint_id_1")

    mock_create_async_sqs_client_delete_queue.__aenter__.assert_called_once()
    mock_create_async_sqs_client_delete_queue.__aenter__.return_value.get_queue_url.assert_called_once_with(
        QueueName="llm-engine-endpoint-id-model_endpoint_id_1"
    )

    delete_call_kwargs = (
        mock_create_async_sqs_client_delete_queue.__aenter__.return_value.delete_queue.call_args.kwargs
    )
    assert delete_call_kwargs["QueueUrl"].endswith("llm-engine-endpoint-id-model_endpoint_id_1")


@pytest.mark.asyncio
async def test_sqs_delete_resources_throws_exception(
    mock_create_async_sqs_client_delete_queue_throws_exception,
):
    with pytest.raises(EndpointResourceInfraException):
        delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
        await delegate.delete_queue(endpoint_id="model_endpoint_id_1")


@pytest.mark.asyncio
async def test_sqs_delete_resources_non_200(
    mock_create_async_sqs_client_delete_queue_returns_non_200,
):
    with pytest.raises(EndpointResourceInfraException):
        delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
        await delegate.delete_queue(endpoint_id="model_endpoint_id_1")


@pytest.mark.asyncio
async def test_sqs_get_queue_attributes(
    mock_create_async_sqs_client_get_queue_attributes,
):
    delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
    response = await delegate.get_queue_attributes(endpoint_id="model_endpoint_id_1")

    mock_create_async_sqs_client_get_queue_attributes.__aenter__.assert_called_once()
    mock_create_async_sqs_client_get_queue_attributes.__aenter__.return_value.get_queue_url.assert_called_once_with(
        QueueName="llm-engine-endpoint-id-model_endpoint_id_1"
    )

    get_queue_attributes_call_kwargs = (
        mock_create_async_sqs_client_get_queue_attributes.__aenter__.return_value.get_queue_attributes.call_args.kwargs
    )
    assert get_queue_attributes_call_kwargs["QueueUrl"].endswith(
        "llm-engine-endpoint-id-model_endpoint_id_1"
    )

    assert response["Attributes"]["QueueArn"].endswith("llm-engine-endpoint-id-model_endpoint_id_1")


@pytest.mark.asyncio
async def test_sqs_get_queue_attributes_queue_not_found(
    mock_create_async_sqs_client_get_queue_attributes_queue_not_found,
):
    with pytest.raises(EndpointResourceInfraException):
        delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
        await delegate.get_queue_attributes(endpoint_id="model_endpoint_id_1")


@pytest.mark.asyncio
async def test_sqs_get_queue_attributes_queue_throws_exception(
    mock_create_async_sqs_client_get_queue_attributes_queue_throws_exception,
):
    with pytest.raises(EndpointResourceInfraException):
        delegate = LiveSQSEndpointResourceDelegate(sqs_profile="foobar")
        await delegate.get_queue_attributes(endpoint_id="model_endpoint_id_1")
