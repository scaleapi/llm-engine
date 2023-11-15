from typing import Any, List
from unittest import mock

import pytest
from model_engine_server.infra.repositories.live_tokenizer_repository import (
    LiveTokenizerRepository,
    ModelInfo,
)


@pytest.fixture
def tokenizer_repository(fake_llm_artifact_gateway):
    repository = LiveTokenizerRepository(fake_llm_artifact_gateway)
    return repository


def mocked_get_models_s3_uri(*args, **kwargs):  # noqa
    return f"s3://fake-bucket/{args[0]}/{args[1]}"


def mocked_auto_tokenizer_from_pretrained(*args, **kwargs):  # noqa
    class mocked_encode:
        def encode(self, input: str) -> List[Any]:
            return [1] * len(input)

    return mocked_encode()


@mock.patch(
    "model_engine_server.infra.repositories.live_tokenizer_repository.SUPPORTED_MODELS_INFO",
    {"llama-7b": ModelInfo("llama-7b", None)},
)
@mock.patch(
    "model_engine_server.infra.repositories.live_tokenizer_repository.list_repo_refs",
    lambda *args, **kwargs: None,  # noqa
)
@mock.patch(
    "model_engine_server.infra.repositories.live_tokenizer_repository.AutoTokenizer.from_pretrained",
    mocked_auto_tokenizer_from_pretrained,
)
def test_load_tokenizer_from_hf(tokenizer_repository):
    tokenizer = tokenizer_repository.load_tokenizer("llama-7b")

    assert tokenizer.encode("fake input") == [1] * len("fake input")


@mock.patch(
    "model_engine_server.infra.repositories.live_tokenizer_repository.SUPPORTED_MODELS_INFO",
    {"llama-7b": ModelInfo(None, "llama-7b")},
)
@mock.patch(
    "model_engine_server.infra.repositories.live_tokenizer_repository.get_models_s3_uri",
    mocked_get_models_s3_uri,
)
@mock.patch(
    "model_engine_server.infra.repositories.live_tokenizer_repository.AutoTokenizer.from_pretrained",
    mocked_auto_tokenizer_from_pretrained,
)
def test_load_tokenizer_from_s3(tokenizer_repository):
    tokenizer = tokenizer_repository.load_tokenizer("llama-7b")

    assert tokenizer.encode("fake input") == [1] * len("fake input")
