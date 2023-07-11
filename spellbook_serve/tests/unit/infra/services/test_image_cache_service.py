from typing import Any

import pytest

from spellbook_serve.infra.services.image_cache_service import ImageCacheService


@pytest.mark.asyncio
async def test_image_cache_success(
    fake_image_cache_service: ImageCacheService,
    model_endpoint_1,
    model_endpoint_2,
    model_endpoint_3,
    model_endpoint_4,
):
    infra_states = {
        model_endpoint_1.record.id: (bool, model_endpoint_1.infra_state),
        model_endpoint_2.record.id: (bool, model_endpoint_2.infra_state),
        model_endpoint_3.record.id: (bool, model_endpoint_3.infra_state),
        model_endpoint_4.record.id: (bool, model_endpoint_4.infra_state),
    }
    repo: Any = fake_image_cache_service.model_endpoint_record_repository
    repo.add_model_endpoint_record(model_endpoint_1.record)
    repo.add_model_endpoint_record(model_endpoint_2.record)
    repo.add_model_endpoint_record(model_endpoint_3.record)
    repo.add_model_endpoint_record(model_endpoint_4.record)

    await fake_image_cache_service.execute(infra_states)  # type: ignore
    gateway: Any = fake_image_cache_service.image_cache_gateway
    assert gateway.cached_images == {
        "a10": [],
        "a100": [],
        "cpu": [],
        "t4": [
            "692474966980.dkr.ecr.us-west-2.amazonaws.com/catalog-gpu:40d3b5fb06d1a8c3d14903390a3b23ae388bdb19",
            "692474966980.dkr.ecr.us-west-2.amazonaws.com/catalog-gpu:e4ea48ddccfb9ca3ef6d846ae9b2d146d7e30b0f",
            "692474966980.dkr.ecr.us-west-2.amazonaws.com/catalog-gpu:9a319cd9b897f02291f3242b1395f2b669993cdf-fd",
        ],
    }
