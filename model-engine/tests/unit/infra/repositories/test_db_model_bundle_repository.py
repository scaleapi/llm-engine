import datetime
from typing import Callable, List, Optional
from unittest.mock import AsyncMock

import pytest
from model_engine_server.common.dtos.model_bundles import ModelBundleOrderBy
from model_engine_server.core.domain_exceptions import ReadOnlyDatabaseException
from model_engine_server.db.models import Bundle
from model_engine_server.domain.entities import (
    CloudpickleArtifactFlavor,
    ModelBundle,
    ModelBundlePackagingType,
    PytorchFramework,
)
from model_engine_server.infra.repositories.db_model_bundle_repository import (
    DbModelBundleRepository,
    OrmModelBundle,
)
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_create_model_bundle(dbsession: Callable[[], AsyncSession], orm_model_bundle: Bundle):
    def mock_model_bundle_create(session: AsyncSession, bundle: Bundle) -> None:
        bundle.id = "test_model_bundle_id"
        bundle.created_at = datetime.datetime(2022, 1, 1)

    def mock_model_bundle_select_by_id(session: AsyncSession, bundle_id: str) -> Optional[Bundle]:
        orm_model_bundle.bundle_id = bundle_id
        return orm_model_bundle

    OrmModelBundle.select_by_id = AsyncMock(side_effect=mock_model_bundle_select_by_id)
    OrmModelBundle.create = AsyncMock(side_effect=mock_model_bundle_create)

    repo = DbModelBundleRepository(session=dbsession, read_only=False)
    model_bundle = await repo.create_model_bundle(
        name="test_model_bundle_name_1",
        created_by="test_user_id",
        owner="test_user_id",
        model_artifact_ids=["test_model_artifact_id"],
        schema_location="test_schema_location",
        metadata={},
        flavor=CloudpickleArtifactFlavor(
            flavor="cloudpickle_artifact",
            framework=PytorchFramework(
                framework_type="pytorch",
                pytorch_image_tag="test_tag",
            ),
            requirements=["numpy==0.0.0"],
            location="test_location",
            app_config=None,
            load_predict_fn="test_load_predict_fn",
            load_model_fn="test_load_model_fn",
        ),
        # LEGACY FIELDS
        location="test_location",
        requirements=[],
        env_params={"framework_type": "pytorch", "pytorch_image_tag": "0.0.0"},
        packaging_type=ModelBundlePackagingType.CLOUDPICKLE,
        app_config=None,
    )

    assert model_bundle


@pytest.mark.asyncio
async def test_create_model_bundle_raises_if_read_only(
    dbsession: Callable[[], AsyncSession], orm_model_bundle: Bundle
):
    def mock_model_bundle_create(session: AsyncSession, bundle: Bundle) -> None:
        bundle.id = "test_model_bundle_id"
        bundle.created_at = datetime.datetime(2022, 1, 1)

    def mock_model_bundle_select_by_id(session: AsyncSession, bundle_id: str) -> Optional[Bundle]:
        orm_model_bundle.bundle_id = bundle_id
        return orm_model_bundle

    OrmModelBundle.select_by_id = AsyncMock(side_effect=mock_model_bundle_select_by_id)
    OrmModelBundle.create = AsyncMock(side_effect=mock_model_bundle_create)

    repo = DbModelBundleRepository(session=dbsession, read_only=True)
    with pytest.raises(ReadOnlyDatabaseException):
        await repo.create_model_bundle(
            name="test_model_bundle_name_1",
            created_by="test_user_id",
            owner="test_user_id",
            model_artifact_ids=["test_model_artifact_id"],
            schema_location="test_schema_location",
            metadata={},
            flavor=CloudpickleArtifactFlavor(
                flavor="cloudpickle_artifact",
                framework=PytorchFramework(
                    framework_type="pytorch",
                    pytorch_image_tag="test_tag",
                ),
                requirements=["numpy==0.0.0"],
                location="test_location",
                app_config=None,
                load_predict_fn="test_load_predict_fn",
                load_model_fn="test_load_model_fn",
            ),
            # LEGACY FIELDS
            location="test_location",
            requirements=[],
            env_params={"framework_type": "pytorch", "pytorch_image_tag": "0.0.0"},
            packaging_type=ModelBundlePackagingType.CLOUDPICKLE,
            app_config=None,
        )


@pytest.mark.asyncio
async def test_list_model_bundles(
    dbsession: Callable[[], AsyncSession],
    orm_model_bundle: Bundle,
    model_bundle_1: ModelBundle,
):
    def mock_model_bundle_select_all_by_name_owner(
        session: AsyncSession, owner: str, name: str
    ) -> List[Bundle]:
        orm_model_bundle.name = name
        orm_model_bundle.owner = owner
        return [orm_model_bundle]

    def mock_model_bundle_select_all_by_owner(session: AsyncSession, owner: str) -> List[Bundle]:
        orm_model_bundle.created_by = owner
        return [orm_model_bundle]

    OrmModelBundle.select_all_by_name_owner = AsyncMock(
        side_effect=mock_model_bundle_select_all_by_name_owner
    )
    OrmModelBundle.select_all_by_owner = AsyncMock(
        side_effect=mock_model_bundle_select_all_by_owner
    )

    repo = DbModelBundleRepository(session=dbsession, read_only=True)
    model_bundles = await repo.list_model_bundles(
        owner="test_user_id",
        name="test_model_bundle_name_1",
        order_by=ModelBundleOrderBy.NEWEST,
    )
    # Use dict comparison because pytest can pretty-print dicts if they are different
    assert len(model_bundles) == 1
    assert model_bundles[0].dict() == model_bundle_1.dict()

    model_bundles = await repo.list_model_bundles(
        owner="test_user_id", name=None, order_by=ModelBundleOrderBy.OLDEST
    )
    assert len(model_bundles) == 1
    assert model_bundles[0].dict() == model_bundle_1.dict()


@pytest.mark.asyncio
async def test_list_model_bundles_team(
    dbsession: Callable[[], AsyncSession],
    orm_model_bundle: Bundle,
    orm_model_bundle_2: Bundle,
    orm_model_bundle_3: Bundle,
    orm_model_bundle_4: Bundle,
    orm_model_bundle_5: Bundle,
    model_bundle_1: ModelBundle,
    model_bundle_2: ModelBundle,
    model_bundle_3: ModelBundle,
    model_bundle_4: ModelBundle,
    model_bundle_5: ModelBundle,
    test_api_key_user_on_other_team: str,
    test_api_key_user_on_other_team_2: str,
    test_api_key_team: str,
):
    orm_model_bundle.created_by = test_api_key_user_on_other_team
    orm_model_bundle.owner = test_api_key_team
    orm_model_bundle_2.created_by = test_api_key_user_on_other_team_2
    orm_model_bundle_2.owner = test_api_key_team
    orm_model_bundle_3.created_by = test_api_key_user_on_other_team_2
    orm_model_bundle_3.owner = test_api_key_team
    orm_model_bundle_4.created_by = test_api_key_user_on_other_team_2
    orm_model_bundle_4.owner = test_api_key_team
    orm_model_bundle_5.created_by = test_api_key_user_on_other_team_2
    orm_model_bundle_5.owner = test_api_key_team

    model_bundle_1.created_by = test_api_key_user_on_other_team
    model_bundle_1.owner = test_api_key_team
    model_bundle_2.created_by = test_api_key_user_on_other_team_2
    model_bundle_2.owner = test_api_key_team
    model_bundle_3.created_by = test_api_key_user_on_other_team_2
    model_bundle_3.owner = test_api_key_team
    model_bundle_4.created_by = test_api_key_user_on_other_team_2
    model_bundle_4.owner = test_api_key_team
    model_bundle_5.created_by = test_api_key_user_on_other_team_2
    model_bundle_5.owner = test_api_key_team

    def mock_model_bundle_select_all_by_name_owner(
        session: AsyncSession, owner: str, name: str
    ) -> List[Bundle]:
        if owner != test_api_key_team:
            return []
        if name == model_bundle_1.name:
            return [orm_model_bundle]
        elif name == model_bundle_2.name:
            return [orm_model_bundle_2]
        elif name == model_bundle_3.name:
            return [orm_model_bundle_3]
        elif name == model_bundle_4.name:
            return [orm_model_bundle_4]
        elif name == model_bundle_5.name:
            return [orm_model_bundle_5]
        else:
            return []

    def mock_model_bundle_select_all_by_owner(session: AsyncSession, owner: str) -> List[Bundle]:
        if owner != test_api_key_team:
            return []
        return [
            orm_model_bundle,
            orm_model_bundle_2,
            orm_model_bundle_3,
            orm_model_bundle_4,
            orm_model_bundle_5,
        ]

    OrmModelBundle.select_all_by_name_owner = AsyncMock(
        side_effect=mock_model_bundle_select_all_by_name_owner
    )
    OrmModelBundle.select_all_by_owner = AsyncMock(
        side_effect=mock_model_bundle_select_all_by_owner
    )

    repo = DbModelBundleRepository(session=dbsession, read_only=True)
    model_bundles = await repo.list_model_bundles(
        owner=test_api_key_team,
        name=model_bundle_1.name,
        order_by=ModelBundleOrderBy.NEWEST,
    )
    assert model_bundles == [model_bundle_1]

    model_bundles = await repo.list_model_bundles(
        owner=test_api_key_team, name=None, order_by=ModelBundleOrderBy.OLDEST
    )
    # Use dict comparison because pytest can pretty-print dicts if they are different
    assert len(model_bundles) == 5
    assert model_bundles[0].dict() == model_bundle_1.dict()
    assert model_bundles[1].dict() == model_bundle_2.dict()
    assert model_bundles[2].dict() == model_bundle_3.dict()
    assert model_bundles[3].dict() == model_bundle_4.dict()
    assert model_bundles[4].dict() == model_bundle_5.dict()


@pytest.mark.asyncio
async def test_get_by_id_success(
    dbsession: Callable[[], AsyncSession],
    orm_model_bundle: Bundle,
    model_bundle_1: ModelBundle,
):
    def mock_model_bundle_select_by_id(session: AsyncSession, bundle_id: str) -> Optional[Bundle]:
        orm_model_bundle.bundle_id = bundle_id
        return orm_model_bundle

    OrmModelBundle.select_by_id = AsyncMock(side_effect=mock_model_bundle_select_by_id)

    repo = DbModelBundleRepository(session=dbsession, read_only=True)
    model_bundle = await repo.get_model_bundle(model_bundle_id="test_model_bundle_id")

    assert model_bundle is not None
    assert model_bundle.dict() == model_bundle_1.dict()


@pytest.mark.asyncio
async def test_get_by_id_returns_none(dbsession: Callable[[], AsyncSession]):
    OrmModelBundle.select_by_id = AsyncMock(return_value=None)

    repo = DbModelBundleRepository(session=dbsession, read_only=True)
    model_bundle = await repo.get_model_bundle(model_bundle_id="test_model_bundle_id")

    assert model_bundle is None


@pytest.mark.asyncio
async def test_get_latest_model_bundle_by_name_success(
    dbsession: Callable[[], AsyncSession],
    orm_model_bundle: Bundle,
    model_bundle_1: ModelBundle,
):
    def mock_model_bundle_select_by_name_owner(
        session: AsyncSession, owner: str, name: str
    ) -> Optional[Bundle]:
        orm_model_bundle.name = name
        orm_model_bundle.owner = owner
        return orm_model_bundle

    OrmModelBundle.select_by_name_owner = AsyncMock(
        side_effect=mock_model_bundle_select_by_name_owner
    )

    repo = DbModelBundleRepository(session=dbsession, read_only=True)
    model_bundle = await repo.get_latest_model_bundle_by_name(
        owner="test_user_id", name="test_model_bundle_name_1"
    )

    assert model_bundle.dict() == model_bundle_1.dict()


@pytest.mark.asyncio
async def test_get_latest_model_bundle_by_name_success_team(
    dbsession: Callable[[], AsyncSession],
    orm_model_bundle: Bundle,
    model_bundle_1: ModelBundle,
    test_api_key_user_on_other_team: str,
    test_api_key_team: str,
):
    orm_model_bundle.created_by = test_api_key_user_on_other_team
    orm_model_bundle.owner = test_api_key_team
    model_bundle_1.created_by = test_api_key_user_on_other_team
    model_bundle_1.owner = test_api_key_team

    def mock_model_bundle_select_by_name_owner(
        session: AsyncSession, owner: str, name: str
    ) -> Optional[Bundle]:
        orm_model_bundle.name = name
        orm_model_bundle.owner = owner
        return orm_model_bundle

    OrmModelBundle.select_by_name_owner = AsyncMock(
        side_effect=mock_model_bundle_select_by_name_owner
    )

    repo = DbModelBundleRepository(session=dbsession, read_only=True)
    model_bundle = await repo.get_latest_model_bundle_by_name(
        owner=test_api_key_team, name="test_model_bundle_name_1"
    )

    assert model_bundle == model_bundle_1


@pytest.mark.asyncio
async def test_get_latest_model_bundle_by_name_returns_none(dbsession: Callable[[], AsyncSession]):
    OrmModelBundle.select_by_name_owner = AsyncMock(return_value=None)

    repo = DbModelBundleRepository(session=dbsession, read_only=True)
    model_bundle = await repo.get_latest_model_bundle_by_name(
        owner="test_user_id", name="test_model_bundle_name_1"
    )

    assert model_bundle is None
