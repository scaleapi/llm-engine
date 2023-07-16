from typing import Any, Dict, List, Optional, Sequence

from spellbook_serve.common import dict_not_none
from spellbook_serve.common.dtos.model_bundles import ModelBundleOrderBy
from spellbook_serve.db.models import Bundle as OrmModelBundle
from spellbook_serve.domain.entities import (
    ModelBundle,
    ModelBundleFlavors,
    ModelBundlePackagingType,
)
from spellbook_serve.domain.repositories import ModelBundleRepository
from spellbook_serve.infra.repositories.db_repository_mixin import (
    DbRepositoryMixin,
    raise_if_read_only,
)

__all__: Sequence[str] = ("DbModelBundleRepository",)


class DbModelBundleRepository(ModelBundleRepository, DbRepositoryMixin):
    """
    Implementation of a ModelBundleRepository that is backed by a relational database.
    """

    @raise_if_read_only
    async def create_model_bundle(
        self,
        *,
        name: str,
        created_by: str,
        owner: str,
        model_artifact_ids: List[str],
        schema_location: Optional[str],
        metadata: Dict[str, Any],
        flavor: ModelBundleFlavors,
        # LEGACY FIELDS
        location: str,
        requirements: List[str],
        env_params: Dict[str, Any],
        packaging_type: ModelBundlePackagingType,
        app_config: Optional[Dict[str, Any]],
    ) -> ModelBundle:
        model_bundle_record = translate_kwargs_to_model_bundle_orm(
            name=name,
            created_by=created_by,
            owner=owner,
            model_artifact_ids=model_artifact_ids,
            schema_location=schema_location,
            metadata=metadata,
            flavor=flavor,
            # LEGACY FIELDS
            location=location,
            requirements=requirements,
            env_params=env_params,
            packaging_type=packaging_type,
            app_config=app_config,
        )
        async with self.session() as session:
            await OrmModelBundle.create(session, model_bundle_record)
            model_bundle_record = await OrmModelBundle.select_by_id(  # type: ignore
                session=session, bundle_id=model_bundle_record.id
            )
        return translate_model_bundle_orm_to_model_bundle(model_bundle_record)

    async def list_model_bundles(
        self, owner: str, name: Optional[str], order_by: Optional[ModelBundleOrderBy]
    ) -> Sequence[ModelBundle]:
        async with self.session() as session:
            if name is not None:
                model_bundle_records = await OrmModelBundle.select_all_by_name_owner(
                    session=session, name=name, owner=owner
                )
            else:
                model_bundle_records = await OrmModelBundle.select_all_by_owner(
                    session=session, owner=owner
                )
        model_bundles = [
            translate_model_bundle_orm_to_model_bundle(mb) for mb in model_bundle_records
        ]

        # TODO(phil): we could use an ORDER_BY operation in the DB instead.
        if order_by == ModelBundleOrderBy.NEWEST:
            model_bundles.sort(key=lambda x: x.created_at, reverse=True)
        elif order_by == ModelBundleOrderBy.OLDEST:
            model_bundles.sort(key=lambda x: x.created_at, reverse=False)

        return model_bundles

    async def get_model_bundle(self, model_bundle_id: str) -> Optional[ModelBundle]:
        async with self.session() as session:
            model_bundle_record = await OrmModelBundle.select_by_id(
                session=session, bundle_id=model_bundle_id
            )
        if not model_bundle_record:
            return None

        return translate_model_bundle_orm_to_model_bundle(model_bundle_record)

    async def get_latest_model_bundle_by_name(self, owner: str, name: str) -> Optional[ModelBundle]:
        async with self.session() as session:
            model_bundle_record = await OrmModelBundle.select_by_name_owner(
                session=session, name=name, owner=owner
            )
        if not model_bundle_record:
            return None
        return translate_model_bundle_orm_to_model_bundle(model_bundle_record)


def translate_model_bundle_orm_to_model_bundle(
    model_bundle_orm: OrmModelBundle,
) -> ModelBundle:
    kwargs = dict_not_none(
        id=model_bundle_orm.id,
        created_at=model_bundle_orm.created_at,
        name=model_bundle_orm.name,
        created_by=model_bundle_orm.created_by,
        owner=model_bundle_orm.owner,
        metadata=model_bundle_orm.bundle_metadata,
        model_artifact_ids=model_bundle_orm.model_artifact_ids or [],
        schema_location=model_bundle_orm.schema_location,
        flavor=dict_not_none(
            flavor=model_bundle_orm.flavor,
            requirements=model_bundle_orm.artifact_requirements,
            location=model_bundle_orm.artifact_location,
            framework=None
            if model_bundle_orm.artifact_framework_type is None
            else dict_not_none(
                framework_type=model_bundle_orm.artifact_framework_type,
                pytorch_image_tag=model_bundle_orm.artifact_pytorch_image_tag,
                tensorflow_version=model_bundle_orm.artifact_tensorflow_version,
                image_repository=model_bundle_orm.artifact_image_repository,
                image_tag=model_bundle_orm.artifact_image_tag,
            ),
            app_config=model_bundle_orm.artifact_app_config,
            load_predict_fn=model_bundle_orm.cloudpickle_artifact_load_predict_fn,
            load_model_fn=model_bundle_orm.cloudpickle_artifact_load_model_fn,
            load_predict_fn_module_path=model_bundle_orm.zip_artifact_load_predict_fn_module_path,
            load_model_fn_module_path=model_bundle_orm.zip_artifact_load_model_fn_module_path,
            repository=model_bundle_orm.runnable_image_repository,
            tag=model_bundle_orm.runnable_image_tag,
            command=model_bundle_orm.runnable_image_command,
            predict_route=model_bundle_orm.runnable_image_predict_route,
            healthcheck_route=model_bundle_orm.runnable_image_healthcheck_route,
            env=model_bundle_orm.runnable_image_env,
            protocol=model_bundle_orm.runnable_image_protocol,
            readiness_initial_delay_seconds=model_bundle_orm.runnable_image_readiness_initial_delay_seconds,
            streaming_command=model_bundle_orm.streaming_enhanced_runnable_image_streaming_command,
            streaming_predict_route=model_bundle_orm.streaming_enhanced_runnable_image_streaming_predict_route,
            triton_model_repository=model_bundle_orm.triton_enhanced_runnable_image_model_repository,
            triton_model_replicas=model_bundle_orm.triton_enhanced_runnable_image_model_replicas,
            triton_num_cpu=model_bundle_orm.triton_enhanced_runnable_image_num_cpu,
            triton_commit_tag=model_bundle_orm.triton_enhanced_runnable_image_commit_tag,
            triton_storage=model_bundle_orm.triton_enhanced_runnable_image_storage,
            triton_memory=model_bundle_orm.triton_enhanced_runnable_image_memory,
            triton_readiness_initial_delay_seconds=model_bundle_orm.triton_enhanced_runnable_image_readiness_initial_delay_seconds,
        ),
        # LEGACY FIELDS
        requirements=model_bundle_orm.requirements,
        location=model_bundle_orm.location,
        env_params=model_bundle_orm.env_params,
        packaging_type=model_bundle_orm.packaging_type,
        app_config=model_bundle_orm.app_config,
    )
    return ModelBundle.parse_obj(kwargs)


def translate_kwargs_to_model_bundle_orm(
    name: str,
    created_by: str,
    owner: str,
    model_artifact_ids: List[str],
    schema_location: Optional[str],
    metadata: Dict[str, Any],
    flavor: ModelBundleFlavors,
    # LEGACY FIELDS
    location: str,
    requirements: List[str],
    env_params: Dict[str, Any],
    packaging_type: ModelBundlePackagingType,
    app_config: Optional[Dict[str, Any]],
) -> OrmModelBundle:
    kwargs = locals()
    flavor_dict = flavor.dict()
    framework_dict = flavor_dict.get("framework", {})
    return OrmModelBundle(
        name=name,
        created_by=created_by,
        owner=owner,
        bundle_metadata=metadata,
        model_artifact_ids=model_artifact_ids,
        schema_location=schema_location,
        flavor=flavor_dict.get("flavor").value,  # type: ignore
        artifact_requirements=flavor_dict.get("requirements"),
        artifact_location=flavor_dict.get("location"),
        artifact_app_config=flavor_dict.get("app_config"),
        artifact_framework_type=framework_dict.get("framework_type"),
        artifact_pytorch_image_tag=framework_dict.get("pytorch_image_tag"),
        artifact_tensorflow_version=framework_dict.get("tensorflow_version"),
        artifact_image_tag=framework_dict.get("image_tag"),
        artifact_image_repository=framework_dict.get("image_repository"),
        cloudpickle_artifact_load_predict_fn=flavor_dict.get("load_predict_fn"),
        cloudpickle_artifact_load_model_fn=flavor_dict.get("load_model_fn"),
        zip_artifact_load_predict_fn_module_path=flavor_dict.get("load_predict_fn_module_path"),
        zip_artifact_load_model_fn_module_path=flavor_dict.get("load_model_fn_module_path"),
        runnable_image_repository=flavor_dict.get("repository"),
        runnable_image_tag=flavor_dict.get("tag"),
        runnable_image_command=flavor_dict.get("command"),
        runnable_image_predict_route=flavor_dict.get("predict_route"),
        runnable_image_healthcheck_route=flavor_dict.get("healthcheck_route"),
        runnable_image_env=flavor_dict.get("env"),
        runnable_image_protocol=flavor_dict.get("protocol"),
        runnable_image_readiness_initial_delay_seconds=flavor_dict.get(
            "readiness_initial_delay_seconds"
        ),
        streaming_enhanced_runnable_image_streaming_command=flavor_dict.get("streaming_command"),
        streaming_enhanced_runnable_image_streaming_predict_route=flavor_dict.get(
            "streaming_predict_route"
        ),
        triton_enhanced_runnable_image_model_repository=flavor_dict.get("triton_model_repository"),
        triton_enhanced_runnable_image_model_replicas=flavor_dict.get("triton_model_replicas"),
        triton_enhanced_runnable_image_num_cpu=flavor_dict.get("triton_num_cpu"),
        triton_enhanced_runnable_image_commit_tag=flavor_dict.get("triton_commit_tag"),
        triton_enhanced_runnable_image_storage=flavor_dict.get("triton_storage"),
        triton_enhanced_runnable_image_memory=flavor_dict.get("triton_memory"),
        triton_enhanced_runnable_image_readiness_initial_delay_seconds=flavor_dict.get(
            "triton_readiness_initial_delay_seconds"
        ),
        # LEGACY FIELDS
        location=location,
        requirements=requirements,
        env_params=env_params,
        packaging_type=packaging_type,
        app_config=app_config,
    )
