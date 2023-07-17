from typing import Optional, Union
from uuid import uuid4

from llm_engine_server.common.dtos.model_bundles import (
    CloneModelBundleV1Request,
    CloneModelBundleV2Request,
    CreateModelBundleV1Request,
    CreateModelBundleV1Response,
    CreateModelBundleV2Request,
    CreateModelBundleV2Response,
    ListModelBundlesV1Response,
    ListModelBundlesV2Response,
    ModelBundleOrderBy,
    ModelBundlePackagingType,
    ModelBundleV1Response,
    ModelBundleV2Response,
)
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import (
    DockerImageNotFoundException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from llm_engine_server.domain.authorization.scale_authorization_module import (
    ScaleAuthorizationModule,
)
from llm_engine_server.domain.entities import (
    ArtifactLike,
    CloudpickleArtifactFlavor,
    CustomFramework,
    ModelBundleFlavors,
    ModelBundleFlavorType,
    ModelBundleFrameworkType,
    PytorchFramework,
    RunnableImageFlavor,
    RunnableImageLike,
    TensorflowFramework,
    ZipArtifactFlavor,
)
from llm_engine_server.domain.gateways import ModelPrimitiveGateway
from llm_engine_server.domain.repositories import DockerRepository, ModelBundleRepository


class CreateModelBundleV1UseCase:
    """
    Use case for creating a Model Bundle.
    """

    def __init__(
        self,
        model_bundle_repository: ModelBundleRepository,
        docker_repository: DockerRepository,
        model_primitive_gateway: ModelPrimitiveGateway,
    ):
        self.authz_module = ScaleAuthorizationModule()
        self.model_bundle_repository = model_bundle_repository
        self.docker_repository = docker_repository
        self.model_primitive_gateway = model_primitive_gateway

    async def execute(
        self, user: User, request: CreateModelBundleV1Request
    ) -> CreateModelBundleV1Response:
        """
        Runs the use case to create a Model Bundle.

        Args:
            user: The user who is creating the Model Bundle.
            request: A request object that contains the creation fields.

        Returns:
            A response object that contains the creation response fields.

        Raises:
            DockerImageNotFoundException: If a model bundle specifies a Docker image that cannot be
                found.
            ObjectNotAuthorizedException: If a user is not authorized to perform the operation.
        """
        if request.env_params.framework_type == ModelBundleFrameworkType.CUSTOM:
            # This should always pass due to Pydantic validation.
            assert request.env_params.ecr_repo and request.env_params.image_tag
            if not self.docker_repository.image_exists(
                image_tag=request.env_params.image_tag,
                repository_name=request.env_params.ecr_repo,
            ):
                raise DockerImageNotFoundException(
                    repository=request.env_params.ecr_repo,
                    tag=request.env_params.image_tag,
                )
        created_by = user.user_id
        owner = user.team_id
        model_artifact_id = await self.model_primitive_gateway.create_model_artifact(
            # guarantee uniqueness
            model_artifact_name=f"{request.name}-{str(uuid4())[-8:]}",
            location=request.location,
            framework_type=request.env_params.framework_type,
            created_by=created_by,
        )
        model_artifact_ids = [model_artifact_id] if model_artifact_id else []

        if not self.authz_module.check_access_create_bundle_v1(user, request):
            raise ObjectNotAuthorizedException

        # Convert the framework type to an object
        framework: Optional[Union[PytorchFramework, TensorflowFramework, CustomFramework]] = None
        if request.env_params.framework_type == ModelBundleFrameworkType.PYTORCH:
            framework = PytorchFramework(
                framework_type=ModelBundleFrameworkType.PYTORCH,
                pytorch_image_tag=str(request.env_params.pytorch_image_tag),
            )
        elif request.env_params.framework_type == ModelBundleFrameworkType.TENSORFLOW:
            framework = TensorflowFramework(
                framework_type=ModelBundleFrameworkType.TENSORFLOW,
                tensorflow_version=str(request.env_params.tensorflow_version),
            )
        elif request.env_params.framework_type == ModelBundleFrameworkType.CUSTOM:
            framework = CustomFramework(
                framework_type=ModelBundleFrameworkType.CUSTOM,
                image_repository=str(request.env_params.ecr_repo),
                image_tag=str(request.env_params.image_tag),
            )

        # Convert the flavor to an object
        flavor: ModelBundleFlavors
        if request.packaging_type == ModelBundlePackagingType.CLOUDPICKLE:
            assert framework is not None
            metadata = request.metadata or {}
            flavor = CloudpickleArtifactFlavor(
                flavor=ModelBundleFlavorType.CLOUDPICKLE_ARTIFACT,
                requirements=request.requirements,
                framework=framework,
                app_config=request.app_config,
                location=request.location,
                load_predict_fn=metadata.get("load_predict_fn", ""),
                load_model_fn=metadata.get("load_model_fn", ""),
            )
        elif request.packaging_type == ModelBundlePackagingType.ZIP:
            assert framework is not None
            metadata = request.metadata or {}
            flavor = ZipArtifactFlavor(
                flavor=ModelBundleFlavorType.ZIP_ARTIFACT,
                requirements=request.requirements,
                framework=framework,
                app_config=request.app_config,
                location=request.location,
                load_predict_fn_module_path=metadata.get("load_predict_fn_module_path", ""),
                load_model_fn_module_path=metadata.get("load_model_fn_module_path", ""),
            )
        else:  # request.packaging_type == ModelBundlePackagingType.CLOUDPICKLE:
            flavor = RunnableImageFlavor(
                flavor=ModelBundleFlavorType.RUNNABLE_IMAGE,
                repository="",  # stub value, not used
                tag="",  # stub value, not used
                command=[],  # stub value, not used
                env=None,  # stub value, not used
                protocol="http",
                readiness_initial_delay_seconds=30,  # stub value, not used
            )

        model_bundle = await self.model_bundle_repository.create_model_bundle(
            name=request.name,
            created_by=created_by,
            owner=owner,
            model_artifact_ids=model_artifact_ids,
            schema_location=request.schema_location,
            metadata=request.metadata or {},
            flavor=flavor,
            # LEGACY FIELDS
            location=request.location,
            requirements=request.requirements,
            env_params=request.env_params.dict(),
            packaging_type=request.packaging_type or ModelBundlePackagingType.CLOUDPICKLE,
            app_config=request.app_config,
        )
        response = CreateModelBundleV1Response(model_bundle_id=model_bundle.id)
        return response


class CloneModelBundleV1UseCase:
    """
    Use case for cloning a Model Bundle from an existing one.
    """

    def __init__(self, model_bundle_repository: ModelBundleRepository):
        self.model_bundle_repository = model_bundle_repository
        self.authz_module = ScaleAuthorizationModule()

    async def execute(
        self,
        user: User,
        request: CloneModelBundleV1Request,
    ) -> CreateModelBundleV1Response:
        """
        Runs the use case to clone a Model Bundle.

        Args:
            user: The user who is cloning the Model Bundle.
            request: A request object that contains the fields for cloning.

        Returns:
            A response object that contains the creation response fields.

        Raises:
            ObjectNotAuthorizedException: If a user is not authorized to perform the operation.
            ObjectNotFoundException: If the source Model Bundle could not be found.
        """

        original_bundle = await self.model_bundle_repository.get_model_bundle(
            model_bundle_id=request.original_model_bundle_id,
        )
        if original_bundle is None:
            raise ObjectNotFoundException

        # Owning a bundle is necessary and sufficient for cloning it.
        if not self.authz_module.check_access_read_owned_entity(user, original_bundle):
            raise ObjectNotAuthorizedException

        new_metadata = original_bundle.metadata.copy()
        new_metadata["__cloned_from_bundle_id"] = original_bundle.id

        env_params_dict = {}
        if (env_params := original_bundle.env_params) is not None:
            env_params_dict = env_params.dict()

        new_model_bundle = await self.model_bundle_repository.create_model_bundle(
            name=original_bundle.name,
            created_by=user.user_id,
            owner=original_bundle.owner,
            model_artifact_ids=original_bundle.model_artifact_ids,
            schema_location=original_bundle.schema_location,
            metadata=new_metadata,
            flavor=original_bundle.flavor,
            # LEGACY FIELDS
            location=original_bundle.location or "",
            requirements=original_bundle.requirements or [],
            env_params=env_params_dict,
            packaging_type=original_bundle.packaging_type or ModelBundlePackagingType.CLOUDPICKLE,
            app_config=request.new_app_config or original_bundle.app_config,
        )
        response = CreateModelBundleV1Response(model_bundle_id=new_model_bundle.id)
        return response


class ListModelBundlesV1UseCase:
    """
    Use case for listing all versions of a Model Bundle of a given user and model bundle name.
    """

    def __init__(self, model_bundle_repository: ModelBundleRepository):
        self.model_bundle_repository = model_bundle_repository

    async def execute(
        self,
        user: User,
        model_name: Optional[str],
        order_by: Optional[ModelBundleOrderBy],
    ) -> ListModelBundlesV1Response:
        """
        Runs the use case to list all Model Bundles owned by the user with the given name.

        Args:
            user: The user making request, on team who owns the model bundle(s).
            model_name: The name of the Model Bundle(s).
            order_by: An optional argument to specify the output ordering of the model bundles.

        Returns:
            A response object that contains the model bundles.
        """
        model_bundles = await self.model_bundle_repository.list_model_bundles(
            user.team_id, model_name, order_by
        )
        return ListModelBundlesV1Response(
            model_bundles=[ModelBundleV1Response.from_orm(mb) for mb in model_bundles]
        )


class GetModelBundleByIdV1UseCase:
    """
    Use case for getting a Model Bundle of a given user by ID.
    """

    def __init__(self, model_bundle_repository: ModelBundleRepository):
        self.model_bundle_repository = model_bundle_repository
        self.authz_module = ScaleAuthorizationModule()

    async def execute(self, user: User, model_bundle_id: str) -> ModelBundleV1Response:
        """
        Runs the use case to get the Model Bundle owned by the user with the given ID.

        Args:
            user: The owner of the model bundle.
            model_bundle_id: The ID of the model bundle.

        Returns:
            A response object that contains the model bundle.

        Raises:
            ObjectNotFoundException: If a model bundle with the given ID could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model bundle.
        """
        model_bundle = await self.model_bundle_repository.get_model_bundle(model_bundle_id)
        if not model_bundle:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, model_bundle):
            raise ObjectNotAuthorizedException
        return ModelBundleV1Response.from_orm(model_bundle)


class GetLatestModelBundleByNameV1UseCase:
    """
    Use case for getting the latest Model Bundle of a given user by name.
    """

    def __init__(self, model_bundle_repository: ModelBundleRepository):
        self.model_bundle_repository = model_bundle_repository

    async def execute(self, user: User, model_name: str) -> ModelBundleV1Response:
        """
        Runs the use case to get the latest Model Bundles owned by the user with the given name.

        Args:
            user: The owner of the model bundle.
            model_name: The name of the model bundle.

        Returns:
            A response object that contains the model bundle.

        Raises:
            ObjectNotFoundException: If a model bundle with the given name/owner could not be found.
        """
        model_bundle = await self.model_bundle_repository.get_latest_model_bundle_by_name(
            user.team_id, model_name
        )
        if model_bundle is None:
            raise ObjectNotFoundException
        return ModelBundleV1Response.from_orm(model_bundle)


class CreateModelBundleV2UseCase:
    """
    Use case for creating a Model Bundle.
    """

    def __init__(
        self,
        model_bundle_repository: ModelBundleRepository,
        docker_repository: DockerRepository,
        model_primitive_gateway: ModelPrimitiveGateway,
    ):
        self.authz_module = ScaleAuthorizationModule()
        self.model_bundle_repository = model_bundle_repository
        self.docker_repository = docker_repository
        self.model_primitive_gateway = model_primitive_gateway

    async def execute(
        self, user: User, request: CreateModelBundleV2Request
    ) -> CreateModelBundleV2Response:
        """
        Runs the use case to create a Model Bundle.

        Args:
            user: The user who is creating the Model Bundle.
            request: A request object that contains the creation fields.

        Returns:
            A response object that contains the creation response fields.

        Raises:
            DockerImageNotFoundException: If a model bundle specifies a Docker image that cannot be
                found.
            ObjectNotAuthorizedException: If a user is not authorized to perform the operation.
        """
        if (
            isinstance(request.flavor, ArtifactLike)
            and isinstance(request.flavor.framework, CustomFramework)
            and not self.docker_repository.image_exists(
                image_tag=request.flavor.framework.image_tag,
                repository_name=request.flavor.framework.image_repository,
            )
        ):
            raise DockerImageNotFoundException(
                repository=request.flavor.framework.image_repository,
                tag=request.flavor.framework.image_tag,
            )
        elif (
            isinstance(request.flavor, RunnableImageLike)
            and self.docker_repository.is_repo_name(request.flavor.repository)
            and not self.docker_repository.image_exists(
                image_tag=request.flavor.tag,
                repository_name=request.flavor.repository,
            )
        ):
            # only check image existance if repository is specified as just repo name
            # if a full image registry is specified, we skip this check to enable pass through of images from private registries
            raise DockerImageNotFoundException(
                repository=request.flavor.repository,
                tag=request.flavor.tag,
            )

        if not self.authz_module.check_access_create_bundle_v2(user, request):
            raise ObjectNotAuthorizedException

        created_by = user.user_id
        owner = user.team_id
        model_artifact_id = None
        if isinstance(request.flavor, ArtifactLike):
            model_artifact_id = await self.model_primitive_gateway.create_model_artifact(
                # guarantee uniqueness
                model_artifact_name=f"{request.name}-{str(uuid4())[-8:]}",
                location=request.flavor.location,
                framework_type=request.flavor.framework.framework_type,
                created_by=created_by,
            )
        model_artifact_ids = [model_artifact_id] if model_artifact_id else []

        # POPULATE LEGACY FIELDS
        if isinstance(request.flavor, ArtifactLike):
            location = request.flavor.location
            requirements = request.flavor.requirements
            env_params = request.flavor.framework.dict()
            # Rename image_repository to ecr_repo for legacy naming
            if "image_repository" in env_params:
                env_params["ecr_repo"] = env_params.pop("image_repository")

            packaging_type = (
                ModelBundlePackagingType.CLOUDPICKLE
                if isinstance(request.flavor, CloudpickleArtifactFlavor)
                else ModelBundlePackagingType.ZIP
            )
            app_config = request.flavor.app_config
        else:
            location = "unused"  # Nonempty to support legacy LLMEngine
            requirements = []
            env_params = {
                "framework_type": ModelBundleFrameworkType.CUSTOM,
                "ecr_repo": request.flavor.repository,
                "image_tag": request.flavor.tag,
            }
            packaging_type = ModelBundlePackagingType.CLOUDPICKLE
            app_config = None

        model_bundle = await self.model_bundle_repository.create_model_bundle(
            name=request.name,
            created_by=created_by,
            owner=owner,
            model_artifact_ids=model_artifact_ids,
            schema_location=request.schema_location,
            metadata=request.metadata or {},
            flavor=request.flavor,
            # LEGACY FIELDS
            location=location,
            requirements=requirements,
            env_params=env_params,
            packaging_type=packaging_type,
            app_config=app_config,
        )
        response = CreateModelBundleV2Response(model_bundle_id=model_bundle.id)
        return response


class CloneModelBundleV2UseCase:
    """
    Use case for cloning a Model Bundle from an existing one.
    """

    def __init__(self, model_bundle_repository: ModelBundleRepository):
        self.model_bundle_repository = model_bundle_repository
        self.authz_module = ScaleAuthorizationModule()

    async def execute(
        self,
        user: User,
        request: CloneModelBundleV2Request,
    ) -> CreateModelBundleV2Response:
        """
        Runs the use case to clone a Model Bundle.

        Args:
            user: The user who is cloning the Model Bundle.
            request: A request object that contains the fields for cloning.

        Returns:
            A response object that contains the creation response fields.

        Raises:
            ObjectNotAuthorizedException: If a user is not authorized to perform the operation.
            ObjectNotFoundException: If the source Model Bundle could not be found.
        """

        original_bundle = await self.model_bundle_repository.get_model_bundle(
            model_bundle_id=request.original_model_bundle_id,
        )
        if original_bundle is None:
            raise ObjectNotFoundException

        # Owning a bundle is necessary and sufficient for cloning it.
        if not self.authz_module.check_access_read_owned_entity(user, original_bundle):
            raise ObjectNotAuthorizedException

        env_params_dict = {}
        if (env_params := original_bundle.env_params) is not None:
            env_params_dict = env_params.dict()

        new_metadata = original_bundle.metadata.copy()
        new_metadata["__cloned_from_bundle_id"] = original_bundle.id
        new_model_bundle = await self.model_bundle_repository.create_model_bundle(
            name=original_bundle.name,
            created_by=user.user_id,
            owner=original_bundle.owner,
            model_artifact_ids=original_bundle.model_artifact_ids,
            schema_location=original_bundle.schema_location,
            metadata=new_metadata,
            flavor=original_bundle.flavor,
            # LEGACY FIELDS
            location=original_bundle.location or "",
            requirements=original_bundle.requirements or [],
            env_params=env_params_dict,
            packaging_type=original_bundle.packaging_type or ModelBundlePackagingType.CLOUDPICKLE,
            app_config=request.new_app_config or original_bundle.app_config,
        )
        response = CreateModelBundleV2Response(model_bundle_id=new_model_bundle.id)
        return response


class ListModelBundlesV2UseCase:
    """
    Use case for listing all versions of a Model Bundle of a given user and model bundle name.
    """

    def __init__(self, model_bundle_repository: ModelBundleRepository):
        self.model_bundle_repository = model_bundle_repository

    async def execute(
        self,
        user: User,
        model_name: Optional[str],
        order_by: Optional[ModelBundleOrderBy],
    ) -> ListModelBundlesV2Response:
        """
        Runs the use case to list all Model Bundles owned by the user with the given name.

        Args:
            user: The user making request, on team who owns the model bundle(s).
            model_name: The name of the Model Bundle(s).
            order_by: An optional argument to specify the output ordering of the model bundles.

        Returns:
            A response object that contains the model bundles.
        """
        model_bundles = await self.model_bundle_repository.list_model_bundles(
            user.team_id, model_name, order_by
        )
        return ListModelBundlesV2Response(
            model_bundles=[ModelBundleV2Response.from_orm(mb) for mb in model_bundles]
        )


class GetModelBundleByIdV2UseCase:
    """
    Use case for getting a Model Bundle of a given user by ID.
    """

    def __init__(self, model_bundle_repository: ModelBundleRepository):
        self.model_bundle_repository = model_bundle_repository
        self.authz_module = ScaleAuthorizationModule()

    async def execute(self, user: User, model_bundle_id: str) -> ModelBundleV2Response:
        """
        Runs the use case to get the Model Bundle owned by the user with the given ID.

        Args:
            user: The owner of the model bundle.
            model_bundle_id: The ID of the model bundle.

        Returns:
            A response object that contains the model bundle.

        Raises:
            ObjectNotFoundException: If a model bundle with the given ID could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model bundle.
        """
        model_bundle = await self.model_bundle_repository.get_model_bundle(model_bundle_id)
        if not model_bundle:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, model_bundle):
            raise ObjectNotAuthorizedException
        return ModelBundleV2Response.from_orm(model_bundle)


class GetLatestModelBundleByNameV2UseCase:
    """
    Use case for getting the latest Model Bundle of a given user by name.
    """

    def __init__(self, model_bundle_repository: ModelBundleRepository):
        self.model_bundle_repository = model_bundle_repository

    async def execute(self, user: User, model_name: str) -> ModelBundleV2Response:
        """
        Runs the use case to get the latest Model Bundles owned by the user with the given name.

        Args:
            user: The owner of the model bundle.
            model_name: The name of the model bundle.

        Returns:
            A response object that contains the model bundle.

        Raises:
            ObjectNotFoundException: If a model bundle with the given name/owner could not be found.
        """
        model_bundle = await self.model_bundle_repository.get_latest_model_bundle_by_name(
            user.team_id, model_name
        )
        if model_bundle is None:
            raise ObjectNotFoundException
        return ModelBundleV2Response.from_orm(model_bundle)
