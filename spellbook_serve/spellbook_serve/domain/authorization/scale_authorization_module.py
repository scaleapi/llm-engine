from spellbook_serve.common.dtos.model_bundles import (
    CreateModelBundleV1Request,
    CreateModelBundleV2Request,
)
from spellbook_serve.core.auth.authentication_repository import User
from spellbook_serve.core.config import ml_infra_config
from spellbook_serve.domain.entities import CustomFramework, ModelBundleFrameworkType, OwnedEntity
from spellbook_serve.domain.entities.model_bundle_entity import RunnableImageLike
from spellbook_serve.domain.entities.model_endpoint_entity import ModelEndpointRecord

SPELLBOOK_SERVE_INTEGRATION_TEST_USER: str = "62bc820451dbea002b1c5421"


class ScaleAuthorizationModule:
    """
    This class contains authorization utilities. All methods expect User objects given from authn.
    """

    @staticmethod
    def check_access_create_bundle_v1(user: User, request: CreateModelBundleV1Request) -> bool:
        """Checks whether the provided user is authorized to create the requested model bundle."""
        # External customers cannot use custom images.
        return (
            user.is_privileged_user
            or request.env_params.framework_type != ModelBundleFrameworkType.CUSTOM
        )

    @staticmethod
    def check_access_create_bundle_v2(user: User, request: CreateModelBundleV2Request) -> bool:
        """Checks whether the provided user is authorized to create the requested model bundle."""
        # External customers cannot use custom images.
        return (
            user.is_privileged_user
            or user.user_id == SPELLBOOK_SERVE_INTEGRATION_TEST_USER
            or (
                not isinstance(request.flavor, RunnableImageLike)
                and not isinstance(request.flavor.framework, CustomFramework)
            )
        )

    @staticmethod
    def check_access_read_owned_entity(user: User, owned_entity: OwnedEntity) -> bool:
        """Check whether the provided user is authorized to read the owned entity."""
        return user.team_id == owned_entity.owner

    @staticmethod
    def check_access_write_owned_entity(user: User, owned_entity: OwnedEntity) -> bool:
        """Check whether the provided user is authorized to write to the owned entity."""
        # TODO: we should create and use owned_entity.owner
        return user.team_id == owned_entity.owner

    @staticmethod
    def get_aws_role_for_user(user: User) -> str:
        """Returns the AWS role that should be assumed with the user's resources."""
        return ml_infra_config().profile_ml_inference_worker

    @staticmethod
    def get_s3_bucket_for_user(user: User) -> str:
        """Returns the AWS role that should be assumed with the user's resources."""
        return ml_infra_config().s3_bucket

    @staticmethod
    def check_endpoint_public_inference_for_user(
        user: User, endpoint_record: ModelEndpointRecord
    ) -> bool:
        """Returns whether the user is allowed to inference with this endpoint publicly."""
        return bool(endpoint_record.public_inference)
