from typing import List

from spellbook_serve.db.base import Session
from spellbook_serve.db.models import Bundle, Model, ModelArtifact, ModelVersion


def test_model_select(dbsession: Session, models: List[Model]):
    models_by_owner = Model.select(dbsession, owner="test_user_id_1")
    assert len(models_by_owner) == 2

    models_by_name = Model.select(dbsession, owner="test_user_id_1", name="test_model_1")
    assert len(models_by_name) == 1

    models_by_created_by = Model.select(
        dbsession, owner="test_user_id_1", created_by="test_user_id_1"
    )
    assert len(models_by_created_by) == 2

    models_by_task_types = Model.select(
        dbsession, owner="test_user_id_1", task_types=["test_task_type_1"]
    )
    assert len(models_by_task_types) == 2

    model_by_id = Model.select_by_id(dbsession, model_id=models[0].id)
    assert model_by_id is not None


def test_model_update(dbsession: Session, models: List[Model]):
    Model.update_by_id(dbsession, models[0].id, description="new description")
    model = Model.select_by_id(dbsession, models[0].id)
    assert model is not None
    assert model.description == "new description"


def test_model_version_select(
    dbsession: Session, models: List[Model], model_versions: List[ModelVersion]
):
    model_versions_by_owner = ModelVersion.select(dbsession, owner="test_user_id_1")
    assert len(model_versions_by_owner) == 2

    model_versions_by_model_id = ModelVersion.select(
        dbsession, owner="test_user_id_1", model_id=models[0].id
    )
    assert len(model_versions_by_model_id) == 2

    model_versions_by_model_name = ModelVersion.select(
        dbsession, owner="test_user_id_1", model_name="test_model_1"
    )
    assert len(model_versions_by_model_name) == 2

    model_versions_by_tags = ModelVersion.select(
        dbsession, owner="test_user_id_1", tags=["test_tag_1"]
    )
    assert len(model_versions_by_tags) == 2

    model_version_by_id = ModelVersion.select_by_id(
        dbsession, model_version_id=model_versions[0].id
    )
    assert model_version_by_id is not None


def test_model_version_select_by_model_id(
    dbsession: Session,
    bundles: List[Bundle],
    models: List[Model],
    model_versions: List[ModelVersion],
):
    model_version_by_bundle_id = ModelVersion.select_by_spellbook_serve_model_bundle_id(
        dbsession, bundles[0].id
    )
    assert model_version_by_bundle_id is not None
    assert model_version_by_bundle_id.spellbook_serve_model_bundle_id == bundles[0].id

    model_version_by_nucleus_model_id = ModelVersion.select_by_nucleus_model_id(
        dbsession, model_versions[2].nucleus_model_id  # type: ignore
    )
    assert model_version_by_nucleus_model_id is not None


def test_model_version_get_highest_version_number(
    dbsession: Session, models: List[Model], model_versions: List[ModelVersion]
):
    version_number = ModelVersion.get_highest_version_number_for_model(
        dbsession,
        model_id=models[0].id,
    )
    assert version_number == 1

    version_number = ModelVersion.get_highest_version_number_for_model(
        dbsession,
        model_id=models[1].id,
    )
    assert version_number is None

    version_number = ModelVersion.get_highest_version_number_for_model(
        dbsession,
        model_id="unknown id",
    )
    assert version_number is None


def test_model_version_update(
    dbsession: Session, models: List[Model], model_versions: List[ModelVersion]
):
    ModelVersion.update_by_id(
        dbsession, model_versions[0].id, nucleus_model_id="test_nucleus_model_id_upd"
    )
    model_version = ModelVersion.select_by_id(dbsession, model_versions[0].id)
    assert model_version is not None
    assert model_version.nucleus_model_id == "test_nucleus_model_id_upd"


def test_model_artifact_select(dbsession: Session, model_artifacts: List[ModelArtifact]):
    model_artifacts_by_owner = ModelArtifact.select(dbsession, owner="test_user_id_1")
    assert len(model_artifacts_by_owner) == 3

    model_artifacts_by_no_owner = ModelArtifact.select(dbsession)
    assert len(model_artifacts_by_no_owner) == 2

    model_artifacts_by_name = ModelArtifact.select(
        dbsession, owner="test_user_id_1", name="test_model_artifact_1"
    )
    assert len(model_artifacts_by_name) == 1

    model_artifacts_by_created_by = ModelArtifact.select(
        dbsession, owner="test_user_id_1", created_by="test_user_id_1"
    )
    assert len(model_artifacts_by_created_by) == 2

    model_artifact_by_id = ModelArtifact.select_by_id(
        dbsession, model_artifact_id=model_artifacts[0].id
    )
    assert model_artifact_by_id is not None


def test_model_artifact_update(dbsession: Session, model_artifacts: List[ModelArtifact]):
    ModelArtifact.update_by_id(dbsession, model_artifacts[0].id, description="new description")
    updated_model_artifact = ModelArtifact.select_by_id(dbsession, model_artifacts[0].id)
    assert updated_model_artifact is not None
    assert updated_model_artifact.description == "new description"
