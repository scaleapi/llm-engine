from typing import Any, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm.session import Session
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import update
from sqlalchemy.sql.schema import UniqueConstraint
from xid import XID

from ..base import Base
from .constants import SHORT_STRING


def get_xid() -> str:
    return XID().string()


class Model(Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("owner", "name", name="model_owner_name_uc"),
        {"schema": "model"},
    )

    id = Column(Text, primary_key=True)
    name = Column(Text, index=True, nullable=False)
    description = Column(Text, index=True, nullable=True)
    task_types = Column(ARRAY(Text), index=True, nullable=False)
    created_by = Column(String(SHORT_STRING), index=True, nullable=False)
    owner = Column(String(SHORT_STRING), index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        task_types: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        owner: Optional[str] = None,
        created_at: Optional[DateTime] = None,
    ):
        self.id = f"mod_{get_xid()}"
        self.name = name
        self.description = description
        self.task_types = task_types
        self.created_by = created_by
        self.owner = owner
        self.created_at = created_at

    @staticmethod
    def create(session: Session, model: "Model"):
        session.connection()
        session.add(model)
        session.commit()

    @staticmethod
    def select(
        session: Session,
        owner: Optional[str] = None,
        name: Optional[str] = None,
        task_types: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ) -> List["Model"]:
        query = select(Model)
        if owner is not None:
            query = query.filter_by(owner=owner)
        if name is not None:
            query = query.filter_by(name=name)
        if task_types is not None:
            query = query.filter(Model.task_types.contains(task_types))
        if created_by is not None:
            query = query.filter_by(created_by=created_by)
        models = session.execute(query).scalars().all()
        return models

    @staticmethod
    def select_by_id(session: Session, model_id: str) -> Optional["Model"]:
        model = session.execute(select(Model).filter_by(id=model_id)).scalar_one_or_none()
        return model

    @staticmethod
    def update_by_id(session: Session, model_id: str, **kwargs: str) -> None:
        stmt = update(Model).where(Model.id == model_id).values(**kwargs)

        session.execute(stmt)
        session.commit()


class ModelVersion(Base):
    __table__ = Table(
        "model_versions",
        Base.metadata,
        Column("id", Text, primary_key=True),
        Column("model_id", Text, ForeignKey("model.models.id"), index=True, nullable=False),
        Column("version_number", Integer, index=True, nullable=False),
        Column(
            "launch_model_bundle_id",
            Text,
            # ForeignKey("spellbook_serve.bundles.id"), # This is currently breaking tests.
            index=True,
            nullable=True,
        ),
        Column("nucleus_model_id", Text, index=True, nullable=True),
        Column("tags", ARRAY(Text), index=True, nullable=False),
        Column("metadata", JSON, index=False, server_default="{}"),
        Column("created_by", String(SHORT_STRING), index=True, nullable=False),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        UniqueConstraint("model_id", "version_number", name="model_id_version_number_uc"),
        UniqueConstraint("launch_model_bundle_id", name="launch_model_bundle_id_uc"),
        UniqueConstraint("nucleus_model_id", name="nucleus_model_id_uc"),
        schema="model",
    )

    def __init__(
        self,
        model_id: Optional[str] = None,
        version_number: Optional[int] = None,
        launch_model_bundle_id: Optional[str] = None,
        nucleus_model_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Any] = None,
        created_by: Optional[str] = None,
        created_at: Optional[DateTime] = None,
    ):
        self.id = f"mov_{get_xid()}"
        self.model_id = model_id
        self.version_number = version_number
        self.launch_model_bundle_id = launch_model_bundle_id
        self.nucleus_model_id = nucleus_model_id
        self.tags = tags or []
        self.metadata = metadata
        self.created_by = created_by
        self.created_at = created_at

    @staticmethod
    def create(session: Session, model_version: "ModelVersion"):
        session.add(model_version)
        session.commit()

    @staticmethod
    def select(
        session: Session,
        owner: Optional[str] = None,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List["ModelVersion"]:
        query = select(ModelVersion).join(Model, Model.id == ModelVersion.model_id)
        if owner is not None:
            query = query.filter(Model.owner == owner)
        if model_id is not None:
            query = query.filter(Model.id == model_id)
        if model_name is not None:
            query = query.filter(Model.name == model_name)
        if tags is not None:
            query = query.filter(ModelVersion.tags.contains(tags))  # type: ignore
        models = session.execute(query).scalars().all()
        return models

    @staticmethod
    def select_by_launch_model_bundle_id(
        session: Session, launch_model_bundle_id: str
    ) -> Optional["ModelVersion"]:
        model_version = session.execute(
            select(ModelVersion).filter_by(launch_model_bundle_id=launch_model_bundle_id)
        ).scalar_one_or_none()
        return model_version

    @staticmethod
    def select_by_nucleus_model_id(
        session: Session, nucleus_model_id: str
    ) -> Optional["ModelVersion"]:
        model_version = session.execute(
            select(ModelVersion).filter_by(nucleus_model_id=nucleus_model_id)
        ).scalar_one_or_none()
        return model_version

    @staticmethod
    def select_by_id(session: Session, model_version_id: str) -> Optional["ModelVersion"]:
        model_version = session.execute(
            select(ModelVersion).filter_by(id=model_version_id)
        ).scalar_one_or_none()
        return model_version

    @staticmethod
    def get_highest_version_number_for_model(session: Session, model_id: str) -> Optional[int]:
        """
        Returns the highest version number among the ModelVersions associated with a given Model.

        Returns:
            If there is at least one existing ModelVersion, then this returns that
            ModelVersion's version_number.
            Otherwise, return None.
        """
        version_number = session.execute(
            select(func.max(ModelVersion.version_number)).filter_by(model_id=model_id)
        ).scalar_one_or_none()
        return version_number

    @staticmethod
    def update_by_id(session: Session, model_version_id: str, **kwargs: Optional[str]) -> None:
        stmt = update(ModelVersion).where(ModelVersion.id == model_version_id).values(**kwargs)

        session.execute(stmt)
        session.commit()


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    __table_args__ = (
        UniqueConstraint("owner", "name", name="model_artifact_owner_name_uc"),
        {"schema": "model"},
    )

    id = Column(Text, primary_key=True)
    name = Column(Text, index=True, nullable=False)
    description = Column(Text, index=True, nullable=True)
    is_public = Column(Boolean, index=True, nullable=False)
    created_by = Column(String(SHORT_STRING), index=True, nullable=False)
    owner = Column(String(SHORT_STRING), index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    input_schema = Column(JSON, index=False, nullable=False)
    output_schema = Column(JSON, index=False, nullable=False)
    config = Column(JSON, index=False, nullable=False)
    location = Column(Text, index=False, nullable=False)
    format = Column(Text, index=True, nullable=False)
    format_metadata = Column(JSON, index=False, nullable=False)
    source = Column(Text, index=True, nullable=False)
    source_metadata = Column(JSON, index=False, nullable=False)

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_public: Optional[bool] = None,
        created_by: Optional[str] = None,
        owner: Optional[str] = None,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        config: Optional[Any] = None,
        location: Optional[str] = None,
        format: Optional[str] = None,  # pylint:disable=redefined-builtin
        format_metadata: Optional[Any] = None,
        source: Optional[str] = None,
        source_metadata: Optional[Any] = None,
        created_at: Optional[DateTime] = None,
    ):
        self.id = f"moa_{get_xid()}"
        self.name = name
        self.description = description
        self.is_public = is_public
        self.created_by = created_by
        self.owner = owner
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.config = config
        self.location = location
        self.format = format
        self.format_metadata = format_metadata
        self.source = source
        self.source_metadata = source_metadata
        self.created_at = created_at

    @staticmethod
    def create(session: Session, model_artifact: "ModelArtifact") -> None:
        session.add(model_artifact)
        session.commit()

    @staticmethod
    def select(
        session: Session,
        owner: Optional[str] = None,
        name: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> List["ModelArtifact"]:
        query = select(ModelArtifact)
        if owner is not None:
            query = query.filter(ModelArtifact.is_public | (ModelArtifact.owner == owner))
        else:
            query = query.filter(ModelArtifact.is_public)

        if name is not None:
            query = query.filter_by(name=name)
        if created_by is not None:
            query = query.filter_by(created_by=created_by)
        model_artifacts = session.execute(query).scalars().all()
        return model_artifacts

    @staticmethod
    def select_by_id(session: Session, model_artifact_id: str) -> Optional["ModelArtifact"]:
        model_artifact = session.execute(
            select(ModelArtifact).filter_by(id=model_artifact_id)
        ).scalar_one_or_none()
        return model_artifact

    @staticmethod
    def update_by_id(session: Session, model_artifact_id: str, **kwargs: str) -> None:
        stmt = update(ModelArtifact).where(ModelArtifact.id == model_artifact_id).values(**kwargs)

        session.execute(stmt)
        session.commit()
