from .base import Base, ml_infra_pg_engine

# we need to import the following for sqlalchemy
# pylint: disable=unused-import
from .models.llm_engine import Bundle, Endpoint  # noqa
from .models.model import Model, ModelArtifact, ModelVersion  # noqa
from .models.train import Execution, Experiment, Job, Snapshot  # noqa

# run this file to create the db models imported
Base.metadata.create_all(ml_infra_pg_engine)
