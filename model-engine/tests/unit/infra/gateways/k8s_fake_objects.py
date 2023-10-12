# Various fake k8s objects to be used in mocking out the python k8s api client
# Only classes are defined here. If you need to add various fields to the classes, please do so here.

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class FakeK8sV1ObjectMeta:
    name: str = "fake_name"
    namespace: str = "fake_namespace"
    annotations: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    creation_timestamp: datetime = datetime(2021, 1, 1, 0, 0, 0, 0)
    # TODO: everything else


@dataclass
class FakeK8sV1PodStatus:
    phase: str = "Running"
    # TODO: everything else


@dataclass
class FakeK8sV1JobStatus:
    active: int = 0
    succeeded: int = 0
    failed: int = 0
    ready: int = 0
    terminating: int = 0
    completion_time: Optional[datetime] = None


@dataclass
class FakeK8sV1Job:
    metadata: FakeK8sV1ObjectMeta = FakeK8sV1ObjectMeta()
    status: FakeK8sV1JobStatus = FakeK8sV1JobStatus()
    # TODO: spec, api_version, kind


@dataclass
class FakeK8sV1JobList:
    items: List[FakeK8sV1Job] = field(default_factory=list)


@dataclass
class FakeK8sV1Pod:
    metadata: FakeK8sV1ObjectMeta = FakeK8sV1ObjectMeta()
    status: FakeK8sV1PodStatus = FakeK8sV1PodStatus()
    # TODO: spec, api_version, kind


@dataclass
class FakeK8sV1PodList:
    items: List[FakeK8sV1Pod] = field(default_factory=list)


@dataclass
class FakeK8sEnvVar:
    name: str
    value: str


@dataclass
class FakeK8sDeploymentContainer:
    env: List[FakeK8sEnvVar]
