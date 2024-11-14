"""
For emitting external monitoring metrics to some sort of store e.g. datadog
Currently distinct from something emitting to a Metrics Store

Used to calculate proportion of successful/unsuccessful requests, differentiates between
docker build vs other errors
"""

from abc import ABC, abstractmethod
from typing import Optional

from model_engine_server.common.dtos.llms import TokenUsage
from model_engine_server.common.pydantic_types import BaseModel
from model_engine_server.core.auth.authentication_repository import User


class MetricMetadata(BaseModel):
    user: User
    model_name: Optional[str] = None


class MonitoringMetricsGateway(ABC):
    @abstractmethod
    def emit_attempted_build_metric(self):
        """
        Service builder attempted metric
        """

    @abstractmethod
    def emit_successful_build_metric(self):
        """
        Service builder succeeded metric
        """

    @abstractmethod
    def emit_build_time_metric(self, duration_seconds: float):
        """
        Service builder build time metric
        """

    @abstractmethod
    def emit_image_build_cache_hit_metric(self, image_type: str):
        """
        Service builder image build cache hit metric
        """

    @abstractmethod
    def emit_image_build_cache_miss_metric(self, image_type: str):
        """
        Service builder image build cache miss metric
        """

    @abstractmethod
    def emit_docker_failed_build_metric(self):
        """
        Service builder docker build failed metric
        """

    @abstractmethod
    def emit_database_cache_hit_metric(self):
        """
        Successful database cache metric
        """

    @abstractmethod
    def emit_database_cache_miss_metric(self):
        """
        Missed database cache metric
        """

    @abstractmethod
    def emit_route_call_metric(self, route: str, metadata: MetricMetadata):
        """
        Route call metric
        """
        pass

    @abstractmethod
    def emit_token_count_metrics(self, token_usage: TokenUsage, metadata: MetricMetadata):
        """
        Token count metrics
        """
        pass

    @abstractmethod
    def emit_http_call_error_metrics(self, endpoint_name: str, error_code: int):
        """
        Sync call timeout metrics, emitted when sync/streaming request
        times out (likely due to scale-from-zero not being fast enough
        or we're out of capacity)
        """
        pass
