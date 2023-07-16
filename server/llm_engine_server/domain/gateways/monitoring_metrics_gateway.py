"""
For emitting external monitoring metrics to some sort of store e.g. datadog
Currently distinct from something emitting to a Metrics Store

Used to calculate proportion of successful/unsuccessful requests, differentiates between
docker build vs other errors
"""

from abc import ABC, abstractmethod


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
