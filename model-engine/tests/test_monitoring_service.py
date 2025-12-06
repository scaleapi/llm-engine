"""
Test Suite for Monitoring Service
==================================

Tests the monitoring, metrics collection, and health checking functionality.

Run with:
  pytest tests/test_monitoring_service.py -v

Coverage:
- InMemoryMetricsCollector
- LLMEngineHealthChecker
- StructuredLogger
- MonitoringService integration
"""

import pytest
import asyncio
import json
from datetime import datetime

from model_engine_server.monitoring_service import (
    Metric,
    MetricType,
    InMemoryMetricsCollector,
    LLMEngineHealthChecker,
    StructuredLogger,
    MonitoringService
)


class TestMetric:
    """Test Metric dataclass"""
    
    def test_metric_creation_with_defaults(self):
        """Test basic metric creation"""
        metric = Metric(
            name="test_metric",
            value=42.0,
            type=MetricType.GAUGE
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.type == MetricType.GAUGE
        assert metric.timestamp is not None
        assert metric.labels == {}
    
    def test_metric_with_labels(self):
        """Test metric with labels"""
        metric = Metric(
            name="http_requests",
            value=100,
            type=MetricType.COUNTER,
            labels={"endpoint": "/health", "method": "GET", "status": "200"}
        )
        
        assert metric.labels["endpoint"] == "/health"
        assert metric.labels["method"] == "GET"
        assert metric.labels["status"] == "200"
    
    def test_metric_to_prometheus_format(self):
        """Test conversion to Prometheus line format"""
        metric = Metric(
            name="test_gauge",
            value=123.45,
            type=MetricType.GAUGE,
            labels={"job": "test"}
        )
        
        prom_format = metric.to_prometheus_format()
        
        assert "test_gauge" in prom_format
        assert "job=" in prom_format
        assert "123.45" in prom_format
    
    def test_metric_prometheus_format_no_labels(self):
        """Test Prometheus format without labels"""
        metric = Metric(
            name="simple_metric",
            value=99.99,
            type=MetricType.HISTOGRAM
        )
        
        prom_format = metric.to_prometheus_format()
        
        assert prom_format.startswith("simple_metric ")
        assert "99.99" in prom_format


class TestInMemoryMetricsCollector:
    """Test InMemoryMetricsCollector"""
    
    @pytest.fixture
    def collector(self):
        """Create a metrics collector"""
        return InMemoryMetricsCollector(max_metrics=1000)
    
    def test_collector_initialization(self, collector):
        """Test collector is initialized properly"""
        assert collector.max_metrics == 1000
        assert len(collector.metrics) == 0
        assert len(collector.counters) == 0
        assert len(collector.gauges) == 0
    
    def test_record_counter_metric(self, collector):
        """Test recording counter metrics"""
        metric = Metric(
            name="total_requests",
            value=1,
            type=MetricType.COUNTER
        )
        
        collector.record_metric(metric)
        collector.record_metric(metric)  # Record again
        
        assert len(collector.metrics) == 2
        assert "total_requests_" in collector.counters
        assert collector.counters["total_requests_"] == 2
    
    def test_record_gauge_metric(self, collector):
        """Test recording gauge metrics"""
        metric = Metric(
            name="active_jobs",
            value=5,
            type=MetricType.GAUGE
        )
        
        collector.record_metric(metric)
        
        assert len(collector.metrics) == 1
        assert "active_jobs_" in collector.gauges
        assert collector.gauges["active_jobs_"] == 5
    
    def test_record_metric_with_labels(self, collector):
        """Test recording metrics with labels"""
        metric = Metric(
            name="request_count",
            value=1,
            type=MetricType.COUNTER,
            labels={"endpoint": "/api/inference", "method": "POST"}
        )
        
        collector.record_metric(metric)
        
        assert len(collector.metrics) == 1
        key = "request_count_/api/inference,POST"
        assert collector.counters[key] == 1
    
    def test_get_metrics(self, collector):
        """Test retrieving all metrics"""
        metrics = [
            Metric(name="m1", value=1, type=MetricType.GAUGE),
            Metric(name="m2", value=2, type=MetricType.COUNTER),
            Metric(name="m3", value=3, type=MetricType.HISTOGRAM)
        ]
        
        for metric in metrics:
            collector.record_metric(metric)
        
        retrieved = collector.get_metrics()
        
        assert len(retrieved) == 3
        assert all(isinstance(m, Metric) for m in retrieved)
    
    def test_export_prometheus_format(self, collector):
        """Test Prometheus format export"""
        metrics = [
            Metric(name="test_counter", value=5, type=MetricType.COUNTER),
            Metric(name="test_gauge", value=10, type=MetricType.GAUGE)
        ]
        
        for metric in metrics:
            collector.record_metric(metric)
        
        prometheus_output = collector.export_prometheus()
        
        assert "# HELP llm_engine" in prometheus_output
        assert "# TYPE llm_engine gauge" in prometheus_output
        assert "test_counter" in prometheus_output or "test_gauge" in prometheus_output
    
    def test_max_metrics_limit(self):
        """Test that max_metrics limit is enforced"""
        collector = InMemoryMetricsCollector(max_metrics=100)
        
        # Record more than max
        for i in range(150):
            metric = Metric(
                name=f"metric_{i}",
                value=i,
                type=MetricType.GAUGE
            )
            collector.record_metric(metric)
        
        # Should keep only latest metrics
        assert len(collector.metrics) <= 100


class TestLLMEngineHealthChecker:
    """Test LLMEngineHealthChecker"""
    
    @pytest.fixture
    def checker(self):
        """Create a health checker"""
        return LLMEngineHealthChecker(
            db_connection_string="postgresql://user:pass@localhost/db",
            redis_connection_string="redis://localhost:6379"
        )
    
    @pytest.mark.asyncio
    async def test_check_database(self, checker):
        """Test database health check"""
        result = await checker.check_component("database")
        
        assert result.component == "database"
        assert result.healthy is True
        assert result.status == "connected"
        assert "host" in result.details
        assert "port" in result.details
    
    @pytest.mark.asyncio
    async def test_check_redis(self, checker):
        """Test Redis health check"""
        result = await checker.check_component("redis")
        
        assert result.component == "redis"
        assert result.healthy is True
        assert result.status == "connected"
        assert "host" in result.details
        assert "memory_mb" in result.details
    
    @pytest.mark.asyncio
    async def test_check_models(self, checker):
        """Test models health check"""
        result = await checker.check_component("models")
        
        assert result.component == "models"
        assert result.healthy is True
        assert result.status == "ready"
        assert "available_models" in result.details
        assert len(result.details["available_models"]) > 0
    
    @pytest.mark.asyncio
    async def test_check_unknown_component(self, checker):
        """Test unknown component returns error"""
        result = await checker.check_component("unknown_service")
        
        assert result.component == "unknown_service"
        assert result.healthy is False
        assert result.error is not None


class TestStructuredLogger:
    """Test StructuredLogger"""
    
    def test_logger_creation(self):
        """Test logger can be created"""
        logger = StructuredLogger(__name__)
        assert logger.logger is not None
    
    def test_log_event(self):
        """Test logging a structured event"""
        logger = StructuredLogger("test_logger")
        
        # Just verify no exception is raised
        logger.log_event("test_event", level="INFO", custom_field="value")
        assert True  # If we got here, no exception
    
    def test_log_job_submitted(self):
        """Test logging job submission"""
        logger = StructuredLogger("test")
        
        # Just verify no exception is raised
        logger.log_job_submitted(
            job_id="job_abc123",
            job_type="fine_tune",
            model="llama-2-7b"
        )
        assert True  # If we got here, no exception
    
    def test_log_job_completed(self):
        """Test logging job completion"""
        logger = StructuredLogger("test")
        
        # Just verify no exception is raised
        logger.log_job_completed(
            job_id="job_abc123",
            duration_seconds=123.45
        )
        assert True  # If we got here, no exception
    
    def test_log_request(self):
        """Test logging API request"""
        logger = StructuredLogger("test")
        
        # Just verify no exception is raised
        logger.log_request(
            endpoint="/api/inference",
            method="POST",
            status_code=200,
            latency_ms=45.5
        )
        assert True  # If we got here, no exception


class TestMonitoringService:
    """Test MonitoringService integration"""
    
    @pytest.fixture
    def monitoring(self):
        """Create monitoring service"""
        return MonitoringService(enable_logging=True)
    
    def test_monitoring_initialization(self, monitoring):
        """Test monitoring service initializes"""
        assert monitoring.metrics is not None
        assert monitoring.health_checker is not None
        assert monitoring.structured_logger is not None
        assert len(monitoring.request_start_times) == 0
    
    def test_record_request_lifecycle(self, monitoring):
        """Test recording a complete request"""
        request_id = "req_123"
        
        # Record start
        monitoring.record_request_start(request_id)
        assert request_id in monitoring.request_start_times
        
        # Simulate some work
        import time
        time.sleep(0.05)
        
        # Record completion
        monitoring.record_request_complete(
            request_id=request_id,
            endpoint="/api/inference",
            method="POST",
            status_code=200
        )
        
        # Should be removed from tracking
        assert request_id not in monitoring.request_start_times
        
        # Metric should be recorded
        metrics = monitoring.metrics.get_metrics()
        assert len(metrics) > 0
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, monitoring):
        """Test all health checks together"""
        health = await monitoring.health_check_all()
        
        assert "healthy" in health
        assert "components" in health
        assert "timestamp" in health
        
        components = health["components"]
        assert "database" in components
        assert "redis" in components
        assert "models" in components
        
        # Each component should have required fields
        for component_name, component_health in components.items():
            assert "healthy" in component_health
            assert "status" in component_health
            assert "details" in component_health
    
    def test_get_prometheus_metrics(self, monitoring):
        """Test Prometheus metrics export"""
        # Record some metrics first
        import time
        for i in range(3):
            request_id = f"req_{i}"
            monitoring.record_request_start(request_id)
            time.sleep(0.01)
            monitoring.record_request_complete(
                request_id,
                f"/endpoint_{i}",
                "GET",
                200
            )
        
        prometheus_output = monitoring.get_prometheus_metrics()
        
        assert isinstance(prometheus_output, str)
        assert "http_request_duration_ms" in prometheus_output or "#" in prometheus_output
    
    def test_record_job_event(self, monitoring):
        """Test recording job events"""
        monitoring.record_job_event(
            event_type="submitted",
            job_id="job_xyz",
            model="llama-2-7b",
            dataset_size=1000
        )
        
        # Should not raise exception
        assert True


class TestMonitoringIntegration:
    """Integration tests for monitoring"""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_flow(self):
        """Test complete monitoring workflow"""
        monitoring = MonitoringService()
        
        # Simulate request
        req_id = "req_full_test"
        monitoring.record_request_start(req_id)
        
        await asyncio.sleep(0.05)
        
        monitoring.record_request_complete(
            req_id,
            "/api/inference",
            "POST",
            200
        )
        
        # Check health
        health = await monitoring.health_check_all()
        assert health["healthy"] is True
        
        # Export metrics
        metrics = monitoring.get_prometheus_metrics()
        assert isinstance(metrics, str)
        
        # Record event
        monitoring.record_job_event(
            "completed",
            "job_123",
            duration_seconds=60
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
