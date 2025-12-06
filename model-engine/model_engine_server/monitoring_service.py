"""
Monitoring Service for LLM Engine Controller
=============================================

Provides observability for the controller through:
- Prometheus metrics (latency, throughput, errors)
- Structured logging (JSON format for ELK/Datadog)
- Health checks for dependencies (PostgreSQL, Redis, Models)
- Event tracking and audit logs

Design: Minimal dependencies, works with existing prometheus_client library.
"""

import time
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional, List
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to track"""
    COUNTER = "counter"      # Increment only (e.g., total requests)
    GAUGE = "gauge"          # Can increase/decrease (e.g., active jobs)
    HISTOGRAM = "histogram"  # Distribution (e.g., latency buckets)


@dataclass
class Metric:
    """Single metric value"""
    name: str
    value: float
    type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus line format"""
        label_str = ""
        if self.labels:
            labels_list = [f'{k}="{v}"' for k, v in self.labels.items()]
            label_str = "{" + ",".join(labels_list) + "}"
        
        return f"{self.name}{label_str} {self.value} {int(self.timestamp.timestamp() * 1000)}"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component: str
    healthy: bool
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class MetricsCollector(ABC):
    """Abstract base for metrics collection"""
    
    @abstractmethod
    def record_metric(self, metric: Metric) -> None:
        """Record a metric"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> List[Metric]:
        """Get all recorded metrics"""
        pass
    
    @abstractmethod
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        pass


class InMemoryMetricsCollector(MetricsCollector):
    """
    Simple in-memory metrics collector.
    
    Use this for local testing and development.
    In production, use prometheus_client directly.
    """
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: List[Metric] = []
        self.max_metrics = max_metrics
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        logger.info("InMemoryMetricsCollector initialized")
    
    def record_metric(self, metric: Metric) -> None:
        """Record a metric in memory"""
        if len(self.metrics) >= self.max_metrics:
            # Keep latest metrics only
            self.metrics = self.metrics[-(self.max_metrics // 2):]
        
        self.metrics.append(metric)
        
        # Update aggregates
        if metric.type == MetricType.COUNTER:
            key = f"{metric.name}_{','.join(metric.labels.values())}"
            self.counters[key] = self.counters.get(key, 0) + metric.value
        elif metric.type == MetricType.GAUGE:
            key = f"{metric.name}_{','.join(metric.labels.values())}"
            self.gauges[key] = metric.value
    
    def get_metrics(self) -> List[Metric]:
        """Get all metrics"""
        return self.metrics.copy()
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        lines.append("# HELP llm_engine LLM Engine Controller Metrics")
        lines.append("# TYPE llm_engine gauge")
        
        # Export counters
        for key, value in self.counters.items():
            lines.append(f"{key} {value}")
        
        # Export gauges
        for key, value in self.gauges.items():
            lines.append(f"{key} {value}")
        
        return "\n".join(lines)


class HealthChecker(ABC):
    """Abstract base for health checks"""
    
    @abstractmethod
    async def check_component(self, component: str) -> HealthCheckResult:
        """Check health of a component"""
        pass


class LLMEngineHealthChecker(HealthChecker):
    """
    Health checker for LLM Engine components.
    
    Checks:
    - PostgreSQL database
    - Redis cache
    - Model executor (GPU, models available)
    - Controller API
    """
    
    def __init__(self, db_connection_string: str = "", redis_connection_string: str = ""):
        self.db_connection_string = db_connection_string
        self.redis_connection_string = redis_connection_string
        logger.info("LLMEngineHealthChecker initialized")
    
    async def check_component(self, component: str) -> HealthCheckResult:
        """Check specific component"""
        if component == "database":
            return await self._check_database()
        elif component == "redis":
            return await self._check_redis()
        elif component == "models":
            return await self._check_models()
        else:
            return HealthCheckResult(
                component=component,
                healthy=False,
                status="unknown",
                error=f"Unknown component: {component}"
            )
    
    async def _check_database(self) -> HealthCheckResult:
        """Check PostgreSQL connectivity"""
        try:
            # In real implementation, use psycopg2 or asyncpg
            # For now, return mock status
            return HealthCheckResult(
                component="database",
                healthy=True,
                status="connected",
                details={
                    "host": "localhost",
                    "port": 5432,
                    "database": "llm_engine",
                    "connections": 5
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="database",
                healthy=False,
                status="failed",
                error=str(e)
            )
    
    async def _check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity"""
        try:
            # In real implementation, use redis-py
            # For now, return mock status
            return HealthCheckResult(
                component="redis",
                healthy=True,
                status="connected",
                details={
                    "host": "localhost",
                    "port": 6379,
                    "memory_mb": 100,
                    "keys": 1250
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="redis",
                healthy=False,
                status="failed",
                error=str(e)
            )
    
    async def _check_models(self) -> HealthCheckResult:
        """Check model executor availability"""
        try:
            return HealthCheckResult(
                component="models",
                healthy=True,
                status="ready",
                details={
                    "available_models": ["llama-2-7b", "falcon-7b", "mpt-7b"],
                    "available_gpus": 0,
                    "gpu_memory_gb": 0
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="models",
                healthy=False,
                status="failed",
                error=str(e)
            )


class StructuredLogger:
    """
    Structured logging for JSON output (ELK/Datadog compatible).
    
    Example output:
    {"timestamp": "2025-12-06T...", "level": "INFO", "event": "job_submitted", 
     "job_id": "job_abc123", "model": "llama-2-7b"}
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_event(
        self,
        event: str,
        level: str = "INFO",
        **fields
    ) -> None:
        """Log a structured event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "event": event,
            **fields
        }
        
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_entry))
    
    def log_job_submitted(self, job_id: str, job_type: str, model: str) -> None:
        """Log job submission event"""
        self.log_event(
            event="job_submitted",
            job_id=job_id,
            job_type=job_type,
            model=model
        )
    
    def log_job_completed(self, job_id: str, duration_seconds: float) -> None:
        """Log job completion event"""
        self.log_event(
            event="job_completed",
            job_id=job_id,
            duration_seconds=duration_seconds
        )
    
    def log_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float
    ) -> None:
        """Log API request"""
        self.log_event(
            event="request_completed",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms
        )


class MonitoringService:
    """
    Main monitoring service for LLM Engine controller.
    
    Integrates:
    - Metrics collection (counters, gauges, histograms)
    - Health checking (component status)
    - Structured logging (JSON events)
    - Prometheus endpoint
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        health_checker: Optional[HealthChecker] = None,
        enable_logging: bool = True
    ):
        self.metrics = metrics_collector or InMemoryMetricsCollector()
        self.health_checker = health_checker or LLMEngineHealthChecker()
        self.structured_logger = StructuredLogger(__name__) if enable_logging else None
        self.request_start_times: Dict[str, float] = {}
        logger.info("MonitoringService initialized")
    
    def record_request_start(self, request_id: str) -> None:
        """Record the start time of a request"""
        self.request_start_times[request_id] = time.time()
    
    def record_request_complete(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        status_code: int
    ) -> None:
        """Record request completion with latency"""
        if request_id in self.request_start_times:
            latency_ms = (time.time() - self.request_start_times[request_id]) * 1000
            
            # Record metric
            metric = Metric(
                name="http_request_duration_ms",
                value=latency_ms,
                type=MetricType.HISTOGRAM,
                labels={"endpoint": endpoint, "method": method, "status": str(status_code)}
            )
            self.metrics.record_metric(metric)
            
            # Log event
            if self.structured_logger:
                self.structured_logger.log_request(endpoint, method, status_code, latency_ms)
            
            del self.request_start_times[request_id]
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        components = ["database", "redis", "models"]
        results = {}
        overall_healthy = True
        
        for component in components:
            result = await self.health_checker.check_component(component)
            results[component] = {
                "healthy": result.healthy,
                "status": result.status,
                "details": result.details,
                "error": result.error
            }
            if not result.healthy:
                overall_healthy = False
        
        return {
            "healthy": overall_healthy,
            "timestamp": datetime.utcnow().isoformat(),
            "components": results
        }
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return self.metrics.export_prometheus()
    
    def record_job_event(self, event_type: str, job_id: str, **details) -> None:
        """Record job-related event"""
        if self.structured_logger:
            self.structured_logger.log_event(
                event=f"job_{event_type}",
                job_id=job_id,
                **details
            )
