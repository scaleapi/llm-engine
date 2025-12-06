"""
Enhanced Engine Controller with Monitoring
===========================================

Integrates monitoring, health checks, and model execution into the controller.
This is the production-ready orchestrator for LLM Engine.

Features:
- Job submission and tracking
- Real-time monitoring and metrics
- Health checks for all dependencies
- Support for mocked inference (local) and Kubernetes (production)

Usage:
    from controller_with_monitoring import EngineControllerWithMonitoring
    from model_executor import MockModelExecutor
    
    executor = MockModelExecutor()
    controller = EngineControllerWithMonitoring(executor)
    
    # Submit a job
    job_id = await controller.submit_fine_tune_job(...)
    
    # Check status
    status = await controller.get_job_status(job_id)
    
    # Get metrics
    metrics = controller.get_metrics()
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

from model_engine_server.model_executor import (
    ModelExecutor,
    MockModelExecutor,
    FineTuneRequest,
    InferenceRequest,
    JobStatus,
    JobResult
)
from model_engine_server.monitoring_service import (
    MonitoringService,
    Metric,
    MetricType
)

logger = logging.getLogger(__name__)


@dataclass
class JobTracker:
    """Track submitted jobs"""
    job_id: str
    job_type: str  # "fine_tune", "inference"
    model: str
    status: JobStatus
    created_at: float
    completed_at: Optional[float] = None
    result: Optional[JobResult] = None


class EngineControllerWithMonitoring:
    """
    Enhanced LLM Engine controller with full monitoring and observability.
    
    Responsibilities:
    - Job orchestration (submission, tracking, status)
    - Metric collection and export
    - Health checks for all components
    - Request logging and audit trails
    """
    
    def __init__(
        self,
        executor: ModelExecutor = None,
        monitoring: MonitoringService = None,
        enable_metrics: bool = True
    ):
        """
        Initialize controller.
        
        Args:
            executor: ModelExecutor instance (uses Mock if not provided)
            monitoring: MonitoringService instance (created if not provided)
            enable_metrics: Whether to collect metrics
        """
        self.executor = executor or MockModelExecutor()
        self.monitoring = monitoring or MonitoringService()
        self.enable_metrics = enable_metrics
        self.jobs: Dict[str, JobTracker] = {}
        
        logger.info(
            f"EngineControllerWithMonitoring initialized "
            f"(executor={self.executor.__class__.__name__}, metrics={enable_metrics})"
        )
    
    async def submit_fine_tune_job(
        self,
        base_model: str,
        dataset_path: str,
        output_path: str,
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> str:
        """
        Submit a fine-tuning job.
        
        Args:
            base_model: Name of model to fine-tune
            dataset_path: Path to training dataset
            output_path: Where to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Job ID for tracking progress
        """
        request_id = str(uuid.uuid4())
        self.monitoring.record_request_start(request_id)
        
        try:
            # Create fine-tune request
            request = FineTuneRequest(
                base_model=base_model,
                dataset_path=dataset_path,
                output_path=output_path,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Submit to executor
            job_id = await self.executor.fine_tune(request)
            
            # Track job
            import time
            self.jobs[job_id] = JobTracker(
                job_id=job_id,
                job_type="fine_tune",
                model=base_model,
                status=JobStatus.SUBMITTED,
                created_at=time.time()
            )
            
            # Record metrics
            if self.enable_metrics:
                self.monitoring.record_job_event(
                    "submitted",
                    job_id=job_id,
                    model=base_model,
                    dataset_path=dataset_path,
                    epochs=epochs
                )
            
            logger.info(f"Fine-tune job submitted: {job_id} (model={base_model})")
            
            return job_id
            
        finally:
            self.monitoring.record_request_complete(
                request_id,
                endpoint="/api/fine-tune",
                method="POST",
                status_code=200
            )
    
    async def submit_inference_job(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Submit an inference request.
        
        Args:
            prompt: Input prompt for the model
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Job ID for tracking
        """
        request_id = str(uuid.uuid4())
        self.monitoring.record_request_start(request_id)
        
        try:
            # Create inference request
            request = InferenceRequest(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # For inference, we typically want direct response
            # But we track it as a job for consistency
            output = await self.executor.infer(request)
            
            job_id = f"inf_{uuid.uuid4().hex[:8]}"
            
            # Track job
            import time
            self.jobs[job_id] = JobTracker(
                job_id=job_id,
                job_type="inference",
                model=model,
                status=JobStatus.COMPLETED,
                created_at=time.time(),
                completed_at=time.time(),
                result=JobResult(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    output={"text": output}
                )
            )
            
            if self.enable_metrics:
                self.monitoring.record_job_event(
                    "completed",
                    job_id=job_id,
                    model=model,
                    tokens_generated=len(output.split())
                )
            
            return job_id
            
        finally:
            self.monitoring.record_request_complete(
                request_id,
                endpoint="/api/infer",
                method="POST",
                status_code=200
            )
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the current status of a job.
        
        Args:
            job_id: ID of the job to query
            
        Returns:
            Status dict with job details
        """
        # Check local tracking first
        if job_id in self.jobs:
            job = self.jobs[job_id]
            return {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "model": job.model,
                "status": job.status.value,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "result": job.result.__dict__ if job.result else None
            }
        
        # Query executor
        try:
            result = await self.executor.get_job_status(job_id)
            
            return {
                "job_id": result.job_id,
                "status": result.status.value,
                "output": result.output,
                "error": result.error,
                "duration_seconds": result.duration_seconds
            }
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {
                "job_id": job_id,
                "status": "unknown",
                "error": str(e)
            }
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: ID of job to cancel
            
        Returns:
            True if cancellation succeeded
        """
        success = await self.executor.cancel_job(job_id)
        
        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.CANCELLED
        
        if self.enable_metrics:
            self.monitoring.record_job_event(
                "cancelled",
                job_id=job_id
            )
        
        logger.info(f"Job cancellation request: {job_id} (success={success})")
        return success
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health check.
        
        Returns:
            Health status of all components
        """
        health = await self.monitoring.health_check_all()
        
        # Add controller-specific health
        health["controller"] = {
            "tracked_jobs": len(self.jobs),
            "active_jobs": len([j for j in self.jobs.values() 
                               if j.status in [JobStatus.SUBMITTED, JobStatus.QUEUED, JobStatus.RUNNING]]),
            "completed_jobs": len([j for j in self.jobs.values() 
                                   if j.status == JobStatus.COMPLETED])
        }
        
        return health
    
    def get_metrics(self) -> str:
        """
        Get Prometheus-formatted metrics.
        
        Returns:
            Metrics in Prometheus text format
        """
        metrics = self.monitoring.get_prometheus_metrics()
        
        # Add controller-specific metrics
        lines = metrics.split("\n")
        lines.append(f"controller_tracked_jobs {len(self.jobs)}")
        lines.append(
            f"controller_active_jobs "
            f"{len([j for j in self.jobs.values() if j.status in [JobStatus.SUBMITTED, JobStatus.QUEUED, JobStatus.RUNNING]])}"
        )
        
        return "\n".join(lines)
    
    async def get_metrics_dict(self) -> Dict[str, Any]:
        """
        Get metrics as a dictionary (JSON-friendly).
        
        Returns:
            Metrics as dict
        """
        job_stats = {
            "total": len(self.jobs),
            "submitted": len([j for j in self.jobs.values() if j.status == JobStatus.SUBMITTED]),
            "running": len([j for j in self.jobs.values() if j.status == JobStatus.RUNNING]),
            "completed": len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]),
            "failed": len([j for j in self.jobs.values() if j.status == JobStatus.FAILED])
        }
        
        return {
            "jobs": job_stats,
            "metrics_raw": self.monitoring.metrics.get_metrics()
        }
    
    async def healthcheck_endpoint(self) -> Dict[str, Any]:
        """
        Endpoint for health checks (suitable for k8s probes).
        
        Returns:
            Health status dict
        """
        executor_health = await self.executor.health_check()
        system_health = await self.get_system_health()
        
        # Overall health is true if all components are healthy
        overall_healthy = (
            executor_health.get("healthy", False) and
            system_health.get("healthy", False)
        )
        
        return {
            "healthy": overall_healthy,
            "executor": executor_health,
            "system": system_health
        }


# ============================================================================
# USAGE EXAMPLES & REFERENCE
# ============================================================================

async def example_usage():
    """
    Example: How to use the controller with monitoring
    """
    
    # Initialize with mock executor (local development)
    from model_executor import MockModelExecutor
    executor = MockModelExecutor(latency_ms=50)
    controller = EngineControllerWithMonitoring(executor)
    
    # Example 1: Submit a fine-tune job
    job_id = await controller.submit_fine_tune_job(
        base_model="llama-2-7b",
        dataset_path="/data/train.jsonl",
        output_path="/models/finetuned_llama",
        epochs=5,
        batch_size=32,
        learning_rate=1e-4
    )
    print(f"Submitted job: {job_id}")
    
    # Example 2: Check job status
    for _ in range(5):
        status = await controller.get_job_status(job_id)
        print(f"Job status: {status['status']}")
        await asyncio.sleep(0.1)
    
    # Example 3: Get system health
    health = await controller.get_system_health()
    print(f"System health: {health['healthy']}")
    
    # Example 4: Get metrics
    metrics = controller.get_metrics()
    print(f"Metrics:\n{metrics}")
    
    # Example 5: Health check endpoint (for k8s)
    hc = await controller.healthcheck_endpoint()
    print(f"Health check: {hc['healthy']}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
