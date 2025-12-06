"""
Abstract Model Executor Interface
==================================

This module defines the interface for model execution (inference, fine-tuning, etc.)
Enables dependency injection for testing (mocks) vs production (Kubernetes).

Design Pattern: Strategy + Dependency Injection
- Separates controller logic from execution implementation
- Allows local testing without GPU
- Supports multiple backends (Mock, Kubernetes, Ray, etc.)
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of an async job"""
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Result of a completed job"""
    job_id: str
    status: JobStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


@dataclass
class InferenceRequest:
    """Request for model inference"""
    prompt: str
    model: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    timeout_seconds: int = 30


@dataclass
class FineTuneRequest:
    """Request to fine-tune a model"""
    base_model: str
    dataset_path: str
    output_path: str
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    timeout_seconds: int = 3600  # 1 hour default


class ModelExecutor(ABC):
    """
    Abstract base class for model execution backends.
    
    Implementations:
    - MockModelExecutor: for local testing (instant, deterministic)
    - KubernetesModelExecutor: for production (runs on K8s)
    - RayModelExecutor: for distributed execution (future)
    """

    @abstractmethod
    async def infer(self, request: InferenceRequest) -> str:
        """
        Run inference on a model.
        
        Args:
            request: InferenceRequest with prompt, model, hyperparams
            
        Returns:
            Generated text output
            
        Raises:
            TimeoutError: If inference exceeds timeout
            ValueError: If model or prompt invalid
        """
        pass

    @abstractmethod
    async def fine_tune(self, request: FineTuneRequest) -> str:
        """
        Submit a fine-tuning job.
        
        Args:
            request: FineTuneRequest with dataset, model, hyperparams
            
        Returns:
            Job ID for tracking progress
        """
        pass

    @abstractmethod
    async def get_job_status(self, job_id: str) -> JobResult:
        """
        Get the status and result of a job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            JobResult with current status and output (if completed)
        """
        pass

    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            True if cancellation succeeded
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check executor health and dependencies.
        
        Returns:
            Dict with health status:
            {
                "healthy": bool,
                "services": {
                    "models": {"available": int},
                    "compute": {"available_gpus": int},
                    "storage": {"available_gb": int}
                },
                "errors": [list of errors if any]
            }
        """
        pass


class MockModelExecutor(ModelExecutor):
    """
    Mock executor for local testing.
    
    Features:
    - Instant responses (simulated with small delay)
    - Deterministic outputs
    - In-memory job tracking
    - No GPU required
    
    Use this for:
    - Local development
    - Unit tests
    - CI/CD on CPU machines
    """

    def __init__(self, latency_ms: int = 100, fail_rate: float = 0.0):
        """
        Initialize mock executor.
        
        Args:
            latency_ms: Simulated latency in milliseconds
            fail_rate: Probability (0.0-1.0) that jobs fail (for chaos testing)
        """
        self.latency_ms = latency_ms
        self.fail_rate = fail_rate
        self.jobs: Dict[str, JobResult] = {}
        logger.info(
            f"MockModelExecutor initialized (latency={latency_ms}ms, fail_rate={fail_rate})"
        )

    async def infer(self, request: InferenceRequest) -> str:
        """Simulate inference with mock output"""
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Mock output based on prompt
        output = f"[Mock inference for model '{request.model}']\n"
        output += f"Prompt: {request.prompt[:50]}...\n"
        output += f"Generated: This is a mock response with {request.max_tokens} max tokens.\n"
        output += f"Temperature: {request.temperature}, Top-p: {request.top_p}"
        
        logger.info(f"Mock inference completed for model '{request.model}'")
        return output

    async def fine_tune(self, request: FineTuneRequest) -> str:
        """Create a mock fine-tuning job"""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Simulate immediate job creation
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        import random
        status = (
            JobStatus.FAILED if random.random() < self.fail_rate
            else JobStatus.SUBMITTED
        )
        
        self.jobs[job_id] = JobResult(
            job_id=job_id,
            status=status,
            output={
                "base_model": request.base_model,
                "dataset_path": request.dataset_path,
                "epochs": request.epochs,
                "batch_size": request.batch_size
            } if status != JobStatus.FAILED else None,
            error="Mock failure" if status == JobStatus.FAILED else None
        )
        
        logger.info(f"Mock fine-tune job submitted: {job_id} (status={status.value})")
        return job_id

    async def get_job_status(self, job_id: str) -> JobResult:
        """Get status of a mock job"""
        if job_id not in self.jobs:
            return JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                error=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        # Simulate job progression
        if job.status == JobStatus.SUBMITTED:
            job.status = JobStatus.QUEUED
        elif job.status == JobStatus.QUEUED:
            job.status = JobStatus.RUNNING
        elif job.status == JobStatus.RUNNING:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.duration_seconds = (
                job.completed_at - job.created_at
            ).total_seconds()
        
        logger.debug(f"Job {job_id} status: {job.status.value}")
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a mock job"""
        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.CANCELLED
            logger.info(f"Job {job_id} cancelled")
            return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Return mock health status"""
        return {
            "healthy": True,
            "executor_type": "mock",
            "services": {
                "models": {"available": 5, "supported": ["llama-2-7b", "falcon-7b"]},
                "compute": {"available_gpus": 0, "available_cpus": 4},
                "storage": {"available_gb": 100}
            },
            "errors": []
        }


class KubernetesModelExecutor(ModelExecutor):
    """
    Production executor using Kubernetes.
    
    Features:
    - Submits jobs to Kubernetes
    - GPU-aware scheduling
    - Distributed execution
    - Job persistence
    
    Use this for:
    - Production deployments
    - Cloud infrastructure
    - GPU workloads
    
    Note: Requires Kubernetes cluster and model-engine deployment.
    """

    def __init__(self, namespace: str = "default", timeout_seconds: int = 3600):
        """
        Initialize Kubernetes executor.
        
        Args:
            namespace: Kubernetes namespace for jobs
            timeout_seconds: Default job timeout
        """
        self.namespace = namespace
        self.timeout_seconds = timeout_seconds
        logger.info(f"KubernetesModelExecutor initialized (namespace={namespace})")

    async def infer(self, request: InferenceRequest) -> str:
        """Submit inference job to Kubernetes"""
        # TODO: Implement K8s API call to create inference job
        raise NotImplementedError(
            "KubernetesModelExecutor requires Kubernetes cluster setup"
        )

    async def fine_tune(self, request: FineTuneRequest) -> str:
        """Submit fine-tuning job to Kubernetes"""
        # TODO: Implement K8s API call to create fine-tune job
        raise NotImplementedError(
            "KubernetesModelExecutor requires Kubernetes cluster setup"
        )

    async def get_job_status(self, job_id: str) -> JobResult:
        """Get status from Kubernetes"""
        # TODO: Implement K8s API call to query job status
        raise NotImplementedError(
            "KubernetesModelExecutor requires Kubernetes cluster setup"
        )

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel job on Kubernetes"""
        # TODO: Implement K8s API call to cancel job
        raise NotImplementedError(
            "KubernetesModelExecutor requires Kubernetes cluster setup"
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check Kubernetes cluster health"""
        # TODO: Implement K8s cluster health check
        raise NotImplementedError(
            "KubernetesModelExecutor requires Kubernetes cluster setup"
        )
