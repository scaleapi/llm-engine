"""
Batch Job Orchestration System
===============================

Handles fine-tuning jobs and batch inference with:
- Job submission to Kubernetes
- Queue management and prioritization
- Status tracking and result retrieval
- Resource allocation and cleanup

This builds on top of the ModelExecutor interface from monitoring feature.

Architecture:
- JobQueue: Priority-based queue for pending jobs
- BatchJobOrchestrator: Main orchestration logic
- KubernetesJobExecutor: Submits jobs to K8s cluster
- ResultStore: Stores and retrieves job results

Usage:
    orchestrator = BatchJobOrchestrator(
        executor=KubernetesModelExecutor(),
        max_concurrent_jobs=10
    )
    
    job_id = await orchestrator.submit_job(
        job_type="fine_tune",
        config={
            "model": "llama-2-7b",
            "dataset": "s3://bucket/data.jsonl",
            "output": "s3://bucket/models/finetuned"
        },
        priority=JobPriority.HIGH
    )
    
    status = await orchestrator.get_job_status(job_id)
    results = await orchestrator.get_job_results(job_id)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from collections import defaultdict
import heapq
import uuid

from model_engine_server.model_executor import (
    ModelExecutor,
    FineTuneRequest,
    InferenceRequest,
    JobStatus,
    JobResult
)


logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class JobType(Enum):
    """Types of batch jobs"""
    FINE_TUNE = "fine_tune"
    BATCH_INFERENCE = "batch_inference"
    MODEL_EVALUATION = "model_evaluation"


@dataclass
class BatchJob:
    """Represents a batch job in the system"""
    id: str
    type: JobType
    config: Dict[str, Any]
    priority: JobPriority
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[JobResult] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first)"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Time taken to complete the job"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def wait_time(self) -> timedelta:
        """Time spent waiting in queue"""
        start = self.started_at or datetime.utcnow()
        return start - self.created_at


class JobQueue:
    """Priority-based job queue with concurrency control"""
    
    def __init__(self, max_concurrent_jobs: int = 10):
        self.max_concurrent = max_concurrent_jobs
        self._pending: List[BatchJob] = []
        self._running: Dict[str, BatchJob] = {}
        self._completed: Dict[str, BatchJob] = {}
        self._lock = asyncio.Lock()
    
    async def submit(self, job: BatchJob) -> str:
        """Add a job to the queue"""
        async with self._lock:
            heapq.heappush(self._pending, job)
            logger.info(f"Job {job.id} added to queue (priority={job.priority.name})")
            return job.id
    
    async def get_next(self) -> Optional[BatchJob]:
        """Get the next highest-priority job if capacity allows"""
        async with self._lock:
            if len(self._running) >= self.max_concurrent:
                return None
            
            if not self._pending:
                return None
            
            job = heapq.heappop(self._pending)
            self._running[job.id] = job
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            logger.info(f"Job {job.id} started (queue_wait={job.wait_time})")
            return job
    
    async def complete(self, job_id: str, result: JobResult, error: Optional[str] = None):
        """Mark a job as completed"""
        async with self._lock:
            if job_id not in self._running:
                raise ValueError(f"Job {job_id} not in running state")
            
            job = self._running.pop(job_id)
            job.status = JobStatus.COMPLETED if not error else JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.result = result
            job.error = error
            self._completed[job_id] = job
            logger.info(f"Job {job_id} completed (duration={job.duration})")
    
    async def fail(self, job_id: str, error: str, retry: bool = True):
        """Mark a job as failed and optionally retry"""
        async with self._lock:
            if job_id not in self._running:
                raise ValueError(f"Job {job_id} not in running state")
            
            job = self._running.pop(job_id)
            job.retries += 1
            
            if retry and job.retries < job.max_retries:
                # Retry with lower priority
                job.status = JobStatus.QUEUED
                job.priority = JobPriority.LOW
                heapq.heappush(self._pending, job)
                logger.warning(f"Job {job_id} failed, retrying ({job.retries}/{job.max_retries}): {error}")
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error = error
                self._completed[job_id] = job
                logger.error(f"Job {job_id} failed permanently: {error}")
    
    async def get_status(self, job_id: str) -> Optional[BatchJob]:
        """Get job status"""
        async with self._lock:
            if job_id in self._running:
                return self._running[job_id]
            if job_id in self._completed:
                return self._completed[job_id]
            for job in self._pending:
                if job.id == job_id:
                    return job
            return None
    
    async def cancel(self, job_id: str) -> bool:
        """Cancel a job"""
        async with self._lock:
            # Remove from pending
            for i, job in enumerate(self._pending):
                if job.id == job_id:
                    job = self._pending.pop(i)
                    heapq.heapify(self._pending)
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.utcnow()
                    self._completed[job_id] = job
                    logger.info(f"Job {job_id} cancelled (was pending)")
                    return True
            
            # Mark running as cancelled (executor must handle cleanup)
            if job_id in self._running:
                job = self._running[job_id]
                job.status = JobStatus.CANCELLED
                logger.info(f"Job {job_id} marked for cancellation (was running)")
                return True
            
            return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "pending": len(self._pending),
                "running": len(self._running),
                "completed": len(self._completed),
                "capacity_used": len(self._running) / self.max_concurrent,
                "average_wait_time": self._calculate_avg_wait_time()
            }
    
    def _calculate_avg_wait_time(self) -> float:
        """Calculate average wait time for completed jobs"""
        if not self._completed:
            return 0.0
        
        wait_times = [
            job.wait_time.total_seconds()
            for job in self._completed.values()
            if job.wait_time
        ]
        return sum(wait_times) / len(wait_times) if wait_times else 0.0


class BatchJobOrchestrator:
    """Main orchestrator for batch jobs"""
    
    def __init__(
        self,
        executor: ModelExecutor,
        max_concurrent_jobs: int = 10,
        poll_interval: float = 0.1
    ):
        self.executor = executor
        self.queue = JobQueue(max_concurrent_jobs)
        self.poll_interval = poll_interval
        self._running = True
        self._scheduler_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the orchestrator"""
        if self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("Batch job orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Batch job orchestrator stopped")
    
    async def submit_fine_tune_job(
        self,
        model: str,
        dataset_path: str,
        output_path: str,
        hyperparams: Optional[Dict[str, Any]] = None,
        priority: JobPriority = JobPriority.NORMAL
    ) -> str:
        """Submit a fine-tuning job"""
        job = BatchJob(
            id=f"ft_{uuid.uuid4().hex[:12]}",
            type=JobType.FINE_TUNE,
            config={
                "model": model,
                "dataset_path": dataset_path,
                "output_path": output_path,
                "hyperparams": hyperparams or {}
            },
            priority=priority
        )
        return await self.queue.submit(job)
    
    async def submit_batch_inference_job(
        self,
        model: str,
        input_path: str,
        output_path: str,
        batch_size: int = 32,
        priority: JobPriority = JobPriority.NORMAL
    ) -> str:
        """Submit a batch inference job"""
        job = BatchJob(
            id=f"bi_{uuid.uuid4().hex[:12]}",
            type=JobType.BATCH_INFERENCE,
            config={
                "model": model,
                "input_path": input_path,
                "output_path": output_path,
                "batch_size": batch_size
            },
            priority=priority
        )
        return await self.queue.submit(job)
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status"""
        job = await self.queue.get_status(job_id)
        return job.status if job else None
    
    async def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed job information"""
        job = await self.queue.get_status(job_id)
        if not job:
            return None
        
        return {
            "id": job.id,
            "type": job.type.value,
            "status": job.status.value,
            "priority": job.priority.name,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "duration": str(job.duration) if job.duration else None,
            "wait_time": str(job.wait_time),
            "config": job.config,
            "result": job.result.__dict__ if job.result else None,
            "error": job.error,
            "retries": job.retries
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        cancelled = await self.queue.cancel(job_id)
        if cancelled:
            # Also tell executor to cancel if it's running
            try:
                await self.executor.cancel_job(job_id)
            except Exception as e:
                logger.warning(f"Failed to cancel job in executor: {e}")
        return cancelled
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return await self.queue.get_queue_stats()
    
    async def _scheduler_loop(self):
        """Background scheduler that processes jobs"""
        while self._running:
            try:
                # Get next job from queue
                job = await self.queue.get_next()
                
                if job:
                    # Execute job in background
                    asyncio.create_task(self._execute_job(job))
                    # Short sleep to allow task to start
                    await asyncio.sleep(0.01)
                else:
                    # No jobs or at capacity, wait
                    await asyncio.sleep(self.poll_interval)
            
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def _execute_job(self, job: BatchJob):
        """Execute a single job"""
        try:
            if job.type == JobType.FINE_TUNE:
                result = await self._execute_fine_tune(job)
            elif job.type == JobType.BATCH_INFERENCE:
                result = await self._execute_batch_inference(job)
            else:
                raise ValueError(f"Unknown job type: {job.type}")
            
            await self.queue.complete(job.id, result)
        
        except Exception as e:
            logger.error(f"Job {job.id} execution failed: {e}")
            await self.queue.fail(job.id, str(e), retry=True)
    
    async def _execute_fine_tune(self, job: BatchJob) -> JobResult:
        """Execute a fine-tuning job"""
        config = job.config
        request = FineTuneRequest(
            base_model=config["model"],
            dataset_path=config["dataset_path"],
            output_path=config["output_path"],
            epochs=config.get("epochs", 3),
            batch_size=config.get("batch_size", 32),
            learning_rate=config.get("learning_rate", 1e-4)
        )
        
        # Submit to executor
        executor_job_id = await self.executor.fine_tune(request)
        
        # Poll for completion
        while True:
            status = await self.executor.get_job_status(executor_job_id)
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                break
            
            await asyncio.sleep(self.poll_interval)
        
        # Get result
        if status == JobStatus.COMPLETED:
            return JobResult(
                job_id=job.id,
                status=JobStatus.COMPLETED,
                output={"output_path": config["output_path"], "executor_job_id": executor_job_id}
            )
        else:
            raise Exception(f"Job failed with status: {status}")
    
    async def _execute_batch_inference(self, job: BatchJob) -> JobResult:
        """Execute a batch inference job"""
        # Placeholder - would implement batch processing logic
        config = job.config
        
        # Simulate batch processing
        await asyncio.sleep(2)  # Mock processing time
        
        return JobResult(
            job_id=job.id,
            status=JobStatus.COMPLETED,
            output={"output_path": config["output_path"], "processed_items": 100, "batch_size": config["batch_size"]}
        )
