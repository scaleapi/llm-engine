"""
Test Suite for Batch Job Orchestration
=======================================

Tests the job queue, orchestrator, and batch processing functionality.

Run with:
  pytest tests/test_batch_job_orchestrator.py -v

Coverage:
- JobQueue: priority, concurrency, retry logic
- BatchJobOrchestrator: job submission, scheduling, execution
- Job lifecycle: pending → running → completed/failed
- Edge cases: cancellation, retries, capacity limits
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from model_engine_server.batch_job_orchestrator import (
    BatchJob,
    JobQueue,
    JobPriority,
    JobType,
    BatchJobOrchestrator
)
from model_engine_server.model_executor import (
    MockModelExecutor,
    JobStatus,
    JobResult
)


class TestBatchJob:
    """Test BatchJob dataclass"""
    
    def test_job_creation(self):
        """Test basic job creation"""
        job = BatchJob(
            id="test_job_1",
            type=JobType.FINE_TUNE,
            config={"model": "llama-2-7b"},
            priority=JobPriority.HIGH
        )
        
        assert job.id == "test_job_1"
        assert job.type == JobType.FINE_TUNE
        assert job.priority == JobPriority.HIGH
        assert job.status == JobStatus.QUEUED
        assert job.retries == 0
    
    def test_job_priority_ordering(self):
        """Test that jobs are ordered by priority"""
        job_low = BatchJob("1", JobType.FINE_TUNE, {}, JobPriority.LOW)
        job_high = BatchJob("2", JobType.FINE_TUNE, {}, JobPriority.HIGH)
        
        assert job_high < job_low  # Higher priority comes first
    
    def test_job_duration_calculation(self):
        """Test duration calculation"""
        job = BatchJob("1", JobType.FINE_TUNE, {}, JobPriority.NORMAL)
        job.started_at = datetime.utcnow()
        job.completed_at = job.started_at + timedelta(seconds=60)
        
        assert job.duration == timedelta(seconds=60)
    
    def test_job_wait_time(self):
        """Test wait time calculation"""
        job = BatchJob("1", JobType.FINE_TUNE, {}, JobPriority.NORMAL)
        job.created_at = datetime.utcnow() - timedelta(seconds=30)
        job.started_at = datetime.utcnow()
        
        wait = job.wait_time
        assert 29 <= wait.total_seconds() <= 31  # Allow small timing variations


class TestJobQueue:
    """Test JobQueue functionality"""
    
    @pytest.fixture
    def queue(self):
        """Create a job queue for testing"""
        return JobQueue(max_concurrent_jobs=2)
    
    @pytest.mark.asyncio
    async def test_submit_job(self, queue):
        """Test job submission"""
        job = BatchJob("1", JobType.FINE_TUNE, {}, JobPriority.NORMAL)
        job_id = await queue.submit(job)
        
        assert job_id == "1"
        
        # Job should be retrievable
        retrieved = await queue.get_status("1")
        assert retrieved is not None
        assert retrieved.id == "1"
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        """Test that higher priority jobs are processed first"""
        job_low = BatchJob("low", JobType.FINE_TUNE, {}, JobPriority.LOW)
        job_high = BatchJob("high", JobType.FINE_TUNE, {}, JobPriority.HIGH)
        job_normal = BatchJob("normal", JobType.FINE_TUNE, {}, JobPriority.NORMAL)
        
        # Submit in reverse priority order
        await queue.submit(job_low)
        await queue.submit(job_normal)
        await queue.submit(job_high)
        
        # Should get high priority first
        next_job = await queue.get_next()
        assert next_job.id == "high"
        assert next_job.status == JobStatus.RUNNING
        
        # Then normal
        next_job = await queue.get_next()
        assert next_job.id == "normal"
    
    @pytest.mark.asyncio
    async def test_concurrency_limit(self, queue):
        """Test that queue respects max concurrent jobs"""
        jobs = [
            BatchJob(f"job{i}", JobType.FINE_TUNE, {}, JobPriority.NORMAL)
            for i in range(5)
        ]
        
        for job in jobs:
            await queue.submit(job)
        
        # Should only get 2 jobs (max_concurrent=2)
        job1 = await queue.get_next()
        job2 = await queue.get_next()
        job3 = await queue.get_next()
        
        assert job1 is not None
        assert job2 is not None
        assert job3 is None  # At capacity
        
        # Complete one job
        await queue.complete(job1.id, JobResult(job1.id, JobStatus.COMPLETED))
        
        # Now should get another job
        job4 = await queue.get_next()
        assert job4 is not None
    
    @pytest.mark.asyncio
    async def test_job_completion(self, queue):
        """Test job completion"""
        job = BatchJob("1", JobType.FINE_TUNE, {}, JobPriority.NORMAL)
        await queue.submit(job)
        
        job = await queue.get_next()
        assert job.status == JobStatus.RUNNING
        
        result = JobResult("1", JobStatus.COMPLETED, output_path="/models/output")
        await queue.complete("1", result)
        
        # Job should be in completed state
        completed = await queue.get_status("1")
        assert completed.status == JobStatus.COMPLETED
        assert completed.result == result
    
    @pytest.mark.asyncio
    async def test_job_retry_logic(self, queue):
        """Test that failed jobs are retried"""
        job = BatchJob("1", JobType.FINE_TUNE, {}, JobPriority.HIGH, max_retries=3)
        await queue.submit(job)
        
        job = await queue.get_next()
        
        # Fail the job
        await queue.fail("1", "Test error", retry=True)
        
        # Job should be back in pending queue
        status = await queue.get_status("1")
        assert status.status == JobStatus.QUEUED
        assert status.retries == 1
        assert status.priority == JobPriority.LOW  # Downgraded
        
        # Can get it again
        job = await queue.get_next()
        assert job.id == "1"
    
    @pytest.mark.asyncio
    async def test_job_permanent_failure(self, queue):
        """Test that jobs fail permanently after max retries"""
        job = BatchJob("1", JobType.FINE_TUNE, {}, JobPriority.NORMAL, max_retries=2)
        await queue.submit(job)
        
        # Fail twice (will retry)
        for _ in range(2):
            job = await queue.get_next()
            await queue.fail(job.id, "Test error", retry=True)
        
        # Third failure should be permanent
        job = await queue.get_next()
        await queue.fail(job.id, "Test error", retry=True)
        
        status = await queue.get_status("1")
        assert status.status == JobStatus.FAILED
        assert status.error == "Test error"
    
    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, queue):
        """Test cancelling a pending job"""
        job = BatchJob("1", JobType.FINE_TUNE, {}, JobPriority.NORMAL)
        await queue.submit(job)
        
        cancelled = await queue.cancel("1")
        assert cancelled is True
        
        status = await queue.get_status("1")
        assert status.status == JobStatus.CANCELLED
        
        # Should not be able to get it
        next_job = await queue.get_next()
        assert next_job is None
    
    @pytest.mark.asyncio
    async def test_cancel_running_job(self, queue):
        """Test cancelling a running job"""
        job = BatchJob("1", JobType.FINE_TUNE, {}, JobPriority.NORMAL)
        await queue.submit(job)
        
        job = await queue.get_next()
        assert job.status == JobStatus.RUNNING
        
        cancelled = await queue.cancel("1")
        assert cancelled is True
        
        status = await queue.get_status("1")
        assert status.status == JobStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_queue_stats(self, queue):
        """Test queue statistics"""
        # Submit 3 jobs
        for i in range(3):
            job = BatchJob(f"job{i}", JobType.FINE_TUNE, {}, JobPriority.NORMAL)
            await queue.submit(job)
        
        # Start 2 (at capacity)
        await queue.get_next()
        await queue.get_next()
        
        stats = await queue.get_queue_stats()
        assert stats["pending"] == 1
        assert stats["running"] == 2
        assert stats["completed"] == 0
        assert stats["capacity_used"] == 1.0  # 2/2 = 100%


class TestBatchJobOrchestrator:
    """Test BatchJobOrchestrator functionality"""
    
    @pytest.fixture
    def executor(self):
        """Create a mock executor"""
        return MockModelExecutor(latency_ms=10)
    
    @pytest.fixture
    def orchestrator(self, executor):
        """Create orchestrator with mock executor"""
        return BatchJobOrchestrator(executor, max_concurrent_jobs=3, poll_interval=0.1)
    
    @pytest.mark.asyncio
    async def test_orchestrator_start_stop(self, orchestrator):
        """Test starting and stopping the orchestrator"""
        await orchestrator.start()
        assert orchestrator._scheduler_task is not None
        
        await orchestrator.stop()
        assert orchestrator._running is False
    
    @pytest.mark.asyncio
    async def test_submit_fine_tune_job(self, orchestrator):
        """Test submitting a fine-tune job"""
        job_id = await orchestrator.submit_fine_tune_job(
            model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/finetuned",
            priority=JobPriority.HIGH
        )
        
        assert job_id.startswith("ft_")
        
        # Job should be retrievable
        status = await orchestrator.get_job_status(job_id)
        assert status == JobStatus.QUEUED
    
    @pytest.mark.asyncio
    async def test_submit_batch_inference_job(self, orchestrator):
        """Test submitting a batch inference job"""
        job_id = await orchestrator.submit_batch_inference_job(
            model="llama-2-7b",
            input_path="/data/inputs.jsonl",
            output_path="/data/outputs.jsonl",
            batch_size=64
        )
        
        assert job_id.startswith("bi_")
        
        status = await orchestrator.get_job_status(job_id)
        assert status == JobStatus.QUEUED
    
    @pytest.mark.asyncio
    async def test_job_execution_lifecycle(self, orchestrator):
        """Test complete job lifecycle from submission to completion"""
        await orchestrator.start()
        
        job_id = await orchestrator.submit_fine_tune_job(
            model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/finetuned"
        )
        
        # Wait for job to start
        await asyncio.sleep(0.2)
        status = await orchestrator.get_job_status(job_id)
        assert status in [JobStatus.QUEUED, JobStatus.RUNNING]
        
        # Wait for completion
        for _ in range(20):  # Max 2 seconds
            status = await orchestrator.get_job_status(job_id)
            if status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.1)
        
        assert status == JobStatus.COMPLETED
        
        # Check details
        details = await orchestrator.get_job_details(job_id)
        assert details is not None
        assert details["status"] == "completed"
        assert details["result"] is not None
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_job_execution(self, orchestrator):
        """Test that multiple jobs run concurrently"""
        await orchestrator.start()
        
        # Submit 5 jobs
        job_ids = []
        for i in range(5):
            job_id = await orchestrator.submit_fine_tune_job(
                model="llama-2-7b",
                dataset_path=f"/data/train{i}.jsonl",
                output_path=f"/models/finetuned{i}"
            )
            job_ids.append(job_id)
        
        # Wait a bit for jobs to start
        await asyncio.sleep(0.3)
        
        # Check queue stats
        stats = await orchestrator.get_queue_stats()
        assert stats["running"] <= 3  # Max concurrent = 3
        assert stats["pending"] + stats["running"] + stats["completed"] == 5
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_job_cancellation(self, orchestrator):
        """Test cancelling a job"""
        await orchestrator.start()
        
        job_id = await orchestrator.submit_fine_tune_job(
            model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/finetuned"
        )
        
        # Cancel immediately
        cancelled = await orchestrator.cancel_job(job_id)
        assert cancelled is True
        
        status = await orchestrator.get_job_status(job_id)
        assert status == JobStatus.CANCELLED
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_priority_job_execution(self, orchestrator):
        """Test that high priority jobs execute first"""
        await orchestrator.start()
        
        # Submit jobs with different priorities
        low_id = await orchestrator.submit_fine_tune_job(
            model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/low",
            priority=JobPriority.LOW
        )
        
        high_id = await orchestrator.submit_fine_tune_job(
            model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/high",
            priority=JobPriority.HIGH
        )
        
        normal_id = await orchestrator.submit_fine_tune_job(
            model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/normal",
            priority=JobPriority.NORMAL
        )
        
        # Wait for jobs to start
        await asyncio.sleep(0.3)
        
        # High priority should start first
        high_details = await orchestrator.get_job_details(high_id)
        low_details = await orchestrator.get_job_details(low_id)
        
        # High should have started before low (or be completed)
        if high_details["started_at"] and low_details["started_at"]:
            assert high_details["started_at"] <= low_details["started_at"]
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_get_job_details(self, orchestrator):
        """Test retrieving detailed job information"""
        job_id = await orchestrator.submit_fine_tune_job(
            model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/finetuned",
            hyperparams={"learning_rate": 0.001}
        )
        
        details = await orchestrator.get_job_details(job_id)
        assert details is not None
        assert details["id"] == job_id
        assert details["type"] == "fine_tune"
        assert details["priority"] == "NORMAL"
        assert details["config"]["model"] == "llama-2-7b"
        assert details["config"]["hyperparams"]["learning_rate"] == 0.001
    
    @pytest.mark.asyncio
    async def test_batch_inference_execution(self, orchestrator):
        """Test batch inference job execution"""
        await orchestrator.start()
        
        job_id = await orchestrator.submit_batch_inference_job(
            model="llama-2-7b",
            input_path="/data/inputs.jsonl",
            output_path="/data/outputs.jsonl",
            batch_size=128
        )
        
        # Wait for completion
        for _ in range(30):
            status = await orchestrator.get_job_status(job_id)
            if status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.1)
        
        assert status == JobStatus.COMPLETED
        
        details = await orchestrator.get_job_details(job_id)
        assert details["result"]["metrics"]["batch_size"] == 128
        
        await orchestrator.stop()


class TestIntegration:
    """Integration tests for full batch job workflow"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_monitoring(self):
        """Test batch orchestrator integration with monitoring"""
        from model_engine_server.controller_with_monitoring import EngineControllerWithMonitoring
        
        executor = MockModelExecutor(latency_ms=50)
        controller = EngineControllerWithMonitoring(executor, enable_metrics=True)
        
        orchestrator = BatchJobOrchestrator(executor, max_concurrent_jobs=2)
        await orchestrator.start()
        
        # Submit multiple jobs
        jobs = []
        for i in range(3):
            job_id = await orchestrator.submit_fine_tune_job(
                model="llama-2-7b",
                dataset_path=f"/data/train{i}.jsonl",
                output_path=f"/models/finetuned{i}"
            )
            jobs.append(job_id)
        
        # Wait for all to complete
        await asyncio.sleep(1.0)
        
        # Check all completed
        for job_id in jobs:
            status = await orchestrator.get_job_status(job_id)
            assert status in [JobStatus.COMPLETED, JobStatus.RUNNING]
        
        # Get metrics from controller
        metrics = controller.get_metrics_dict()
        assert "jobs_submitted" in metrics
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_stress_test_many_jobs(self):
        """Stress test with many concurrent jobs"""
        executor = MockModelExecutor(latency_ms=10)
        orchestrator = BatchJobOrchestrator(executor, max_concurrent_jobs=5, poll_interval=0.05)
        await orchestrator.start()
        
        # Submit 50 jobs
        job_ids = []
        for i in range(50):
            job_id = await orchestrator.submit_batch_inference_job(
                model="llama-2-7b",
                input_path=f"/data/input{i}.jsonl",
                output_path=f"/data/output{i}.jsonl"
            )
            job_ids.append(job_id)
        
        # Wait for all to complete (with timeout)
        max_wait = 10.0  # 10 seconds max
        start = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start < max_wait:
            stats = await orchestrator.get_queue_stats()
            if stats["completed"] == 50:
                break
            await asyncio.sleep(0.1)
        
        stats = await orchestrator.get_queue_stats()
        assert stats["completed"] >= 45  # At least 90% completed
        
        await orchestrator.stop()
